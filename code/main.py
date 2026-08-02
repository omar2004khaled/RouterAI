"""
WhatsApp Message Notification Router — entry point.

Run from the repo root:
    python code/main.py
    python code/main.py --eval
    python code/main.py --error-analysis

All pipeline modules live flat in code/ alongside this file.
There is intentionally no code/__init__.py: Python's stdlib has a module
called 'code' that is required by torch/easyocr/faster-whisper internally,
and adding __init__.py here would shadow it and crash at import time.
Instead, code/ is added to sys.path so its modules are importable directly.
"""

import os
import sys
import csv
import argparse
import logging
import time
import warnings

# Suppress non-critical library warnings
warnings.filterwarnings("ignore", message="pin_memory")
warnings.filterwarnings("ignore", message="symlinks")
warnings.filterwarnings("ignore", message="huggingface_hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Add code/ to sys.path so sibling modules are importable without a package prefix.
# Add repo root too so dataset/ relative paths resolve from the working directory.
_this_dir  = os.path.dirname(os.path.abspath(__file__))   # …/code/
_repo_root = os.path.dirname(_this_dir)                    # …/Hackerrank/
for _p in (_this_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import config
from utils import setup_logging
from models import RawMessage
from context_builder import ContextBuilder
from feature_engineering import FeatureExtractor
from retriever import EvidenceRetriever
from rule_engine import RuleEngine
from reasoning_engine import ReasoningEngine
from confidence import ConfidenceCalibrator
from csv_writer import CSVWriter
from evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="WhatsApp Message Notification Router")
    parser.add_argument("--dataset-dir",    type=str, default=config.dataset_dir)
    parser.add_argument("--output-path",    type=str, default=config.output_path)
    parser.add_argument("--eval",           action="store_true",
                        help="Run evaluation benchmark after routing")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Generate ERROR_ANALYSIS.md")
    return parser.parse_args()


def load_input_messages(filepath: str):
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                fwd = int(r.get("forwarded_count", "0").strip())
            except ValueError:
                fwd = 0
            messages.append(RawMessage(
                message_id       = r.get("message_id",       "").strip(),
                user_id          = r.get("user_id",          "").strip(),
                conversation_type= r.get("conversation_type","").strip(),
                group_id         = r.get("group_id",         "").strip(),
                business_id      = r.get("business_id",      "").strip(),
                sender_user_id   = r.get("sender_user_id",   "").strip(),
                created_at       = r.get("created_at",       "").strip(),
                message_text     = r.get("message_text",     "").strip(),
                media_type       = r.get("media_type",       "").strip(),
                media_id         = r.get("media_id",         "").strip(),
                forwarded_count  = fwd,
            ))
    return messages


def main():
    setup_logging()
    args = parse_args()

    messages_path = os.path.join(args.dataset_dir, "messages.csv")
    if not os.path.exists(messages_path):
        logger.error(f"Required dataset file not found: {messages_path}")
        sys.exit(1)

    # ── Initialise pipeline (loaded once, reused per message) ─────────────────
    logger.info("Initialising pipeline modules …")
    t0      = time.perf_counter()
    builder = ContextBuilder(dataset_dir=args.dataset_dir)
    builder_elapsed = time.perf_counter() - t0

    feature_extractor = FeatureExtractor()
    retriever         = EvidenceRetriever()
    rule_engine       = RuleEngine()
    reasoning_engine  = ReasoningEngine()
    calibrator        = ConfidenceCalibrator()
    writer            = CSVWriter(output_path=args.output_path)

    # ── Load messages ──────────────────────────────────────────────────────────
    input_messages = load_input_messages(messages_path)
    logger.info(f"Loaded {len(input_messages)} messages to route.")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    decisions = []
    timing = {
        "context_builder":     builder_elapsed,
        "feature_engineering": 0.0,
        "retriever":           0.0,
        "rule_engine":         0.0,
        "reasoning":           0.0,
        "confidence":          0.0,
        "csv_writer":          0.0,
    }

    t_start = time.perf_counter()
    for msg in input_messages:
        ctx = builder.build_context(msg)

        t = time.perf_counter()
        fs = feature_extractor.extract(ctx)
        timing["feature_engineering"] += time.perf_counter() - t

        t = time.perf_counter()
        ev_ids = retriever.retrieve(ctx, top_k=3)
        timing["retriever"] += time.perf_counter() - t
        ev_str = ";".join(ev_ids) if ev_ids else "none"

        t = time.perf_counter()
        decision   = rule_engine.evaluate(ctx, fs, ev_str)
        from_rule  = decision is not None
        rule_label = decision.reason if from_rule else "No pre-rule signal"
        timing["rule_engine"] += time.perf_counter() - t

        if not from_rule:
            t = time.perf_counter()
            decision = reasoning_engine.evaluate(ctx, fs, ev_ids, rule_signal=rule_label)
            timing["reasoning"] += time.perf_counter() - t

        t = time.perf_counter()
        calibrated = calibrator.calibrate(decision, ctx, fs, from_rule)
        timing["confidence"] += time.perf_counter() - t
        decisions.append(calibrated)

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Routed {len(decisions)} messages in {elapsed:.2f}s "
        f"({len(decisions)/max(elapsed, 0.001):.0f} msg/s)."
    )

    t = time.perf_counter()
    writer.write_decisions(decisions)
    timing["csv_writer"] = time.perf_counter() - t
    logger.info("Timing: " + ", ".join(f"{k}={v:.3f}s" for k, v in timing.items()))

    # ── Optional evaluation (only with --eval flag) ────────────────────────────
    if args.eval:
        sample_path = os.path.join(args.dataset_dir, "sample_messages.csv")
        evaluate_predictions(
            sample_path=sample_path,
            output_path=args.output_path,
            error_analysis=args.error_analysis,
        )


if __name__ == "__main__":
    main()
