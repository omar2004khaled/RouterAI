import os
import sys
import csv
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

logger = logging.getLogger(__name__)

ACTIONS      = ["notify", "digest", "mute"]
MSG_TYPES    = [
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
]


def _load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for r in reader:
            if not r or not r.get("message_id", "").strip():
                continue
            rows.append(r)
    return rows


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def evaluate_predictions(
    sample_path: str = config.sample_messages_path,
    output_path: str = config.output_path,
    error_analysis: bool = False
) -> Dict[str, float]:

    if not os.path.exists(sample_path) or not os.path.exists(output_path):
        logger.error("Ground-truth or prediction file missing — skipping evaluation.")
        return {}

    gt_rows   = _load_csv(sample_path)
    pred_rows = _load_csv(output_path)

    gt   = {r["message_id"]: r for r in gt_rows}
    pred = {r["message_id"]: r for r in pred_rows}
    ids  = [mid for mid in gt if mid in pred]
    n    = len(ids)
    if n == 0:
        logger.warning("No overlapping message IDs between GT and predictions.")
        return {}

    # ── Per-class tallies ──
    action_tp  = defaultdict(int)
    action_fp  = defaultdict(int)
    action_fn  = defaultdict(int)
    type_tp    = defaultdict(int)
    type_fp    = defaultdict(int)
    type_fn    = defaultdict(int)

    # ── Confusion matrices (action 3×3, type 11×11) ──
    a_idx = {a: i for i, a in enumerate(ACTIONS)}
    t_idx = {t: i for i, t in enumerate(MSG_TYPES)}
    cm_action = [[0]*len(ACTIONS)    for _ in ACTIONS]
    cm_type   = [[0]*len(MSG_TYPES)  for _ in MSG_TYPES]

    correct_action   = 0
    correct_type     = 0
    correct_evidence = 0
    confidences      = []
    errors           = []          # (mid, gt_action, pred_action, gt_type, pred_type, reason)

    for mid in ids:
        g = gt[mid]
        p = pred[mid]

        ga = g["action"].strip().lower()
        pa = p["action"].strip().lower()
        gt_ = g["message_type"].strip().lower()
        pt_ = p["message_type"].strip().lower()

        # Action accuracy
        if ga == pa:
            correct_action += 1
            action_tp[ga] += 1
        else:
            action_fp[pa] += 1
            action_fn[ga] += 1
            errors.append((mid, ga, pa, gt_, pt_, p.get("reason", "")))

        # Message-type accuracy
        if gt_ == pt_:
            correct_type += 1
            type_tp[gt_] += 1
        else:
            type_fp[pt_] += 1
            type_fn[gt_] += 1

        # Confusion matrices
        gi_a = a_idx.get(ga, 0);  pi_a = a_idx.get(pa, 0)
        gi_t = t_idx.get(gt_, len(MSG_TYPES)-1)
        pi_t = t_idx.get(pt_,  len(MSG_TYPES)-1)
        cm_action[gi_a][pi_a] += 1
        cm_type[gi_t][pi_t]   += 1

        # Evidence accuracy — partial credit: any GT evidence ID appears in prediction
        ge = set(g.get("evidence_message_ids", "none").split(";")) - {"none", ""}
        pe = set(p.get("evidence_message_ids", "none").split(";")) - {"none", ""}
        if ge and pe and ge.intersection(pe):
            correct_evidence += 1
        elif not ge and not pe:
            correct_evidence += 1

        try:
            confidences.append(float(p["confidence"]))
        except (ValueError, KeyError):
            pass

    action_acc   = correct_action   / n * 100
    type_acc     = correct_type     / n * 100
    evidence_acc = correct_evidence / n * 100
    conf_arr     = np.array(confidences) if confidences else np.array([0.0])
    conf_mean    = float(conf_arr.mean())
    conf_std     = float(conf_arr.std())

    # Per-class metrics
    action_metrics = {}
    for a in ACTIONS:
        tp = action_tp[a]; fp = action_fp[a]; fn = action_fn[a]
        p_, r_, f1_ = _precision_recall_f1(tp, fp, fn)
        action_metrics[a] = {"precision": p_, "recall": r_, "f1": f1_}

    type_metrics = {}
    for t in MSG_TYPES:
        tp = type_tp[t]; fp = type_fp[t]; fn = type_fn[t]
        p_, r_, f1_ = _precision_recall_f1(tp, fp, fn)
        type_metrics[t] = {"precision": p_, "recall": r_, "f1": f1_}

    # ── Print report ──
    sep = "=" * 64
    print(sep)
    print("   WhatsApp Notification Router - Evaluation Report")
    print(sep)
    print(f"  Samples evaluated        : {n}")
    print(f"  Action accuracy          : {action_acc:.2f}%  ({correct_action}/{n})")
    print(f"  Message-type accuracy    : {type_acc:.2f}%  ({correct_type}/{n})")
    print(f"  Evidence accuracy        : {evidence_acc:.2f}%  ({correct_evidence}/{n})")
    print(f"  Confidence mean          : {conf_mean:.3f}")
    print(f"  Confidence std dev       : {conf_std:.3f}")
    print()
    print("  Action-level Precision / Recall / F1")
    for a, m in action_metrics.items():
        print(f"    {a:<10}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
    print()
    print("  Confidence histogram (10 buckets 0.0-1.0)")
    hist, edges = np.histogram(conf_arr, bins=10, range=(0.0, 1.0))
    max_cnt = max(hist.max(), 1)
    for i, cnt in enumerate(hist):
        bar = '#' * int(cnt * 30 / max_cnt)
        lo, hi = f"{edges[i]:.1f}", f"{edges[i+1]:.1f}"
        print(f"    [{lo}-{hi}]  {bar}  ({cnt})")
    print(sep)

    # ── Error analysis markdown ──
    if error_analysis and errors:
        _write_error_analysis(errors, n, action_acc, type_acc)

    return {
        "n": n,
        "action_accuracy": action_acc,
        "type_accuracy": type_acc,
        "evidence_accuracy": evidence_acc,
        "confidence_mean": conf_mean,
        "confidence_std": conf_std,
        "action_metrics": action_metrics,
        "type_metrics": type_metrics,
    }


def _write_error_analysis(errors: list, n: int, action_acc: float, type_acc: float):
    lines = [
        "# Error Analysis Report\n",
        f"- Total samples evaluated: **{n}**\n",
        f"- Action accuracy: **{action_acc:.2f}%**\n",
        f"- Message-type accuracy: **{type_acc:.2f}%**\n",
        f"- Error count: **{len(errors)}**\n\n",
        "## Misclassified Messages\n\n",
        "| message_id | GT action | Pred action | GT type | Pred type | Pred reason |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for mid, ga, pa, gt_, pt_, reason in errors[:50]:
        lines.append(f"| `{mid}` | {ga} | {pa} | {gt_} | {pt_} | {reason[:80]} |\n")

    # Action confusion breakdown
    from collections import Counter
    action_errors = Counter((ga, pa) for mid, ga, pa, *_ in errors)
    lines += [
        "\n## Action Confusion Pairs\n\n",
        "| GT action | Pred action | Count |\n|---|---|---|\n",
    ]
    for (ga, pa), cnt in action_errors.most_common(20):
        lines.append(f"| {ga} | {pa} | {cnt} |\n")

    out_path = os.path.join("code", "ERROR_ANALYSIS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info(f"Error analysis written to {out_path}")


if __name__ == "__main__":
    evaluate_predictions(error_analysis=True)
