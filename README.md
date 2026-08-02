# WhatsApp Message Notification Router

HackerRank Orchestrate — August 2026 submission.

---

## What This System Does

For every message in `dataset/messages.csv` the pipeline decides:

| Action | Meaning |
|--------|---------|
| `notify` | Interrupt the user now |
| `digest` | Batch for later review |
| `mute` | Suppress (spam, scam, opted-out, low-value) |

Decisions are personalised using the user's notification history, business relationships, group membership, prior message-event reactions, OCR from image attachments, and speech transcription from voice notes.

---

## Architecture

```
messages.csv
    │
    ▼
ContextBuilder          loads all relational tables once:
                        users, groups, group_members,
                        businesses, user_business_history,
                        message_history + message_events,
                        images, voice_notes, daily_summary
    │
    ▼
MediaProcessor          OCR on images  (EasyOCR → PIL fallback)
                        ASR on voice   (faster-whisper-small)
    │
    ▼
FeatureExtractor        ~100 engineered signals:
                        user engagement rates, domain trust,
                        group mute state, quiet-hours,
                        scam/urgency/promotion keyword scores,
                        media category, prior reaction history
    │
    ▼
EvidenceRetriever       multi-signal scoring of message_history:
                        relationship (sender/business/group) ×
                        user reactions (replied/opened/dismissed/reported) ×
                        TF-IDF cosine similarity × recency decay
                        → diverse top-3 evidence IDs
    │
    ▼
RuleEngine              90 deterministic rules (scam detection,
                        domain mismatch, business category routing,
                        quiet hours, opt-out, urgency overrides)
    │
    ▼ (fallthrough only)
ReasoningEngine         Gemini-flash prompt with structured context
                        (skipped if no API key → heuristic fallback)
    │
    ▼
ConfidenceCalibrator    agreement/conflict model:
                        verified + history + evidence + fast-replies → high
                        mixed / unknown / no-context → low
    │
    ▼
output.csv
```

---

## Repository Layout

```
.
├── code/
│   └── main.py                 entry point (delegates to router/)
├── router/                     pipeline package
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── context_builder.py
│   ├── media_processor.py
│   ├── ocr.py
│   ├── voice_processor.py
│   ├── feature_engineering.py
│   ├── retriever.py
│   ├── rule_engine.py
│   ├── reasoning_engine.py
│   ├── confidence.py
│   ├── csv_writer.py
│   ├── evaluation.py
│   └── prompts.py
├── models_cache/               pre-downloaded OCR + ASR models
│   ├── easyocr/                craft_mlt_25k.pth, english_g2.pth
│   └── whisper-small/          model.bin, tokenizer.json, …
├── dataset/                    official HackerRank dataset (read-only)
├── setup.py                    one-time model download helper
├── requirements.txt
└── README.md
```

**Note on `code/` vs `router/`:** Python's stdlib has a module called `code`. Since
`torch`/`easyocr` internally do `import code`, naming the package `code/` with an
`__init__.py` would shadow the stdlib and crash at import time. The solution is that
`code/` contains only `main.py` (no `__init__.py`), and all pipeline logic lives in
`router/`. Running `python code/main.py` works correctly from the repo root.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Download models (once, ~600 MB)

```bash
python setup.py
```

This downloads EasyOCR and faster-whisper-small into `models_cache/` so the
pipeline runs fully offline. Skip this step if you have internet access at
inference time — models will be downloaded automatically on first run.

### 3 — Run the pipeline

```bash
python code/main.py
```

Reads `dataset/messages.csv`, writes `dataset/output.csv`. No flags required.

Optional flags:

```bash
python code/main.py --eval            # run local benchmark against sample_messages.csv
python code/main.py --error-analysis  # write ERROR_ANALYSIS.md
```

### 4 — Optional: LLM reasoning

Set a Gemini API key to enable the LLM fallback for messages that don't match
any deterministic rule:

```bash
set GEMINI_API_KEY=your_key_here     # Windows
export GEMINI_API_KEY=your_key_here  # macOS / Linux
python code/main.py
```

Without the key the pipeline uses heuristic reasoning only — results are still
fully valid.

---

## Output Format

`dataset/output.csv` — one row per message in `dataset/messages.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string | Matches `messages.csv` |
| `action` | `notify` / `digest` / `mute` | Routing decision |
| `message_type` | string | e.g. `urgent`, `payment`, `promotion`, `scam` |
| `reason` | string | Human-readable explanation |
| `confidence` | float 0–1 | Signal-agreement calibrated confidence |
| `evidence_message_ids` | string | `;`-separated historical IDs, or `none` |

---

## Runtime

| Stage | Time (110 messages, CPU) |
|-------|--------------------------|
| Context loading | ~0.1 s |
| OCR (14 images) | ~60 s (EasyOCR, CPU) |
| ASR (9 voice notes) | ~80 s (Whisper-small, CPU) |
| Rules + retrieval | ~0.5 s |
| Total | ~2–3 min |

On a GPU machine OCR and ASR are approximately 10× faster.
On a CPU-only machine without cached models, add ~5 min for model download on first run.

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `scikit-learn` | TF-IDF retrieval |
| `Pillow` | Image loading fallback |
| `easyocr` | Image OCR |
| `torch` | Required by EasyOCR |
| `faster-whisper` | Voice note ASR |
| `google-generativeai` | LLM reasoning (optional) |

---

## Submission Checklist

- [x] `dataset/output.csv` — 110 rows, correct columns, no synthetic data
- [x] `requirements.txt` — all imports listed with pinned versions
- [x] `setup.py` — one-time model cache download
- [x] `code/main.py` — runs cleanly from repo root
- [x] No hardcoded labels or organizer-only files used
- [x] No API keys in source code
- [x] `models_cache/` — pre-downloaded models for offline evaluation
