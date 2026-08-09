# RouterAI — Notification Intelligence System

HackerRank Orchestrate · August 2026 · Omar Khaled Hussein

---

## What It Does

RouterAI classifies every incoming message — WhatsApp or Gmail — into one of three actions:

| Action | Meaning |
|--------|---------|
| `notify` | Interrupt the user now |
| `digest` | Batch for later review |
| `mute` | Suppress — spam, scam, opted-out, or low-value |

Decisions are personalised. The system knows who sent the message, whether the user has replied to that sender before, whether the business is verified, whether the user is in quiet hours, and what the message actually says — including text extracted from images (OCR) and voice notes (ASR).

---

## Two Modes

### 1 — WhatsApp Dataset Pipeline (`python code/main.py`)

Processes `dataset/messages.csv` using the full relational context from 10 CSV tables. Writes `dataset/output.csv`. This is the HackerRank submission path — unchanged.

### 2 — Live Gmail Intelligence (dashboard)

Connects to a real Gmail inbox via OAuth 2.0 (read-only). Builds a synthetic context from email metadata and runs the same Python pipeline — the same 90 rules, the same feature extractor, the same confidence calibrator — on real emails. Includes a feedback loop: corrections you make are saved to `dataset/feedback.csv` and applied on the next analysis run.

---

## Pipeline Architecture

```
Input (WhatsApp message or Gmail email)
    │
    ▼
ContextBuilder          Loads all relational tables once at startup:
                        users, groups, group_members, businesses,
                        user_business_history, message_history,
                        message_events, images, voice_notes,
                        daily_notification_summary
                        ── For Gmail: builds a synthetic Context
                           from sender domain, labels, and feedback
    │
    ▼
MediaProcessor          OCR on image attachments  (EasyOCR → PIL fallback)
                        ASR on voice notes        (faster-whisper-small)
    │
    ▼
FeatureExtractor        ~100 engineered signals per message:
                        user engagement rates (open/reply/dismiss/report)
                        business domain trust + verification status
                        group mute state + direct-mention detection
                        quiet-hours window check
                        weighted keyword scores: urgency / scam /
                          payment / promotion / greeting / event
                        media category + transcript quality
                        prior reaction history (fast-reply bonus)
    │
    ▼
EvidenceRetriever       Scores message_history for relevance:
                        relationship weight (sender / business / group)
                        × user reaction signals (replied / opened /
                          dismissed / reported)
                        × TF-IDF cosine similarity (sklearn)
                        × exponential recency decay (30-day half-life)
                        → diverse top-3 evidence IDs
    │
    ▼
RuleEngine              90 deterministic rules, evaluated in priority order:
                        emergency & family safety overrides
                        scam / fraud / phishing detection
                        domain mismatch + young-domain checks
                        business opt-out enforcement
                        verified business category routing
                        delivery / payment / event classification
                        voice note urgency detection
                        quiet-hours batching
    │
    ▼  (fallthrough only — when no rule fires)
ReasoningEngine         Gemini-2.5-flash with structured JSON prompt
                        (falls back to weighted heuristic if no API key)
    │
    ▼
ConfidenceCalibrator    Agreement / conflict model (not a fixed score):
                        agreement signals: rule fired, verified sender,
                          active relationship, prior replies, fast replies,
                          evidence count, media confidence, scam/urgency match
                        conflict signals: unknown type, no context,
                          unverified + payment, domain mismatch,
                          prior reports, high forward count
                        output range: 0.38 – 0.96
    │
    ▼
Output (output.csv  or  API JSON response)
```

---

## Gmail Intelligence — How It Works

Standard email analysis tools dump raw HTML into a text classifier. RouterAI does not do that.

For each Gmail email the system:

1. **Checks feedback first.** If you have previously corrected a decision for this sender, that correction is applied immediately at 0.95 confidence. No pipeline needed.

2. **Builds a synthetic Context.** Sender domain is looked up in a trusted-domain table (Google, LinkedIn, PayPal, GitHub, etc.) to construct a `BusinessAccount` with real `is_verified`, `category`, and `account_age_days` values. Gmail labels (`IMPORTANT`, `STARRED`, `SPAM`, `CATEGORY_PROMOTIONS`, etc.) are translated into synthetic history events — a `STARRED` email gets a fake prior-reply record so the confidence calibrator sees engagement signal. An `IMPORTANT` label boosts urgency. A `SPAM` label injects a prior-report record.

3. **Runs the full pipeline.** The same `FeatureExtractor → RuleEngine → ReasoningEngine → ConfidenceCalibrator` chain that processes WhatsApp messages runs on the synthetic context. The rule engine fires real rules — R03 (production incident), R17 (verified bank OTP), R29 (unverified promotion), R50 (verified business category routing) — not keyword shortcuts.

4. **Uses clean text only.** The message body passed to the pipeline is `Subject + Gmail snippet` (≤ 3000 chars). Raw HTML, tracking URLs, and unsubscribe footers are never passed in, so the scam detector does not false-positive on legitimate marketing emails.

### Feedback Loop

Every correction you make in the dashboard is appended to `dataset/feedback.csv`:

```
message_id, sender, subject, original_action, correct_action, timestamp
```

On the next "Analyze Gmail Emails" run, `_load_feedback()` reads this file and builds a `{sender → action}` override map. Corrections accumulate across server restarts. The more you correct, the more accurate the system becomes for your inbox.

---

## Repository Layout

```
.
├── code/
│   ├── main.py                 WhatsApp pipeline entry point
│   ├── api_server.py           FastAPI adapter + Gmail intelligence
│   ├── gmail_service.py        OAuth 2.0 flow + Gmail API client
│   ├── config.py
│   ├── models.py               All dataclasses (RawMessage, Context, FeatureSet, Decision, …)
│   ├── context_builder.py      Loads 10 CSV tables, assembles Context per message
│   ├── feature_engineering.py  ~100 signals → FeatureSet
│   ├── retriever.py            TF-IDF + reaction-weighted evidence retrieval
│   ├── rule_engine.py          90 deterministic routing rules
│   ├── reasoning_engine.py     Gemini-2.5-flash + heuristic fallback
│   ├── confidence.py           Agreement/conflict confidence calibration
│   ├── media_processor.py      Dispatches OCR / ASR
│   ├── ocr.py                  EasyOCR with PIL fallback
│   ├── voice_processor.py      faster-whisper-small ASR
│   ├── prompts.py              Gemini system + user prompt templates
│   ├── csv_writer.py           Writes output.csv
│   └── evaluation.py           Local benchmark against sample_messages.csv
├── frontend/                   React + Vite + TypeScript dashboard
│   └── src/
│       ├── main.tsx            All UI components (single file)
│       └── services/api.ts     All fetch calls to FastAPI
├── models_cache/               Pre-downloaded models (offline evaluation)
│   ├── easyocr/                craft_mlt_25k.pth, english_g2.pth
│   └── whisper-small/          model.bin, tokenizer.json, …
├── dataset/                    HackerRank dataset (read-only except feedback.csv)
│   ├── messages.csv            110 WhatsApp messages to route
│   ├── output.csv              Pipeline decisions (generated by main.py)
│   └── feedback.csv            User corrections (generated at runtime)
├── .env                        Secrets — never commit
├── .env.example                Template
├── requirements.txt
├── setup.py                    One-time model download
└── README.md
```

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

Downloads EasyOCR and faster-whisper-small into `models_cache/` for fully offline inference. Skip if you have internet access — models download automatically on first run.

### 3 — Run the WhatsApp pipeline

```bash
python code/main.py
```

Reads `dataset/messages.csv`, writes `dataset/output.csv`. No flags required.

```bash
python code/main.py --eval            # benchmark against sample_messages.csv
python code/main.py --error-analysis  # write ERROR_ANALYSIS.md
```

### 4 — Start the dashboard

```bash
# Terminal 1 — API server
uvicorn api_server:app --app-dir code --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard shows live routing stats, lets you test individual messages through the full pipeline, and connects to Gmail.

### 5 — Optional: Gemini LLM reasoning

```bash
set GEMINI_API_KEY=your_key_here     # Windows
export GEMINI_API_KEY=your_key_here  # macOS / Linux
```

Without the key the pipeline uses heuristic reasoning — results are fully valid.

---

## Gmail Setup

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create or select a project.
2. Enable the **Gmail API** under APIs & Services.
3. Configure the OAuth consent screen. Add yourself as a test user.
4. Create a **Web application** OAuth 2.0 client.
5. Add `http://localhost:8000/api/auth/gmail/callback` as an authorized redirect URI.
6. Copy `.env.example` to `.env` and fill in `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `SESSION_SECRET`, `FRONTEND_URL`.

RouterAI requests `gmail.readonly` only. It never sends, deletes, labels, moves, marks read, or modifies email in any way. The access token is stored server-side in the session only and is never returned to the browser.

---

## Output Format

`dataset/output.csv` — one row per message:

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string | Matches `messages.csv` |
| `action` | `notify` / `digest` / `mute` | Routing decision |
| `message_type` | string | `urgent`, `payment`, `scam`, `promotion`, `event`, … |
| `reason` | string | Human-readable explanation of the decision |
| `confidence` | float 0–1 | Signal-agreement calibrated score |
| `evidence_message_ids` | string | `;`-separated historical message IDs, or `none` |

---

## Runtime

| Stage | Time (110 messages, CPU) |
|-------|--------------------------|
| Context loading | ~0.1 s |
| OCR (14 images) | ~60 s (EasyOCR, CPU) |
| ASR (9 voice notes) | ~80 s (Whisper-small, CPU) |
| Rules + retrieval | ~0.5 s |
| Total | ~2–3 min |

GPU: OCR and ASR are ~10× faster. First run without cached models: add ~5 min for download.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn` | API server |
| `pandas`, `numpy` | Data loading |
| `scikit-learn` | TF-IDF evidence retrieval |
| `easyocr`, `torch` | Image OCR |
| `faster-whisper` | Voice note ASR |
| `Pillow` | Image loading fallback |
| `google-generativeai` | Gemini LLM reasoning (optional) |
| `google-auth-oauthlib` | Gmail OAuth 2.0 |
| `google-api-python-client` | Gmail API client |
| `starlette` | Session middleware |
| `python-dotenv` | Environment variable loading |

---

## Submission Checklist

- [x] `dataset/output.csv` — 110 rows, correct columns, no synthetic data
- [x] `requirements.txt` — all imports listed with pinned versions
- [x] `setup.py` — one-time model cache download
- [x] `code/main.py` — runs cleanly from repo root
- [x] No hardcoded labels or organizer-only files used
- [x] No API keys in source code
- [x] `models_cache/` — pre-downloaded models for offline evaluation
