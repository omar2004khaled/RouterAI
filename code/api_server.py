"""FastAPI adapter for the existing WhatsApp notification routing pipeline.

Run from the repository root:
    uvicorn api_server:app --app-dir code --reload
"""
from __future__ import annotations

import csv
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import sentry_sdk
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Allow HTTP redirect URIs in local development (oauthlib rejects non-HTTPS by default).
if os.getenv("OAUTHLIB_INSECURE_TRANSPORT"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
if os.getenv("OAUTHLIB_RELAX_TOKEN_SCOPE"):
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Sentry initialisation ────────────────────────────────────────────────────
# DSN is read from the SENTRY_DSN environment variable.
# If the variable is not set, Sentry is silently disabled — the app still works.
_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        # Capture 100 % of transactions in development; lower this in production.
        traces_sample_rate=1.0,
        # Attach the request body to error events so you can see what was sent.
        send_default_pii=False,
    )

from confidence import ConfidenceCalibrator
from context_builder import ContextBuilder
from feature_engineering import FeatureExtractor
from models import RawMessage
from reasoning_engine import ReasoningEngine
from retriever import EvidenceRetriever
from rule_engine import RuleEngine
from gmail_service import authorization_url, credentials_from_code, fetch_messages, profile, serialize


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    sender: str = ""
    conversation: str = ""
    message_type: Literal["text", "image", "voice", "document"] = "text"
    timestamp: datetime | None = None
    has_media: bool = False
    # Optional existing dataset identifiers preserve the full relationship context.
    user_id: str = "u_002"
    conversation_type: Literal["personal", "group", "business"] = "personal"
    sender_user_id: str = ""
    group_id: str = ""
    business_id: str = ""
    forwarded_count: int = Field(default=0, ge=0)


class RouterService:
    def __init__(self) -> None:
        self.builder = ContextBuilder(dataset_dir=str(ROOT / "dataset"))
        self.features = FeatureExtractor()
        self.retriever = EvidenceRetriever()
        self.rules = RuleEngine()
        self.reasoner = ReasoningEngine()
        self.confidence = ConfidenceCalibrator()
        self.messages = self._read_csv("messages.csv")
        self.decisions = self._read_csv("output.csv")
        self.history = {item.message_id: item for item in self.builder.message_history}

    @staticmethod
    def _read_csv(name: str) -> list[dict[str, str]]:
        with (ROOT / "dataset" / name).open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def analyze(self, request: AnalyzeRequest) -> dict:
        started = time.perf_counter()
        created_at = (request.timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        message = RawMessage(
            message_id=f"api_{uuid4().hex[:12]}", user_id=request.user_id,
            conversation_type=request.conversation_type, group_id=request.group_id,
            business_id=request.business_id, sender_user_id=request.sender_user_id,
            created_at=created_at, message_text=request.text, media_type=request.message_type if request.has_media else "",
            forwarded_count=request.forwarded_count,
        )
        context = self.builder.build_context(message)
        features = self.features.extract(context)
        evidence_ids = self.retriever.retrieve(context, top_k=3)
        evidence_string = ";".join(evidence_ids) if evidence_ids else "none"
        decision = self.rules.evaluate(context, features, evidence_string)
        from_rule = decision is not None
        if decision is None:
            decision = self.reasoner.evaluate(context, features, evidence_ids)
        decision = self.confidence.calibrate(decision, context, features, from_rule)
        evidence = [{"id": item_id, "text": self.history[item_id].message_text, "signal": "Retrieved historical message"}
                    for item_id in evidence_ids if item_id in self.history]
        priority = "Critical" if decision.message_type == "urgent" else ("High" if decision.action == "notify" else "Normal")
        return {"action": decision.action.upper(), "confidence": decision.confidence,
                "message_type": request.message_type, "priority": priority,
                "reasoning": decision.reason, "evidence": evidence,
                "rules_triggered": [decision.reason] if from_rule else ["LLM / heuristic reasoning fallback"],
                "processing_time": round(time.perf_counter() - started, 4)}

    def message_rows(self) -> list[dict]:
        decision_by_id = {row["message_id"]: row for row in self.decisions}
        rows = []
        for message in self.messages:
            decision = decision_by_id.get(message["message_id"])
            if not decision:
                continue
            rows.append({"id": message["message_id"], "sender": message["sender_user_id"] or message["business_id"] or "Group member",
                         "conversation": message["group_id"] or message["conversation_type"], "text": message["message_text"],
                         "type": message["media_type"] or "text", "source": "dataset", "action": decision["action"].upper(),
                         "confidence": float(decision["confidence"]), "timestamp": message["created_at"]})
        return sorted(rows, key=lambda row: row["timestamp"], reverse=True)

    def analytics(self) -> dict:
        rows = self.message_rows()
        counts = {key: sum(row["action"] == key for row in rows) for key in ("NOTIFY", "DIGEST", "MUTE")}
        colors = {"NOTIFY": "#2dd4bf", "DIGEST": "#818cf8", "MUTE": "#fb7185"}
        trend: dict[str, dict] = {}
        for row in rows:
            day = row["timestamp"][:10]
            trend.setdefault(day, {"day": day, "notify": 0, "digest": 0, "mute": 0})[row["action"].lower()] += 1
        return {"routing": [{"name": key.title(), "value": value, "color": colors[key]} for key, value in counts.items()],
                "trend": list(trend.values())}


service: RouterService | None = None
_startup_time: float = time.time()
# Development-only in-memory storage. In production replace this with encrypted,
# persistent server-side storage (and use a strong SESSION_SECRET).
gmail_sessions: dict[str, object] = {}
gmail_results: dict[str, list[dict]] = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = RouterService()
    yield
    service = None

app = FastAPI(
    title="RouterAI — WhatsApp Notification Router API",
    version="1.0.0",
    description=(
        "AI-powered pipeline that classifies every incoming WhatsApp message "
        "into **notify**, **digest**, or **mute**.\n\n"
        "The pipeline runs: ContextBuilder → FeatureExtractor → EvidenceRetriever "
        "→ RuleEngine (90 rules) → ReasoningEngine (Gemini / heuristic) → ConfidenceCalibrator.\n\n"
        "A Gmail intelligence mode is also available via OAuth 2.0 (read-only)."
    ),
    contact={"name": "Omar Khaled Hussein"},
    lifespan=lifespan,
    openapi_tags=[
        {"name": "routing",  "description": "Core message routing — analyze a message or browse dataset decisions."},
        {"name": "dashboard","description": "Aggregate statistics and trend data for the dashboard UI."},
        {"name": "gmail",    "description": "Gmail OAuth 2.0 connection and inbox analysis."},
        {"name": "system",   "description": "Health checks and system status."},
        {"name": "debug",    "description": "Development-only endpoints. Disabled in production."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "local-development-change-this"), same_site="lax", https_only=False)

def router() -> RouterService:
    if service is None:
        raise HTTPException(503, "Router is still initializing")
    return service

# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    tags=["system"],
    summary="Health check",
    description="Returns pipeline readiness, uptime in seconds, and version. "
                "Deployment platforms use this endpoint to verify the service is alive.",
)
def health_check() -> dict:
    return {
        "status": "ok",
        "pipeline_ready": service is not None,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "version": "1.0.0",
    }

@app.get(
    "/api/system",
    tags=["system"],
    summary="System status",
    description="High-level system status string used by the frontend status bar.",
)
def get_system() -> dict:
    return {"status": "Operational", "version": "1.0.0", "backend": "FastAPI connected to Python router", "latency": "Live"}

# ---------------------------------------------------------------------------
# Debug (development only — disabled when DEBUG_MODE != true)
# ---------------------------------------------------------------------------

@app.get(
    "/api/debug/sentry-test",
    tags=["debug"],
    summary="Trigger a test Sentry error",
    description="Raises a deliberate ZeroDivisionError so you can verify Sentry "
                "is receiving events. **Only active when `DEBUG_MODE=true` in the "
                "environment.** Returns 404 in production.",
)
def sentry_test() -> dict:
    if os.getenv("DEBUG_MODE", "false").lower() != "true":
        raise HTTPException(404, "Not found")
    # This intentional error will be captured by Sentry.
    result = 1 / 0  # noqa: F841  — deliberate ZeroDivisionError
    return {"ok": True}  # never reached

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@app.get(
    "/api/messages",
    tags=["routing"],
    summary="List all routed messages",
    description="Returns all 110 messages from the dataset with their routing decisions "
                "(action, confidence, timestamp). Sorted newest-first.",
)
def get_messages() -> list[dict]: return router().message_rows()

@app.post(
    "/api/analyze",
    tags=["routing"],
    summary="Route a single message",
    description="Runs the full 7-stage pipeline on the supplied message text and returns "
                "the routing decision with action, confidence, message type, reasoning, "
                "and supporting evidence IDs.",
)
def analyze_message(request: AnalyzeRequest) -> dict: return router().analyze(request)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get(
    "/api/dashboard",
    tags=["dashboard"],
    summary="Dataset routing summary",
    description="Aggregate counts of notify / digest / mute decisions across the 110-message dataset, "
                "plus average confidence score.",
)
def get_dashboard() -> dict:
    rows = router().message_rows()
    total = len(rows)
    return {"total": total, "notify": sum(x["action"] == "NOTIFY" for x in rows), "digest": sum(x["action"] == "DIGEST" for x in rows), "mute": sum(x["action"] == "MUTE" for x in rows), "confidence": round(sum(x["confidence"] for x in rows) / total, 2) if total else 0, "processingTime": None}

@app.get(
    "/api/dashboard/gmail",
    tags=["dashboard"],
    summary="Gmail analysis summary",
    description="Aggregate routing stats for the currently connected Gmail inbox. "
                "Returns `connected: false` if no Gmail session is active.",
)
def get_dashboard_gmail(request: Request) -> dict:
    key = request.session.get("gmail_session")
    session = gmail_sessions.get(key) if key else None
    if not session:
        return {"connected": False}
    results = gmail_results.get(key, [])
    total = len(results)
    if total == 0:
        return {"connected": True, "email": session["email"], "total": 0, "notify": 0, "digest": 0, "mute": 0, "confidence": 0, "analyzed": False}
    confidence = round(sum(float(r.get("confidence", 0)) for r in results) / total, 2)
    return {"connected": True, "email": session["email"], "total": total,
            "notify": sum(1 for r in results if r.get("action", "").upper() == "NOTIFY"),
            "digest": sum(1 for r in results if r.get("action", "").upper() == "DIGEST"),
            "mute": sum(1 for r in results if r.get("action", "").upper() == "MUTE"),
            "confidence": confidence, "analyzed": True}

@app.get(
    "/api/analytics",
    tags=["dashboard"],
    summary="Routing trend data",
    description="Returns daily routing counts (notify / digest / mute) for the trend chart "
                "and a pie-chart breakdown by action.",
)
def get_analytics() -> dict: return router().analytics()

def _session_id(request: Request) -> str:
    key = request.session.get("gmail_session")
    if not key or key not in gmail_sessions:
        raise HTTPException(401, "Gmail is not connected. Connect Gmail before requesting inbox data.")
    return key

@app.get(
    "/api/auth/gmail",
    tags=["gmail"],
    summary="Start Gmail OAuth flow",
    description="Redirects the browser to Google's OAuth consent screen. "
                "After the user grants access, Google redirects back to `/api/auth/gmail/callback`.",
)
def gmail_auth(request: Request):
    url, state = authorization_url()
    request.session["gmail_oauth_state"] = state
    return RedirectResponse(url)

@app.get(
    "/api/auth/gmail/callback",
    tags=["gmail"],
    summary="Gmail OAuth callback",
    description="Handles the redirect from Google after the user grants or denies access. "
                "On success, stores credentials in the server session and redirects to the frontend.",
)
def gmail_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    import traceback
    frontend = os.getenv("FRONTEND_URL", "http://localhost:5173")
    if error or not code or not state or state != request.session.get("gmail_oauth_state"):
        print(f"[gmail_callback] early exit: error={error!r} code={bool(code)} state_match={state == request.session.get('gmail_oauth_state')}")
        return RedirectResponse(f"{frontend}?gmail=error")
    try:
        credentials = credentials_from_code(code, state)
        account = profile(credentials)
    except Exception:
        traceback.print_exc()
        return RedirectResponse(f"{frontend}?gmail=error")
    key = uuid4().hex
    gmail_sessions[key] = {"credentials": credentials, "email": account["email"]}
    request.session.pop("gmail_oauth_state", None)
    request.session["gmail_session"] = key
    return RedirectResponse(f"{frontend}?gmail=connected")

@app.get(
    "/api/gmail/status",
    tags=["gmail"],
    summary="Gmail connection status",
    description="Returns whether a Gmail account is currently connected in this session, "
                "and the connected email address if so.",
)
def gmail_status(request: Request) -> dict:
    key = request.session.get("gmail_session")
    session = gmail_sessions.get(key) if key else None
    return {"connected": bool(session), **({"email": session["email"]} if session else {})}

@app.delete(
    "/api/gmail/connection",
    tags=["gmail"],
    summary="Disconnect Gmail",
    description="Clears the Gmail session and all cached analysis results. "
                "Does not modify, delete, or send any Gmail messages.",
)
def gmail_disconnect(request: Request) -> dict:
    key = request.session.pop("gmail_session", None)
    if key:
        gmail_sessions.pop(key, None); gmail_results.pop(key, None)
    return {"disconnected": True, "message": "Gmail disconnected. Your Gmail messages were not modified."}

@app.get(
    "/api/gmail/messages",
    tags=["gmail"],
    summary="Fetch Gmail inbox",
    description="Returns up to `limit` emails from the connected Gmail inbox. "
                "Supports pagination via `page_token` and Gmail search syntax via `query`.",
)
def gmail_messages(request: Request, limit: int = 25, page_token: str | None = None, query: str | None = None) -> dict:
    key = _session_id(request)
    try:
        emails, next_token = fetch_messages(gmail_sessions[key]["credentials"], limit, page_token, query)
        return {"messages": [serialize(email) for email in emails], "next_page_token": next_token}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(502, "Unable to read Gmail messages. Reconnect Gmail and try again.") from exc

@app.post(
    "/api/gmail/analyze",
    tags=["gmail"],
    summary="Analyze Gmail inbox",
    description="Fetches up to `limit` emails and runs the full RouterAI pipeline on each one "
                "using a synthetic context built from sender domain, Gmail labels, and feedback history. "
                "Results are cached in the session for the dashboard.",
)
def gmail_analyze(request: Request, limit: int = 25, query: str | None = None) -> dict:
    key = _session_id(request)
    try: emails, _ = fetch_messages(gmail_sessions[key]["credentials"], limit, None, query)
    except Exception as exc: raise HTTPException(502, "Unable to read Gmail messages. Reconnect Gmail and try again.") from exc
    feedback = _load_feedback()
    results = []
    for email in emails:
        try:
            outcome = _analyze_email(email, feedback)
            results.append({"id": email.id, "thread_id": email.thread_id, "sender": email.sender,
                            "recipient": email.recipient, "subject": email.subject,
                            "message": email.body or email.snippet, "preview": email.snippet,
                            "timestamp": email.timestamp, "labels": email.labels, "source": "gmail", **outcome})
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"id": email.id, "thread_id": email.thread_id, "sender": email.sender,
                            "recipient": email.recipient, "subject": email.subject,
                            "message": email.body or email.snippet, "preview": email.snippet,
                            "timestamp": email.timestamp, "labels": email.labels, "source": "gmail",
                            "action": "DIGEST", "confidence": 0.5, "message_type": "text",
                            "priority": "Normal", "reasoning": f"Analysis error: {exc}",
                            "evidence": [], "rules_triggered": [], "processing_time": 0})
    gmail_results[key] = results
    return {"total": len(results), "results": results}

# ---------------------------------------------------------------------------
# Gmail-aware analysis helpers
# ---------------------------------------------------------------------------

import re as _re

# Trusted sender domains → category used to build synthetic BusinessAccount
_TRUSTED_DOMAINS: dict[str, tuple[str, str]] = {
    # domain: (category, brand_name)
    "accounts.google.com": ("security",      "Google"),
    "google.com":          ("security",      "Google"),
    "linkedin.com":        ("professional",  "LinkedIn"),
    "github.com":          ("professional",  "GitHub"),
    "amazon.com":          ("ecommerce_delivery", "Amazon"),
    "amazon.eg":           ("ecommerce_delivery", "Amazon"),
    "paypal.com":          ("finance",        "PayPal"),
    "stripe.com":          ("finance",        "Stripe"),
    "apple.com":           ("professional",  "Apple"),
    "microsoft.com":       ("professional",  "Microsoft"),
    "zoom.us":             ("education",     "Zoom"),
    "calendly.com":        ("education",     "Calendly"),
    "udemy.com":           ("education",     "Udemy"),
    "coursera.org":        ("education",     "Coursera"),
    "pinterest.com":       ("fashion",       "Pinterest"),
    "discover.pinterest.com": ("fashion",    "Pinterest"),
    "twitter.com":         ("professional",  "Twitter/X"),
    "facebook.com":        ("professional",  "Facebook"),
    "instagram.com":       ("professional",  "Instagram"),
    "notion.so":           ("professional",  "Notion"),
    "slack.com":           ("professional",  "Slack"),
}

# Gmail label → (urgency_boost, is_spam, is_important)
_LABEL_SIGNALS: dict[str, tuple[float, bool, bool]] = {
    "SPAM":                 (0.0,  True,  False),
    "IMPORTANT":            (0.3,  False, True),
    "STARRED":              (0.4,  False, True),
    "CATEGORY_PROMOTIONS":  (-0.3, False, False),
    "CATEGORY_SOCIAL":      (-0.1, False, False),
    "CATEGORY_UPDATES":     (0.0,  False, False),
    "CATEGORY_FORUMS":      (-0.2, False, False),
}

# Subject patterns → urgency boost
_URGENT_SUBJECT_RE = _re.compile(
    r"security alert|sign.in attempt|new sign.in|unusual activity|"
    r"verify your|action required|account suspended|password reset|"
    r"interview|offer letter|job offer|internship offer|"
    r"urgent|deadline|expires today|last chance",
    _re.I
)


def _sender_domain(sender: str) -> str:
    m = _re.search(r"@([\w.-]+)", sender or "")
    return m.group(1).lower() if m else ""


def _load_feedback() -> dict[str, str]:
    """Return {sender_string_or_domain: correct_action} from feedback.csv."""
    path = ROOT / "dataset" / "feedback.csv"
    if not path.exists():
        return {}
    corrections: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sender = (row.get("sender") or "").strip()
            action = (row.get("correct_action") or "").strip().upper()
            if sender and action in ("NOTIFY", "DIGEST", "MUTE"):
                corrections[sender] = action
                domain = _sender_domain(sender)
                if domain:
                    corrections.setdefault(domain, action)
    return corrections


def _build_gmail_context(email, feedback: dict[str, str]):
    """Build a synthetic Context from email metadata so the full pipeline runs
    with real signals instead of all-None defaults."""
    from models import (
        RawMessage, Context, UserProfile, BusinessAccount,
        UserBusinessHistory, MessageHistoryItem, DailyNotificationSummary
    )

    sender  = email.sender or ""
    domain  = _sender_domain(sender)
    labels  = [l.upper() for l in (email.labels or [])]
    subject = email.subject or ""

    # ── Parse timestamp ───────────────────────────────────────────────────
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if email.timestamp:
        try:
            dt = datetime.fromisoformat(email.timestamp)
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    # ── Clean text: subject + snippet only, no raw HTML ───────────────────
    clean_text = f"Subject: {subject}\n\n{email.snippet or ''}".strip() or subject

    # ── Synthetic RawMessage ──────────────────────────────────────────────
    msg = RawMessage(
        message_id=f"gmail_{email.id}",
        user_id="gmail_user",
        conversation_type="business" if domain else "personal",
        business_id=domain or "",
        sender_user_id=sender,
        created_at=ts_str,
        message_text=clean_text[:3000],
        forwarded_count=0,
    )

    # ── Synthetic UserProfile — moderate engagement baseline ─────────────
    user = UserProfile(
        user_id="gmail_user",
        quiet_hours_start="23:00", quiet_hours_end="07:00",
        messages_opened_30d=60, messages_replied_30d=15,
        notifications_dismissed_30d=20, messages_reported_30d=1,
        open_rate=0.65, reply_rate=0.25, dismissal_rate=0.20, report_rate=0.01,
    )

    # ── Synthetic BusinessAccount from domain trust table ─────────────────
    business = None
    user_biz = None
    trusted = _TRUSTED_DOMAINS.get(domain, {})
    if domain:
        is_verified   = bool(trusted)
        category      = trusted[0] if trusted else "unknown"
        brand         = trusted[1] if trusted else domain
        report_count  = 0 if is_verified else 2
        domain_age    = 3650 if is_verified else 180
        business = BusinessAccount(
            business_id=domain,
            display_name=brand, brand_name=brand,
            category=category,
            is_verified=is_verified,
            official_domain=domain,
            domain_used_by_sender=domain,
            account_age_days=domain_age,
            messages_sent_30d=50 if is_verified else 5,
            user_reports_30d=report_count,
            domain_used_by_sender_age_days=domain_age,
        )
        # Relationship strength: feedback corrections mean the user has engaged
        has_prior = sender in feedback or domain in feedback
        user_biz = UserBusinessHistory(
            user_id="gmail_user", business_id=domain,
            activity_count_180d=10 if is_verified else (3 if has_prior else 0),
            messages_opened_30d=5 if is_verified else 1,
            messages_dismissed_30d=0 if is_verified else 2,
            messages_replied_30d=2 if has_prior else 0,
            allows_promotions=False,
        )

    # ── Label signals → synthetic history items ───────────────────────────
    # Inject fake history events so ConfidenceCalibrator sees engagement signals
    relevant_history: list[MessageHistoryItem] = []
    urgency_boost = 0.0
    is_spam_label = False
    is_important  = False

    for label in labels:
        sig = _LABEL_SIGNALS.get(label)
        if sig:
            boost, spam, important = sig
            urgency_boost += boost
            is_spam_label  = is_spam_label or spam
            is_important   = is_important  or important

    if is_important:
        # Simulate prior open+reply so pipeline treats this as high-engagement
        relevant_history.append(MessageHistoryItem(
            message_id="synth_imp", user_id="gmail_user",
            sender_user_id=sender, business_id=domain,
            created_at=ts_str, message_text=subject,
            message_opened=True, message_replied=True, reaction_time_minutes=2.0,
        ))

    if is_spam_label:
        relevant_history.append(MessageHistoryItem(
            message_id="synth_spam", user_id="gmail_user",
            sender_user_id=sender, business_id=domain,
            created_at=ts_str, message_text=subject,
            message_reported=True, muted_after_message=True,
        ))

    # ── Urgency boost via message text injection ──────────────────────────
    # If subject has urgent keywords, prepend them so the rule engine sees them
    if _URGENT_SUBJECT_RE.search(subject):
        msg.message_text = f"URGENT ACTION REQUIRED: {msg.message_text}"
    elif urgency_boost < 0:  # promotions
        msg.message_text = f"sale discount promo offer: {msg.message_text}"

    # ── Daily summary — moderate load ────────────────────────────────────
    daily = DailyNotificationSummary(
        user_id="gmail_user", date=ts_str[:10],
        notifications_sent=30, notifications_dismissed=8,
    )

    return Context(
        message=msg, user=user, business=business,
        user_business_history=user_biz, daily_summary=daily,
        relevant_history=relevant_history,
    )


def _analyze_email(email, feedback: dict[str, str]) -> dict:
    """Route a GmailEmail through the full pipeline with a synthetic context."""
    import time as _time
    started = _time.perf_counter()

    sender = email.sender or ""
    domain = _sender_domain(sender)

    # 1. Feedback override — user already corrected this sender
    for key in (sender, domain):
        if key in feedback:
            action = feedback[key]
            mtype  = "personal" if action == "NOTIFY" else ("promotion" if action == "MUTE" else "business_update")
            return {"action": action, "confidence": 0.95, "message_type": mtype,
                    "priority": "High" if action == "NOTIFY" else "Normal",
                    "reasoning": f"User feedback: corrected to {action} for '{sender}'.",
                    "evidence": [], "rules_triggered": ["Feedback override"],
                    "processing_time": round(_time.perf_counter() - started, 4)}

    # 2. Build synthetic context and run the full pipeline
    ctx      = _build_gmail_context(email, feedback)
    svc      = router()
    features = svc.features.extract(ctx)
    ev_ids   = svc.retriever.retrieve(ctx, top_k=3)
    ev_str   = ";".join(ev_ids) if ev_ids else "none"
    decision = svc.rules.evaluate(ctx, features, ev_str)
    from_rule = decision is not None
    if decision is None:
        decision = svc.reasoner.evaluate(ctx, features, ev_ids)
    decision = svc.confidence.calibrate(decision, ctx, features, from_rule)

    priority = "Critical" if decision.message_type == "urgent" else (
        "High" if decision.action == "notify" else "Normal")
    return {
        "action":          decision.action.upper(),
        "confidence":      decision.confidence,
        "message_type":    decision.message_type,
        "priority":        priority,
        "reasoning":       decision.reason,
        "evidence":        [],
        "rules_triggered": [decision.reason] if from_rule else ["Heuristic / LLM reasoning"],
        "processing_time": round(_time.perf_counter() - started, 4),
    }


@app.get(
    "/api/gmail/results",
    tags=["gmail"],
    summary="Get cached Gmail analysis results",
    description="Returns the results from the most recent `/api/gmail/analyze` call for this session.",
)
def gmail_saved_results(request: Request) -> dict:
    return {"results": gmail_results.get(_session_id(request), [])}

class FeedbackRequest(BaseModel):
    message_id: str
    original_action: str
    correct_action: str
    subject: str = ""
    sender: str = ""

@app.post(
    "/api/gmail/feedback",
    tags=["gmail"],
    summary="Submit a routing correction",
    description="Saves a user correction to `dataset/feedback.csv`. "
                "On the next analysis run, corrections for this sender are applied automatically at 0.95 confidence.",
)
def gmail_feedback(body: FeedbackRequest, request: Request) -> dict:
    _session_id(request)
    feedback_path = ROOT / "dataset" / "feedback.csv"
    write_header = not feedback_path.exists()
    with feedback_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["message_id", "sender", "subject", "original_action", "correct_action", "timestamp"])
        if write_header:
            writer.writeheader()
        writer.writerow({"message_id": body.message_id, "sender": body.sender, "subject": body.subject,
                         "original_action": body.original_action.upper(), "correct_action": body.correct_action.upper(),
                         "timestamp": datetime.now().isoformat()})
    return {"saved": True}

@app.get(
    "/api/gmail/feedback",
    tags=["gmail"],
    summary="Get all saved feedback",
    description="Returns all routing corrections previously saved to `dataset/feedback.csv`.",
)
def gmail_get_feedback(request: Request) -> dict:
    _session_id(request)
    feedback_path = ROOT / "dataset" / "feedback.csv"
    if not feedback_path.exists():
        return {"feedback": []}
    with feedback_path.open(encoding="utf-8", newline="") as f:
        return {"feedback": list(csv.DictReader(f))}
