"""
EvidenceRetriever — multi-signal retrieval for the WhatsApp notification router.

Scoring formula (raw points, then normalised to 0-1 before threshold check):

  Relationship signals (who sent it / where)
    same sender               +3.5
    same business             +3.0
    same group                +2.0
    same user (always true)   +0.5  (tiebreak only)

  User reaction signals (did this user engage positively or negatively before?)
    user replied              +3.0
    user opened quickly (<5m) +2.0
    user opened (any)         +1.0
    user dismissed            -1.0
    user muted after          -1.5
    user reported             -2.5

  Content signals
    TF-IDF cosine similarity  × 4.0  (sklearn) or Jaccard × 2.5 (fallback)
    same message category     +1.5
    lexical overlap ratio     × 1.2

  Recency decay
    exponential decay, half-life = 30 days, applied to the whole score

  Domain-trust bonus (business messages only)
    domain matches official   +1.0
    domain mismatch           -2.0

All scores are clamped to ≥ 0 before sorting. A normalised score must exceed
`score_threshold` (default 0.15) to appear in the final evidence list.
"""

import logging
import math
import re
from datetime import datetime
from typing import List, Tuple, Optional

from models import Context, MessageHistoryItem

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Coarse category vocabulary — used only as a lightweight alignment signal
# ---------------------------------------------------------------------------
_CATEGORY_VOCAB = {
    "urgent":    ["urgent", "emergency", "immediately", "asap", "water leak",
                  "server crash", "critical", "alert", "hospital", "ambulance"],
    "payment":   ["invoice", "payment", "receipt", "total due", "statement",
                  "overdue", "bank", "bill", "amount due", "transaction"],
    "promotion": ["sale", "discount", "promo", "offer", "coupon", "clearance",
                  "deal", "collect now"],
    "event":     ["meeting", "webinar", "appointment", "schedule", "tomorrow at",
                  "school board", "pta"],
    "greeting":  ["good morning", "hello", "hi there", "happy weekend", "family"],
    "delivery":  ["delivery", "driver", "arriving", "package", "order #",
                  "tracking", "shipped"],
    "scam":      ["otp", "lottery", "free gift", "claim now", "bit.ly",
                  "tinyurl", "crypto", "roi", "win $", "processing fee",
                  "gift card", "wire transfer"],
}


def _infer_category(text: str, media_type: str = "") -> Optional[str]:
    t = (text or "").lower()
    for cat, keywords in _CATEGORY_VOCAB.items():
        if any(kw in t for kw in keywords):
            return cat
    if media_type == "voice":
        return "voice"
    if media_type == "image":
        return "image"
    return None


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    ts = ts.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _recency_decay(current_ts: str, history_ts: str, half_life_days: float = 30.0) -> float:
    dt_curr = _parse_ts(current_ts)
    dt_hist = _parse_ts(history_ts)
    if dt_curr is None or dt_hist is None:
        return 1.0
    delta = max(0.0, (dt_curr - dt_hist).total_seconds() / 86400.0)
    return math.exp(-0.693 * delta / half_life_days)


class EvidenceRetriever:
    def __init__(self, score_threshold: float = 0.15):
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, ctx: Context, top_k: int = 3) -> List[str]:
        """
        Return up to top_k message_ids from ctx.relevant_history that best
        support the routing decision for ctx.message.
        """
        if not ctx.relevant_history:
            return []

        candidates = self._filter_candidates(ctx)
        if not candidates:
            return []

        scored = self._score_candidates(ctx, candidates)

        # Normalise scores to 0-1 so the threshold is scale-independent
        max_score = max(s for s, _ in scored) if scored else 1.0
        if max_score <= 0:
            return []

        # Diversify: prefer results spread across different relationship types
        # rather than top-k from the same entity (e.g. 3 messages from same promo).
        seen_ids: set = set()
        seen_buckets: set = set()   # "sender", "business", "group", "category"
        result: List[str] = []

        # Build a lookup for bucket classification
        msg = ctx.message
        bucket_map: dict = {}
        for h in candidates:
            if msg.sender_user_id and h.sender_user_id == msg.sender_user_id:
                bucket_map[h.message_id] = "sender"
            elif msg.business_id and h.business_id == msg.business_id:
                bucket_map[h.message_id] = "business"
            elif msg.group_id and h.group_id == msg.group_id:
                bucket_map[h.message_id] = "group"
            else:
                bucket_map[h.message_id] = "other"

        # First pass: one result per bucket (ensures diversity)
        for raw_score, mid in scored:
            if mid in seen_ids:
                continue
            normalised = raw_score / max_score
            if normalised < self.score_threshold:
                continue
            bucket = bucket_map.get(mid, "other")
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                seen_ids.add(mid)
                result.append(mid)
                if len(result) >= top_k:
                    break

        # Second pass: fill remaining slots with next-best scores (any bucket)
        if len(result) < top_k:
            for raw_score, mid in scored:
                if mid in seen_ids:
                    continue
                normalised = raw_score / max_score
                if normalised < self.score_threshold:
                    continue
                seen_ids.add(mid)
                result.append(mid)
                if len(result) >= top_k:
                    break

        return result

    # ------------------------------------------------------------------
    # Stage 1: filter to same-user candidates only
    # (context_builder already filters to same user + matching entity,
    #  so relevant_history is already scoped — we just deduplicate here)
    # ------------------------------------------------------------------

    def _filter_candidates(self, ctx: Context) -> List[MessageHistoryItem]:
        seen: set = set()
        out: List[MessageHistoryItem] = []
        for h in ctx.relevant_history:
            if not h.message_id or h.message_id in seen:
                continue
            # Skip completely empty records
            if not h.message_text and not h.media_type:
                continue
            seen.add(h.message_id)
            out.append(h)
        return out

    # ------------------------------------------------------------------
    # Stage 2: score each candidate
    # ------------------------------------------------------------------

    def _score_candidates(
        self, ctx: Context, candidates: List[MessageHistoryItem]
    ) -> List[Tuple[float, str]]:
        msg = ctx.message
        query_text = (msg.message_text or "").strip()
        query_category = _infer_category(query_text, msg.media_type)

        # Compute all semantic similarities in one batch (fast)
        sem_scores = self._semantic_batch(query_text, candidates)

        scored: List[Tuple[float, str]] = []
        for idx, h in enumerate(candidates):
            score = 0.0
            h_text = (h.message_text or "").strip()

            # ── Relationship signals ──────────────────────────────────────
            if msg.sender_user_id and h.sender_user_id == msg.sender_user_id:
                score += 3.5
            if msg.business_id and h.business_id == msg.business_id:
                score += 3.0
            if msg.group_id and h.group_id == msg.group_id:
                score += 2.0
            # same user is always true (filtered by context_builder), small tiebreak
            score += 0.5

            # ── Domain trust bonus (business messages) ────────────────────
            biz = ctx.business
            if biz and msg.business_id and h.business_id == msg.business_id:
                if not biz.domain_mismatch:
                    score += 1.0
                else:
                    score -= 2.0

            # ── User reaction signals ─────────────────────────────────────
            # Use the real boolean fields from message_events
            if h.message_replied:
                score += 3.0
                # Extra bonus for fast replies — strong engagement signal
                if h.reaction_time_minutes > 0 and h.reaction_time_minutes <= 5:
                    score += 1.0
            elif h.message_opened:
                if h.reaction_time_minutes > 0 and h.reaction_time_minutes <= 5:
                    score += 2.0   # opened very quickly
                else:
                    score += 1.0
            if h.notification_dismissed:
                score -= 1.0
            if h.muted_after_message:
                score -= 1.5
            if h.message_reported:
                score -= 2.5

            # ── Content / semantic signals ────────────────────────────────
            sim = float(sem_scores[idx]) if idx < len(sem_scores) else 0.0
            score += sim * 4.0

            # Lexical overlap on top of semantic (rewards direct phrase reuse)
            q_tok = set(_tokens(query_text))
            h_tok = set(_tokens(h_text))
            if q_tok:
                overlap = len(q_tok & h_tok) / len(q_tok)
                score += overlap * 1.2

            # Category alignment
            h_category = _infer_category(h_text, h.media_type)
            if query_category and h_category and query_category == h_category:
                score += 1.5

            # ── Recency decay ─────────────────────────────────────────────
            decay = _recency_decay(msg.created_at, h.created_at, half_life_days=30.0)
            score *= decay

            # Floor at 0
            score = max(0.0, score)
            scored.append((score, h.message_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Semantic similarity helpers
    # ------------------------------------------------------------------

    def _semantic_batch(
        self, query_text: str, candidates: List[MessageHistoryItem]
    ) -> List[float]:
        """Return cosine similarity scores for each candidate vs. the query."""
        texts = [query_text] + [(h.message_text or "").strip() for h in candidates]

        if _SKLEARN and len(candidates) > 0:
            try:
                vec = TfidfVectorizer(stop_words="english", norm="l2", min_df=1)
                tfidf = vec.fit_transform(texts)
                sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                return [max(0.0, float(s)) for s in sims]
            except Exception as exc:
                logger.debug(f"TF-IDF failed, falling back to Jaccard: {exc}")

        # Jaccard fallback
        q_words = set(_tokens(query_text))
        result = []
        for h in candidates:
            h_words = set(_tokens(h.message_text))
            union = q_words | h_words
            inter = q_words & h_words
            result.append(len(inter) / max(1, len(union)))
        return result
