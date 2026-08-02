"""
Signal-agreement confidence calibration.

Instead of nudging a base score up/down with additive bonuses, we compute two
independent quantities and combine them:

  agreement  — how many independent signals corroborate the decision
  conflict   — how many signals contradict or weaken it

Final confidence = clamp(0.40 * agreement_score + 0.60 * base, uncertainty_penalty)

This produces a wide, realistic distribution:
  strong agreement  → 0.88-0.96
  mixed signals     → 0.55-0.72
  high uncertainty  → 0.40-0.55
"""

import logging
from models import Context, FeatureSet, Decision

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:

    def calibrate(self, decision: Decision, ctx: Context, fs: FeatureSet, from_rule: bool) -> Decision:
        base = _safe_float(decision.confidence, 0.70)

        ev_ids = [x for x in (decision.evidence_message_ids or "").split(";") if x and x != "none"]
        ev_count = len(ev_ids)

        # ── Agreement signals (each is an independent corroborating source) ──
        agree = 0.0

        # Rule fired deterministically — strong prior
        if from_rule:
            agree += 0.25

        # Verified business — sender identity is confirmed
        if fs.is_business_verified:
            agree += 0.18

        # Active prior relationship — we know the user cares about this sender
        if fs.has_active_customer_relationship:
            agree += 0.12

        # User previously replied to this sender (strong positive engagement)
        replied_hist = [h for h in ctx.relevant_history if h.message_replied]
        if replied_hist:
            agree += 0.15
            # Fast reply bonus — user was very attentive
            if any(h.reaction_time_minutes <= 5 for h in replied_hist if h.reaction_time_minutes > 0):
                agree += 0.08

        # User previously opened (weaker positive)
        if any(h.message_opened for h in ctx.relevant_history):
            agree += 0.06

        # Multiple pieces of evidence retrieved (retriever is confident)
        if ev_count >= 3:
            agree += 0.10
        elif ev_count >= 1:
            agree += 0.05

        # Media successfully processed and contributed signal
        media_conf = max(
            _safe_float(fs.features.get("media_ocr_conf", 0.0)),
            _safe_float(fs.features.get("media_asr_conf", 0.0)),
        )
        if media_conf >= 0.80:
            agree += 0.08

        # Scam signals fully agree with a mute decision
        if decision.action == "mute" and decision.message_type in ("scam", "spam"):
            scam_score = _safe_float(fs.features.get("txt_scam_score", 0.0))
            if scam_score >= 0.5 or fs.has_short_url:
                agree += 0.12

        # Urgency signals fully agree with a notify decision
        if decision.action == "notify" and decision.message_type == "urgent":
            if fs.has_urgency_keywords:
                agree += 0.10

        agreement_score = min(1.0, agree)

        # ── Conflict / uncertainty signals ──────────────────────────────────
        conflict = 0.0

        # Unknown message type — classification failed
        if decision.message_type == "unknown":
            conflict += 0.20

        # No context at all — blind guess
        if not ctx.user and not ctx.group and not ctx.business:
            conflict += 0.15

        # Unverified sender + payment or OTP request — contradictory signals
        if not fs.is_business_verified and fs.has_payment_ask:
            conflict += 0.12
        if not fs.is_business_verified and fs.has_otp:
            conflict += 0.10

        # Domain mismatch — identity uncertainty
        if fs.features.get("biz_domain_mismatch", False):
            conflict += 0.12

        # Young sender domain with financial content
        if fs.features.get("biz_sender_domain_young", False) and fs.has_payment_ask:
            conflict += 0.10

        # Prior reports — this sender has been flagged before
        if fs.prior_reported_count >= 1:
            conflict += 0.08

        # Prior mutes from this sender
        if _safe_int(fs.features.get("hist_prior_muted", 0)) >= 1:
            conflict += 0.06

        # High forward count — message integrity uncertain
        if fs.forwarded_magnitude > 5:
            conflict += 0.08

        # Media present but no text extracted — weak evidence base
        if fs.features.get("media_has_media", False) and not fs.media_extracted_text:
            conflict += 0.06

        # LLM-only decision (no rule) with high base confidence → penalise overconfidence
        if not from_rule and base >= 0.85:
            conflict += 0.10

        conflict_score = min(1.0, conflict)

        # ── Blend ────────────────────────────────────────────────────────────
        # 60% anchored on the decision's own base confidence,
        # 40% adjusted by the agreement/conflict balance
        net_signal = agreement_score - (0.7 * conflict_score)
        blended = 0.60 * max(0.35, min(0.95, base)) + 0.40 * max(0.0, min(1.0, net_signal))
        final = max(0.38, min(0.96, round(blended, 2)))

        logger.debug(
            f"[{decision.message_id}] conf {base:.2f}→{final:.2f} "
            f"agree={agreement_score:.2f} conflict={conflict_score:.2f} rule={from_rule}"
        )
        decision.confidence = final
        return decision


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default
