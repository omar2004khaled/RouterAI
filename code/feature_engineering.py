import re
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from models import Context, FeatureSet
from media_processor import MediaProcessor

# These regexes are intentionally narrow and used to compute weighted signals,
# rather than to create a large set of weak binary flags.
URGENCY_PATTERNS = [
    "urgent", "emergency", "immediately", "asap", "water leak", "server crash",
    "critical", "alert", "security breach", "deadline today", "action required",
    "hospital", "ambulance", "police", "help", "alarm"
]

PAYMENT_PATTERNS = [
    "payment", "invoice", "receipt", "total due", "account statement", "refund",
    "transaction", "paid", "bill", "due date", "amount due", "overdue", "bank"
]

PROMOTION_PATTERNS = [
    "sale", "discount", "promo", "offer", "coupon", "clearance", "deal", "collect now"
]

GREETINGS_PATTERNS = [
    "good morning", "hello", "hi there", "happy weekend", "have a great weekend"
]

EVENT_PATTERNS = [
    "meeting", "webinar", "appointment", "schedule", "tomorrow at", "school board", "pta"
]

SCAM_PATTERNS = [
    "verify your bank otp", "send your otp", "claim refund", "guaranteed roi",
    "crypto signal", "500% roi", "send 0.1 btc", "account suspended", "lottery winner",
    "free gift", "bit.ly", "tinyurl", "click link", "processing fee", "claim now",
    "win $10,000", "wire transfer", "gift card", "bonus gift"
]


class FeatureExtractor:
    def __init__(self):
        self.media_processor = MediaProcessor()

    def extract(self, ctx: Context) -> FeatureSet:
        msg = ctx.message
        f_dict: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # User profile signals
        # ------------------------------------------------------------------
        if ctx.user:
            f_dict["usr_open_rate"] = self._safe_float(ctx.user.open_rate)
            f_dict["usr_reply_rate"] = self._safe_float(ctx.user.reply_rate)
            f_dict["usr_dismissal_rate"] = self._safe_float(ctx.user.dismissal_rate)
            f_dict["usr_report_rate"] = self._safe_float(ctx.user.report_rate)
            f_dict["usr_quiet_start"] = self._safe_str(ctx.user.quiet_hours_start)
            f_dict["usr_quiet_end"] = self._safe_str(ctx.user.quiet_hours_end)
            f_dict["usr_is_quiet_hours"] = self._check_quiet_hours(msg.created_at, ctx.user.quiet_hours_start, ctx.user.quiet_hours_end)
            f_dict["usr_rel_strength"] = self._normalize(0.6 * ctx.user.reply_rate + 0.4 * ctx.user.open_rate - 0.5 * ctx.user.dismissal_rate)
            f_dict["usr_trust_score"] = self._normalize(1.0 - (ctx.user.report_rate * 4.0 + ctx.user.dismissal_rate * 0.6))
        else:
            f_dict["usr_open_rate"] = 0.5
            f_dict["usr_reply_rate"] = 0.2
            f_dict["usr_dismissal_rate"] = 0.3
            f_dict["usr_report_rate"] = 0.0
            f_dict["usr_quiet_start"] = "22:00"
            f_dict["usr_quiet_end"] = "07:00"
            f_dict["usr_is_quiet_hours"] = False
            f_dict["usr_rel_strength"] = 0.5
            f_dict["usr_trust_score"] = 0.7

        if ctx.daily_summary:
            daily_load = ctx.daily_summary.notifications_sent
            daily_dismissed = ctx.daily_summary.notifications_dismissed
            f_dict["usr_daily_load"] = daily_load
            f_dict["usr_daily_high_prio"] = max(0, daily_load - daily_dismissed)
            f_dict["usr_daily_muted"] = daily_dismissed
            f_dict["usr_fatigue_index"] = self._normalize(daily_load / 200.0)
            f_dict["usr_muted_bias"] = self._normalize(daily_dismissed / max(1, daily_load))
        else:
            f_dict["usr_daily_load"] = 20
            f_dict["usr_daily_high_prio"] = 2
            f_dict["usr_daily_muted"] = 5
            f_dict["usr_fatigue_index"] = 0.1
            f_dict["usr_muted_bias"] = 0.25

        # ------------------------------------------------------------------
        # Group and group-member signals
        # ------------------------------------------------------------------
        if ctx.group:
            f_dict["grp_size"] = self._safe_int(ctx.group.size)
            f_dict["grp_type"] = self._safe_str(ctx.group.group_type)
            f_dict["grp_activity"] = self._safe_str(ctx.group.activity_level)
            f_dict["grp_admin_count"] = ctx.group.admin_count
            f_dict["grp_is_sender_admin"] = bool(ctx.sender_is_group_admin)
            f_dict["grp_is_broadcast"] = ctx.group.group_type == "broadcast"
            f_dict["grp_is_community"] = ctx.group.group_type == "community"
            f_dict["grp_is_family"] = ctx.group.group_type == "family"
            f_dict["grp_is_work"] = ctx.group.group_type == "work"
        else:
            f_dict["grp_size"] = 0
            f_dict["grp_type"] = "none"
            f_dict["grp_activity"] = "none"
            f_dict["grp_admin_count"] = 0
            f_dict["grp_is_sender_admin"] = False
            f_dict["grp_is_broadcast"] = False
            f_dict["grp_is_community"] = False
            f_dict["grp_is_family"] = False
            f_dict["grp_is_work"] = False

        if ctx.group_member:
            f_dict["grp_mem_role"] = self._safe_str(ctx.group_member.role)
            f_dict["grp_mem_is_muted"] = bool(ctx.group_member.is_muted)
            f_dict["grp_mem_is_user_admin"] = ctx.group_member.role == "admin"
            f_dict["grp_mem_activity"] = self._safe_str(ctx.group_member.activity_level)
        else:
            f_dict["grp_mem_role"] = "member"
            f_dict["grp_mem_is_muted"] = False
            f_dict["grp_mem_is_user_admin"] = False
            f_dict["grp_mem_activity"] = "medium"

        f_dict["grp_is_direct_mention"] = bool(msg.user_id and f"@{msg.user_id}" in (msg.message_text or ""))
        f_dict["grp_mention_strength"] = 1.0 if f_dict["grp_is_direct_mention"] else 0.0

        # ------------------------------------------------------------------
        # Business and relationship signals
        # ------------------------------------------------------------------
        if ctx.business:
            f_dict["biz_verified"] = bool(ctx.business.is_verified)
            f_dict["biz_report_count"] = self._safe_int(ctx.business.user_reports_30d)
            f_dict["biz_account_age"] = self._safe_int(ctx.business.account_age_days)
            f_dict["biz_domain"] = self._safe_str(ctx.business.domain_used_by_sender)
            f_dict["biz_brand"] = self._safe_str(ctx.business.brand_name)
            f_dict["biz_category"] = self._safe_str(ctx.business.category)
            f_dict["biz_domain_mismatch"] = bool(ctx.business.domain_mismatch)
            _sender_domain = (ctx.business.domain_used_by_sender or "").lower()
            f_dict["biz_suspicious_domain"] = (
                ctx.business.domain_mismatch or
                any(ext in _sender_domain for ext in [".top", ".xyz", ".info", ".site"])
            )
            f_dict["biz_sender_domain_age_days"] = self._safe_int(ctx.business.domain_used_by_sender_age_days)
            f_dict["biz_sender_domain_young"] = ctx.business.domain_used_by_sender_age_days < 90
            f_dict["biz_trust_score"] = self._normalize(
                (1.0 if ctx.business.is_verified else 0.0)
                - min(0.6, ctx.business.user_reports_30d / 50.0)
                - (0.3 if ctx.business.domain_mismatch else 0.0)
                - (0.2 if ctx.business.domain_used_by_sender_age_days < 90 else 0.0)
            )
        else:
            f_dict["biz_verified"] = False
            f_dict["biz_report_count"] = 0
            f_dict["biz_account_age"] = 0
            f_dict["biz_domain"] = ""
            f_dict["biz_brand"] = ""
            f_dict["biz_category"] = ""
            f_dict["biz_domain_mismatch"] = False
            f_dict["biz_suspicious_domain"] = False
            f_dict["biz_sender_domain_age_days"] = 0
            f_dict["biz_sender_domain_young"] = False
            f_dict["biz_trust_score"] = 0.2

        if ctx.user_business_history:
            ubiz = ctx.user_business_history
            opt_in_status = ubiz.opt_in_status  # derived property
            f_dict["ubiz_activity_180d"] = ubiz.activity_count_180d
            f_dict["ubiz_orders"] = ubiz.activity_count_180d   # compat alias
            f_dict["ubiz_bookings"] = 0
            f_dict["ubiz_payments"] = 0
            f_dict["ubiz_opened_30d"] = ubiz.messages_opened_30d
            f_dict["ubiz_dismissed_30d"] = ubiz.messages_dismissed_30d
            f_dict["ubiz_replied_30d"] = ubiz.messages_replied_30d
            f_dict["ubiz_opt_in"] = opt_in_status
            f_dict["ubiz_is_opted_out"] = opt_in_status == "opted_out"
            f_dict["ubiz_is_opted_in"] = opt_in_status == "opted_in"
            f_dict["ubiz_allows_promotions"] = ubiz.allows_promotions
            f_dict["ubiz_why_known"] = ubiz.why_user_knows_account
            f_dict["ubiz_has_active_relationship"] = (
                ubiz.activity_count_180d > 0
                or ubiz.messages_replied_30d > 0
                or opt_in_status == "opted_in"
            )
            # Engagement ratio: how often does user open vs dismiss biz messages
            total_biz = max(1, ubiz.messages_opened_30d + ubiz.messages_dismissed_30d)
            f_dict["ubiz_engagement_ratio"] = round(ubiz.messages_opened_30d / total_biz, 3)
            f_dict["ubiz_relationship_strength"] = self._normalize(
                (ubiz.activity_count_180d * 0.02)
                + (ubiz.messages_replied_30d * 0.15)
                + (ubiz.messages_opened_30d * 0.05)
                + (0.3 if opt_in_status == "opted_in" else 0.0)
                - (ubiz.messages_dismissed_30d * 0.03)
            )
            f_dict["ubiz_recent_delivery_signal"] = 1.0 if ubiz.activity_count_180d > 0 else 0.0
            f_dict["ubiz_payment_signal"] = 0.0
        else:
            f_dict["ubiz_activity_180d"] = 0
            f_dict["ubiz_orders"] = 0
            f_dict["ubiz_bookings"] = 0
            f_dict["ubiz_payments"] = 0
            f_dict["ubiz_opened_30d"] = 0
            f_dict["ubiz_dismissed_30d"] = 0
            f_dict["ubiz_replied_30d"] = 0
            f_dict["ubiz_opt_in"] = "none"
            f_dict["ubiz_is_opted_out"] = False
            f_dict["ubiz_is_opted_in"] = False
            f_dict["ubiz_allows_promotions"] = False
            f_dict["ubiz_why_known"] = ""
            f_dict["ubiz_has_active_relationship"] = False
            f_dict["ubiz_engagement_ratio"] = 0.0
            f_dict["ubiz_relationship_strength"] = 0.0
            f_dict["ubiz_recent_delivery_signal"] = 0.0
            f_dict["ubiz_payment_signal"] = 0.0

        # ------------------------------------------------------------------
        # Text and entity signals using weighted scoring instead of broad flags
        # ------------------------------------------------------------------
        text = self._safe_str(msg.message_text)
        media_signals = self.media_processor.process(ctx.media)
        combined_text = " ".join(part for part in [text, media_signals.get("extracted_text", "")] if part).strip()
        lower_comb = combined_text.lower()

        urgency_score = self._weighted_score(lower_comb, URGENCY_PATTERNS, 0.55)
        payment_score = self._weighted_score(lower_comb, PAYMENT_PATTERNS, 0.45)
        promotion_score = self._weighted_score(lower_comb, PROMOTION_PATTERNS, 0.45)
        greeting_score = self._weighted_score(lower_comb, GREETINGS_PATTERNS, 0.45)
        event_score = self._weighted_score(lower_comb, EVENT_PATTERNS, 0.45)
        scam_score = self._weighted_score(lower_comb, SCAM_PATTERNS, 0.5)

        # Weighted entity extraction with safe numeric normalization.
        money_count = len(re.findall(r"[$€£₹]\d+(?:\.\d+)?|\b\d+\s?(usd|eur|gbp|btc|eth)\b", lower_comb))
        phone_match = bool(re.search(r"\+?\d{10,12}", combined_text))
        deadline_hit = bool(re.search(r"\b(?:today|tomorrow|next\s+day|due\s+today|deadline)\b", lower_comb))
        date_hit = bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}\b", combined_text, re.I))
        time_hit = bool(re.search(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", combined_text, re.I))
        suspicious_url = bool(re.search(r"https?://(bit\.ly|tinyurl\.com|[a-z0-9-]+\.(?:top|info|xyz|site))", lower_comb))
        qr_indicator = bool(media_signals.get("has_qr_code", False) or "qr" in lower_comb or "scan" in lower_comb or "barcode" in lower_comb)
        otp_hit = bool(re.search(r"\b(?:otp|one[- ]time password|verification code)\b", lower_comb))

        f_dict["txt_length"] = len(text)
        f_dict["txt_word_count"] = len(text.split())
        f_dict["txt_forwarded_count"] = msg.forwarded_count
        f_dict["txt_urgency_score"] = self._clamp(urgency_score, 0.0, 1.0)
        f_dict["txt_payment_score"] = self._clamp(payment_score, 0.0, 1.0)
        f_dict["txt_promotion_score"] = self._clamp(promotion_score, 0.0, 1.0)
        f_dict["txt_greeting_score"] = self._clamp(greeting_score, 0.0, 1.0)
        f_dict["txt_event_score"] = self._clamp(event_score, 0.0, 1.0)
        f_dict["txt_scam_score"] = self._clamp(scam_score, 0.0, 1.0)
        f_dict["txt_has_urgency"] = urgency_score >= 0.55 or bool(media_signals.get("is_urgent", False))
        f_dict["txt_has_payment"] = payment_score >= 0.5 or money_count > 0
        f_dict["txt_has_scam"] = scam_score >= 0.5 or suspicious_url or qr_indicator
        f_dict["txt_has_otp"] = otp_hit or bool(re.search(r"\b\d{4,6}\b", lower_comb))
        f_dict["txt_has_short_url"] = suspicious_url
        f_dict["txt_has_phone"] = phone_match
        f_dict["txt_has_currency"] = money_count > 0
        f_dict["txt_has_date"] = date_hit
        f_dict["txt_has_time"] = time_hit
        f_dict["txt_has_deadline"] = deadline_hit
        f_dict["txt_has_qr_indicator"] = qr_indicator
        f_dict["txt_money_amount"] = min(money_count, 5)
        f_dict["txt_has_delivery"] = bool(re.search(r"\bdelivery\b|\bdriver\b|\border\s*#\b", lower_comb))
        f_dict["txt_has_lottery"] = bool(re.search(r"\blottery\b|\bwon\b|\bbonus\b\s+\bgift\b", lower_comb))
        f_dict["txt_has_meeting"] = bool(re.search(r"\bmeeting\b|\bwebinar\b|\bappointment\b|\bschedule\b|\bpta\b", lower_comb))

        # ------------------------------------------------------------------
        # Media and transcript quality signals
        # ------------------------------------------------------------------
        f_dict["media_has_media"] = bool(media_signals.get("has_media", False))
        f_dict["media_type"] = media_signals.get("media_type", "empty")
        f_dict["media_category"] = media_signals.get("category", "none")
        f_dict["media_is_image"] = media_signals.get("media_type") == "image"
        f_dict["media_is_voice"] = media_signals.get("media_type") == "voice"
        f_dict["media_has_qr"] = bool(media_signals.get("has_qr_code", False))
        f_dict["media_ocr_conf"] = self._safe_float(media_signals.get("ocr_confidence", 0.0))
        f_dict["media_asr_conf"] = self._safe_float(media_signals.get("transcription_confidence", 0.0))
        f_dict["media_voice_duration"] = self._safe_float(media_signals.get("voice_duration", 0.0))

        # ------------------------------------------------------------------
        # History and sender relationship quality
        # ------------------------------------------------------------------
        relevant_history = list(ctx.relevant_history or [])
        f_dict["hist_matches_count"] = len(relevant_history)
        f_dict["hist_prior_reported"] = sum(1 for h in relevant_history if h.message_reported)
        f_dict["hist_prior_replied"] = sum(1 for h in relevant_history if h.message_replied)
        f_dict["hist_prior_opened"] = sum(1 for h in relevant_history if h.message_opened)
        f_dict["hist_prior_dismissed"] = sum(1 for h in relevant_history if h.notification_dismissed)
        f_dict["hist_prior_muted"] = sum(1 for h in relevant_history if h.muted_after_message)
        f_dict["hist_fast_replies"] = sum(1 for h in relevant_history if h.message_replied and 0 < h.reaction_time_minutes <= 5)
        f_dict["hist_sender_history_count"] = sum(1 for h in relevant_history if h.sender_user_id == msg.sender_user_id)
        replied_from_sender = sum(1 for h in relevant_history if h.sender_user_id == msg.sender_user_id and h.message_replied)
        opened_from_sender = sum(1 for h in relevant_history if h.sender_user_id == msg.sender_user_id and h.message_opened)
        f_dict["hist_sender_trust_score"] = self._normalize(
            (replied_from_sender * 0.5 + opened_from_sender * 0.2)
            / max(1, f_dict["hist_sender_history_count"])
        )
        f_dict["hist_repeated_suspicion"] = 1.0 if f_dict["hist_prior_reported"] >= 1 else 0.0
        f_dict["hist_engagement_signal"] = self._normalize(
            (f_dict["hist_prior_replied"] * 0.6)
            + (f_dict["hist_fast_replies"] * 0.3)
            + (f_dict["hist_prior_opened"] * 0.2)
            - (f_dict["hist_prior_reported"] * 0.8)
            - (f_dict["hist_prior_muted"] * 0.4)
        )

        # ------------------------------------------------------------------
        # Build the compatible dataclass shape used by downstream modules.
        # ------------------------------------------------------------------
        fs = FeatureSet(
            features=f_dict,
            is_quiet_hours=f_dict["usr_is_quiet_hours"],
            user_dismissal_rate=f_dict["usr_dismissal_rate"],
            user_report_rate=f_dict["usr_report_rate"],
            daily_notification_load=f_dict["usr_daily_load"],
            is_group_muted=f_dict["grp_mem_is_muted"],
            is_user_group_admin=f_dict["grp_mem_is_user_admin"],
            is_sender_group_admin=f_dict["grp_is_sender_admin"],
            group_size=f_dict["grp_size"],
            is_direct_mention=f_dict["grp_is_direct_mention"],
            is_business_verified=f_dict["biz_verified"],
            business_report_count=f_dict["biz_report_count"],
            is_opted_out=f_dict["ubiz_is_opted_out"],
            has_active_customer_relationship=f_dict["ubiz_has_active_relationship"],
            has_urgency_keywords=f_dict["txt_has_urgency"],
            has_payment_ask=f_dict["txt_has_payment"],
            has_otp=f_dict["txt_has_otp"],
            has_short_url=f_dict["txt_has_short_url"],
            forwarded_magnitude=msg.forwarded_count,
            media_type=media_signals.get("media_type", "empty"),
            media_extracted_text=media_signals.get("extracted_text", ""),
            media_category=media_signals.get("category", "none"),
            history_matches_count=f_dict["hist_matches_count"],
            prior_reported_count=f_dict["hist_prior_reported"],
            prior_opened_count=f_dict["hist_prior_opened"]
        )

        return fs

    def _weighted_score(self, text: str, patterns: List[str], threshold: float) -> float:
        # Count how many relevant phrase hits are present. This produces a softer
        # signal than a one-shot binary keyword test and reduces noise.
        hits = 0
        for p in patterns:
            if p in text:
                hits += 1
        score = min(1.0, hits / max(1, len(patterns) // 3))
        return self._clamp(score, 0.0, 1.0)

    def _normalize(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _safe_str(self, value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    def _check_quiet_hours(self, created_at_str: str, start_str: str, end_str: str) -> bool:
        if not start_str or not end_str or not created_at_str:
            return False
        try:
            dt = datetime.strptime(created_at_str.split()[1] if " " in created_at_str else created_at_str, "%H:%M:%S")
            start_t = datetime.strptime(start_str, "%H:%M").time()
            end_t = datetime.strptime(end_str, "%H:%M").time()
            msg_t = dt.time()

            if start_t <= end_t:
                return start_t <= msg_t <= end_t
            else:
                return msg_t >= start_t or msg_t <= end_t
        except Exception:
            return False
