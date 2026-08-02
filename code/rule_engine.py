import re
from typing import Optional, List, Tuple
from models import Context, FeatureSet, Decision

# Business category → preferred message_type mapping
_BIZ_CATEGORY_TO_TYPE = {
    "ecommerce_delivery": "business_update",
    "bank": "payment",
    "finance": "payment",
    "insurance": "payment",
    "healthcare": "event",
    "pharmacy": "event",
    "food_delivery": "business_update",
    "fashion": "promotion",
    "electronics": "promotion",
    "travel": "business_update",
    "education": "event",
    "telecom": "business_update",
    "utility": "payment",
    "real_estate": "event",
}


class RuleEngine:
    def __init__(self):
        self.rule_count = 90

    def evaluate(self, ctx: Context, fs: FeatureSet, evidence_ids: str) -> Optional[Decision]:
        msg = ctx.message
        comb_text = ((msg.message_text or "") + " " + (fs.media_extracted_text or "")).lower()

        # ---------------------------------------------------------------------
        # EMERGENCY & FAMILY SAFETY RULES
        # ---------------------------------------------------------------------
        if self._contains_any(comb_text, ["hospital", "ambulance", "emergency room", "medical emergency"]):
            return self._decision(msg, evidence_ids, "notify", "urgent", "R01: Critical hospital/medical emergency alert.", 0.99)

        if self._contains_any(comb_text, ["water leak", "fire alarm", "power outage", "gas leak"]):
            return self._decision(msg, evidence_ids, "notify", "urgent", "R02: Critical infrastructure hazard/emergency alert.", 0.99)

        if self._contains_any(comb_text, ["server crash", "production incident", "security breach", "system outage"]):
            return self._decision(msg, evidence_ids, "notify", "urgent", "R03: Critical DevOps production incident/outage.", 0.98)

        if fs.features.get("grp_is_family") and self._contains_any(comb_text, ["urgent", "call me", "help", "please respond immediately"]):
            return self._decision(msg, evidence_ids, "notify", "urgent", "R04: Urgent family emergency communication.", 0.97)

        # ---------------------------------------------------------------------
        # DIRECT MENTION / MUTED GROUP OVERRIDES
        # ---------------------------------------------------------------------
        if fs.is_group_muted and (fs.is_direct_mention or fs.has_urgency_keywords):
            return self._decision(msg, evidence_ids, "notify", "urgent", "R26: Direct @mention or emergency alert in muted group overrides mute.", 0.98)

        # ---------------------------------------------------------------------
        # SCAM / FRAUD DETECTION
        # ---------------------------------------------------------------------
        suspicious_url = fs.has_short_url
        qr_risk = bool(fs.features.get("media_has_qr", False))
        otp_risk = fs.has_otp
        payment_risk = fs.has_payment_ask
        domain_mismatch = bool(fs.features.get("biz_domain_mismatch", False))
        domain_young = bool(fs.features.get("biz_sender_domain_young", False))
        high_risk_fraud = suspicious_url or otp_risk or payment_risk or qr_risk

        # Domain mismatch on a financial/bank business is a strong scam signal
        if domain_mismatch and self._contains_any(comb_text, ["bank", "payment", "otp", "verify", "account", "login", "security"]):
            return self._decision(msg, evidence_ids, "mute", "scam",
                f"R10: Sender domain does not match official business domain — likely phishing.", 0.97)

        if (fs.business_report_count >= 5 or fs.prior_reported_count >= 1) and high_risk_fraud:
            return self._decision(msg, evidence_ids, "mute", "scam",
                f"R11: Known bad entity ({fs.business_report_count} reports) with suspicious payment/security link.", 0.99)

        if (not fs.is_business_verified) and (fs.forwarded_magnitude >= 5) and high_risk_fraud:
            return self._decision(msg, evidence_ids, "mute", "scam",
                f"R12: Unverified sender with high forwarded count ({fs.forwarded_magnitude}) requesting OTP/payment.", 0.98)

        if self._is_lottery_or_winner_scam(comb_text):
            return self._decision(msg, evidence_ids, "mute", "scam", "R13: Fake lottery / reward winner scam requesting fee.", 0.99)

        if self._is_crypto_investment_scam(comb_text):
            return self._decision(msg, evidence_ids, "mute", "scam", "R14: Fraudulent high-yield crypto investment scam.", 0.98)

        if qr_risk and self._contains_any(comb_text, ["claim", "bonus", "gift", "win", "verify", "login", "otp"]):
            return self._decision(msg, evidence_ids, "mute", "scam", "R15: QR-driven reward or verification scam.", 0.97)

        if not fs.is_business_verified and (otp_risk or suspicious_url) and self._contains_any(comb_text, ["bank", "login", "security", "account", "verify"]):
            return self._decision(msg, evidence_ids, "mute", "scam", "R16: Unverified banking message requests OTP or uses suspicious link.", 0.97)

        # Young sender domain + financial request = scam
        if domain_young and not fs.is_business_verified and self._contains_any(comb_text, ["payment", "bank", "otp", "verify", "transfer"]):
            return self._decision(msg, evidence_ids, "mute", "scam", "R18: Very new sender domain with financial request — high phishing risk.", 0.95)

        # Verified bank OTP is legitimate
        if fs.is_business_verified and otp_risk and self._contains_any(comb_text, ["bank", "login", "two-factor", "otp", "verification"]):
            return self._decision(msg, evidence_ids, "notify", "payment", "R17: Authentic 2FA login verification code from verified bank.", 0.96)

        # ---------------------------------------------------------------------
        # BUSINESS OPT-OUTS & PROMOTIONAL MUTE RULES
        # ---------------------------------------------------------------------
        if fs.is_opted_out and not fs.has_urgency_keywords:
            biz_name = ctx.business.brand_name if ctx.business else "business"
            return self._decision(msg, evidence_ids, "mute", "promotion",
                f"R28: User explicitly opted out of marketing from {biz_name}.", 0.96)

        if fs.is_group_muted and not fs.is_direct_mention and self._contains_any(comb_text, ["promo", "discount", "sale", "coupon", "offer"]):
            return self._decision(msg, evidence_ids, "mute", "spam", "R27: Unsolicited promotional chatter in muted group.", 0.96)

        # Unverified business sending promotional content → mute
        if not fs.is_business_verified and msg.conversation_type == "business" and \
                float(fs.features.get("txt_promotion_score", 0)) >= 0.33 and \
                not fs.has_active_customer_relationship:
            return self._decision(msg, evidence_ids, "mute", "promotion",
                "R29: Unsolicited promotion from unverified business with no prior relationship.", 0.88)

        # ---------------------------------------------------------------------
        # DELIVERY, PAYMENT, AND BUSINESS UPDATE RULES
        # ---------------------------------------------------------------------
        if fs.is_business_verified and fs.has_active_customer_relationship and \
                self._contains_any(comb_text, ["order", "driver", "deliv", "arriving", "package", "tracking", "shipped"]):
            biz_name = ctx.business.brand_name if ctx.business else "verified business"
            return self._decision(msg, evidence_ids, "notify", "business_update",
                f"R41: Verified business {biz_name} — active delivery/order update.", 0.97)

        if fs.is_business_verified and \
                self._contains_any(comb_text, ["statement", "invoice", "total due", "amount due", "bill", "payment reminder"]):
            return self._decision(msg, evidence_ids, "digest", "payment",
                "R42: Financial statement/invoice from verified business.", 0.88)

        # Business category-driven routing: verified sender + known category
        if fs.is_business_verified and ctx.business:
            biz_type = self._infer_type_from_business(ctx.business.category, comb_text, fs)
            if biz_type:
                action = self._action_for_type(biz_type, fs)
                return self._decision(msg, evidence_ids, action, biz_type,
                    f"R50: Verified {ctx.business.category} business — routed by category.", 0.84)

        # ---------------------------------------------------------------------
        # EVENT / CALENDAR RULES
        # ---------------------------------------------------------------------
        if self._contains_any(comb_text, ["pta", "school board", "parent meeting", "meeting", "webinar",
                                           "appointment", "schedule", "tomorrow at", "rescheduled"]):
            return self._decision(msg, evidence_ids, "digest", "event", "R43: Scheduled event or meeting notice.", 0.86)

        # ---------------------------------------------------------------------
        # VOICE NOTE RULES
        # ---------------------------------------------------------------------
        if fs.media_type == "voice" and fs.has_urgency_keywords:
            return self._decision(msg, evidence_ids, "notify", "urgent", "R44: Voice note contains urgency alert.", 0.97)

        if fs.media_type == "voice" and not fs.has_urgency_keywords and \
                self._contains_any(comb_text, ["good morning", "hello", "weekend", "family", "happy"]):
            return self._decision(msg, evidence_ids, "digest", "greeting", "R45: Casual voice greeting.", 0.84)

        # ---------------------------------------------------------------------
        # QUIET HOURS RULE
        # ---------------------------------------------------------------------
        if fs.is_quiet_hours and not fs.has_urgency_keywords and not fs.is_direct_mention:
            msg_type = self._classify_message_type(msg, ctx, fs, comb_text)
            dnd = ctx.user.do_not_disturb_window if ctx.user else "DND"
            return self._decision(msg, evidence_ids, "digest", msg_type,
                f"R46: Non-urgent message during quiet hours ({dnd}).", 0.85)

        return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _infer_type_from_business(self, category: str, comb_text: str, fs: FeatureSet) -> Optional[str]:
        """Map business category + text signals to a message_type."""
        cat = (category or "").lower()
        mapped = _BIZ_CATEGORY_TO_TYPE.get(cat)
        if mapped:
            return mapped
        # Fuzzy fallbacks
        if "bank" in cat or "financ" in cat:
            return "payment"
        if "health" in cat or "pharma" in cat or "clinic" in cat:
            return "event"
        if "deliver" in cat or "logistics" in cat or "courier" in cat:
            return "business_update"
        if "fashion" in cat or "retail" in cat or "shop" in cat:
            return "promotion"
        return None

    def _action_for_type(self, msg_type: str, fs: FeatureSet) -> str:
        """Given a message_type, pick action using engagement/urgency signals."""
        if msg_type in ("urgent",):
            return "notify"
        if msg_type == "business_update":
            return "notify" if fs.has_active_customer_relationship else "digest"
        if msg_type == "payment":
            return "digest"
        if msg_type in ("promotion", "spam"):
            return "mute" if not fs.features.get("ubiz_allows_promotions", False) else "digest"
        if msg_type == "event":
            return "digest"
        return "digest"

    def _classify_message_type(self, msg, ctx: Context, fs: FeatureSet, comb_text: str) -> str:
        """Best-effort message_type when falling through to quiet-hours rule."""
        if msg.conversation_type == "business" and ctx.business:
            t = self._infer_type_from_business(ctx.business.category, comb_text, fs)
            if t:
                return t
        if float(fs.features.get("txt_urgency_score", 0)) >= 0.4:
            return "urgent"
        if float(fs.features.get("txt_payment_score", 0)) >= 0.4:
            return "payment"
        if float(fs.features.get("txt_promotion_score", 0)) >= 0.4:
            return "promotion"
        if float(fs.features.get("txt_event_score", 0)) >= 0.4:
            return "event"
        if msg.conversation_type == "personal":
            return "personal"
        return "business_update"

    def _decision(self, msg, evidence_ids: str, action: str, message_type: str,
                  reason: str, confidence: float) -> Decision:
        return Decision(
            message_id=msg.message_id,
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=evidence_ids
        )

    def _contains_any(self, text: str, phrases: List[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _is_lottery_or_winner_scam(self, text: str) -> bool:
        return self._contains_any(text, [
            "lottery", "you won", "free gift", "claim now", "claim-reward-now",
            "win $10,000", "processing fee", "gift card", "bonus gift"
        ])

    def _is_crypto_investment_scam(self, text: str) -> bool:
        return self._contains_any(text, [
            "crypto", "500% roi", "0.1 btc", "pump signal", "roi-1000", "guaranteed roi"
        ])

