import os
import json
import logging
import re
from typing import Optional, List, Dict, Any
from models import Context, FeatureSet, Decision
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class ReasoningEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash", max_retries: int = 3):
        self.model_name = model_name
        self.max_retries = max_retries

    def evaluate(self, ctx: Context, fs: FeatureSet, evidence_ids: List[str], rule_signal: str = "No pre-LLM rule fired") -> Decision:
        prompt = self._format_prompt(ctx, fs, evidence_ids, rule_signal)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response_text = self._call_llm(prompt, api_key)
                    parsed = self._parse_json(response_text)
                    if parsed:
                        ev_str = ";".join(parsed.get("evidence_message_ids", [])) if parsed.get("evidence_message_ids") else ("none" if not evidence_ids else ";".join(evidence_ids))
                        return Decision(
                            message_id=ctx.message.message_id,
                            action=parsed.get("action", "digest"),
                            message_type=parsed.get("message_type", "unknown"),
                            reason=parsed.get("reason", "Structured LLM context reasoning decision."),
                            confidence=float(parsed.get("confidence", 0.85)),
                            evidence_message_ids=ev_str
                        )
                except Exception as e:
                    logger.warning(f"LLM API attempt {attempt}/{self.max_retries} failed ({e}).")

        return self._heuristic_reasoning(ctx, fs, evidence_ids, rule_signal)

    def _format_prompt(self, ctx: Context, fs: FeatureSet, evidence_ids: List[str], rule_signal: str) -> str:
        msg = ctx.message
        user = ctx.user
        group = ctx.group
        biz = ctx.business
        ubiz = ctx.user_business_history

        history_lines = []
        for h in ctx.relevant_history:
            if h.message_id in evidence_ids:
                history_lines.append(f"- [{h.message_id}] Action={h.action_taken}, Text='{h.message_text}'")
        history_summary = "\n".join(history_lines) if history_lines else "No relevant history available."

        feature_summary = self._build_feature_summary(fs)
        evidence_summary = ";".join(evidence_ids) if evidence_ids else "none"

        return USER_PROMPT_TEMPLATE.format(
            message_id=msg.message_id,
            user_id=msg.user_id,
            conversation_type=msg.conversation_type,
            sender_user_id=msg.sender_user_id or "N/A",
            group_id=msg.group_id or "N/A",
            business_id=msg.business_id or "N/A",
            created_at=msg.created_at,
            message_text=msg.message_text,
            forwarded_count=msg.forwarded_count,
            quiet_hours_start=user.quiet_hours_start if user else "N/A",
            quiet_hours_end=user.quiet_hours_end if user else "N/A",
            is_quiet_hours=fs.is_quiet_hours,
            daily_notification_load=fs.daily_notification_load,
            open_rate=user.open_rate if user else "N/A",
            reply_rate=user.reply_rate if user else "N/A",
            dismissal_rate=user.dismissal_rate if user else "N/A",
            report_rate=user.report_rate if user else "N/A",
            group_name=group.group_name if group else "N/A",
            group_type=group.group_type if group else "N/A",
            group_size=fs.group_size,
            is_group_muted=fs.is_group_muted,
            is_direct_mention=fs.is_direct_mention,
            brand_name=biz.brand_name if biz else "N/A",
            is_verified=fs.is_business_verified,
            domain=biz.domain_used_by_sender if biz else "N/A",
            report_count=fs.business_report_count,
            orders_count=ubiz.activity_count_180d if ubiz else 0,
            bookings_count=0,
            payments_count=0,
            opt_in_status=ubiz.opt_in_status if ubiz else "none",
            media_type=fs.media_type or "none",
            media_category=fs.media_category,
            media_extracted_text=fs.media_extracted_text or "none",
            has_qr_code=fs.features.get("media_has_qr", False),
            has_phone_number=fs.features.get("txt_has_phone", False),
            history_summary=history_summary,
            evidence_summary=evidence_summary,
            rule_signals_summary=rule_signal,
            feature_summary=feature_summary
        )

    def _call_llm(self, prompt: str, api_key: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt)
        return resp.text

    def _parse_json(self, text: str) -> Optional[dict]:
        if not text:
            return None

        cleaned = self._clean_response_text(text)
        for candidate in self._json_candidates(cleaned):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    repaired = self._repair_json(candidate)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
        return None

    def _clean_response_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    def _json_candidates(self, text: str) -> List[str]:
        candidates = []
        if text.startswith("{") and text.endswith("}"):
            candidates.append(text)

        start = text.find("{")
        if start != -1:
            end = self._find_matching_brace(text, start)
            if end != -1:
                candidates.append(text[start : end + 1])

        if not candidates:
            candidates.append(text)
        return candidates

    def _find_matching_brace(self, text: str, start_index: int) -> int:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start_index, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _repair_json(self, candidate: str) -> str:
        repaired = candidate.strip()
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"(?<![\"'])\b([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', repaired)
        repaired = repaired.strip()
        return repaired

    def _build_feature_summary(self, fs: FeatureSet) -> str:
        summary: Dict[str, Any] = {
            "urgency": round(float(fs.features.get("txt_urgency_score", 0.0)), 3),
            "payment": round(float(fs.features.get("txt_payment_score", 0.0)), 3),
            "scam": round(float(fs.features.get("txt_scam_score", 0.0)), 3),
            "promotion": round(float(fs.features.get("txt_promotion_score", 0.0)), 3),
            "event": round(float(fs.features.get("txt_event_score", 0.0)), 3),
            "greeting": round(float(fs.features.get("txt_greeting_score", 0.0)), 3),
            "sender_trust": round(float(fs.features.get("usr_rel_strength", 0.0)), 3),
            "business_trust": round(float(fs.features.get("biz_trust_score", 0.0)), 3),
            "relationship_strength": round(float(fs.features.get("ubiz_relationship_strength", 0.0)), 3),
            "quiet_hours": bool(fs.is_quiet_hours),
            "direct_mention": bool(fs.is_direct_mention),
            "is_business_verified": bool(fs.is_business_verified),
            "is_opted_out": bool(fs.is_opted_out),
            "has_short_url": bool(fs.features.get("txt_has_short_url", False)),
            "has_phone": bool(fs.features.get("txt_has_phone", False)),
            "has_qr": bool(fs.features.get("txt_has_qr_indicator", False)),
            "has_otp": bool(fs.features.get("txt_has_otp", False)),
            "forwarded_magnitude": int(fs.forwarded_magnitude),
            "history_matches": int(fs.history_matches_count),
            "prior_reported_count": int(fs.prior_reported_count),
        }
        return json.dumps(summary, sort_keys=True)

    def _heuristic_reasoning(self, ctx: Context, fs: FeatureSet, evidence_ids: List[str], rule_signal: str = "No pre-LLM rule fired") -> Decision:
        msg = ctx.message
        text_lower = (msg.message_text or "").lower()
        ev_str = ";".join(evidence_ids) if evidence_ids else "none"

        urgency = float(fs.features.get("txt_urgency_score", 0.0))
        payment = float(fs.features.get("txt_payment_score", 0.0))
        scam = float(fs.features.get("txt_scam_score", 0.0))
        promotion = float(fs.features.get("txt_promotion_score", 0.0))
        greeting = float(fs.features.get("txt_greeting_score", 0.0))
        event = float(fs.features.get("txt_event_score", 0.0))
        sender_trust = float(fs.features.get("usr_rel_strength", 0.5))
        business_trust = float(fs.features.get("biz_trust_score", 0.5))
        relationship_strength = float(fs.features.get("ubiz_relationship_strength", 0.0))
        forwarded = int(fs.forwarded_magnitude)
        history_count = int(fs.history_matches_count)
        reported_count = int(fs.prior_reported_count)

        risk_score = min(1.0, (scam * 0.45) + (float(fs.features.get("txt_has_phone", False)) * 0.15) + (float(fs.features.get("txt_has_short_url", False)) * 0.15) + (float(fs.features.get("txt_has_otp", False)) * 0.15) + (float(fs.features.get("txt_has_qr_indicator", False)) * 0.10))
        risk_score += max(0.0, (forwarded - 1) * 0.03)
        risk_score = min(1.0, risk_score)

        rule_label = (rule_signal or "").lower()
        if any(token in rule_label for token in ("mute", "scam", "spam", "otp", "lottery", "phish", "crypto")):
            return Decision(
                message_id=msg.message_id,
                action="mute",
                message_type="scam" if scam >= 0.45 or "scam" in rule_label else "spam",
                reason=f"Deterministic rule precedence and weighted risk signals indicate suspicious or malicious content. Risk={risk_score:.2f}.",
                confidence=max(0.82, min(0.97, 0.82 + risk_score * 0.12)),
                evidence_message_ids=ev_str
            )

        if any(token in rule_label for token in ("notify", "urgent", "emergency", "alert")):
            return Decision(
                message_id=msg.message_id,
                action="notify",
                message_type="urgent" if urgency >= 0.45 else "personal",
                reason=f"Deterministic rule precedence marks this as high-priority. Urgency={urgency:.2f}, risk={risk_score:.2f}.",
                confidence=max(0.78, min(0.95, 0.78 + urgency * 0.12)),
                evidence_message_ids=ev_str
            )

        if fs.is_quiet_hours and urgency < 0.5 and not fs.is_direct_mention and sender_trust >= 0.35:
            return Decision(
                message_id=msg.message_id,
                action="digest",
                message_type="business_update" if msg.conversation_type == "business" else "personal",
                reason=f"Non-urgent message during quiet hours, with low urgency and moderate sender trust. Quiet hours should batch this content.",
                confidence=0.84,
                evidence_message_ids=ev_str
            )

        if payment >= 0.45 or "invoice" in text_lower or "statement" in text_lower or fs.media_category in ("invoice", "receipt"):
            action = "notify" if urgency >= 0.45 else "digest"
            return Decision(
                message_id=msg.message_id,
                action=action,
                message_type="payment",
                reason=f"Payment or billing-related signal is present with relationship strength {relationship_strength:.2f} and business trust {business_trust:.2f}.",
                confidence=max(0.68, min(0.9, 0.70 + payment * 0.15 + business_trust * 0.05)),
                evidence_message_ids=ev_str
            )

        if greeting >= 0.45 and urgency < 0.35:
            return Decision(
                message_id=msg.message_id,
                action="digest",
                message_type="greeting",
                reason="Casual greeting pattern with low urgency and low risk.",
                confidence=0.80,
                evidence_message_ids=ev_str
            )

        if event >= 0.45 and urgency < 0.5:
            return Decision(
                message_id=msg.message_id,
                action="digest",
                message_type="event",
                reason="Event or calendar style signal is present and does not indicate high urgency or risk.",
                confidence=0.79,
                evidence_message_ids=ev_str
            )

        if promotion >= 0.45 and risk_score < 0.45:
            return Decision(
                message_id=msg.message_id,
                action="mute",
                message_type="promotion",
                reason=f"Promotion-like signal is present but the message is not urgent and the risk score is low ({risk_score:.2f}).",
                confidence=max(0.65, min(0.86, 0.65 + promotion * 0.13)),
                evidence_message_ids=ev_str
            )

        if msg.conversation_type == "personal" or fs.is_direct_mention:
            action = "notify" if urgency >= 0.45 or fs.media_type == "voice" else "digest"
            mtype = "urgent" if urgency >= 0.45 else "personal"
            return Decision(
                message_id=msg.message_id,
                action=action,
                message_type=mtype,
                reason=f"Direct personal or @mention context with sender trust {sender_trust:.2f} and interaction history count {history_count}.",
                confidence=max(0.68, min(0.90, 0.68 + sender_trust * 0.12 + urgency * 0.10)),
                evidence_message_ids=ev_str
            )

        if business_trust <= 0.25 and risk_score >= 0.4:
            return Decision(
                message_id=msg.message_id,
                action="mute",
                message_type="scam",
                reason=f"Low business trust and elevated risk score ({risk_score:.2f}) suggest suspicious content.",
                confidence=max(0.74, min(0.93, 0.74 + risk_score * 0.15)),
                evidence_message_ids=ev_str
            )

        if reported_count >= 1 and risk_score >= 0.45:
            return Decision(
                message_id=msg.message_id,
                action="mute",
                message_type="spam",
                reason=f"Prior reported history plus elevated risk signals indicate suspicious handling should be suppressed.",
                confidence=0.83,
                evidence_message_ids=ev_str
            )

        default_action = "notify" if urgency >= 0.45 else "digest"
        default_type = "urgent" if urgency >= 0.45 else ("business_update" if msg.conversation_type == "business" else "event")
        return Decision(
            message_id=msg.message_id,
            action=default_action,
            message_type=default_type,
            reason=f"Weighted multimodal evidence favors a {default_action} decision. Urgency={urgency:.2f}, scam={scam:.2f}, payment={payment:.2f}, risk={risk_score:.2f}.",
            confidence=max(0.63, min(0.86, 0.63 + urgency * 0.10 + (1.0 - risk_score) * 0.05)),
            evidence_message_ids=ev_str
        )
