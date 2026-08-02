from typing import Optional, Dict, Any
from models import MediaItem
from ocr import OCRProcessor
from voice_processor import VoiceProcessor

class MediaProcessor:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.voice_processor = VoiceProcessor()

    def process(self, media: Optional[MediaItem]) -> Dict[str, Any]:
        if not media:
            return {
                "has_media": False,
                "media_type": "empty",
                "extracted_text": "",
                "category": "none",
                "is_urgent": False,
                "is_spam": False,
                "is_scam": False,
                "has_qr_code": False,
                "has_phone_number": False,
                "urls": [],
                "ocr_confidence": 0.0,
                "transcription_confidence": 0.0
            }

        if media.media_type == "image":
            ocr_meta = self.ocr_processor.process_image(media.extracted_text, media.file_path, media.category)
            return {
                "has_media": True,
                "media_type": "image",
                "extracted_text": ocr_meta["extracted_text"],
                "category": ocr_meta["category"],
                "is_urgent": ocr_meta["category"] == "warning",
                "is_spam": False,
                "is_scam": "short_url" in ocr_meta["urls"] or "bit.ly" in ocr_meta["extracted_text"],
                "has_qr_code": ocr_meta["has_qr_code"],
                "has_phone_number": ocr_meta["has_phone_number"],
                "urls": ocr_meta["urls"],
                "ocr_confidence": ocr_meta.get("warning_prob", 0.8),
                "poster_prob": ocr_meta.get("poster_prob", 0.0),
                "receipt_prob": ocr_meta.get("receipt_prob", 0.0),
                "invoice_prob": ocr_meta.get("invoice_prob", 0.0),
                "transcription_confidence": 0.0
            }
        elif media.media_type == "voice":
            v_signals = self.voice_processor.process_voice_note(media.extracted_text, media.duration_seconds, media.file_path)
            return {
                "has_media": True,
                "media_type": "voice",
                "extracted_text": v_signals["transcript"],
                "category": "voice_note",
                "is_urgent": v_signals["is_urgent"],
                "is_payment": v_signals["is_payment"],
                "is_event": v_signals["is_event"],
                "is_greeting": v_signals["is_greeting"],
                "is_spam": v_signals["is_spam"],
                "is_scam": v_signals["is_scam"],
                "has_qr_code": False,
                "has_phone_number": False,
                "urls": [],
                "ocr_confidence": 0.0,
                "voice_duration": media.duration_seconds,
                "transcription_confidence": v_signals["transcription_confidence"]
            }

        return {
            "has_media": False,
            "media_type": "unknown",
            "extracted_text": "",
            "category": "unknown",
            "is_urgent": False,
            "is_spam": False,
            "is_scam": False,
            "has_qr_code": False,
            "has_phone_number": False,
            "urls": [],
            "ocr_confidence": 0.0,
            "transcription_confidence": 0.0
        }
