import os
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Dynamic ASR module imports with fallbacks
FASTER_WHISPER_AVAILABLE = False
WHISPER_AVAILABLE = False
SPEECH_REC_AVAILABLE = False

try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    pass

def _whisper_model_dir() -> str:
    """Return the faster-whisper model directory, preferring the bundled cache."""
    # setup.py downloads to models_cache/whisper-small/
    # __file__ is code/voice_processor.py → two levels up = repo root
    bundled = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models_cache", "whisper-small"
    )
    # The actual model.bin should be directly in this folder
    if os.path.isdir(bundled) and os.path.exists(os.path.join(bundled, "model.bin")):
        return bundled
    # Fallback: let faster-whisper download via HF hub
    return "small"


class VoiceProcessor:
    def __init__(self):
        self._whisper_model = None
        self._faster_whisper_model = None

    def process_voice_note(self, transcript_csv: str, duration_seconds: float, file_path: str = "") -> Dict[str, Any]:
        """
        Process voice note. Priority:
        1. Pre-populated transcript from CSV if available.
        2. Live ASR speech-to-text decoding using faster-whisper -> whisper -> speech_recognition.
        Extracts structured signals (urgent, payment, event, greeting, spam, scam, business, personal, confidence).
        """
        transcript = transcript_csv.strip() if transcript_csv else ""
        asr_confidence = 0.95 if transcript else 0.50

        # If transcript is empty, attempt live ASR on audio file
        if not transcript and file_path and os.path.exists(file_path):
            transcript, asr_confidence = self._run_live_asr(file_path)

        return self._extract_audio_signals(transcript, duration_seconds, asr_confidence)

    def _run_live_asr(self, file_path: str) -> tuple[str, float]:
        # Engine Priority 1: faster-whisper
        if FASTER_WHISPER_AVAILABLE:
            try:
                if self._faster_whisper_model is None:
                    self._faster_whisper_model = faster_whisper.WhisperModel(
                        _whisper_model_dir(), device="cpu", compute_type="int8"
                    )
                segments, info = self._faster_whisper_model.transcribe(file_path)
                texts = [seg.text for seg in segments]
                if texts:
                    return " ".join(texts).strip(), 0.92
            except Exception as e:
                logger.debug(f"faster-whisper execution failed: {e}")

        # Engine Priority 2: whisper
        if WHISPER_AVAILABLE:
            try:
                if self._whisper_model is None:
                    self._whisper_model = whisper.load_model("small")
                result = self._whisper_model.transcribe(file_path)
                txt = result.get("text", "").strip()
                if txt:
                    return txt, 0.90
            except Exception as e:
                logger.debug(f"openai-whisper execution failed: {e}")

        # Engine Priority 3: speech_recognition
        if SPEECH_REC_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(file_path) as source:
                    audio = r.record(source)
                txt = r.recognize_google(audio)
                if txt:
                    return txt, 0.85
            except Exception as e:
                logger.debug(f"speech_recognition execution failed: {e}")

        return "", 0.30

    def _extract_audio_signals(self, transcript: str, duration_seconds: float, asr_confidence: float) -> Dict[str, Any]:
        text = transcript.strip()
        lower = text.lower()

        is_urgent = any(w in lower for w in ["emergency", "urgent", "immediately", "asap", "crashed", "help", "alarm", "hospital"])
        is_payment = any(w in lower for w in ["payment", "money", "paid", "invoice", "bank", "transfer", "fee"])
        is_event = any(w in lower for w in ["meeting", "appointment", "schedule", "tomorrow", "party", "webinar"])
        is_greeting = any(w in lower for w in ["good morning", "hello", "hi there", "have a great weekend", "family"])
        is_spam = any(w in lower for w in ["free gift", "won", "reward", "claim now", "crypto", "investment", "lottery"])
        is_scam = is_spam and any(w in lower for w in ["otp", "pin", "fee", "wire", "btc", "click"])
        is_business = any(w in lower for w in ["customer", "order", "delivery", "support", "invoice", "service"])
        is_personal = not is_business and not is_spam and not is_scam

        return {
            "transcript": text,
            "duration_seconds": duration_seconds,
            "transcription_confidence": asr_confidence,
            "is_urgent": is_urgent,
            "is_payment": is_payment,
            "is_event": is_event,
            "is_greeting": is_greeting,
            "is_spam": is_spam,
            "is_scam": is_scam,
            "is_business": is_business,
            "is_personal": is_personal
        }
