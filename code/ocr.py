import os
import re
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# Dynamic imports with graceful fallbacks
PADDLE_AVAILABLE = False
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False
PIL_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass

def _easyocr_model_dir() -> str:
    """Return the EasyOCR model directory, preferring the bundled cache."""
    # __file__ is code/ocr.py → two levels up = repo root
    bundled = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models_cache", "easyocr"
    )
    if os.path.isdir(bundled) and os.listdir(bundled):
        return bundled
    return os.environ.get("EASYOCR_MODULE_PATH", os.path.expanduser("~/.EasyOCR"))


class OCRProcessor:
    def __init__(self):
        self._easyocr_reader = None
        self._paddleocr_engine = None

    def process_image(self, ocr_text_csv: str, image_path: str = "", category_hint: str = "") -> Dict[str, Any]:
        """
        Process image text. Priority:
        1. Pre-populated ocr_text from CSV if valid.
        2. Live image file OCR using PaddleOCR -> EasyOCR -> PyTesseract -> PIL heuristic fallback.
        Extracts structured metadata (text, categories, URLs, phone numbers, QR indicators, probabilities).
        """
        raw_text = ocr_text_csv.strip() if ocr_text_csv else ""
        
        # If no CSV text, attempt live OCR engine pipeline on file
        if not raw_text and image_path and os.path.exists(image_path):
            raw_text = self._run_live_ocr(image_path)

        return self._extract_structured_metadata(raw_text, category_hint)

    def _run_live_ocr(self, image_path: str) -> str:
        # Engine Priority 1: PaddleOCR
        if PADDLE_AVAILABLE:
            try:
                if self._paddleocr_engine is None:
                    self._paddleocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                res = self._paddleocr_engine.ocr(image_path, cls=True)
                lines = []
                if res and isinstance(res, list):
                    for line in res[0]:
                        lines.append(line[1][0])
                if lines:
                    return " ".join(lines)
            except Exception as e:
                logger.debug(f"PaddleOCR execution failed: {e}")

        # Engine Priority 2: EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                if self._easyocr_reader is None:
                    self._easyocr_reader = easyocr.Reader(
                        ['en'], gpu=False, model_storage_directory=_easyocr_model_dir()
                    )
                res = self._easyocr_reader.readtext(image_path, detail=0)
                if res:
                    return " ".join(res)
            except Exception as e:
                logger.debug(f"EasyOCR execution failed: {e}")

        # Engine Priority 3: PyTesseract
        if TESSERACT_AVAILABLE and PIL_AVAILABLE:
            try:
                img = Image.open(image_path)
                txt = pytesseract.image_to_string(img)
                if txt.strip():
                    return txt.strip()
            except Exception as e:
                logger.debug(f"PyTesseract execution failed: {e}")

        return ""

    def _extract_structured_metadata(self, text: str, category_hint: str = "") -> Dict[str, Any]:
        lower_text = text.lower()
        
        # Extract Entities: URLs & Phone numbers
        urls = re.findall(r"https?://[^\s]+|bit\.ly/[^\s]+|tinyurl\.com/[^\s]+|[a-z0-9-]+\.(?:top|info|xyz|site)", lower_text)
        phones = re.findall(r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}", text)
        has_qr = "qr" in lower_text or "scan" in lower_text or "barcode" in lower_text

        # Compute Category Probabilities
        poster_score = 0.8 if any(w in lower_text for w in ["sale", "off", "discount", "promo", "code", "clearance"]) else 0.1
        invoice_score = 0.9 if any(w in lower_text for w in ["invoice", "total due", "bill", "due date", "amount due"]) else 0.1
        receipt_score = 0.9 if any(w in lower_text for w in ["receipt", "payment received", "paid", "subtotal", "tax"]) else 0.1
        warning_score = 0.95 if any(w in lower_text for w in ["warning", "suspended", "alert", "security", "unauthorized"]) else 0.05
        ad_score = 0.7 if any(w in lower_text for w in ["shop", "buy", "fashion", "collection", "exclusive"]) else 0.1
        meeting_score = 0.8 if any(w in lower_text for w in ["meeting", "webinar", "join", "agenda", "conference"]) else 0.1

        # Final Classification
        cat = category_hint.lower() if category_hint else "unknown"
        if cat == "unknown":
            scores = {
                "warning": warning_score,
                "invoice": invoice_score,
                "receipt": receipt_score,
                "poster": poster_score,
                "meeting": meeting_score,
                "advertisement": ad_score,
                "qr_code": 0.85 if has_qr else 0.0
            }
            best_cat, max_s = max(scores.items(), key=lambda x: x[1])
            cat = best_cat if max_s > 0.4 else "unknown"

        return {
            "extracted_text": text,
            "category": cat,
            "has_qr_code": has_qr,
            "has_phone_number": len(phones) > 0,
            "urls": urls,
            "poster_prob": poster_score,
            "invoice_prob": invoice_score,
            "receipt_prob": receipt_score,
            "warning_prob": warning_score,
            "meeting_prob": meeting_score,
            "advertisement_prob": ad_score
        }
