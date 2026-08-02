"""
setup.py — one-time model download for the WhatsApp Notification Router.

Run this ONCE before running the pipeline:
    python setup.py

This downloads:
  - EasyOCR English models  (~130 MB) → models_cache/easyocr/
  - faster-whisper-small     (~486 MB) → models_cache/whisper-small/

After setup, the pipeline runs fully offline with no internet required.
"""

import os
import sys
import shutil
import pathlib
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = pathlib.Path(__file__).parent.resolve()
CACHE = ROOT / "models_cache"
EASYOCR_CACHE = CACHE / "easyocr"
WHISPER_CACHE  = CACHE / "whisper-small"

EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)
WHISPER_CACHE.mkdir(parents=True, exist_ok=True)


def download_easyocr():
    print("[1/2] Downloading EasyOCR English models → models_cache/easyocr/ ...")
    try:
        import easyocr
        reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=str(EASYOCR_CACHE),
            verbose=False,
        )
        # Warm up with a tiny blank image to confirm the model loads
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (100, 30), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), "test", fill=(0, 0, 0))
        tmp = CACHE / "_warmup.png"
        img.save(str(tmp))
        reader.readtext(str(tmp), detail=0)
        tmp.unlink(missing_ok=True)
        print("    ✓ EasyOCR ready")
    except Exception as e:
        print(f"    ✗ EasyOCR download failed: {e}")
        print("    The pipeline will fall back to PIL heuristics for images.")


def download_whisper():
    print("[2/2] Downloading faster-whisper-small → models_cache/whisper-small/ ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Systran/faster-whisper-small",
            local_dir=str(WHISPER_CACHE),
            local_dir_use_symlinks=False,
        )
        print("    ✓ faster-whisper-small ready")
    except ImportError:
        # Fallback: let faster-whisper download it itself and then copy
        try:
            import faster_whisper
            model = faster_whisper.WhisperModel(
                "small", device="cpu", compute_type="int8",
                download_root=str(WHISPER_CACHE)
            )
            del model
            print("    ✓ faster-whisper-small ready (via faster-whisper download)")
        except Exception as e:
            print(f"    ✗ Whisper download failed: {e}")
            print("    The pipeline will attempt to download at runtime or skip ASR.")
    except Exception as e:
        print(f"    ✗ Whisper download failed: {e}")
        print("    The pipeline will attempt to download at runtime or skip ASR.")


def verify():
    print("\nVerifying cached models ...")
    easyocr_ok = any(EASYOCR_CACHE.rglob("*.pth"))
    whisper_ok  = any(WHISPER_CACHE.rglob("*.bin")) or any(WHISPER_CACHE.rglob("model.bin"))
    print(f"  EasyOCR: {'✓ found' if easyocr_ok else '✗ not found'}")
    print(f"  Whisper: {'✓ found' if whisper_ok else '✗ not found (will download at first run)'}")
    if easyocr_ok or whisper_ok:
        print("\nSetup complete. Run the pipeline with:")
        print("    python code/main.py")
    else:
        print("\nWarning: no models cached. The pipeline will still run but will")
        print("attempt to download models on first use (requires internet).")


if __name__ == "__main__":
    print("WhatsApp Notification Router — model setup")
    print("=" * 50)
    download_easyocr()
    download_whisper()
    verify()
