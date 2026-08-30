"""
Quick OCR sanity check.

Runs Tesseract on a real photo of an ingredient label and prints exactly
what it read -- no cleanup, no correction. This is the raw material for
testing ocr_bridge.py against, instead of hand-typed guesses at what OCR
noise looks like.

Usage:
    python run_ocr.py path\\to\\your\\photo.jpg
"""

import sys
import pytesseract
from PIL import Image

# Tesseract isn't on PATH on this machine, so point pytesseract straight at
# the exe. Only needed because of that -- if PATH ever gets fixed later,
# this line becomes harmless (it just points at the same place PATH would).
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_ocr.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        img = Image.open(image_path)
    except Exception as exc:
        print(f"Could not open image: {exc}")
        sys.exit(1)

    raw_text = pytesseract.image_to_string(img)

    print("=" * 60)
    print("RAW OCR OUTPUT (exactly what Tesseract read, unmodified):")
    print("=" * 60)
    print(raw_text)
    print("=" * 60)
    print(f"Character count: {len(raw_text)}")


if __name__ == "__main__":
    main()
