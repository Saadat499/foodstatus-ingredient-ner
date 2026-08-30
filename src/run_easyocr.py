"""
Standalone EasyOCR test -- for your own experimentation, separate from the
main pipeline. Lets you compare EasyOCR's raw output against Tesseract's
(from run_ocr.py) on the same photo, side by side.

First run will be slow -- it downloads EasyOCR's model files (a few hundred
MB) the first time only.

On your machine, gpu=True should work since PyTorch+CUDA is already set up
(confirmed by your BERT model running on device: cuda). This sandbox is
CPU-only, which is why testing here has been slow/resource-constrained --
your actual hardware (RTX 3050) should run this meaningfully faster.

Usage:
    python run_easyocr.py path\\to\\your\\photo.jpg
"""

import sys
import easyocr


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_easyocr.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading EasyOCR (slow on first run only, downloads model files)...")
    reader = easyocr.Reader(["en"], gpu=True)  # set gpu=False if this errors on your setup

    print("Running OCR...")
    results = reader.readtext(image_path, detail=1)

    print("=" * 60)
    print("EASYOCR OUTPUT (each detected text region, with confidence):")
    print("=" * 60)
    for bbox, text, confidence in results:
        print(f"  [{confidence:.2f}] {text}")

    print("=" * 60)
    full_text = " ".join(text for _, text, _ in results)
    print("Combined text:")
    print(full_text)


if __name__ == "__main__":
    main()
