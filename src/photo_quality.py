"""
Photo quality check -- catches bad photos BEFORE they reach OCR, instead of
silently producing bad results and needing to explain why afterward.

Two separate checks, because they catch different problems:

1. BLUR: calibrated against two real phone photos -- one that OCR'd nearly
   perfectly (blur score ~6280) and one that dropped multiple words
   entirely (blur score ~155). Roughly a 40x gap.

2. TEXT SIZE: added after a real webcam capture passed the blur check
   (score 459, above threshold) but produced almost no usable OCR text.
   The first fix attempt checked raw image dimensions and was WRONG -- it
   flagged a real working photo (a tight crop, small in total pixels but
   with large text filling the frame) as bad. What actually distinguishes
   a good photo from a bad one isn't image size, it's how many pixels the
   TEXT ITSELF occupies. Measured directly via Tesseract's own detected
   word-box heights on all three real cases:
       - working tight crop:        median text height ~18px
       - blurry-but-big-text photo: median text height ~66px (blur was
         the real problem there, not size -- confirms these are genuinely
         separate failure modes needing separate checks)
       - broken webcam capture:     median text height ~10px
   10px is not enough for OCR to work with; 18px is. The threshold below
   sits between them.

Both thresholds are calibrated on a small number of real examples, not a
large validated dataset -- reasonable starting points to refine as more
real photos are tested, not final numbers.
"""

import statistics
from PIL import Image, ImageFilter
import pytesseract
from pytesseract import Output

BLUR_THRESHOLD = 300
MIN_TEXT_HEIGHT_PX = 15  # see calibration note above


def blur_score(image: Image.Image) -> float:
    """
    Higher = sharper. Uses edge-detection variance as a lightweight,
    pure-PIL proxy for the standard 'variance of Laplacian' blur-detection
    technique -- avoids adding OpenCV as a new dependency for one check.
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    return statistics.variance(pixels)


def median_text_height(image: Image.Image, min_confidence: int = 40) -> float:
    """
    Median height (in pixels) of confidently-detected text, using
    Tesseract's own word-level bounding boxes. Returns 0 if nothing was
    detected with reasonable confidence at all.
    """
    data = pytesseract.image_to_data(image, config="--psm 6", output_type=Output.DICT)
    heights = [
        data["height"][i] for i in range(len(data["text"]))
        if data["text"][i].strip() and data["conf"][i] > min_confidence
    ]
    return statistics.median(heights) if heights else 0


def assess_photo_quality(image: Image.Image) -> dict:
    """
    Returns:
        {
            "blur_score": float,
            "text_height_px": float,
            "is_too_blurry": bool,
            "is_text_too_small": bool,
            "message": str or None,  # user-facing explanation if rejected
        }
    """
    score = blur_score(image)
    too_blurry = score < BLUR_THRESHOLD

    text_height = median_text_height(image)
    text_too_small = text_height < MIN_TEXT_HEIGHT_PX

    message = None
    if text_too_small:
        message = (
            "The ingredients text looks too small in this photo for reliable "
            "reading. Try moving closer so the text fills more of the frame."
        )
    elif too_blurry:
        message = (
            "This photo looks too blurry to read reliably. Try holding the "
            "camera steadier, moving a bit closer, or improving the lighting, "
            "then capture again."
        )

    return {
        "blur_score": score,
        "text_height_px": text_height,
        "is_too_blurry": too_blurry,
        "is_text_too_small": text_too_small,
        "message": message,
    }
