"""Photo-quality gate. Uses synthetic images and never calls Tesseract."""

from PIL import Image, ImageFilter

import photo_quality
from photo_quality import BLUR_THRESHOLD, assess_photo_quality, blur_score


def checkerboard(size=300, square=4):
    img = Image.new("L", (size, size))
    img.putdata([255 if (x // square + y // square) % 2 else 0
                 for y in range(size) for x in range(size)])
    return img.convert("RGB")


def test_sharp_image_scores_above_threshold():
    assert blur_score(checkerboard()) > BLUR_THRESHOLD


def test_blurred_image_scores_below_threshold():
    blurred = checkerboard().filter(ImageFilter.GaussianBlur(8))
    assert blur_score(blurred) < BLUR_THRESHOLD


def test_small_text_is_rejected_with_message(monkeypatch):
    monkeypatch.setattr(photo_quality, "median_text_height", lambda image: 10)
    result = assess_photo_quality(checkerboard())
    assert result["is_text_too_small"] is True
    assert "too small" in result["message"]


def test_blurry_photo_is_rejected_with_message(monkeypatch):
    monkeypatch.setattr(photo_quality, "median_text_height", lambda image: 40)
    blurred = checkerboard().filter(ImageFilter.GaussianBlur(8))
    result = assess_photo_quality(blurred)
    assert result["is_too_blurry"] is True
    assert "blurry" in result["message"]


def test_good_photo_has_no_message(monkeypatch):
    monkeypatch.setattr(photo_quality, "median_text_height", lambda image: 40)
    result = assess_photo_quality(checkerboard())
    assert result["message"] is None
