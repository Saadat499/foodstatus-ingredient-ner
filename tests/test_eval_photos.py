"""Tests for the metric helpers in tools/eval_photos.py (no photos needed)."""

import importlib.util
import os

import conftest

_spec = importlib.util.spec_from_file_location(
    "eval_photos", os.path.join(conftest.ROOT, "tools", "eval_photos.py"))
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def test_cer_identical_is_zero():
    assert ep.cer("Water, Sugar", "water sugar") == 0


def test_cer_counts_character_errors():
    # "sugar" vs "suqar": 1 edit over 5 characters
    assert ep.cer("suqar", "sugar") == 0.2


def test_cer_ignores_text_around_the_list():
    # the true list appears verbatim inside longer OCR text (e.g. nutrition table)
    ocr = "calories 75 total fat 4g water sugar salt serving size 15g"
    assert ep.cer(ocr, "Water, Sugar, Salt") == 0
    assert ep.cer_full_text(ocr, "Water, Sugar, Salt") > 1


def test_cer_is_bounded_by_one():
    assert ep.cer("zzzz", "water sugar") <= 1


def test_cer_counts_errors_inside_the_matched_stretch():
    assert ep.cer("xx suqar xx", "sugar") == 0.2


def test_cer_empty_truth_is_none():
    assert ep.cer("anything", "") is None


def test_split_ignores_commas_inside_brackets_and_qualifiers():
    assert ep.split_ingredients("Water, Chocolate (cocoa, sugar), Gelatin (Bovine)") == [
        "water", "chocolate", "gelatin"]


def test_recall_exact_and_near():
    exact, near = ep.ingredient_recall("water, gelatine, salt", "Water, Gelatin, Salt, Yeast")
    assert exact == 0.5      # water, salt
    assert near == 0.75      # + gelatine ~ gelatin; yeast missing


def test_recall_empty_truth_is_none():
    assert ep.ingredient_recall("water", "") == (None, None)
