"""Exact-match OCR correction (ocr_bridge) and the ARED -> LARD regression.

Fuzzy matching was removed because it turned the packaging fragment "ARED"
(from "PREPARED BY") into the haram term "lard", giving a false HARAM
verdict. These tests make sure that cannot come back.
"""

from classifier import classify_ingredient, load_e_number_table
from ocr_bridge import clean_ocr_text
from pipeline import analyse

E_TABLE = load_e_number_table("data/sanha_e_numbers_all.csv")


def cleaned(raw):
    return clean_ocr_text(raw)["cleaned_text"]


def test_zero_for_o_confusion_is_corrected():
    assert cleaned("c0c0a butter") == "cocoa butter"


def test_eight_for_b_confusion_is_corrected():
    assert cleaned("Sugar, Salt, 8acon Fat") == "Sugar, Salt, bacon Fat"


def test_one_for_l_confusion_is_corrected():
    assert cleaned("sa1t, sugar") == "salt, sugar"


def test_near_miss_is_left_unchanged():
    # Not an exact match after undoing confusions -> no "closest guess".
    result = clean_ocr_text("Sugarr, salt, suger")
    assert result["cleaned_text"] == "Sugarr, salt, suger"
    assert result["corrections"] == []


def test_ared_is_not_corrected_to_lard():
    result = clean_ocr_text("ARED")
    assert "lard" not in result["cleaned_text"].lower()
    assert result["corrections"] == []


def test_ared_fragment_is_not_haram():
    assert classify_ingredient("ared", None, E_TABLE)["status"] != "haram"


def test_prepared_by_line_does_not_produce_haram():
    # Packaging text is not stripped as boilerplate (it passes through as an
    # unrecognised segment -> syubhah), but it must never become haram.
    text = cleaned("Sugar, Salt, PREPARED BY ACME FOODS LTD")
    report = analyse(text, E_TABLE)
    assert report["overall"] != "haram"
    assert "lard" not in text.lower()
