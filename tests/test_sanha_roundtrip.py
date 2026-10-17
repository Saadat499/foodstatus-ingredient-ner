"""Every SANHA entry must come back from the classifier with SANHA's verdict.

This is the regression guard for the substring bug where "e100" (curcumin,
halal) matched inside "e1000" (cholic acid, mashbooh) and returned a
confident halal. It runs over the whole table, so any entry whose verdict is
overridden by a generic keyword fails here.
"""

import csv
import os

import conftest
from classifier import classify_ingredient, load_e_number_table

PATH = os.path.join(conftest.ROOT, "data", "sanha_e_numbers_all.csv")
E_TABLE = load_e_number_table(PATH)
with open(PATH, encoding="utf-8", newline="") as _f:
    ROWS = list(csv.DictReader(_f))


def _verdict(text):
    return classify_ingredient(text, None, E_TABLE)["status"]


def test_every_e_number_returns_the_table_verdict():
    wrong = [(r["e_number"], r["status"], _verdict(r["e_number"]))
             for r in ROWS if _verdict(r["e_number"]) != r["status"].strip().lower()]
    assert not wrong, wrong[:10]


def test_every_entry_name_returns_the_table_verdict():
    wrong = [(r["name"], r["status"], _verdict(r["name"]))
             for r in ROWS if _verdict(r["name"]) != r["status"].strip().lower()]
    assert not wrong, wrong[:10]


def test_e1000_is_not_mistaken_for_e100():
    assert _verdict("E1000") == "syubhah"
    assert _verdict("E100") == "halal"
