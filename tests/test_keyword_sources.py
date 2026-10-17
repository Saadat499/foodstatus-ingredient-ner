import csv
import os

import conftest

PATH = os.path.join(conftest.ROOT, "data", "keyword_sources.csv")
QURAN_STATUSES = {"QURAN_VERSE_DIRECT", "QURAN_VERSE_CATEGORY"}


def _rows():
    with open(PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_all_84_keywords_present_and_unique():
    rows = _rows()
    assert len(rows) == 84
    assert len({r["keyword"] for r in rows}) == 84


def test_no_keyword_still_waits_for_a_citation():
    assert all(r["verification_status"] != "NEEDS_EXACT_CITATION" for r in _rows())


def test_quran_rows_cite_a_verse_reference():
    quran = [r for r in _rows() if r["verification_status"] in QURAN_STATUSES]
    assert len(quran) == 29
    for r in quran:
        assert any(ref in r["basis"] for ref in ("2:173", "5:3", "5:90")), r["keyword"]
