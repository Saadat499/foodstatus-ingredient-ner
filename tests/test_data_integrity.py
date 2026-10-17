"""Integrity checks over the data files the classifier reads.

These guard against the kind of silent corruption found in the early audit:
unquoted commas that split one field into three, duplicate keys, and rows
with no source.
"""

import csv
import os

import conftest

D = os.path.join(conftest.ROOT, "data")
VERDICTS = {"halal", "haram", "syubhah"}
TIERS = {"T1_CERTIFIER_RULING", "T2_SCRIPTURAL", "T3_PRINCIPLE", "T4_JUDGMENT"}


def _read(name):
    with open(os.path.join(D, name), encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _dicts(name):
    with open(os.path.join(D, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assert_consistent_field_count(name):
    header, rows = _read(name)
    bad = [i + 2 for i, r in enumerate(rows) if len(r) != len(header)]
    assert not bad, f"{name}: rows with the wrong number of fields: {bad}"


def test_ingredient_reference_field_counts():
    _assert_consistent_field_count("ingredient_reference.csv")


def test_ingredient_reference_values():
    rows = _dicts("ingredient_reference.csv")
    for r in rows:
        assert r["verdict"] in VERDICTS, r["term"]
        assert r["evidence_tier"] in TIERS, r["term"]
        assert r["source"].strip(), f"{r['term']}: empty source"


def test_ingredient_reference_terms_are_unique():
    terms = [r["term"].strip().lower() for r in _dicts("ingredient_reference.csv")]
    dupes = {t for t in terms if terms.count(t) > 1}
    assert not dupes, dupes


def test_sanha_table_field_counts():
    _assert_consistent_field_count("sanha_e_numbers_all.csv")


def test_sanha_table_values_and_unique_codes():
    rows = _dicts("sanha_e_numbers_all.csv")
    assert len(rows) == 419
    codes = [r["e_number"].strip().lower() for r in rows]
    assert len(set(codes)) == len(codes)
    for r in rows:
        assert r["status"].strip().lower() in VERDICTS, r["e_number"]
        assert r["name"].strip(), r["e_number"]
        assert r["citation"].strip(), f"{r['e_number']}: empty citation"


def test_keyword_sources_field_counts_and_verdicts():
    _assert_consistent_field_count("keyword_sources.csv")
    for r in _dicts("keyword_sources.csv"):
        assert r["verdict"] in VERDICTS, r["keyword"]
        assert r["basis"].strip(), r["keyword"]
