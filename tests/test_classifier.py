"""Classifier behaviour that must not change by accident.

These tests pin the conservative design: unknown input is never "halal",
and a specific certifier ruling outranks a generic keyword.
"""

import pytest

from classifier import classify_ingredient, load_e_number_table

E_TABLE = load_e_number_table("data/sanha_e_numbers_all.csv")


def classify(name, qualifier=None, **kw):
    return classify_ingredient(name, qualifier, E_TABLE, **kw)


def test_known_halal_term():
    r = classify("sugar")
    assert r["status"] == "halal"


def test_known_haram_term():
    r = classify("lard")
    assert r["status"] == "haram"
    assert r["confidence"] == "high"


def test_known_syubhah_term_with_source_qualifier():
    # Gelatin is source-dependent: naming the animal does not clear it.
    r = classify("gelatin", "bovine")
    assert r["status"] == "syubhah"


@pytest.mark.parametrize("name", ["xyzzy", "qwerty plugh", "natural flavouring"])
def test_unrecognised_term_never_becomes_halal(name):
    r = classify(name)
    assert r["status"] == "syubhah"
    assert r["confidence"] == "none"  # "no evidence", not "evidence of doubt"


def test_unknown_policy_muis_returns_cannot_be_certified():
    assert classify("xyzzy", unknown_policy="muis")["status"] == "cannot_be_certified"


def test_invalid_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        classify("sugar", unknown_policy="not-a-policy")


def test_specific_certifier_ruling_outranks_keyword():
    # "carmine" is a haram keyword, but SANHA lists Indigo Carmine (E132,
    # a synthetic dye) as halal. The specific ruling must win.
    r = classify("Indigo Carmine")
    assert r["status"] == "halal"
    assert r["evidence_tier"] == "T1_CERTIFIER_RULING"
