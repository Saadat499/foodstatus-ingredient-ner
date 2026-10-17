"""Product-level verdict: the most severe ingredient verdict wins.

Precedence (from pipeline.SEVERITY): halal < cannot_be_certified < syubhah < haram.
"""

from classifier import load_e_number_table
from pipeline import SEVERITY, analyse

E_TABLE = load_e_number_table("data/sanha_e_numbers_all.csv")


def overall(text):
    return analyse(text, E_TABLE)["overall"]


def test_severity_order():
    order = sorted(SEVERITY, key=SEVERITY.get)
    assert order == ["halal", "cannot_be_certified", "syubhah", "haram"]


def test_all_halal_is_halal():
    assert overall("Sugar, Salt") == "halal"


def test_one_unrecognised_ingredient_makes_product_syubhah():
    assert overall("Sugar, Xyzzy") == "syubhah"


def test_haram_outranks_unrecognised_and_halal():
    assert overall("Sugar, Lard, Xyzzy") == "haram"


def test_source_dependent_ingredient_makes_product_syubhah():
    assert overall("Sugar, Gelatin (Bovine)") == "syubhah"
