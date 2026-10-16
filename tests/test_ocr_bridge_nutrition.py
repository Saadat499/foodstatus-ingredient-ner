"""
Tests for nutrition-table stripping in ocr_bridge.clean_ocr_text.
 
Why this exists: the boilerplate pattern originally matched only
"Nutrition Facts". Labels that say "Nutrition Information" were passed
through to the classifier, so table rows (Protein, Carbohydrate, Sugars...)
appeared as unrecognized ingredients.
 
Run from the repository root:
    python tests/test_ocr_bridge_nutrition.py
or, if pytest is installed:
    pytest tests/
"""
 
import os
import sys
 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
 
from ocr_bridge import clean_ocr_text  # noqa: E402
 
 
def cleaned(raw: str) -> str:
    return clean_ocr_text(raw)["cleaned_text"]
 
 
def test_nutrition_information_table_is_stripped():
    raw = ("NUTRITION INFORMATION\nServing size 40g\nProtein 4g\n"
           "Carbohydrate 25g\nSugars 1g\nDietary Fibre 4g")
    assert cleaned(raw) == ""
 
 
def test_nutritional_information_variant_is_stripped():
    assert cleaned("NUTRITIONAL INFORMATION\nProtein 4g") == ""
 
 
def test_nutrition_facts_still_stripped():
    # Existing behaviour must not change.
    assert cleaned("NUTRITION FACTS\nProtein 4g\nSugars 1g") == ""
 
 
def test_ingredients_before_nutrition_information_are_kept():
    raw = "INGREDIENTS: Organic Whole Grain Oats.\nNUTRITION INFORMATION\nProtein 4g\nSugars 1g"
    assert cleaned(raw) == "Organic Whole Grain Oats."
 
 
def test_ingredients_marker_after_nutrition_information_is_kept():
    # The marker cut happens first, so a table read BEFORE the ingredients
    # does not delete the ingredient list.
    raw = "NUTRITION INFORMATION Serving size 40g Protein 4g INGREDIENTS: Oats, Sugar, Salt"
    assert cleaned(raw) == "Oats, Sugar, Salt"
 
 
def test_real_ingredient_containing_nutritional_is_untouched():
    # "Nutritional yeast" is an ingredient, not a table heading.
    assert cleaned("Ingredients: Water, Nutritional yeast, Salt") == "Water, Nutritional yeast, Salt"
 
 
def test_unrelated_information_text_is_untouched():
    raw = "Ingredients: Oats, Sugar. Information about allergens: contains gluten."
    assert cleaned(raw) == "Oats, Sugar. Information about allergens: contains gluten."
 
 
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
 
