"""Bracket-aware segmentation and hard-case flagging."""

from segmentation import flag_hard_cases, parse_segment, segment_ingredient_list


def test_comma_inside_brackets_does_not_split():
    segs = segment_ingredient_list("Water, Natural Flavor (Contains Soy, Wheat), Sugar")
    assert [s["name"] for s in segs] == ["Water", "Natural Flavor", "Sugar"]
    assert segs[1]["qualifier"] == "Contains Soy, Wheat"


def test_parse_segment_splits_name_and_qualifier():
    seg = parse_segment("Gelatin (Bovine)")
    assert seg["name"] == "Gelatin"
    assert seg["qualifier"] == "Bovine"


def test_parse_segment_without_brackets_has_no_qualifier():
    assert parse_segment("Sugar")["qualifier"] is None


def test_hard_cases_are_flagged_for_ner():
    segs = flag_hard_cases(segment_ingredient_list(
        "Water, Natural Flavor (Contains Soy, Wheat), "
        "May contain traces of nuts, Enzyme Treated Modified Starch"))
    flags = {s["name"]: s["needs_ner_review"] for s in segs}
    assert flags["Water"] is False
    assert flags["Natural Flavor"] is True          # comma inside the qualifier
    assert flags["May contain traces of nuts"] is True
    assert flags["Enzyme Treated Modified Starch"] is True  # 4+ word name


def test_simple_qualifier_is_not_flagged():
    segs = flag_hard_cases(segment_ingredient_list("Gelatin (Bovine)"))
    assert segs[0]["needs_ner_review"] is False
