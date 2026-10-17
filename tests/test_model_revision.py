"""Model revision pinning. No torch or network needed."""

from pipeline import revision_for

HASH = "6182865bc94cc3ed4f53f2159d8c349a8337efae"


def test_hub_repo_id_gets_the_pinned_revision():
    assert revision_for("Saadat57/foodstatus-ingredient-ner", HASH) == HASH


def test_no_revision_means_latest():
    assert revision_for("Saadat57/foodstatus-ingredient-ner", None) is None
    assert revision_for("Saadat57/foodstatus-ingredient-ner", "") is None


def test_local_folder_is_never_given_a_revision(tmp_path):
    assert revision_for(str(tmp_path), HASH) is None
