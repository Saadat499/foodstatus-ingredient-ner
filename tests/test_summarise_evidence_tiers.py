import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from summarise_evidence_tiers import summarise  # noqa: E402


def _ing(tier, conf, status):
    return {"evidence_tier": tier, "confidence": conf, "status": status}


def test_counts_only_resolved_ingredients():
    reports = [{"ingredients": [
        _ing("T3_PRINCIPLE", "medium", "halal"),
        _ing("T3_PRINCIPLE", "medium", "halal"),
        _ing("T1_CERTIFIER_RULING", "high", "syubhah"),
        _ing(None, "none", "syubhah"),
    ]}]
    rows, n_total, n_resolved = summarise(reports)
    assert (n_total, n_resolved) == (4, 3)
    assert ("T3_PRINCIPLE", "medium", "halal", 2) in rows
    assert ("T1_CERTIFIER_RULING", "high", "syubhah", 1) in rows
    assert all(r[0] != "none" for r in rows)


def test_committed_off_results_match_readme_figures():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "off_results.json")
    with open(path, encoding="utf-8") as f:
        reports = json.load(f)
    rows, n_total, n_resolved = summarise(reports)
    assert (n_total, n_resolved) == (853, 556)
    by_tier = {}
    for t, _c, _s, n in rows:
        by_tier[t] = by_tier.get(t, 0) + n
    assert by_tier == {"T3_PRINCIPLE": 417, "T1_CERTIFIER_RULING": 139}
