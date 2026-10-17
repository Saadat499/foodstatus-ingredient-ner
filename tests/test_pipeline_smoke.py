"""Run the README's rules-only example end to end through the CLI.

This needs no model and no Tesseract, so it runs in CI. It catches a broken
import, a changed CLI flag or a changed output format.
"""

import subprocess
import sys

import conftest

EXAMPLE = "Water, Sugar, Gelatin (Bovine), Vanilla, Lard"


def test_readme_no_ner_example():
    result = subprocess.run(
        [sys.executable, "src/pipeline.py", "--no-ner", "--text", EXAMPLE],
        cwd=conftest.ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "rules only" in out
    assert "PRODUCT VERDICT: HARAM" in out  # lard decides the overall verdict
    for line, verdict in (("Sugar", "halal"), ("Gelatin (Bovine)", "syubhah"),
                          ("Lard", "haram")):
        row = next(ln for ln in out.splitlines() if ln.strip().startswith(line))
        assert verdict in row, row
