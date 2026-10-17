"""
Break down the resolved ingredients in a pipeline results file by evidence tier.

"Resolved" means the pipeline gave the ingredient a confidence other than
"none", the same definition tools/analyse_off_results.py uses for coverage.
The point is to show how much of the coverage figure rests on a certifier's
ruling (T1) and how much on the permissibility default (T3).

Uses the standard library only.

Usage (from the repository root):
    python tools/summarise_evidence_tiers.py
    python tools/summarise_evidence_tiers.py --results data/off_results.json \
        --out data/off_evidence_tiers.csv
"""

import argparse
import csv
import json
from collections import Counter


def summarise(reports):
    """Count (evidence_tier, confidence, status) over resolved ingredients.

    Returns (rows, n_total, n_resolved) with rows as a list of
    (tier, confidence, status, count) sorted by tier then status.
    """
    ingredients = [i for r in reports for i in r["ingredients"]]
    resolved = [i for i in ingredients if i.get("confidence") != "none"]
    counts = Counter(
        (i.get("evidence_tier") or "none", i["confidence"], i["status"])
        for i in resolved
    )
    rows = sorted((t, c, s, n) for (t, c, s), n in counts.items())
    return rows, len(ingredients), len(resolved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/off_results.json")
    p.add_argument("--out", default="data/off_evidence_tiers.csv")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        reports = json.load(f)

    rows, n_total, n_resolved = summarise(reports)
    print(f"{n_resolved} of {n_total} ingredients resolved")
    for t, c, s, n in rows:
        print(f"  {t:22s} confidence={c:7s} status={s:8s} {n:4d} "
              f"({n / n_resolved:.1%} of resolved)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["evidence_tier", "confidence", "status", "count"])
        w.writerows(rows)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
