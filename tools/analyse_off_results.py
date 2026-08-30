"""
Summarise pipeline results on real product labels.

Turns data/off_results.json into the statistics needed for the domain-gap
section: coverage, verdict distribution, how often NER was invoked, and --
most usefully -- which ingredients the system failed to recognise, ranked
by frequency.

There is no gold annotation for real labels, so this does NOT produce an
F1. What it produces is coverage and failure analysis, which for this
project is arguably the more decision-relevant number: an ingredient the
system cannot resolve is a gap regardless of what any F1 says.

Usage (from the HalalGuard folder):
    python tools/analyse_off_results.py
    python tools/analyse_off_results.py --results data/off_results.json --top 30
"""

import argparse
import csv
import json
import os
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/off_results.json")
    p.add_argument("--top", type=int, default=25,
                   help="How many unrecognised ingredients to list.")
    p.add_argument("--out", default="data/off_coverage_summary.csv")
    p.add_argument("--gaps_out", default="data/off_unrecognised.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results):
        print(f"Not found: {args.results}")
        print("Run this first:")
        print("  python src/pipeline.py --file data/off_sample.txt "
              "--json data/off_results.json")
        return

    with open(args.results, encoding="utf-8") as f:
        reports = json.load(f)

    ingredients = [ing for r in reports for ing in r["ingredients"]]
    n_labels, n_ing = len(reports), len(ingredients)

    if not n_ing:
        print("No ingredients found in the results file.")
        return

    status = Counter(i["status"] for i in ingredients)
    conf = Counter(i["confidence"] for i in ingredients)
    routed = Counter(i.get("resolved_by", "unknown") for i in ingredients)
    overall = Counter(r["overall"] for r in reports)
    risk = Counter(i.get("verification_risk", "unrated") for i in ingredients)

    # An ingredient counts as RESOLVED if the system had actual evidence
    # for it. confidence == "none" means no reference list matched, i.e.
    # the verdict is a default, not a finding.
    resolved = sum(v for k, v in conf.items() if k != "none")
    coverage = resolved / n_ing * 100

    print("=" * 66)
    print("DOMAIN-GAP TEST: real product labels (Open Food Facts)")
    print("=" * 66)
    print(f"  labels analysed      : {n_labels}")
    print(f"  ingredients extracted: {n_ing}")
    print(f"  mean per label       : {n_ing/n_labels:.1f}")
    print()
    print(f"  COVERAGE: {resolved}/{n_ing} = {coverage:.1f}% resolved with evidence")
    print(f"            {n_ing-resolved}/{n_ing} = {100-coverage:.1f}% fell back to a default")

    print("\n--- ingredient verdicts ---")
    for k, v in status.most_common():
        print(f"  {k:<22} {v:>5}  ({v/n_ing*100:5.1f}%)")

    print("\n--- confidence ---")
    for k, v in conf.most_common():
        print(f"  {k:<22} {v:>5}  ({v/n_ing*100:5.1f}%)")

    print("\n--- which stage resolved it ---")
    for k, v in routed.most_common():
        print(f"  {k:<22} {v:>5}  ({v/n_ing*100:5.1f}%)")

    print("\n--- MUIS verification risk ---")
    for k, v in risk.most_common():
        print(f"  {k:<22} {v:>5}  ({v/n_ing*100:5.1f}%)")

    print("\n--- product-level verdicts ---")
    for k, v in overall.most_common():
        print(f"  {k:<22} {v:>5}  ({v/n_labels*100:5.1f}%)")

    # The actionable part: what the system could not resolve.
    unresolved = [i["ingredient"].strip().lower()
                  for i in ingredients if i["confidence"] == "none"]
    gaps = Counter(unresolved)

    print(f"\n--- TOP {args.top} UNRECOGNISED INGREDIENTS ---")
    print("    (each is a concrete candidate for the keyword list)")
    for name, count in gaps.most_common(args.top):
        print(f"  {count:>4}x  {name[:60]}")

    # Segmentation health: implausibly long "ingredients" usually mean the
    # splitter failed on that label's punctuation rather than that a real
    # ingredient has 12 words.
    long_ing = [i["ingredient"] for i in ingredients
                if len(i["ingredient"].split()) >= 8]
    print(f"\n--- possible segmentation failures ---")
    print(f"  {len(long_ing)} extracted 'ingredients' are 8+ words "
          f"({len(long_ing)/n_ing*100:.1f}%)")
    for x in long_ing[:5]:
        print(f"    {x[:70]}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["labels_analysed", n_labels])
        w.writerow(["ingredients_extracted", n_ing])
        w.writerow(["coverage_percent", f"{coverage:.2f}"])
        for k, v in status.items():
            w.writerow([f"status_{k}", v])
        for k, v in routed.items():
            w.writerow([f"resolved_by_{k}", v])

    with open(args.gaps_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ingredient", "occurrences"])
        for name, count in gaps.most_common():
            w.writerow([name, count])

    print(f"\nWritten: {args.out}")
    print(f"Written: {args.gaps_out}  ({len(gaps)} distinct unrecognised ingredients)")


if __name__ == "__main__":
    main()
