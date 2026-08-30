"""
Measure classifier accuracy without circularity.

Testing the classifier against the SANHA table it reads would prove
nothing -- it would just confirm a lookup returns what was looked up.

Instead this HOLDS OUT a random slice of the 419 SANHA entries, loads the
classifier with only the remainder, and asks whether it still reaches
SANHA's verdict on the held-out entries using keywords, the reference
file, and qualifier logic alone.

That measures the thing that actually matters: how well does the system
reason when the lookup table does NOT contain the answer -- which is the
situation on every real product label.

Held-out entries will often return syubhah with confidence "none". That
is a correct, honest answer, not a failure, so results are reported in
three buckets rather than as a single accuracy figure:

    correct    -- matched SANHA's verdict
    abstained  -- returned syubhah with no evidence (no claim made)
    wrong      -- asserted a verdict that contradicts SANHA

Only `wrong` is a genuine error. A system that abstains rather than
guesses is behaving as intended.

Usage (from the HalalGuard folder):
    python tools/evaluate_classifier_holdout.py
    python tools/evaluate_classifier_holdout.py --holdout 0.3 --seed 7
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import classifier as clf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="data/sanha_e_numbers_all.csv")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/classifier_holdout_results.csv")
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()

    with open(args.table, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.seed(args.seed)
    shuffled = rows[:]
    random.shuffle(shuffled)
    n_hold = int(len(shuffled) * args.holdout)
    held, kept = shuffled[:n_hold], shuffled[n_hold:]

    # Rebuild the lookup table from the KEPT rows only.
    partial = {}
    for row in kept:
        e = row["e_number"].strip().lower()
        n = row["name"].strip().lower()
        partial[e] = row
        partial[n] = row
        partial[e.replace("(", "").replace(")", "")] = row
        for alt in row.get("alternative_names", "").split(";"):
            alt = alt.strip().lower()
            if alt and alt != "n/a":
                partial[alt] = row

    print(f"SANHA entries      : {len(rows)}")
    print(f"  visible to system: {len(kept)}")
    print(f"  held out         : {len(held)}  (seed={args.seed})")
    print()

    results, buckets = [], Counter()
    by_status = {}

    for row in held:
        truth = row["status"].strip().lower()
        # Query by NAME, not E-number: on a real label an additive usually
        # appears by name, and the name is what the reasoning must handle.
        got = clf.classify_ingredient(row["name"], None, partial)
        pred, conf = got["status"], got["confidence"]

        if pred == truth:
            bucket = "correct"
        elif pred == "syubhah" and conf == "none":
            bucket = "abstained"
        else:
            bucket = "wrong"

        buckets[bucket] += 1
        by_status.setdefault(truth, Counter())[bucket] += 1
        results.append({
            "e_number": row["e_number"], "name": row["name"],
            "sanha_verdict": truth, "predicted": pred,
            "confidence": conf, "bucket": bucket,
            "reason": got["reason"][:160],
        })

    total = len(held)
    print("=" * 62)
    print("HELD-OUT PERFORMANCE (lookup table cannot answer these)")
    print("=" * 62)
    for b in ("correct", "abstained", "wrong"):
        print(f"  {b:<12} {buckets[b]:>4}  ({buckets[b]/total*100:5.1f}%)")
    print()
    decided = buckets["correct"] + buckets["wrong"]
    if decided:
        print(f"  Of the {decided} it was willing to decide, "
              f"{buckets['correct']/decided*100:.1f}% matched SANHA.")
    print(f"  It abstained on {buckets['abstained']/total*100:.1f}% rather than guess.")

    print("\n--- by SANHA verdict ---")
    for status, counts in sorted(by_status.items()):
        tot = sum(counts.values())
        print(f"  {status:<10} (n={tot:>3})  "
              f"correct {counts['correct']:>3} | "
              f"abstained {counts['abstained']:>3} | "
              f"wrong {counts['wrong']:>3}")

    wrong = [r for r in results if r["bucket"] == "wrong"]
    if wrong:
        print(f"\n--- ERRORS ({len(wrong)}) -- these are the real failures ---")
        for r in wrong[:15]:
            print(f"  {r['e_number']:<9} {r['name'][:30]:<30} "
                  f"SANHA={r['sanha_verdict']:<8} got={r['predicted']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
