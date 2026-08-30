"""
Merge hand-collected SANHA source descriptions into the main E-number table.

28 of the 419 entries have no source_description: SANHA's list-page card
excerpts truncate before the "Source:" field on entries whose alternative-
name lists are long. The text exists on each entry's own SANHA page, but
only there.

Workflow:
  1. Open data/missing_descriptions_TEMPLATE.csv
  2. For each row, open that E-number's page on sanha.org.za and copy the
     "Source:" text into source_description_FILL_ME. Leave blank to skip.
  3. python tools/merge_descriptions.py

Only fills blanks -- never overwrites an existing description, so it is
safe to re-run as you work through the list in batches.
"""

import csv
import shutil
import sys

MAIN = "data/sanha_e_numbers_all.csv"
TEMPLATE = "data/missing_descriptions_TEMPLATE.csv"


def main() -> int:
    try:
        filled = {
            r["e_number"].strip(): r["source_description_FILL_ME"].strip()
            for r in csv.DictReader(open(TEMPLATE, encoding="utf-8"))
            if r.get("source_description_FILL_ME", "").strip()
        }
    except FileNotFoundError:
        print(f"Template not found: {TEMPLATE}")
        return 1

    if not filled:
        print("Nothing filled in yet -- no changes made.")
        return 0

    rows = list(csv.DictReader(open(MAIN, encoding="utf-8")))
    fields = list(rows[0].keys())

    updated, skipped, unknown = 0, 0, []
    known = {r["e_number"].strip() for r in rows}

    for code in filled:
        if code not in known:
            unknown.append(code)

    for r in rows:
        code = r["e_number"].strip()
        if code in filled:
            if r["source_description"].strip():
                # Never clobber text already sourced from the list pages.
                skipped += 1
            else:
                r["source_description"] = filled[code]
                r["citation"] = (
                    f"SANHA {r['status'].title()} E-Numbers list "
                    f"(description from individual entry page), sanha.org.za"
                )
                updated += 1

    # Back up before writing -- this file is not regenerable by script alone.
    shutil.copy(MAIN, MAIN + ".bak")

    with open(MAIN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Updated : {updated}")
    print(f"Skipped : {skipped} (already had a description)")
    if unknown:
        print(f"UNKNOWN E-numbers, not in main table: {unknown}")
    remaining = sum(1 for r in rows if not r["source_description"].strip())
    print(f"Still empty: {remaining} of {len(rows)}")
    print(f"Backup written to {MAIN}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
