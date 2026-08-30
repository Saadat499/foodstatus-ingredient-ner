"""
Parse raw SANHA E-number page dumps into the project's CSV schema.

Input : sources_halaal.txt / sources_haraam.txt / sources_mashbooh.txt
        (browser Ctrl+A copies of the three SANHA category pages)
Output: sanha_e_numbers_all.csv

Status is assigned by SOURCE PAGE, not by the inline "Status:" field --
the cards are truncated with "[...]" and the status text is sometimes cut
off, but the page an entry appears on is itself the classification.
The inline status, where present, is checked as a consistency test.
"""

import csv
import re
import sys

FIELD_LABELS = [
    "E-number:",
    "Name:",
    "Alternative Names :",
    "Alternative Names:",
    "Alternative Name:",
    "Function:",
    "Status:",
    "Source:",
]

# Header line looks like: "E101(a) (Riboflavin-5′-phosphate)"
HEADER_RE = re.compile(r"^(E[\dA-Za-z()\-]+)\s*\((.*)\)\s*$")


def split_fields(detail: str) -> dict:
    """Split a run-together detail line on its known labels."""
    pattern = "(" + "|".join(re.escape(l) for l in FIELD_LABELS) + ")"
    parts = re.split(pattern, detail)
    out, current = {}, None
    for chunk in parts:
        if chunk in FIELD_LABELS:
            current = chunk.rstrip(":").rstrip().rstrip(":").strip()
            current = current.replace("Alternative Names", "Alternative Name")
            out.setdefault(current, "")
        elif current:
            out[current] += chunk
    return {k: v.strip() for k, v in out.items()}


def clean(text: str) -> str:
    text = text.replace("[...]", " ")
    # These trailing site widgets are not part of the source description.
    text = re.sub(r"Health Info!!.*$", "", text)
    text = re.sub(r"Health Code:.*$", "", text)
    # SANHA's own dumps sometimes repeat the label inside the value.
    text = re.sub(r"^Source:\s*", "", text)
    return re.sub(r"\s+", " ", text).strip(" .;")


def parse_file(path: str, status: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\r\n") for l in f]

    rows, mismatches = [], []
    i = 0
    while i < len(lines):
        header = HEADER_RE.match(lines[i].strip())
        if not header:
            i += 1
            continue

        # The detail line is the next non-blank line.
        detail, j = "", i + 1
        while j < len(lines):
            if lines[j].strip():
                detail = lines[j].strip()
                break
            j += 1

        if "E-number:" not in detail:
            i += 1
            continue

        f = split_fields(detail)
        e_number = header.group(1).strip()

        inline = f.get("Status", "")
        inline_norm = inline.strip().lower()
        # "Mushbooh" is SANHA's spelling; the project uses "syubhah".
        expected = {"halal": "halaal", "haram": "haraam", "syubhah": "mushbooh"}[status]
        if inline_norm and not inline_norm.startswith(expected):
            mismatches.append(f"{e_number}: page={status} inline='{inline}'")

        rows.append({
            "e_number": e_number,
            "name": clean(f.get("Name", header.group(2))),
            "alternative_names": clean(f.get("Alternative Name", "")),
            "function": clean(f.get("Function", "")),
            "status": status,
            "source_description": clean(f.get("Source", "")),
            "citation": f"SANHA {status.title()} E-Numbers list, sanha.org.za/e-numbers/",
        })
        i = j + 1

    return rows, mismatches


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
    jobs = [
        (f"{src}/sources_halaal.txt", "halal"),
        (f"{src}/sources_haraam.txt", "haram"),
        (f"{src}/sources_mashbooh.txt", "syubhah"),
    ]

    all_rows, all_mismatch = [], []
    for path, status in jobs:
        rows, mism = parse_file(path, status)
        print(f"{path.split('/')[-1]:24} -> {len(rows):4} entries as '{status}'")
        all_rows += rows
        all_mismatch += mism

    print(f"\nTotal parsed: {len(all_rows)}")

    if all_mismatch:
        print(f"\nInline-status mismatches ({len(all_mismatch)}):")
        for m in all_mismatch:
            print("  ", m)
    else:
        print("\nNo inline-status mismatches -- page assignment agrees with inline text.")

    # Cross-list contradictions: same E-number on more than one page.
    seen = {}
    for r in all_rows:
        seen.setdefault(r["e_number"].lower(), set()).add(r["status"])
    conflicts = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"\nE-numbers appearing on multiple pages: {len(conflicts)}")
    for k, v in sorted(conflicts.items()):
        print(f"   {k}: {sorted(v)}")

    out = "/home/claude/halalguard/data/sanha_e_numbers_all.csv"
    fields = ["e_number", "name", "alternative_names", "function",
              "status", "source_description", "citation"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out}")
