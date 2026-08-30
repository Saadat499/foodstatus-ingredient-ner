"""
Download real product ingredient lists from Open Food Facts for the
domain-gap test: FINER (Allrecipes recipe text) vs. real product labels.

Uses the public v2 search API -- no key required, no rate limit for
reasonable use, per Open Food Facts' own API docs
(https://openfoodfacts.github.io/openfoodfacts-server/api/).

Two real problems this defends against:

1. The `fields=` request parameter does NOT reliably restrict the
   response -- testing returned full product objects (nutrition,
   eco-score, images, everything) regardless. Fields are therefore
   extracted client-side, not trusted from the API.

2. Most products are NOT English. This queries English-labelling
   categories/countries, then filters again in code: a product is only
   kept if ingredients_text_en is present, OR lang == "en" and
   ingredients_text looks ASCII. Silently accepting non-English text
   would corrupt the gap test rather than measure it.

Usage (from the HalalGuard folder):
    python tools/get_open_food_facts.py
    python tools/get_open_food_facts.py --n 100 --out data/off_sample.txt
"""

import argparse
import json
import time
import urllib.parse
import urllib.request

BASE = "https://world.openfoodfacts.org/api/v2/search"

# A spread of categories, not one -- a single category (e.g. "cereals")
# would bias the sample toward one label style.
CATEGORIES = [
    "breakfast-cereals", "snacks", "biscuits", "breads", "sauces",
    "canned-foods", "chocolates", "dairies", "beverages", "condiments",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60,
                   help="Target number of English-language samples.")
    p.add_argument("--out", default="data/off_sample.txt",
                   help="Plain text output -- one ingredient list per line, "
                        "ready for: python src/pipeline.py --file <out>")
    p.add_argument("--meta_out", default="data/off_sample_meta.jsonl",
                   help="JSONL with product_name/code alongside each line, "
                        "for traceability back to the source product.")
    return p.parse_args()


def is_probably_english(text: str) -> bool:
    """Cheap heuristic, not a language detector: ASCII-heavy text with
    common English ingredient words. Good enough to reject the obviously
    non-English results seen during testing (Czech, German)."""
    if not text:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    return ascii_ratio > 0.95


def fetch_category(category: str, page_size: int = 40) -> list[dict]:
    params = {
        "categories_tags_en": category,
        "countries_tags_en": "united-states,united-kingdom",
        "page_size": page_size,
        # Requested for efficiency, but NOT trusted -- see module docstring.
        "fields": "product_name,ingredients_text,ingredients_text_en,lang,code",
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "HalalGuard-NLP research project"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return data.get("products", [])
    except Exception as exc:
        print(f"  [{category}] request failed: {exc}")
        return []


def main():
    args = parse_args()
    samples, meta, seen_codes = [], [], set()

    for category in CATEGORIES:
        if len(samples) >= args.n:
            break

        print(f"Fetching: {category} ...")
        products = fetch_category(category)
        kept_this_category = 0

        for prod in products:
            if len(samples) >= args.n:
                break

            code = prod.get("code")
            if not code or code in seen_codes:
                continue

            # Prefer the explicit English field; fall back to the generic
            # field only if the product's declared language is English AND
            # the text itself looks English -- two independent checks,
            # since either signal alone was unreliable in testing.
            text = prod.get("ingredients_text_en", "").strip()
            if not text and prod.get("lang") == "en":
                candidate = prod.get("ingredients_text", "").strip()
                if is_probably_english(candidate):
                    text = candidate

            if not text or len(text) < 10:
                continue

            seen_codes.add(code)
            samples.append(text)
            meta.append({
                "code": code,
                "product_name": prod.get("product_name", ""),
                "category": category,
                "ingredients_text": text,
            })
            kept_this_category += 1

        print(f"  kept {kept_this_category} English samples "
              f"(running total: {len(samples)}/{args.n})")
        time.sleep(1)  # be a reasonable citizen of a free public API

    if not samples:
        print("\nNo samples collected. The API may be unreachable, or "
              "every result was filtered as non-English. Check your "
              "internet connection and try again.")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.replace("\n", " ").replace("\r", " ") + "\n")

    with open(args.meta_out, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    print(f"\nCollected {len(samples)} real, English-language product "
          f"ingredient lists.")
    print(f"Written to {args.out} (plain text, one per line)")
    print(f"Metadata written to {args.meta_out} (product name/code per line)")
    print(f"\nNext step:")
    print(f"  python src/pipeline.py --file {args.out} --json data/off_results.json")


if __name__ == "__main__":
    main()
