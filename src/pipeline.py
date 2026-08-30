"""
End-to-end HalalGuard pipeline.

Joins the stages that until now only ran in isolation:

    raw label text
        -> [1] segmentation.py   rule-based comma/bracket splitting
        -> [2] train_bert NER    only for segments the rules flag as hard
        -> [4] classifier.py     halal / haram / syubhah per ingredient
        -> aggregate product verdict

Design notes:

* NER is NOT run on everything. Stage 1 resolves the easy majority
  deterministically; the model is invoked only on segments flagged
  needs_ner_review (disclaimers, multi-part qualifiers, long compound
  names). This is the whole point of the hybrid design -- and it means
  the pipeline still works, with reduced coverage, when no model is
  available.

* NER emits token spans, the classifier wants {name, qualifier}. The
  handoff extracts ING spans from a hard segment and treats each as its
  own ingredient, inheriting the segment's qualifier if it had one.

* Product-level aggregation is deliberately conservative: any haram
  ingredient makes the product haram; any syubhah makes it syubhah;
  only an all-halal ingredient list yields halal. Doubt propagates
  upward rather than being averaged away.

Usage (from the HalalGuard folder):
    python src/pipeline.py --text "Water, Sugar, Gelatin (Bovine), E471"
    python src/pipeline.py --text "..." --no-ner        # rules only
    python src/pipeline.py --file labels.txt --json out.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation import segment_ingredient_list
from classifier import (
    load_e_number_table,
    classify_ingredient,
    normalise_text,
    INGREDIENT_REFERENCE,
    CANNOT_CERTIFY_DISCLAIMER,
    UNKNOWN_POLICIES,
)

try:
    from segmentation import flag_hard_cases
except ImportError:  # older segmentation.py without the flagging step
    flag_hard_cases = None

# Severity order for aggregation. Higher wins.
SEVERITY = {"halal": 0, "cannot_be_certified": 1, "syubhah": 2, "haram": 3}


def clean_span(span: str) -> str:
    """Strip punctuation NER leaves attached to span edges.

    Spans arrive as "flour,", "vanillin)", "bicarbonate," -- the trailing
    comma or bracket is tokenisation debris, not part of the ingredient
    name, and it caused otherwise-known ingredients to miss the reference.
    """
    return span.strip().strip(",;:.()[]{}\"'").strip()


def resolve_fragment(span: str, segment: str, e_table, unknown_policy):
    """Classify a span, but let the surrounding segment disambiguate it.

    NER sometimes splits a compound name: "chocolate liquor" came back as
    "semisweet chocolate" plus a separate "liquor", and the bare fragment
    then read as an intoxicant. If the segment contains a reference term
    that CONTAINS this span, that fuller term is what the label actually
    says, so it decides the verdict.
    """
    verdict = classify_ingredient(span, None, e_table,
                                  unknown_policy=unknown_policy)

    seg_lower = normalise_text(segment.lower())
    span_lower = span.lower()
    best = None
    for term, row in INGREDIENT_REFERENCE.items():
        if span_lower in term and span_lower != term and term in seg_lower:
            if best is None or len(term) > len(best):
                best = term
    if best:
        fuller = classify_ingredient(best, None, e_table,
                                     unknown_policy=unknown_policy)
        fuller["reason"] = (
            f"Span '{span}' resolved using the fuller term '{best}' found in "
            f"the same segment. " + fuller["reason"]
        )
        return fuller
    return verdict


def spans_from_tags(words: list[str], tags: list[str]) -> list[str]:
    """Join B-ING/I-ING sequences into ingredient surface strings.

    Kept separate from the model so it can be tested without one. Handles
    the IOB2 edge cases present in FINER: an orphan I-ING with no preceding
    B-ING starts a span rather than being dropped, and two adjacent B-ING
    tags start two spans rather than merging.
    """
    spans, current = [], []
    for word, tag in zip(words, tags):
        if tag == "B-ING":
            if current:
                spans.append(" ".join(current))
            current = [word]
        elif tag == "I-ING":
            # Orphan I- (no B- before it) still opens a span; FINER
            # contains 59 such violations, so dropping them loses entities.
            current.append(word)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


def parse_args():
    p = argparse.ArgumentParser(description="HalalGuard end-to-end pipeline.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Raw ingredient list to analyse.")
    src.add_argument("--file", help="File with one ingredient list per line.")
    p.add_argument("--table", default="data/sanha_e_numbers_all.csv")
    p.add_argument("--model_dir", default=os.environ.get("HALALGUARD_MODEL_DIR", "models/bert-ner"))
    p.add_argument("--no-ner", action="store_true",
                   help="Skip stage 2 entirely; rules only.")
    p.add_argument("--unknown-policy", choices=UNKNOWN_POLICIES, default="syubhah")
    p.add_argument("--json", help="Write full results to this JSON file.")
    args, _ = p.parse_known_args()
    return args


class NERExtractor:
    """Wraps the fine-tuned model. Loaded lazily so the pipeline runs
    without torch/transformers installed when --no-ner is used."""

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def extract_ingredients(self, text: str) -> list[str]:
        """Return ING spans found in text, as surface strings."""
        words = text.split()
        if not words:
            return []

        enc = self.tokenizer(
            [words], is_split_into_words=True, truncation=True,
            max_length=64, return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            logits = self.model(**enc).logits
        preds = self.torch.argmax(logits, dim=2)[0].cpu().tolist()

        # Only the first subword of each word carries a prediction, matching
        # the convention training used.
        word_ids = enc.word_ids(batch_index=0)
        tags, seen = {}, set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            tags[wid] = self.id2label[preds[pos]]

        tag_seq = [tags.get(i, "O") for i in range(len(words))]
        return spans_from_tags(words, tag_seq)


def analyse(text, e_table, ner=None, unknown_policy="syubhah"):
    """Run one ingredient list through all stages."""
    segments = segment_ingredient_list(text)
    if flag_hard_cases:
        segments = flag_hard_cases(segments)

    results = []
    for seg in segments:
        hard = seg.get("needs_ner_review", False)

        # Stage 2 only fires on hard segments, and only if a model is loaded.
        if hard and ner is not None:
            spans = ner.extract_ingredients(seg["raw"])
            if spans:
                # A parenthetical is only a SOURCE qualifier when it
                # describes the ingredient ("Gelatin (Bovine)"). When NER
                # pulls several ingredients out of one segment, the
                # parenthetical is a sub-ingredient LIST, not a qualifier
                # of each -- e.g. "chocolate (chocolate liquor, sugar,
                # soy lecithin)". Propagating it made one bad word in the
                # list poison every span extracted from that segment.
                inherited = seg.get("qualifier") if len(spans) == 1 else None
                for raw_span in spans:
                    span = clean_span(raw_span)
                    if not span:
                        continue
                    if inherited is not None:
                        verdict = classify_ingredient(
                            span, inherited, e_table,
                            unknown_policy=unknown_policy)
                    else:
                        verdict = resolve_fragment(
                            span, seg["raw"], e_table, unknown_policy)
                    results.append({
                        "source_segment": seg["raw"],
                        "ingredient": span,
                        "qualifier": inherited,
                        "resolved_by": "ner",
                        **verdict,
                    })
                continue
            # No ING span found -- fall through to the rule-based name
            # rather than dropping the segment silently.

        verdict = classify_ingredient(
            seg["name"], seg.get("qualifier"), e_table,
            unknown_policy=unknown_policy)
        results.append({
            "source_segment": seg["raw"],
            "ingredient": seg["name"],
            "qualifier": seg.get("qualifier"),
            "resolved_by": "ner_no_span" if (hard and ner) else "rules",
            "flagged_hard": hard,
            **verdict,
        })

    overall = "halal"
    for r in results:
        if SEVERITY[r["status"]] > SEVERITY[overall]:
            overall = r["status"]

    return {"input": text, "overall": overall, "ingredients": results}


def print_report(report):
    print(f"\nINPUT: {report['input']}")
    print("-" * 78)
    for r in report["ingredients"]:
        q = f" ({r['qualifier']})" if r.get("qualifier") else ""
        print(f"  {r['ingredient'] + q:<34} {r['status']:<20} "
              f"[{r['confidence']:<6}] via {r['resolved_by']}")
    print("-" * 78)
    print(f"  PRODUCT VERDICT: {report['overall'].upper()}")
    print("  (conservative aggregation: the most severe ingredient verdict wins)")
    if any(r["status"] == "cannot_be_certified" for r in report["ingredients"]):
        print(f"\n  {CANNOT_CERTIFY_DISCLAIMER}")


def main():
    args = parse_args()

    e_table = load_e_number_table(args.table)

    ner = None
    if not args.no_ner:
        try:
            ner = NERExtractor(args.model_dir)
            print(f"NER model loaded from {args.model_dir} "
                  f"(device: {ner.device})")
        except Exception as exc:
            print(f"Could not load NER model ({exc}); continuing rules-only.")
    else:
        print("Stage 2 (NER) disabled by --no-ner; rules only.")

    if args.text:
        inputs = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            inputs = [l.strip() for l in f if l.strip()]

    reports = []
    for text in inputs:
        rep = analyse(text, e_table, ner, args.unknown_policy)
        reports.append(rep)
        print_report(rep)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"\nWritten to {args.json}")


if __name__ == "__main__":
    main()
