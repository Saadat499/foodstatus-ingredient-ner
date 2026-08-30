"""
OCR cleanup bridge for HalalGuard-NLP.

Sits between the OCR stage (raw text read off a photographed ingredient
label) and the existing segmentation.py / classifier.py pipeline. Its job
is narrow: turn noisy OCR output into a comma-separated ingredient
statement that looks like the labels the rest of the pipeline was built
and tested on. It does not classify anything itself -- classification
still happens entirely in classifier.py.

    raw OCR text
        -> [1] structural cleanup   strip boilerplate, fix line wraps,
                                     balance dropped brackets
        -> [2] segment splitting    comma-split while respecting "(...)"
        -> [3] vocabulary matching  fuzzy-correct each base ingredient name
                                     against your existing reference terms
        -> cleaned ingredient statement, ready for segmentation.py

Zero third-party dependencies (stdlib only), so it drops into src/
alongside your existing files without touching requirements.txt.

Usage once wired into your project:

    from ocr_bridge import clean_ocr_text
    result = clean_ocr_text(raw_ocr_text)
    # result["cleaned_text"] is what you feed to segmentation.py / pipeline.py
    # result["corrections"] is a log of what changed, for your evaluation step
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from typing import Iterable

try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker()
except ImportError:
    _SPELL = None  # degrade gracefully -- see _is_real_word below

# ---------------------------------------------------------------------------
# Stage 1: structural cleanup
# ---------------------------------------------------------------------------

# Lines/prefixes that are packaging boilerplate, not ingredients. Applied
# before splitting so they don't get mistaken for ingredient segments.
# Note: the leading "Ingredients:" marker is handled separately, by
# _cut_before_ingredients_marker below, not in this list -- see its
# docstring for why.
_BOILERPLATE_PATTERNS = [
    r"(?i)\bnutrition(al)?\s+facts?\b.*$",     # trailing nutrition table junk
    r"(?i)\bbest\s+before\b.*$",
    r"(?i)\bnet\s+wt\.?\b.*$",
    r"(?i)\ballerg(en|y)\s+advice\b.*$",
    # Common dietary-claim phrases that real OCR frequently glues onto the
    # last real ingredient when the punctuation between them gets misread
    # (a period where a comma should be). Removing just the phrase itself,
    # not "everything after" -- these are short, self-contained claims,
    # not markers that the rest of the label is safe to discard.
    r"(?i)\bgluten\s+free\.?",
    r"(?i)\bsuitable\s+for\s+vegetarians?\.?",
    r"(?i)\bsuitable\s+for\s+vegans?\.?",
]


def _dehyphenate(text: str) -> str:
    """Join words split across a line wrap: 'gela-\\ntin' -> 'gelatin'."""
    return re.sub(r"-\s*\n\s*", "", text)


def _cut_before_ingredients_marker(text: str) -> str:
    """
    Find "Ingredients:" ANYWHERE in the text (not just at the true start)
    and discard everything before it, including the marker itself.

    This replaces an earlier version that only matched "Ingredients:" when
    it was anchored to the very start of the string. That worked for a
    single-column label but broke on a real photo with two panels side by
    side (Nutrition Facts on the right, Ingredients on the left): OCR read
    "Nutrition Facts" first, so "Ingredients:" was no longer at the start,
    the old anchor never fired, and the later "strip everything after
    Nutrition Facts" step deleted the ENTIRE real ingredient list along
    with it -- confirmed by testing the exact real output, not assumed.

    Searching for the marker anywhere and cutting before it fixes this
    regardless of which panel OCR happens to read first, and safely does
    nothing if no "Ingredients:" marker is found at all (better to leave
    unrelated noise in than risk cutting into real content).
    """
    match = re.search(r"(?i)\bingredients?\s*[:\-]\s*", text)
    if match:
        return text[match.end():]
    return text


def _strip_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        # DOTALL is required: without it, "." can't cross a newline, so
        # ".*$" silently stops at the first line break and the rest of a
        # multi-line trailer (address, "best before", etc.) survives
        # untouched. Confirmed this was happening on real OCR output --
        # not a hypothetical edge case.
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


def _balance_parentheses(text: str) -> str:
    """
    OCR frequently drops one paren of a pair, which would otherwise make
    segmentation.py misread the rest of the label as "inside brackets".
    If counts are off by exactly one, patch it. This is a heuristic, not
    a guarantee -- log entries where it fires if you want to audit it.
    """
    opens, closes = text.count("("), text.count(")")
    if opens == closes + 1:
        text = text + ")"
    elif closes == opens + 1:
        text = text.replace(")", "", 1)
    return text


def structural_cleanup(raw_text: str) -> str:
    text = _dehyphenate(raw_text)
    text = _cut_before_ingredients_marker(text)
    text = _strip_boilerplate(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = _balance_parentheses(text)
    return text


# ---------------------------------------------------------------------------
# Stage 2: segment splitting (comma-aware, bracket-respecting)
# ---------------------------------------------------------------------------

def _split_respecting_parens(text: str) -> list[str]:
    """Split on commas, but never inside "(...)" -- qualifiers like
    "Gelatin (Bovine, Halal-Certified)" must stay one segment."""
    segments, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        segments.append("".join(current))
    return segments


_ARTIFACT_TOKENS = {"", ".", ",", "|", "-", "_", "*", "%"}


def _clean_segment_artifacts(segment: str) -> str:
    """Drop stray punctuation tokens a scan sometimes leaves behind."""
    words = segment.split()
    words = [w for w in words if w not in _ARTIFACT_TOKENS]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Stage 3: vocabulary-based fuzzy correction (word level)
# ---------------------------------------------------------------------------

# Conservative character-confusion table. Used only to generate a
# *candidate* spelling for vocabulary matching -- never applied blindly to
# the text that gets returned, so a correct word is never mangled.
_CONFUSIONS = {
    "0": "o", "1": "l", "5": "s", "8": "b", "|": "l",
    "!": "i", "$": "s", "rn": "m", "vv": "w",
}

# Words that matter for classification but usually won't appear in an
# ingredient-name reference file, since they describe *source*, not the
# ingredient itself. Getting "Bovine" vs "Porcine" wrong is a correctness
# problem, not a cosmetic one, so these are corrected too.
_DEFAULT_QUALIFIER_WORDS = {
    "bovine", "porcine", "ovine", "avian", "vegetable", "vegetarian",
    "synthetic", "artificial", "natural", "animal", "plant", "certified",
    "halal", "kosher", "derived", "source",
}

_PUNCT_STRIP = "()[]{}.,;:!\"'"


def _candidate_spellings(token: str) -> list[str]:
    """A handful of plausible alternate spellings for fuzzy matching."""
    lowered = token.lower()
    swapped = lowered
    for wrong, right in _CONFUSIONS.items():
        swapped = swapped.replace(wrong, right)
    return list({lowered, swapped})


def _build_word_vocab(
    vocabulary: Iterable[str], qualifier_vocabulary: Iterable[str] | None = None
) -> set[str]:
    """
    Word-level vocabulary: every whole phrase AND every individual word
    inside it. Word-level matters because OCR errors land inside compound
    names ("Cocoa Butter", "Bacon Fat") and correcting only whole phrases
    misses those unless the whole phrase happens to survive intact.
    """
    words: set[str] = set()
    for phrase in vocabulary:
        phrase = phrase.lower().strip()
        if not phrase:
            continue
        words.add(phrase)
        words.update(phrase.split())
    words.update(qualifier_vocabulary or _DEFAULT_QUALIFIER_WORDS)
    return words


# E-numbers (E100-E1599, optionally with a trailing letter like E472e) are
# too short and densely packed for fuzzy matching to be safe: several
# valid codes can sit at the identical edit-distance from a garbled
# token, so picking "the closest one" is really picking one at random
# among several with very different halal implications.
#
# Checking the shape of the INPUT doesn't work -- OCR noise is exactly
# what destroys that shape (a garbled "E471" can easily contain a letter
# where a digit used to be, e.g. "E4T1"). What matters is the shape of
# the proposed CORRECTION: if the best fuzzy match looks like an
# E-number, refuse it unless it was an exact hit (checked separately,
# above, and always trustworthy). An ambiguous E-number should surface
# downstream as unrecognized and need review, not silently become the
# wrong one.
_CLEAN_ENUMBER_PATTERN = re.compile(r"^e\d{3,4}[a-z]?$")


def _is_risky_enumber_guess(candidate_word: str) -> bool:
    return bool(_CLEAN_ENUMBER_PATTERN.match(candidate_word))


def _length_gap_too_large(a: str, b: str) -> bool:
    """
    Real OCR corruption (dropped/added/swapped characters) rarely changes
    a word's length by more than one character -- every legitimate
    correction found in testing has a length difference of 0 or 1
    ("buttor"/"butter": 0, "bacn"/"bacon": 1, "watr"/"water": 1). A larger
    gap is a signal that two genuinely different words just happen to
    share a lot of characters, not that one is a garbled version of the
    other -- this is exactly how "flavour" (a real, correctly-spelled
    word, not in the vocabulary) got wrongly matched to the unrelated
    vocabulary word "flour" (length difference of 2).
    """
    return abs(len(a) - len(b)) > 1


# The default pyspellchecker dictionary is American-English only and
# doesn't recognize common British spellings ("flavour", "colour") --
# confirmed by testing, not assumed. That matters a lot here: British
# spelling is the norm on labels from the UK and Commonwealth/JAKIM-region
# markets this project targets, so relying on the dictionary alone would
# silently fail on exactly the words most likely to appear. This is a
# small, deliberately narrow supplement (food-label vocabulary only, not
# a general British/American spelling converter).
_BRITISH_SPELLING_EXTRAS = {
    "flavour", "flavours", "flavoured", "flavouring", "flavourings",
    "colour", "colours", "coloured", "colouring", "colourings",
    "fibre", "fibres", "sulphite", "sulphites", "sulphate", "sulphates",
    "aluminium", "mould", "moulds", "moulded",
}


def _is_real_word(word: str) -> bool:
    """
    True if `word` is a legitimate, correctly-spelled English word --
    meaning it should never be treated as a candidate for correction, no
    matter how close a fuzzy match to some unrelated vocabulary word looks.
    This addresses the actual root cause of the "flavour" -> "flour" class
    of bug: the word was never garbled in the first place. If the
    spellchecker package isn't installed, this always returns False,
    which just falls back to the previous (length-gap-guarded) behavior
    rather than failing outright.
    """
    if _SPELL is None:
        return False
    return bool(_SPELL.known([word])) or word in _BRITISH_SPELLING_EXTRAS

# A confusion-swap guess (e.g. "flav0ur" -> "flavour") that doesn't land
# on an exact vocabulary word is a hypothesis, not confirmed text.
# Fuzzy-matching that hypothesis against the vocabulary needs a much
# higher bar than fuzzy-matching what was actually observed, or it can
# drift onto an unrelated real word -- this constant is set with a
# comfortable margin above a verified good case (guessed "sodum" ~
# "sodium" scores 0.909) and below a verified bad one (guessed "flavour"
# ~ the unrelated word "flour" scores 0.833).
_GUESS_MATCH_CUTOFF = 0.87


def _correct_token(token: str, word_vocab: set[str], cutoff: float) -> tuple[str, bool]:
    """
    Correct a single whitespace-separated token, preserving any leading or
    trailing punctuation (so "(Bovine)" corrects the word, not the parens)
    and roughly preserving the original capitalisation style.
    """
    prefix, core, suffix = "", token, ""
    while core and core[0] in _PUNCT_STRIP:
        prefix, core = prefix + core[0], core[1:]
    while core and core[-1] in _PUNCT_STRIP:
        suffix, core = core[-1] + suffix, core[:-1]

    # Skip short tokens and pure numbers -- too easy to "correct" into
    # something confidently wrong with too little to go on. The threshold
    # is 4, not the original 3: real testing showed a correctly-spelled,
    # unrelated 3-letter word like "for" can score a coincidental 0.75
    # fuzzy match against an unrelated 5-letter vocabulary word ("flour"),
    # silently damaging text that was never wrong. 4-letter words (e.g.
    # "Watr" -> "Water") are still worth correcting and haven't shown this
    # problem in testing so far.
    if len(core) < 4 or core.lower() in word_vocab or core.isdigit():
        return token, False

    candidates = _candidate_spellings(core)  # {raw lowered, confusion-swapped}
    raw = core.lower()

    # Exact hits are always trustworthy, whichever candidate produced them
    # -- including landing exactly on an E-number, which is fine since
    # there's no ambiguity to resolve.
    for candidate in candidates:
        if candidate in word_vocab:
            return _apply_case(prefix, candidate, suffix, core), True

    # No exact hit, and the word is already legitimate English -- this is
    # the actual fix for the "flavour" -> "flour" class of bug: a word
    # that was never garbled should never be a candidate for correction,
    # regardless of how close a fuzzy match to some vocabulary word looks.
    if _is_real_word(raw):
        return token, False

    # No exact hit, and it's not a recognizable word: fuzzy-match the RAW
    # observed token at the normal cutoff (anchored to reality), and any
    # confusion-swap GUESS only at the much stricter cutoff (anchored to a
    # hypothesis). Keep whichever clears its bar with the higher ratio --
    # but never accept a fuzzy match that lands on an E-number-shaped word
    # (see _is_risky_enumber_guess) or one with an implausible length gap
    # (see _length_gap_too_large) -- kept as defense in depth even with
    # the real-word check in place, since that check only helps when the
    # OBSERVED token happens to be a dictionary word itself.
    best_ratio, best_match = 0.0, None

    for match in difflib.get_close_matches(raw, word_vocab, n=1, cutoff=cutoff):
        if _is_risky_enumber_guess(match) or _length_gap_too_large(raw, match):
            continue
        ratio = difflib.SequenceMatcher(None, raw, match).ratio()
        if ratio > best_ratio:
            best_ratio, best_match = ratio, match

    for candidate in candidates:
        if candidate == raw:
            continue
        for match in difflib.get_close_matches(candidate, word_vocab, n=1, cutoff=_GUESS_MATCH_CUTOFF):
            if _is_risky_enumber_guess(match) or _length_gap_too_large(candidate, match):
                continue
            ratio = difflib.SequenceMatcher(None, candidate, match).ratio()
            if ratio > best_ratio:
                best_ratio, best_match = ratio, match

    if best_match:
        return _apply_case(prefix, best_match, suffix, core), True
    return token, False


def _apply_case(prefix: str, corrected_core: str, suffix: str, original_core: str) -> str:
    if original_core.isupper():
        corrected_core = corrected_core.upper()
    elif original_core[:1].isupper():
        corrected_core = corrected_core.capitalize()
    return prefix + corrected_core + suffix


def correct_segment_words(
    segment: str, word_vocab: set[str], cutoff: float = 0.75
) -> tuple[str, list[dict]]:
    """Run word-level correction across a whole segment, parens included."""
    if segment.strip().lower() in word_vocab:
        return segment, []  # already an exact known phrase, leave untouched

    tokens = segment.split(" ")
    out_tokens: list[str] = []
    corrections: list[dict] = []
    for tok in tokens:
        new_tok, changed = _correct_token(tok, word_vocab, cutoff)
        if changed:
            corrections.append({"original": tok, "corrected": new_tok})
        out_tokens.append(new_tok)
    return " ".join(out_tokens), corrections


# ---------------------------------------------------------------------------
# Vocabulary loading -- reuses your existing data, no duplicate list
# ---------------------------------------------------------------------------

def load_default_vocabulary() -> set[str]:
    """
    Pull known ingredient names straight from your existing classifier's
    ingredient_reference.csv, so this module never maintains its own
    duplicate ingredient list.

    Deliberately does NOT rely on classifier.py's own module-level
    INGREDIENT_REFERENCE constant, because that constant is computed at
    import time using a path relative to classifier.py's default argument
    ("data/ingredient_reference.csv") -- which resolves relative to
    whatever directory the command was RUN from, not relative to
    classifier.py's own location. Run the command from inside src/ and
    that default silently resolves to src/data/..., finds nothing, and
    classifier.py's own loader just returns {} (no exception). This
    function instead tries a few sensible, script-relative locations
    directly, so it works the same regardless of your working directory.

    Falls back to a tiny 14-term demo list only if none of that works --
    and always prints a warning to stderr when it does, since a silent
    fallback here is worse than no fallback at all: it makes every
    "correction" look like it worked while actually testing against a
    placeholder vocabulary instead of your real reference file.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(module_dir, "..", "data", "ingredient_reference.csv"),  # sibling data/
        os.path.join(module_dir, "data", "ingredient_reference.csv"),        # data/ inside src/
        os.path.join("data", "ingredient_reference.csv"),                    # cwd-relative, last resort
    ]

    try:
        sys.path.insert(0, module_dir)
        from classifier import load_ingredient_reference  # your existing function
        for path in candidate_paths:
            ref = load_ingredient_reference(path)
            if ref:
                return set(ref.keys())
    except Exception as exc:
        print(f"ocr_bridge: could not import classifier.py ({exc})", file=sys.stderr)

    print(
        "ocr_bridge: WARNING -- falling back to a 14-term demo vocabulary. "
        "ingredient_reference.csv was not found in any of: "
        + ", ".join(os.path.normpath(p) for p in candidate_paths)
        + ". Corrections below are NOT using your real reference file.",
        file=sys.stderr,
    )
    return {
        "water", "sugar", "salt", "gelatin", "bacon", "vanilla extract",
        "milk", "butter", "cocoa butter", "soy lecithin", "wheat flour",
        "citric acid", "sodium bicarbonate", "natural flavour",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_ocr_text(
    raw_text: str,
    vocabulary: Iterable[str] | None = None,
    qualifier_vocabulary: Iterable[str] | None = None,
    cutoff: float = 0.75,
) -> dict:
    """
    Turn raw OCR output into a cleaned, comma-separated ingredient
    statement, plus a log of what changed. The log matters for your
    evaluation step: you want to measure how often this stage fires and
    whether its corrections are actually right, not just trust it silently.

    Returns:
        {
            "cleaned_text": str,        # feed this into segmentation.py
            "corrections": list[dict],  # [{"original": ..., "corrected": ...}]
        }
    """
    word_vocab = _build_word_vocab(
        vocabulary or load_default_vocabulary(), qualifier_vocabulary
    )
    structural = structural_cleanup(raw_text)
    raw_segments = _split_respecting_parens(structural)

    cleaned_segments: list[str] = []
    corrections: list[dict] = []

    for raw_seg in raw_segments:
        seg = _clean_segment_artifacts(raw_seg)
        if not seg:
            continue
        new_seg, seg_corrections = correct_segment_words(seg, word_vocab, cutoff)
        corrections.extend(seg_corrections)
        cleaned_segments.append(new_seg)

    return {
        "cleaned_text": ", ".join(cleaned_segments),
        "corrections": corrections,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEMO_SAMPLES = [
    "INGREDIENTS: Watar, Sugai, Getatin (8ovine), Vanilla Extrsct, Sodurn Bicarbo-\nnate, 8acon Fat.",
    "watef, suqar, salt, gelatin(Bovine) c0c0a buttor, N4TURAL FLAV0UR",
]


def _run_demo(cutoff: float) -> None:
    for sample in _DEMO_SAMPLES:
        result = clean_ocr_text(sample, cutoff=cutoff)
        print("Raw OCR text:")
        print(" ", repr(sample))
        print("Cleaned text:")
        print(" ", result["cleaned_text"])
        if result["corrections"]:
            print("Corrections applied:")
            for c in result["corrections"]:
                print(f"   {c['original']!r} -> {c['corrected']!r}")
        else:
            print("No corrections were needed.")
        print("-" * 60)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Clean OCR-scanned ingredient text before classification."
    )
    parser.add_argument("--text", help="Raw OCR text to clean")
    parser.add_argument("--file", help="Path to a text file of raw OCR output")
    parser.add_argument(
        "--cutoff", type=float, default=0.75,
        help="Fuzzy-match similarity threshold, 0-1 (default 0.75)",
    )
    parser.add_argument(
        "--pipeline-cmd",
        help="Optional: path to your pipeline.py, to pipe the cleaned text "
             "straight into it, e.g. --pipeline-cmd src/pipeline.py",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on built-in noisy OCR samples instead of --text/--file",
    )
    args = parser.parse_args()

    if args.demo:
        _run_demo(args.cutoff)
        return

    if args.file:
        raw = open(args.file, encoding="utf-8").read()
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()

    result = clean_ocr_text(raw, cutoff=args.cutoff)
    print("Cleaned text:")
    print(" ", result["cleaned_text"])
    if result["corrections"]:
        print("\nCorrections applied:")
        for c in result["corrections"]:
            print(f"  {c['original']!r} -> {c['corrected']!r}")
    else:
        print("\nNo corrections were needed.")

    if args.pipeline_cmd:
        print("\nPiping into", args.pipeline_cmd, "...\n")
        subprocess.run([sys.executable, args.pipeline_cmd, "--text", result["cleaned_text"]])


if __name__ == "__main__":
    _cli()
