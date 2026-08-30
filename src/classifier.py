"""
Stage 2: Halal/Haram/Syubhah classification.

Hybrid approach: rule-table lookup + qualifier detection on the surrounding
text. The table (data/sanha_e_numbers_all.csv, 419 entries) is parsed from
all three of SANHA's published E-number category pages -- Halaal (354),
Mashbooh (64) and Haraam (1) -- so an entry's own status decides how it is
handled. Only entries SANHA itself files as doubtful go through qualifier
resolution.

Qualifier detection follows SANHA's own stated rule for source-dependent
ingredients (sanha.org.za/2012/12/14/gelatine/): pork is always haraam,
fish/marine always halaal, and bovine/poultry depend on slaughter method,
so naming the animal alone does not resolve the doubt.

A separate `verification_risk` field applies MUIS Annex A's four-tier
ingredient risk classification. It is an independent axis and never changes
the halal/haram/syubhah verdict.

NOTE ON JAKIM: no JAKIM data is used here. JAKIM publishes no E-number
list -- MPPHM 2020 is a certification procedure manual only. The
"source determines status, not the code" principle was sourced from
JHEAINS, a Malaysian *state* religious department FAQ. JAKIM's actual
relevance is that it formally recognises SANHA as a certification body,
as does MUIS.

This is deliberately NOT trained on the Kaggle datasets -- we proved those
labels are keyword-triggered rather than source-verified, and training on
them would launder that bias rather than fix it.
"""

import csv
import re
import os

# Source words that flip a "syubhah -- check source" entry toward a
# confident verdict, based on what's actually written in the ingredient text.
HARAM_SOURCE_WORDS = {"pork", "porcine", "pig", "swine", "lard"}

# These genuinely CLOSE the doubt SANHA's list exists for: either the
# source isn't animal at all (so the slaughter-method question doesn't
# apply), or it's an explicit certification claim -- the actual resolving
# mechanism SANHA itself points to.
#
# fish/marine are included on SANHA's OWN explicit authority. Their
# gelatine FAQ (sanha.org.za/2012/12/14/gelatine/) states: "If it is
# derived from fish (marine) sources, it will always be deemed Halaal."
# Marine sources require no ritual slaughter, so no doubt remains.
RESOLVES_DOUBT_WORDS = {
    "plant", "vegetable", "vegan", "synthetic", "microbial", "fermentation",
    "soy", "sunflower", "coconut", "halal-certified", "certified",
    "fish", "marine",
}

# These are animal-sourced, but merely naming the animal does NOT confirm
# how it was slaughtered -- which is the actual thing SANHA's doubt is
# about.
#
# This is NOT an inference. SANHA states it directly in the same FAQ:
# "When derived from beef (bovine) or chicken (poultry), we need to
# determine whether the raw material has come from a Halaal slaughtered
# source or not." Naming the animal is therefore insufficient on its own.
ANIMAL_SOURCED_UNCONFIRMED_WORDS = {
    "beef", "bovine", "cattle", "lamb", "mutton", "chicken", "poultry", "animal",
}

# Definite triggers for ingredients written by NAME rather than E-number.
# Prof. Husna: "some products don't have E numbers, they just list the
# ingredients" -- so name-level matching is required, not optional.
#
# Basis for each term is recorded in data/keyword_sources.csv rather than
# inline here. Kept conservative: only terms that are haram/halal by their
# own nature, never terms that merely *suggest* a source. Anything
# source-dependent (whey, rennet, mono- and diglycerides) is deliberately
# absent -- those must fall through to the SANHA table or to syubhah, not
# be settled by a keyword.
DEFINITE_HARAM_KEYWORDS = {
    # Swine and swine-derived. Haram by nature regardless of processing.
    "pork", "bacon", "ham", "swine", "porcine", "pig", "piggy",
    "lard", "gammon", "pancetta", "prosciutto", "speck", "guanciale",
    # Intoxicants.
    "wine", "beer", "rum", "whisky", "whiskey", "vodka", "brandy",
    "liqueur", "liquor", "cognac", "sherry", "champagne", "ale",
    "kirsch", "marsala", "sake",
    # Insect-derived colourants (SANHA Haraam list).
    "e120", "cochineal", "carmine", "carminic acid",
    # Blood and carrion, prohibited by nature.
    "blood plasma", "blood meal", "carrion",
}

# Terms that are TRADITIONALLY swine or alcohol based but not necessarily
# so -- beef salami, turkey pepperoni and non-alcoholic mirin all exist.
# Calling these haram outright would be a false positive, so they resolve
# to syubhah: genuinely doubtful pending the source.
AMBIGUOUS_SOURCE_TERMS = {
    "salami", "chorizo", "pepperoni", "bratwurst", "mirin", "frankfurter",
}

# Compounds where a haram keyword appears but the substance is not haram.
# SANHA states plainly that wine is haraam while vinegar is halaal, so
# "wine vinegar" must not inherit the "wine" trigger.
HARAM_TERM_EXCEPTIONS = {
    "wine vinegar", "red wine vinegar", "white wine vinegar",
    "cider vinegar", "apple cider vinegar", "balsamic vinegar",
    # Confectionery named after a flavour it does not contain.
    "wine gum", "wine gums", "rum flavouring", "rum flavoring",
    # NOTE: "chocolate liquor" / "cocoa liquor" are NOT listed here. They
    # are in data/ingredient_reference.csv as halal instead, so subsumption
    # resolves them ("liquor" is a substring of "chocolate liquor" and is
    # therefore discarded). Listing them here would skip the reference
    # lookup entirely and leave them unresolved.
}

# Explicit negation. Without this, "pork-free gelatin" and "alcohol-free
# vanilla" were returned as HARAM -- the trigger word fired even though
# the label states the opposite. Matched before any haram lookup.
NEGATION_PATTERNS = [
    r"\b{t}[- ]free\b",
    r"\bfree[- ]from[- ]{t}\b",
    r"\bno[- ]{t}\b",
    r"\bnon[- ]?{t}\b",
    r"\bwithout[- ]{t}\b",
    r"\bnon[- ]?alcoholic\b" ,
]

# Plant-based substitutes borrow the name of the thing they replace.
# "Vegetarian bacon" contains no swine; treating it as haram is a false
# positive on a product specifically made to be permissible.
SUBSTITUTE_MARKERS = {
    "vegetarian", "vegan", "plant-based", "plant based",
    "meat-free", "meat free", "veggie", "imitation", "mock",
}

# Cured-meat names are swine by default but exist in other meats. "Beef
# bacon" and "turkey salami" are real products. Animal-sourced, so the
# slaughter question applies -- syubhah, not haram.
NON_PORK_MEAT_PREFIXES = {"beef", "turkey", "chicken", "lamb", "mutton", "veal"}


def _is_negated(text: str, term: str) -> bool:
    """True if the text explicitly denies containing the term."""
    esc = re.escape(term)
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern.format(t=esc), text):
            return True
    return False
DEFINITE_HALAL_KEYWORDS = {
    # Confirmed on SANHA's Halaal E-Numbers page.
    "water", "salt", "sugar", "e100", "curcumin", "e300", "ascorbic acid",
    "e330", "citric acid", "e407", "carrageenan", "e415", "xanthan gum",
    "e904", "shellac",
    # Plant/mineral ingredients with no animal-source pathway.
    "wheat flour", "corn starch", "cornstarch", "rice flour", "oat",
    "barley", "soy flour", "cocoa", "cocoa butter", "olive oil",
    "sunflower oil", "canola oil", "rapeseed oil", "palm oil",
    "coconut oil", "sesame oil", "agar", "agar-agar", "guar gum",
    "pectin", "dextrose", "fructose", "maltodextrin", "sea salt",
    "baking soda", "sodium bicarbonate", "yeast extract",
    # SANHA FAQ states plainly: "Wine is Haraam & Vinegar is Halaal."
    "vinegar",
}

# The fourth status, approved by Prof. Husna with the condition that it
# carries a disclaimer explaining what it is and why.
#
# CRITICAL: this is NOT a fiqh ruling. Islamically there are three
# categories -- halal, haram, syubhah. "cannot_be_certified" is a
# PROCESS/EVIDENCE state meaning "no certifying authority has ruled on
# this, so no verdict can be issued". It is emitted with every such result
# so the distinction can never be lost downstream.
CANNOT_CERTIFY_DISCLAIMER = (
    "NOTE: 'cannot_be_certified' is not an Islamic ruling. Islamic law "
    "recognises three categories: halal, haram, syubhah. This status means "
    "only that no consulted certifying authority has published a ruling on "
    "this ingredient, so the system has no evidential basis to issue one. "
    "It indicates absence of evidence, not evidence of prohibition."
)

# What to do when an ingredient matches nothing. Both options are traceable
# to a named authority rather than being an arbitrary default.
#
#   "syubhah" -- unresolvable status is doubtful. Conservative, and stays
#                inside the three Islamic categories.
#   "muis"    -- MUIS certification conditions state that ingredients not
#                supported by the relevant documents are not certifiable.
#                Maps to cannot_be_certified, NOT to haram: MUIS's rule is
#                about certifiability, not permissibility, and conflating
#                the two would overstate what MUIS actually says.
UNKNOWN_POLICIES = ("syubhah", "muis")

# MUIS Annex A risk-tier keywords -- kept as literal as possible to MUIS's
# own named examples, not inferred/extended categories. A second, independent
# axis alongside the halal/haram/syubhah verdict. Checked high-to-low since
# an ingredient can match more than one tier.
HIGH_RISK_KEYWORDS = {
    "flavouring", "flavoring", "gelatine", "gelatin", "meat", "poultry",
    "chicken skin", "chicken fat", "beef extract", "beef tallow",
    "canned", "confectionery", "pastry", "dairy", "processed seafood",
    "sauce", "condiment", "pure vinegar",
}
MEDIUM_HIGH_RISK_KEYWORDS = {"enzyme", "yeast", "cheese"}
MEDIUM_LOW_RISK_KEYWORDS = {
    "soy bean", "flour", "olive oil", "sesame oil", "vegetable oil",
    "synthetic vinegar", "noodle", "pasta",
}
LOW_RISK_KEYWORDS = {
    "vegetable", "fruit", "pure seafood", "legume", "lentil", "rice",
    "salt", "sugar", "ice", "spice", "synthetic chemical",
}


def estimate_verification_risk(name: str, qualifier: str | None) -> str:
    """MUIS Annex A risk tier: low / medium-low / medium-high / high.
    Independent of the halal/haram/syubhah verdict -- this measures how
    much documentation an ingredient category inherently warrants, not
    whether it's actually halal."""
    combined = f"{name} {qualifier or ''}".strip().lower()
    if any(kw in combined for kw in HIGH_RISK_KEYWORDS):
        return "high"
    if any(kw in combined for kw in MEDIUM_HIGH_RISK_KEYWORDS):
        return "medium-high"
    if any(kw in combined for kw in MEDIUM_LOW_RISK_KEYWORDS):
        return "medium-low"
    if any(kw in combined for kw in LOW_RISK_KEYWORDS):
        return "low"
    return "unrated"


def load_ingredient_reference(csv_path: str = "data/ingredient_reference.csv") -> dict:
    """Load the externalised term reference.

    Knowledge lives in data, not in code. Each term declares the evidence
    it rests on, so a verdict can state its own basis rather than
    presenting a certifier ruling and an engineering guess identically:

      T1_CERTIFIER_RULING -- a named body published this exact ruling
      T2_SCRIPTURAL       -- Quranic/hadith prohibition
      T3_PRINCIPLE        -- original permissibility; no animal pathway
      T4_JUDGMENT         -- engineering decision, no external source

    Returns {} if the file is absent, so the hardcoded sets still apply
    and the classifier degrades rather than breaking.
    """
    ref = {}
    if not os.path.exists(csv_path):
        return ref
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row.get("term", "").strip().lower()
            if term:
                ref[term] = row
    return ref


# Loaded once at import. Tests and callers may pass their own.
INGREDIENT_REFERENCE = load_ingredient_reference()

# Ordered longest-first so "skimmed milk powder" wins over "milk".
_REFERENCE_TERMS = sorted(INGREDIENT_REFERENCE, key=len, reverse=True)


def lookup_reference(text: str, ref: dict | None = None) -> dict | None:
    """Resolve a term using specificity first, then severity.

    Two competing failure modes had to be reconciled:

      "Vegetable Oil and Bacon Fat" -- two INDEPENDENT ingredients in one
      string. Longest-match returned HALAL because "vegetable oil" is
      longer than "bacon". Severity must win here.

      "Cocoa Butter" -- ONE ingredient whose name CONTAINS another term.
      Severity-first returned SYUBHAH because dairy "butter" outranks
      plant "cocoa butter". Specificity must win here.

    The distinction is subsumption: if a matched term is a proper
    substring of another matched term, the shorter one is not a separate
    ingredient, it is part of the longer name -- so it is discarded.
    Severity then decides among whatever genuinely independent matches
    remain, so a false halal is still impossible.
    """
    ref = INGREDIENT_REFERENCE if ref is None else ref
    matched = [t for t in ref if _contains_term(text, t)]
    if not matched:
        return None

    # Drop any match subsumed by a more specific match.
    specific = [t for t in matched
                if not any(t != other and t in other for other in matched)]

    order = {"haram": 0, "syubhah": 1, "halal": 2}
    specific.sort(key=lambda t: (order.get(ref[t].get("verdict"), 3), -len(t)))
    return ref[specific[0]]


def load_e_number_table(csv_path: str) -> dict:
    """Load the SANHA-sourced E-number table into a lookup dict keyed by
    both E-number and ingredient name (lowercased) for flexible matching."""
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e_num = row["e_number"].strip().lower()
            name = row["name"].strip().lower()
            table[e_num] = row
            table[name] = row
            # SANHA writes sub-codes as "E440(a)"; labels and users write
            # "E440a". Index both so either form resolves to the same row.
            table[e_num.replace("(", "").replace(")", "")] = row
            for alt in row.get("alternative_names", "").split(";"):
                alt = alt.strip().lower()
                if alt and alt != "n/a":
                    table[alt] = row
    return table


def _normalize_e_number(text: str) -> str | None:
    """Extract a normalized 'e123' style key from text like 'E-471',
    'E 471', '(E471)', or plain '471' if it looks like an additive code."""
    # The trailing group also captures SANHA-style sub-codes -- "440(a)"
    # and "440a" are the same additive, so both must normalise to one key.
    # The trailing group also captures SANHA-style sub-codes. These come in
    # several written forms for the same additive -- "440(a)", "440a",
    # "440 (a)" -- and roman-numeral variants like "952(iii)". All must
    # normalise to a single key.
    match = re.search(r"e[\s\-]?(\d{3,4})\s*(\([a-z]+\)|[a-z]\b)?", text.lower())
    if match:
        suffix = (match.group(2) or "").replace("(", "").replace(")", "")
        return f"e{match.group(1)}{suffix}"
    return None


def _contains_term(text: str, term: str) -> bool:
    """Whole-word/whole-code match, NOT a bare substring match.

    Critical: a plain `term in text` check silently misclassifies real
    entries. 'e100' (curcumin, definite halal) is a substring of 'e1000'
    (cholic acid -- an animal bile acid that SANHA lists as syubhah), so
    substring matching returned a confident 'halal' verdict for E1000.
    \\b word boundaries prevent that class of collision entirely.
    """
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def normalise_text(text: str) -> str:
    """Strip markup that real label sources embed in ingredient text.

    Open Food Facts wraps allergens in underscores, so labels arrive as
    '_wheat_ flour' and 'malted _barley_ extract'. Those underscores
    broke matching and showed up as unrecognised ingredients that the
    reference file already covered -- a normalisation bug masquerading
    as a coverage gap.
    """
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def _sanha_desc(entry: dict) -> str:
    """Render SANHA's source description for use inside a reason string.

    28 of the 419 entries have no description -- SANHA's list-page card
    excerpts truncate before the Source field on entries with long
    alternative-name lists. Returning an empty parenthetical "()" looks
    like a bug, so say plainly that the detail wasn't published in the
    list view instead.
    """
    desc = (entry.get("source_description") or "").strip()
    if desc:
        return f" ({desc})"
    return " (no source detail in SANHA's list-view entry)"


def classify_ingredient(name: str, qualifier: str | None, e_table: dict,
                        unknown_policy: str = "syubhah") -> dict:
    """Classify a single segmented ingredient. Returns status, confidence,
    and a plain-language reason -- explainability matters here as much as
    the verdict itself.

    unknown_policy controls only the no-match fallback; see UNKNOWN_POLICIES.
    """
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(f"unknown_policy must be one of {UNKNOWN_POLICIES}")
    name_lower = name.strip().lower()
    qualifier_lower = (qualifier or "").strip().lower()
    combined_text = normalise_text(f"{name_lower} {qualifier_lower}")
    risk = estimate_verification_risk(name, qualifier)

    # 1. Check definite keyword triggers first (highest confidence, no
    #    ambiguity by definition). Exceptions are checked before triggers:
    #    "wine vinegar" contains "wine" but is not an intoxicant.
    is_exception = any(_contains_term(combined_text, ex)
                       for ex in HARAM_TERM_EXCEPTIONS)

    # 0. SANHA's own ruling on this exact named substance takes precedence
    #    over every generic heuristic below. Without this, keyword matching
    #    fires on substrings of chemical names and contradicts the
    #    certifier: "Indigo Carmine" (E132, SANHA: halaal) matched the
    #    'carmine' haram keyword; "Amidated Pectin" (E440(b), SANHA:
    #    mashbooh) matched the 'pectin' halal keyword. A specific published
    #    ruling must outrank a general rule of thumb.
    _early_key = _normalize_e_number(name_lower)
    _early = e_table.get(_early_key) if _early_key else e_table.get(name_lower)
    if _early is None:
        _early = e_table.get(name_lower)
    if _early is not None and _early.get("status", "").strip().lower() in ("halal", "haram"):
        st = _early["status"].strip().lower()
        return {
            "status": st,
            "confidence": "high",
            "verification_risk": estimate_verification_risk(
                f"{name} {_early['name']}", qualifier),
            "evidence_tier": "T1_CERTIFIER_RULING",
            "source": _early.get("citation", "SANHA E-Numbers list"),
            "reason": (
                f"{_early['name']} ({_early['e_number']}) is listed on SANHA's "
                f"{st.title()} E-Numbers page"
                + (f" ({_early['source_description']})"
                   if _early.get("source_description") else "")
                + "."
            ),
        }

    if _early is None and not is_exception:
        # 0. Explicit negation and plant-based substitutes are checked
        #    BEFORE any haram lookup. A label that says "pork-free" or
        #    "vegetarian" is asserting the opposite of the trigger word,
        #    and firing on the trigger anyway is a false positive.
        is_substitute = any(_contains_term(combined_text, m)
                            for m in SUBSTITUTE_MARKERS)
        if is_substitute:
            return {
                "status": "halal",
                "confidence": "medium",
                "verification_risk": risk,
                "evidence_tier": "T3_PRINCIPLE",
                "source": "Plant-based substitute; named after the product it replaces",
                "reason": (
                    "Labelled as a vegetarian/vegan/imitation substitute, so the "
                    "meat or alcohol term in the name refers to the flavour "
                    "profile, not the contents."
                ),
            }

        # "Beef bacon" and "turkey salami" are animal but not swine --
        # the slaughter question applies, so syubhah rather than haram.
        for meat in NON_PORK_MEAT_PREFIXES:
            if _contains_term(combined_text, meat):
                for cured in ("bacon", "salami", "pepperoni", "ham",
                              "chorizo", "frankfurter", "pancetta"):
                    if _contains_term(combined_text, cured):
                        return {
                            "status": "syubhah",
                            "confidence": "low",
                            "verification_risk": risk,
                            "evidence_tier": "T4_JUDGMENT",
                            "source": "Non-swine cured meat",
                            "reason": (
                                f"'{cured}' is normally swine, but '{meat}' is "
                                f"stated as the source. Not haram on that basis; "
                                f"remains syubhah because the slaughter method is "
                                f"unconfirmed."
                            ),
                        }

        hit = lookup_reference(combined_text)
        # A negated term must not produce its own verdict.
        if hit and hit.get("verdict") == "haram" and _is_negated(combined_text, hit["term"]):
            hit = None
        if hit:
            tier = hit.get("evidence_tier", "T4_JUDGMENT")
            # Confidence is derived from evidence tier, not asserted: a
            # certifier ruling and an engineering guess must not both
            # read as "high".
            conf = {"T1_CERTIFIER_RULING": "high", "T2_SCRIPTURAL": "high",
                    "T3_PRINCIPLE": "medium", "T4_JUDGMENT": "low"}.get(tier, "low")
            return {
                "status": hit["verdict"],
                "confidence": conf,
                "verification_risk": risk,
                "evidence_tier": tier,
                "source": hit.get("source", ""),
                "reason": (
                    f"Matched reference term '{hit['term']}' -> {hit['verdict']}. "
                    f"Basis [{tier}]: {hit.get('source','')}"
                    + (f" NOTE: {hit['notes']}" if hit.get("notes") else "")
                ),
            }

        for kw in sorted(DEFINITE_HARAM_KEYWORDS):
            if _contains_term(combined_text, kw) and not _is_negated(combined_text, kw):
                return {
                    "status": "haram",
                    "confidence": "high",
                    "verification_risk": risk,
                    "reason": f"Contains a definite-haram trigger term ('{kw}').",
                }

        # Traditionally swine/alcohol based, but halal variants exist --
        # doubtful rather than prohibited.
        for kw in sorted(AMBIGUOUS_SOURCE_TERMS):
            if _contains_term(combined_text, kw):
                return {
                    "status": "syubhah",
                    "confidence": "low",
                    "verification_risk": risk,
                    "reason": (
                        f"'{kw}' is traditionally swine- or alcohol-based, but "
                        f"halal variants exist (e.g. beef salami, non-alcoholic "
                        f"mirin). Source must be confirmed; not haram outright."
                    ),
                }
    # Guarded by _early: if SANHA has published a ruling on this named
    # substance, a generic halal keyword must not override it. "Amidated
    # Pectin" (SANHA: mashbooh) contains 'pectin'; "Glycine & its Sodium
    # Salt" (SANHA: mashbooh) contains 'salt'. Both were returning halal.
    if _early is None:
        for kw in sorted(DEFINITE_HALAL_KEYWORDS):
            if _contains_term(combined_text, kw):
                return {
                    "status": "halal",
                    "confidence": "high",
                    "verification_risk": risk,
                    "reason": f"Matches a definite-halal reference term ('{kw}').",
                }

    # 2. Look up against the SANHA table (all three categories), by code or name.
    e_key = _normalize_e_number(name_lower)
    entry = e_table.get(e_key) if e_key else e_table.get(name_lower)

    if entry is None:
        # Not in any reference list. Which verdict this produces depends on
        # the configured policy -- see UNKNOWN_POLICIES. Under either policy
        # this is "no evidence", never "evidence of prohibition".
        if unknown_policy == "muis":
            return {
                "status": "cannot_be_certified",
                "confidence": "none",
                "verification_risk": risk,
                "reason": (
                    "Not found in any reference list. Under MUIS documentation "
                    "policy, an ingredient not supported by the relevant "
                    "documents is not certifiable. This concerns certifiability, "
                    "not permissibility -- the ingredient is not being called "
                    "haram."
                ),
                "disclaimer": CANNOT_CERTIFY_DISCLAIMER,
            }
        return {
            "status": "syubhah",
            "confidence": "none",
            "verification_risk": risk,
            "reason": (
                "Not found in any reference list (SANHA table or definite "
                "halal/haram terms). Unverifiable status is treated as syubhah "
                "rather than guessed. Note this is 'no evidence', not 'evidence "
                "of doubt' -- confidence is 'none' for exactly this reason."
            ),
        }

    # Recompute risk now that the E-number has resolved to a real ingredient
    # name (e.g. "E441" -> "Gelatin") -- otherwise a code lookup would miss
    # keyword matches that the plain name would have caught.
    risk = estimate_verification_risk(f"{name} {entry['name']}", qualifier)

    # 3. The table now spans all three SANHA category pages, so an entry's
    #    own status decides how it is handled. Only rows SANHA itself files
    #    as doubtful go through qualifier resolution below; rows SANHA
    #    states outright are returned as stated.
    entry_status = entry.get("status", "syubhah").strip().lower()

    if entry_status == "halal":
        return {
            "status": "halal",
            "confidence": "high",
            "verification_risk": risk,
            "reason": (
                f"{entry['name']} ({entry['e_number']}) is listed on SANHA's Halaal "
                f"E-Numbers page"
                + (f" ({entry['source_description']})" if entry["source_description"] else "")
                + "."
            ),
        }

    if entry_status == "haram":
        return {
            "status": "haram",
            "confidence": "high",
            "verification_risk": risk,
            "reason": (
                f"{entry['name']} ({entry['e_number']}) is listed on SANHA's Haraam "
                f"E-Numbers page"
                + (f" ({entry['source_description']})" if entry["source_description"] else "")
                + "."
            ),
        }

    # 3b. Entry is one SANHA files as doubtful -- apply qualifier detection to
    #    try to resolve it beyond a flat "doubtful" answer, but only when
    #    the qualifier actually resolves the specific doubt, not just when
    #    it adds detail.
    if any(w in qualifier_lower for w in HARAM_SOURCE_WORDS):
        return {
            "status": "haram",
            "confidence": "medium",
            "verification_risk": risk,
            "reason": (
                f"{entry['name']} ({entry['e_number']}) is source-dependent per SANHA"
                f"{_sanha_desc(entry)}, and the stated qualifier indicates "
                f"a haram source."
            ),
        }
    if any(w in qualifier_lower for w in RESOLVES_DOUBT_WORDS):
        return {
            "status": "halal",
            "confidence": "medium",
            "verification_risk": risk,
            "reason": (
                f"{entry['name']} ({entry['e_number']}) is source-dependent per SANHA"
                f"{_sanha_desc(entry)}. The stated qualifier ('{qualifier}') "
                f"either rules out an animal source entirely or is an explicit "
                f"certification claim, which directly resolves the doubt SANHA flags "
                f"-- not just adds detail without answering it."
            ),
        }
    if any(w in qualifier_lower for w in ANIMAL_SOURCED_UNCONFIRMED_WORDS):
        return {
            "status": "syubhah",
            "confidence": "low",
            "verification_risk": risk,
            "reason": (
                f"{entry['name']} ({entry['e_number']}) is source-dependent per SANHA"
                f"{_sanha_desc(entry)}. The stated qualifier ('{qualifier}') "
                f"confirms an animal source but NOT the slaughter method, which is the "
                f"actual doubt SANHA flags -- naming the animal alone doesn't resolve it, "
                f"so this remains syubhah rather than likely halal."
            ),
        }

    # 4. No qualifier information available -- fall back to the honest
    #    syubhah verdict rather than guessing, per the source-over-code
    #    principle (JHEAINS FAQ; see module docstring on JAKIM attribution).
    return {
        "status": "syubhah",
        "confidence": "low",
        "verification_risk": risk,
        "reason": (
            f"{entry['name']} ({entry['e_number']}) is source-dependent per SANHA"
            f"{_sanha_desc(entry)}. No source qualifier was present in "
            f"the ingredient text to resolve this further."
        ),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HalalGuard rule-based classifier.")
    ap.add_argument("--unknown-policy", choices=UNKNOWN_POLICIES, default="syubhah",
                    help="Fallback for ingredients not found in any reference "
                         "list. 'syubhah' (default) keeps to the three Islamic "
                         "categories; 'muis' applies MUIS documentation policy "
                         "and returns cannot_be_certified.")
    ap.add_argument("--table", default="data/sanha_e_numbers_all.csv")
    args, _ = ap.parse_known_args()

    e_table = load_e_number_table(args.table)

    test_cases = [
        # E-number driven
        ("Gelatin", "Bovine"),
        ("Gelatin", "Pork"),
        ("Gelatin", None),
        ("E471", None),
        ("E471", "Plant-derived"),
        ("E441", "Halal-certified"),
        # Name-driven -- products that list ingredients without E-numbers
        ("Water", None),
        ("Bacon", None),
        ("Prosciutto", None),
        ("Cooking Wine", None),
        ("Wheat Flour", None),
        ("Palm Oil", None),
        ("Xanthan Gum", None),
        # No match anywhere -- exercises the policy switch
        ("Natural Flavour", None),
        ("Whey Powder", None),
    ]

    print(f"[unknown-policy = {args.unknown_policy}]\n")
    shown = False
    for name, qualifier in test_cases:
        result = classify_ingredient(name, qualifier, e_table,
                                      unknown_policy=args.unknown_policy)
        label = f"{name} ({qualifier})" if qualifier else name
        print(f"{label:28s} -> {result['status']:20s} [{result['confidence']:6s}] "
              f"risk={result['verification_risk']:11s} {result['reason'][:70]}")
        if "disclaimer" in result and not shown:
            shown = True

    if shown:
        print(f"\n{CANNOT_CERTIFY_DISCLAIMER}")
