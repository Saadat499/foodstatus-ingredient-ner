# HalalGuard-NLP — Source Provenance & Transparency Audit

**Status of this document.** This is a record of a source audit of
`classifier.py`, written when the SANHA table had 65 Mashbooh entries. The
date of the audit was not recorded. This revision (2026-10-17) changes the
wording and flags what is out of date; it does not redo the audit. For the
current datasets, counts and review status, see [`docs/DATA.md`](../docs/DATA.md).
Where this page and `docs/DATA.md` disagree, `docs/DATA.md` is current.

Covers `classifier.py` only. The NER/BERT side uses the FINER dataset and involves
no halal rulings.

**Output vocabulary is strictly three values: `halal`, `haram`, `syubhah`.**
Verified programmatically across 139 test cases (all 65 SANHA entries by code,
all 65 by name, plus 9 edge cases). No fourth status can be produced.

---

## 1. Access links

| Body | URL | Notes |
|---|---|---|
| SANHA — Mashbooh E-numbers list | `https://sanha.org.za/e-numbers/mashbooh-e-numbers/` | Blocks automated fetching; open in a browser |
| SANHA — individual entries | `https://sanha.org.za/2009/12/10/e120-carminic-acid/`, `.../e904-shellac/`, `.../2009/12/15/e330-citric-acid/` | Each has explicit Status + Source fields |
| MUIS — Halal Certification Conditions v4.1 | `https://file.go.gov.sg/hcc-fpa.pdf` | Annex A, p.28. Effective 9 Mar 2026 |
| JAKIM (via JHEAINS, Negeri Sembilan) | `https://jheains.ns.gov.my/help/soalan-lazim` | Malay-language FAQ, item #4 |

---

## 2. What each source contributes

### SANHA → the E-number lookup table AND the definite halal/haram terms

At the time of the audit this was `data/sanha_mashbooh_e_numbers_verified.csv`,
65 entries. The repository now holds `data/sanha_e_numbers_all.csv` with 419
entries across SANHA's Halaal, Mashbooh and Haraam pages (see `docs/DATA.md`),
so the statement below about a Mashbooh-only table describes the earlier
version. In that version every entry carried
`status = syubhah` because SANHA's Mashbooh list *is* the doubtful category by
definition — it contains no halal or haram entries. The `source_description`
field is SANHA's own wording, used verbatim in the classifier's output.

Three individual SANHA pages were fetched and used directly for definite terms:

| Term | SANHA's stated status | In code as |
|---|---|---|
| E120 / carminic acid | Haraam ("From insect Cochineal") | `DEFINITE_HARAM_KEYWORDS` |
| E904 / shellac | Halaal ("resin of lac insect") | `DEFINITE_HALAL_KEYWORDS` |
| E330 / citric acid | Halaal ("microbiological fermentation; citrus fruits") | `DEFINITE_HALAL_KEYWORDS` |

### MUIS → the `verification_risk` field only

MUIS Annex A's four-tier risk classification. **This is a separate axis and never
changes the halal/haram/syubhah verdict.** It measures how much documentation an
ingredient category warrants. Keywords match MUIS's literal Annex A wording:

- **High**: Flavourings, Gelatine, Meat, Poultry, Beef extracts, Beef tallow, Chicken skin, Chicken fat, Canned foods, Confectionery & pastry, Dairy products, Processed seafood, Sauces & condiments, Pure vinegars
- **Medium-High**: Enzymes, Yeast, Cheese & byproducts
- **Medium-Low**: Soy bean products, Flour, Olive oil, Sesame oil, Vegetable oil, Synthetic vinegar, Noodles, Pasta
- **Low**: Vegetables, Fruits, Pure seafood, Legumes & lentils, Rice, Salt & sugar, Ice, Spices, Synthetic chemicals

MUIS distinguishes "Pure seafood" (Low) from "Processed seafood" (High), and
"Synthetic vinegar" (Medium-Low) from "Pure vinegars" (High). The code preserves
both distinctions rather than collapsing them.

### JAKIM/JHEAINS → design principle, plus MHMS 2020 citations in the reference table

This section was written before the reference table cited JAKIM MHMS 2020.
The classifier code contains no JAKIM E-number data. The author's finding is
that JAKIM publishes no E-number list; no dated source for that finding was
recorded, so treat it as unconfirmed. `data/ingredient_reference.csv` cites
MHMS 2020 in 99 of 165 rows (see [`docs/DATA.md`](../docs/DATA.md)). The
JHEAINS FAQ below remains the basis for inspecting the source qualifier.
JHEAINS FAQ item #4 states (Malay):

> "Kod 'e' sebenarnya merujuk kepada aditif makanan... Bahan yang digunakan
> merangkumi dari sumber haiwan, tumbuhan atau kimia. Tidak semestinya kod 'e'
> itu datangnya dari sumber babi semata-mata."

("The 'e' code refers to food additives... spanning animal, plant, or chemical
sources. It doesn't necessarily come from a pork source alone.")

This justifies *why* the classifier inspects the source qualifier rather than
using a flat code lookup. It contributes no rulings.

---

## 3. Confidence levels — what they mean

`confidence` is metadata about evidence strength, not a fourth status:

- **high** — matched a definite term traced to a specific SANHA page
- **medium** — SANHA table hit, plus a qualifier that resolves the doubt
- **low** — SANHA table hit, doubt unresolved
- **none** — no match anywhere; syubhah by default, not by evidence

---

## 4. Bugs found and fixed during this audit

**A. E904/Shellac was wrongly hardcoded as haram.** SANHA's own page states
Halaal. Corrected. This was a factual error, not a judgement call.

**B. Substring collision produced a false halal verdict on a real entry.**
Matching used `term in text`, so `"e100"` (curcumin, halal) matched inside
`"e1000"` — Cholic Acid, an animal bile acid that SANHA lists as syubhah. E1000
was being returned as **halal, high confidence**. Fixed with regex word-boundary
matching (`\b`); E1000 now correctly returns syubhah.

**C. Eight of 65 CSV rows were silently corrupted.** Unquoted commas in
`source_description` split one field into three. On load, descriptions were
truncated at the first comma *and* the `citation` field was overwritten with
stray fragments. Affected E472d, E472e, E472f, E473, E631, E632, E633, E915.
Example — E631 displayed "Meat extract" but should read "Meat extract, dried
sardines, or microbiological fermentation", which changes the meaning
materially: the truncated version implies meat-only origin and hides that a
microbial (non-animal) source is possible. All eight repaired.

**D. Editorial text inside a source-attributed field.** E441's
`source_description` — a field labelled as SANHA's wording — contained
"(This is the example your professor raised directly.)". Removed.

**E. `unknown` status eliminated.** Now resolves to syubhah with
`confidence: none`.

---

## 5. Known limitations and open points (as of the audit)

The items below were open when the audit was written. Items marked
**superseded** have since changed; the rest have not been re-checked.

**Interpretive rule, not any source's words.** `RESOLVES_DOUBT_WORDS` vs
`ANIMAL_SOURCED_UNCONFIRMED_WORDS` encodes the rule that "Bovine" leaves the
doubt open (slaughter method unconfirmed) while "Plant-derived" closes it. This
follows from SANHA's own descriptions and JAKIM's principle, but no source
states this mapping explicitly. It is the single largest interpretive step in
the code.

**Terms not individually verified against a fetched page.** Four terms sat in
`DEFINITE_HALAL_KEYWORDS` without a confirmed SANHA page: `E100/curcumin`,
`E300/ascorbic acid`, `E407/carrageenan`, `E415/xanthan gum`. The pork and
alcohol terms are uncontested in Islamic law but were not tied to one fetched
document. `data/keyword_sources.csv` now records the basis and verification
status of each keyword, and `docs/DATA.md` reports that 36 of 84 keywords still
need an exact citation or review.

**Two entries with possible residual editorial text.** The E471 and E570
`source_description` fields end with "Note: [commercial variant] is stated as
halaal". This is plausibly SANHA's own wording, but it was not confirmed. Both
should be checked against the SANHA pages.

**E1000 has no source description.** It reads "No detailed source description
given in source document", which is honest but makes the reason string
uninformative.

**E120 is disputed.** SANHA says haram and the code follows that. Peer-reviewed
literature (Halalsphere, IIUM) documents a Maliki position permitting it. The
paper should acknowledge the dispute.

**Coverage was limited.** At the time there were 65 E-numbers plus about 25
keywords, so most real ingredient names returned syubhah with
`confidence: none`. That is correct behaviour, but most inputs got a
low-information answer. **Superseded in part:** the reference table now has 165
rows and the SANHA table 419 entries; the README reports coverage on real
labels.

**SANHA covered only the doubtful category (superseded).** The 65-entry table
contained no halal or haram entries. The current 419-entry table includes
SANHA's Halaal and Haraam pages as well.

**No accuracy measurement existed (superseded in part).** There was no labelled
evaluation set for the classifier. The README now reports a held-out test on
83 SANHA entries (16 correct, 66 abstained, 1 wrong) and coverage on 60 real
labels. There is still no gold annotation for real labels, so no accuracy
figure for them.

**MUIS risk tiers are not a halal ruling.** An ingredient can be `halal` with
`risk=high`, meaning permissible but documentation-heavy.

**Provenance was not machine-readable.** The tier distinctions in this document
lived here, not in the code. The reference table now carries an evidence tier
and a source per row, and `data/keyword_sources.csv` records the basis of each
keyword.
