# HalalGuard-NLP — Source Provenance & Transparency Audit

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

`data/sanha_mashbooh_e_numbers_verified.csv`, 65 entries. Every entry carries
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

### JAKIM/JHEAINS → design principle only, no data

**No JAKIM data is in the code.** JHEAINS FAQ item #4 states (Malay):

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

## 5. Flags you must be able to defend

**Tier 2 — my inference, not any source's words.** `RESOLVES_DOUBT_WORDS` vs
`ANIMAL_SOURCED_UNCONFIRMED_WORDS` is the rule that "Bovine" leaves the doubt
open (slaughter method unconfirmed) while "Plant-derived" closes it. This follows
from SANHA's own descriptions and JAKIM's principle, but **no source states this
mapping explicitly.** It is the single largest interpretive step in the code.

**Tier 3 — not individually verified against a fetched page.** These four sit in
`DEFINITE_HALAL_KEYWORDS` without a confirmed SANHA page:
`E100/curcumin`, `E300/ascorbic acid`, `E407/carrageenan`, `E415/xanthan gum`.
Check `sanha.org.za` for each before publication. Also the pork/alcohol terms —
uncontested in Islamic law, but not tied to one fetched document here.

**Two entries with possible residual editorial text.** E471 and E570
`source_description` fields end with "Note: [commercial variant] is stated as
halaal". Plausibly SANHA's own wording (their list does note commercial
variants), but I could not confirm. Verify both directly.

**E1000 has no source description.** Reads "No detailed source description given
in source document" — honest, but it means the reason string is uninformative.

**E120 is not unanimous.** SANHA says haram and the code follows that. Peer-
reviewed literature (Halalsphere, IIUM) documents a Maliki position permitting
it. Your code takes SANHA's position; the paper should acknowledge the dispute.

**Coverage is 65 E-numbers plus ~25 keywords.** Most real ingredient names
return syubhah with `confidence: none` — correct behaviour, but the practical
consequence is that most inputs get a low-information answer.

**SANHA covers only the doubtful category.** There is no SANHA halal or haram
list in the CSV, so the halal/haram verdicts rest on far thinner evidence than
the syubhah ones.

**No accuracy measurement exists.** No labelled evaluation set, so no precision/
recall/F1 can be reported for the classifier. The Kaggle datasets were correctly
rejected as ground truth, which leaves this gap open.

**MUIS risk tiers are not a halal ruling** and shouldn't be presented as one.
An ingredient can be `halal` + `risk=high` — meaning permissible but
documentation-heavy.

**Provenance is not machine-readable.** The tier distinctions in this document
live here, not in the code. Consider adding a `source` field per rule if a
reviewer asks for per-rule attribution.
