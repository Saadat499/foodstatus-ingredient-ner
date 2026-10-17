# Data sources and review status

This page records where each dataset in the repository comes from, how far it
has been checked, and what is still open. The counts below were computed from
the files in `data/` on the commit this page was written against. Statements
that come from the author's project notes and have not been re-verified
against the primary source are marked **(per project notes)**. Facts the
author could not supply are stated as "not recorded".

**ScanHalal is a prototype. Its verdicts are a best effort and are not a halal
certification.** No one with religious-scholarly (fiqh) qualification has
reviewed the reference table; see [Review status](#review-status).

## Summary

| Dataset | File(s) | Size | Source | Review status |
|---|---|---|---|---|
| Ingredient reference table | `data/ingredient_reference.csv` | 165 rows | SANHA, JAKIM MHMS 2020, Qur'an, a jurisprudential default, author judgment | Compiled by the author; method confirmed by the academic supervisor; no fiqh review |
| SANHA E-number table | `data/sanha_e_numbers_all.csv` | 419 entries | SANHA's published Halaal / Mashbooh / Haraam pages | Parsed from the pages; 22 entries have an empty `source_description` |
| Keyword basis | `data/keyword_sources.csv` | 84 keywords | Mixed (see below) | 36 of 84 not yet fully cited |
| Open Food Facts sample | `data/off_sample.txt`, `data/off_sample_meta.jsonl`, `data/off_results.json` | 60 products | Open Food Facts | Used for evaluation only |
| NER training data (FINER) | `data/train.jsonl`, `val.jsonl`, `test.jsonl`, `finer.conll` | 145,576 / 18,197 / 18,197 sentences | FINER dataset on Figshare | CC BY 4.0 (per the Figshare page) |
| Real-photo set | none in the repository | n/a | n/a | Not yet collected or evaluated |

## Ingredient reference table

`data/ingredient_reference.csv` has 165 rows. Each row has a `verdict`, an
`evidence_tier`, a `source` and optional `notes`. The tiers keep different
kinds of evidence apart:

| Tier | Rows | What it is |
|---|---|---|
| `T1_CERTIFIER_RULING` | 41 | A ruling taken from a certifier. 29 rows cite SANHA; 12 rows (milk, butter, cream, yoghurt and similar dairy terms) cite JAKIM MHMS 2020 |
| `T2_SCRIPTURAL` | 31 | A substance named directly in the Qur'an (pork, blood, carrion, intoxicants and similar) |
| `T3_PRINCIPLE` | 87 | A plant-based or otherwise neutral ingredient classified as permissible by the default *al-asl fi al-ashya' al-ibahah* (grounded in Al-A'raf 7:157) |
| `T4_JUDGMENT` | 6 | A judgment call where no source and no general principle settled the case |

### What each tier does and does not claim

- **T1** is the only tier backed by a certifier for each individual term. It is
  not a review of the whole table by SANHA or JAKIM.
- **T2** is scriptural, not inference: the text names the substance.
- **T3** is the author's application of a general principle to 87 specific
  ingredients. It is not 87 separately sourced rulings. Each T3 row also
  carries a corroborating citation to JAKIM MHMS 2020, Lampiran J, which
  classifies plant and natural raw materials as not requiring a halal
  certificate. That corroborates the principle; it does not rule on each
  ingredient by name.
- **T4** has no external source. These rows are engineering judgment calls and
  are disclosed as such.

### JAKIM

JAKIM MHMS 2020 (*Sistem Pengurusan Halal Malaysia*) is cited in 99 of the 165
rows: all 87 T3 rows (as corroboration) and 12 T1 rows (as the direct source
for the dairy verdicts). The 12 dairy rows are `syubhah` because MHMS 2020
treats animal-source raw materials as requiring a halal certificate that the
system cannot verify from a label.

JAKIM publishes no E-number list, so all E-number rulings come from SANHA.

### Citation gaps

- Every row has a non-empty `source` value, and no row is currently marked in
  the `review_flag` column.
- T3 rows have a principle-level citation, not an individual one.
- T4 rows have no citation by design.
- The T1 and T2 rows are meant to be individually sourced. The source text is
  present for all of them, but I have not re-checked each quoted source against
  the live SANHA pages or the MHMS 2020 document.
  The date of the last check against the live sources was not recorded.

## SANHA E-number table

`data/sanha_e_numbers_all.csv` has 419 entries parsed from SANHA's published
pages: 354 Halaal, 64 Mashbooh (mapped to `syubhah`) and 1 Haraam.

- SANHA's website blocks automated fetching, so the pages were read in a
  browser, around 25 to 30 August 2026 (author's recollection; the exact
  date was not recorded).
- 22 entries have an empty `source_description`. SANHA's list-page cards cut
  off before the "Source:" field on these entries, and the text exists only on
  each entry's own page. Six of the original 28 were filled from the individual
  SANHA pages (E161(g), E336, E472(a), E472(b), E472(d), E472(e)), copied
  verbatim. E1000 (Cholic Acid) was checked and its page has no "Source:" line,
  so it stays blank. The other 21 empty entries are halal entries whose
  verdict is already settled; `data/missing_descriptions_TEMPLATE.csv` lists
  them and `tools/merge_descriptions.py` merges any that get filled.

## Keyword basis

`data/keyword_sources.csv` records the basis for each of the 84 hardcoded
keywords and its verification status:

| Verification status | Keywords |
|---|---|
| `NEEDS_EXACT_CITATION` | 29 |
| `GENERAL_CONSENSUS` | 17 |
| `MUIS_ANNEX_A` | 12 |
| `SANHA_HALAAL_LIST` | 10 |
| `REVIEW_NEEDED` | 7 |
| `SANHA_HARAAM_LIST` | 4 |
| SANHA single-page or FAQ sources (`SANHA_E904_PAGE` 2, `SANHA_GELATINE_FAQ` 1, `SANHA_E330_PAGE` 1, `SANHA_FAQ` 1) | 5 |

36 of the 84 keywords (`NEEDS_EXACT_CITATION` plus `REVIEW_NEEDED`) still
need an exact citation or a review. MUIS Annex A feeds a separate
`verification_risk` field and never changes the halal / haram / syubhah verdict.

## Open Food Facts sample

- 60 English-language product ingredient lists, used for the out-of-domain
  test reported in the README.
- Collected through the Open Food Facts public API with
  `tools/get_open_food_facts.py`, around 25 to 30 August 2026 (author's
  recollection; the exact date was not recorded). The API returns live data, so
  re-running the script today may return different products.
- Query: products from the United States and United Kingdom, English-language
  ingredient text only, up to 40 products per category page. The script walks
  ten categories in order (breakfast-cereals, snacks, biscuits, breads, sauces,
  canned-foods, chocolates, dairies, beverages, condiments) until it has 60
  samples. The 60 samples in `data/off_sample_meta.jsonl` all came from the
  first two: 37 snacks and 23 breakfast-cereals. The sample therefore covers
  only those two product types, not the wider spread the script is designed to
  collect.
- Licence: Open Database License (ODbL) 1.0, which the author confirmed
  (per project notes). Verify at <https://world.openfoodfacts.org/terms-of-use>.
- Used for evaluation only. It was not used for training.

## NER training data (FINER)

- Source: the FINER dataset on Figshare, DOI `10.6084/m9.figshare.20222361`
  (per project notes, by Komariah, Purnomo and Sin).
- Checked: the sentence count in this repository (145,576 + 18,197 + 18,197 =
  181,970) matches the figure the author recorded for the dataset
  (per project notes).
- Licence: **CC BY 4.0**, as shown on the Figshare record (checked by the
  author on 17 October 2026, version 3, first posted 2022-09-07). CC BY
  requires credit to the creators, a link to the licence, and a note of any
  changes. Credit: Komariah, K. S., Purnomo, A. T. and Sin, B.-K., *FINER: Food
  Ingredient NER Dataset*, Figshare, <https://doi.org/10.6084/m9.figshare.20222361>.
  The files in `data/` (`finer.conll`, `train.jsonl`, `val.jsonl`,
  `test.jsonl`) are a converted and re-split copy of that dataset, so this
  credit applies to them. I believe CC BY allows redistribution on those terms,
  but I am not a lawyer; check the licence text if the repository is reused
  commercially.
- Domain gap: the text is recipe-style ingredient listing (per project notes,
  drawn from Allrecipes.com recipe text), not terse packaged-product label
  text. This gap is why the real-label evaluation on Open Food Facts is
  reported separately.

## Real-photo set

No photo evaluation set is in the repository yet. The photo-quality gate and
the end-to-end photo behaviour are therefore not measured. The README lists
this under "Not yet measured".

## Review status

- **Compiled by:** the author, a student. This covers pulling SANHA's listings,
  identifying scripturally direct terms, and applying the permissibility
  principle to the remaining ingredients.
- **Academic supervisor:** confirmed that applying the
  permissibility principle to the T3 plant-based ingredients and the six T4
  judgment calls is methodologically reasonable. This is an academic sign-off
  on method. It is not a religious ruling and not a certifier review.
- **Not reviewed by:** anyone with fiqh or muftī qualification, or by SANHA,
  JAKIM, MUIS or any other certifier. SANHA and JAKIM are sources for
  individual entries. Neither has endorsed the compiled table.

For a real halal decision, rely on a recognised certification body.
