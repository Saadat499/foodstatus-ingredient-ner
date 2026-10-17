# ScanHalal

Hi everyone! I am Saadat here : - )  and it's my project named ScanHalal, so basically what it does is you point a camera at an ingredient label, or upload a photo of one, and get an
instant **halal / haram / syubhah** verdict with the actual reasoning
behind every decision and not a black-box yes/no.

Here's the link for live demo:
**Live demo:** https://foodstatus-ingredient-ner-jiqrb4emmsyjcvs6x95xvx.streamlit.app

> **Disclaimer:** This is a personal/academic portfolio project, not an
> official halal certification tool. It should not be relied on as a
> substitute for recognized certification bodies (e.g. JAKIM) for any
> real dietary or religious decision.

## What my project does at a glance

| | |
|---|---|
| **Pipeline** | Photo-quality gate → Tesseract OCR → exact-match cleanup → rule engine + fine-tuned BERT NER → most-severe-verdict aggregation → Streamlit UI |
| **NER model** | `bert-base-uncased` fine-tuned on FINER (181,970 sentences, 5 entity types). **ING entity F1 0.9669** on the held-out test set, vs 0.9536 for a BiLSTM-CRF baseline trained on identical splits |
| **Classifier** | 165-row evidence-tiered reference table + 419-entry SANHA E-number table + keyword rules. Every verdict carries its evidence tier and source |
| **Real-label coverage** | On 60 real English product ingredient lists from Open Food Facts webstie, 556 of 853 ingredients (65.18%) were resolved with actual evidence; the rest were never guessed, they defaulted to "needs review" |
| **Design stance** | my method was conservative, meaning unknown or doubtful input never becomes "halal" or "haram" |

---

## Screenshots

| Home | Result with reasoning |
|---|---|
| ![Home screen](images/ui-home.png) | ![Result screen showing per-ingredient verdicts](images/ui-result.png) |

| Photo-quality rejection |
|---|
| ![Warning shown when label text is too small to read](images/ui-quality-rejection.png) |

Each ingredient gets its own verdict and an expandable **Detail** with the reason. Blurry or too-small photos are rejected before OCR instead of producing a wrong answer.

## Table of contents

- [What it does](#what-it-does)
- [How it works — the full pipeline](#how-it-works--the-full-pipeline)
- [Try it without a model or a browser](#try-it-without-a-model-or-a-browser)
- [Results](#results)
- [Tech stack](#tech-stack)
- [Engineering notes](#engineering-notes)
- [Data and sources](#data-and-sources)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)
- [Running it locally](#running-it-locally)
- [Deployment](#deployment)
- [Project background](#project-background)

---

## What it does

1. A user such as me or you can scan or upload a photo of an ingredient label
2. A photo-quality check rejects blurry photos, or ones where the text is
   too small to read reliably, *before* wasting time on OCR
3. OCR reads the raw text off the photo and automatically checks multiple
   candidate orientations (in case the photo is sideways or upside down)
   and keeping whichever produces the highest real OCR confidence, rather
   than trusting a single guess
4. A cleanup layer strips packaging boilerplate such as nutrition tables,
   addresses, "best before" dates, and storage instructions and corrects OCR
   noise, but only when a token matches a known ingredient term *exactly*
   after undoing a small set of well-documented OCR character confusions.
   It does not guess at "close enough" matches (see *Engineering notes*
   below for why)
5. A rule-based classifier, assisted by a fine-tuned BERT NER model on
   harder segments, identifies each ingredient and classifies it as halal,
   haram, or syubhah (doubtful/needs review). Rulings come from SANHA's
   published E-number lists, scriptural references, and the *al-asl fi
   al-ashya' al-ibahah* principle (original permissibility) for plant and
   mineral ingredients. See [Data and sources](#data-and-sources)
6. The most severe verdict across all ingredients determines the overall
   product verdict which is a conservative, "when in doubt, don't assume safe"
   aggregation
7. Results are shown as a clear verdict with a plain-language reason for
   each ingredient. Full jurisprudential detail (evidence tier, source
   citation) is available per ingredient on request, not hidden by
   default, just not shown all at once. If an ingredient couldn't be read
   clearly, the app says so directly and suggests a clearer retake,
   rather than silently guessing at what it might say

## How it works — the full pipeline

```
   Photo
     │
     ▼
[1] Photo quality check ──── rejects blurry / too-small-text photos
     │                         before OCR even runs
     ▼
[2] OCR (Tesseract) ───────── tries multiple orientations, keeps
     │                         whichever gives the highest real
     │                         confidence score
     ▼
[3] Cleanup (ocr_bridge.py) ─ strips boilerplate (nutrition tables,
     │                         addresses, storage instructions),
     │                         corrects OCR noise (exact-match only)
     ▼
[4] Classification ────────── rules segment the list; BERT NER is
     │  (pipeline.py +          invoked on "hard" segments (e.g. long
     │   classifier.py)         compound names, "may contain" clauses);
     │                          a rule engine classifies each ingredient
     │                          against a tiered reference table (T1-T4
     │                          evidence provenance) plus an E-number
     │                          lookup table
     ▼
[5] Aggregation ────────────── most severe verdict across all
     │                          ingredients wins (conservative default)
     ▼
   Verdict + per-ingredient reasoning, shown in the UI
```

Each stage is independently testable — `ocr_bridge.py` and
`pipeline.py` both have their own CLI entry points, so any stage can be
run and inspected in isolation without going through a browser.

**Verdict precedence.** Per product: `haram` > `syubhah` > `halal`. An
ingredient that matches nothing in any reference list is reported as
`syubhah` with confidence `none` ("unrecognized, needs review"), so it can
raise a product from halal to syubhah but can never make a product look
safer than it is.

## Try it without a model or a browser

The rule engine runs with the Python standard library only. From the
repository root:

```bash
python src/pipeline.py --no-ner --text "Water, Sugar, Gelatin (Bovine), Vanilla, Lard"
```

```
INPUT: Water, Sugar, Gelatin (Bovine), Vanilla, Lard
------------------------------------------------------------------------------
  Water                              halal                [medium] via rules
  Sugar                              halal                [medium] via rules
  Gelatin (Bovine)                   syubhah              [low   ] via rules
  Vanilla                            syubhah              [none  ] via rules
  Lard                               haram                [high  ] via rules
------------------------------------------------------------------------------
  PRODUCT VERDICT: HARAM
  (conservative aggregation: the most severe ingredient verdict wins)
```

Note the three kinds of answer: `Gelatin (Bovine)` is doubtful *with*
evidence (confidence `low`: SANHA files it as source-dependent, and naming
the animal doesn't confirm the slaughter method), `Vanilla` is simply not
in any reference list (confidence `none`), and `Lard` is a definite
prohibition (confidence `high`).

Run commands from the repository root: the data paths are relative.

## Results

All numbers below are copied from the files committed in `data/`, and can be
regenerated with the scripts in `tools/` (the trained model weights are
hosted on Hugging Face and are not in this repository).

### NER model: BERT vs BiLSTM-CRF baseline

Held-out test split (18,197 sentences), entity-level scores (seqeval).
Both models use identical splits, label set and metric, so the only
difference is the architecture.

| Entity | Support | BERT F1 | BiLSTM-CRF F1 |
|---|---:|---:|---:|
| **ING** (feeds the classifier) | 21,104 | **0.9669** | 0.9536 |
| PRODUCT | 1,749 | 0.9280 | 0.9085 |
| QUANTITY | 21,022 | 0.9992 | 0.9989 |
| STATE | 13,459 | 0.9636 | 0.9492 |
| UNIT | 17,585 | 0.9983 | 0.9973 |
| **micro avg** | 74,919 | **0.9818** | 0.9747 |

The BiLSTM-CRF has 883,994 parameters against BERT-base's roughly 110
million, and trained in about 21 minutes on a GPU. BERT's ING gain is
modest (about 1.3 F1 points). Raw outputs:
[`per_entity_results.csv`](data/per_entity_results.csv),
[`bilstm_crf_per_entity.csv`](data/bilstm_crf_per_entity.csv),
[`training_curve.csv`](data/training_curve.csv), and the terminal captures
[`images/BERT_results.png`](images/BERT_results.png) and
[`images/BiLSTM_RESULTS.png`](images/BiLSTM_RESULTS.png).

**Caveats.** The split is a random sentence-level split with a fixed seed.
It is not deduplicated: 167 of the 18,197 test sentences (0.9%) also appear
verbatim in the training set. FINER is recipe text scraped from Allrecipes
(not product labels), which is why the real-label test below matters.

### Real product labels (domain-gap test)

60 English-language ingredient lists from Open Food Facts, run through the
full pipeline (`tools/get_open_food_facts.py`, `tools/analyse_off_results.py`):

| | Count |
|---|---:|
| Ingredients extracted | 853 |
| Resolved with evidence | 556 (65.18%) |
| Fell back to a default (`syubhah`, confidence `none`) | 297 |
| Resolved by rules / by NER / NER found no span | 496 / 350 / 7 |

At the product level, 58 of the 60 labels came out `syubhah` (needs
review), 2 came out `halal`, and none came out `haram`. This follows from the
aggregation rule (the most severe ingredient verdict wins): one ingredient
the system cannot resolve is enough to make the whole product `syubhah`.

There is no gold annotation for these labels, so this measures
**coverage**, not accuracy. The most frequent unrecognized items were
generic terms such as *emulsifier* (13), *flavourings* (7), *raising
agents* (6), *natural flavor* (5), *vanillin* (5) and *soy lecithin* (5):
see [`off_unrecognised.csv`](data/off_unrecognised.csv). They are shown as
"needs review", never silently classified.

### Classifier held-out test (no circularity)

Testing the classifier against the SANHA table it reads would only prove a
lookup returns what was looked up. Instead, `tools/evaluate_classifier_holdout.py`
hides 20% of the 419 SANHA entries (83, seed 42), removes them from the
table, and asks whether the classifier still reaches SANHA's verdict
from keywords, the reference file and qualifier logic alone:

| Outcome | Count | Meaning |
|---|---:|---|
| Correct | 16 | matched SANHA's verdict |
| Abstained | 66 | returned `syubhah` with no evidence (no claim made) |
| Wrong | 1 | asserted a verdict that contradicts SANHA |

The one wrong answer is listed, with its reasoning string, in
[`classifier_holdout_results.csv`](data/classifier_holdout_results.csv).

### Reading the results

- **The 0.9669 ING F1 is on recipe text.** It shows the model learned the
  FINER task well. It does not say how well the system reads product labels.
  The real-label test is the check on that.
- **65.18% is coverage, not accuracy.** It counts how many extracted
  ingredients were resolved with evidence instead of falling back to
  `syubhah`. Nothing in this repository checks whether those 556 verdicts are
  correct for the real labels, because there is no gold annotation for them.
- **The product-level result is conservative to the point of limited use on
  real products.** With 297 of 853 ingredients unresolved, 58 of 60 products
  could not be cleared. The system's value today is what it refuses to claim,
  not how many products it clears.
- **The held-out test is small and one-sided.** It ran on 83 entries and the
  classifier decided only 17 of them (16 correct, 1 wrong). It shows the
  classifier abstains instead of guessing. It does not show how accurate it is
  across the whole table.
- **No claim above covers photographs.** Every number above starts from typed
  or API-sourced text, so none of them include the photo-quality check or
  OCR errors. A small photo pilot is reported separately below.

### Raw outputs

Every number in this section comes from one of these files. The last column
says what is needed to regenerate it. Run commands from the repository root.

| File | What it holds | Regenerate with | Needs |
|---|---|---|---|
| [`data/per_entity_results.csv`](data/per_entity_results.csv) | BERT per-entity precision, recall, F1 | `python tools/evaluate_ner.py` | torch, transformers, seqeval and the trained model in `models/bert-ner` |
| [`data/bilstm_crf_per_entity.csv`](data/bilstm_crf_per_entity.csv) | BiLSTM-CRF per-entity scores | `src/train_bilstm_crf.py` (retrains) | torch, a GPU is recommended; **committed result, not re-run as part of this check** |
| [`data/training_curve.csv`](data/training_curve.csv) | BERT validation curve | `python tools/training_curve.py` | the original training checkpoints (`trainer_state.json`), which are not in this repository |
| [`data/off_sample.txt`](data/off_sample.txt), [`off_sample_meta.jsonl`](data/off_sample_meta.jsonl) | The 60 Open Food Facts ingredient lists | `python tools/get_open_food_facts.py` | network access; the live API may return different products today |
| [`data/off_results.json`](data/off_results.json) | Full pipeline output for those 60 labels | `python src/pipeline.py --file data/off_sample.txt --json data/off_results.json` | torch, transformers and the model. **Without them the run is rules-only and overwrites this file with different numbers** |
| [`data/off_coverage_summary.csv`](data/off_coverage_summary.csv), [`off_unrecognised.csv`](data/off_unrecognised.csv) | Coverage figures and the unrecognised-ingredient ranking | `python tools/analyse_off_results.py` | the standard library only |
| [`data/classifier_holdout_results.csv`](data/classifier_holdout_results.csv) | The 83 held-out SANHA entries and the outcome for each | `python tools/evaluate_classifier_holdout.py` | the standard library only |

Re-running a script can rewrite a file with different line endings, which
shows up as a whole-file change in `git diff` even when the numbers are
identical.

### Photo pilot (small, not an accuracy measurement)

A full end-to-end accuracy evaluation on *photographed* labels has not been
done. That is the main evidence gap in this project. What exists is a small
pilot, run with [`tools/eval_photos.py`](tools/eval_photos.py), which scores
the quality gate, OCR cleanup character error rate (CER) and ingredient recall
per photo against hand-typed ground truth. It needs a folder of photos and a
CSV of that ground truth (template: `results/photo_metadata_TEMPLATE.csv`),
and no model.

**What the pilot is.** 8 photos of 7 distinct packaged products from the
author's own kitchen, one or two shots each (the chili sauce was photographed
twice). All 8 ground-truth ingredient lists were typed from the images, English
text only, and checked against the physical packs by the author. The photos
were not selected at random: they are the packaged items that were to hand
and had a legible English list. The photos themselves are third-party
packaging and are not included in the repository.

| Product | Ingredients found exactly | Found, allowing small spelling errors |
|---|---|---|
| Biscuit box | 40% | 50% |
| Olive oil | 0% | 0% |
| Grated parmesan | 100% | 100% |
| Processed cheese slices | 12.5% | 12.5% |
| Chili sauce, shot 1 | 0% | 33% |
| Chili sauce, shot 2 | 33% | 56% |
| Garlic sauce | 17% | 17% |
| Orange juice | 0% | 0% |
| **Mean over the 8 photos** | **25%** | **34%** |

"Found exactly" is the share of the true ingredients (as typed from the pack)
that appear, with the same normalised name, in the ingredients the pipeline
read from the photo. The last column also counts a close spelling (difflib
similarity of 0.85 or more) as a match. These rows were scored with
`--ocr_rejected`, meaning OCR also ran on photos that the quality gate
rejected. `results/photo_eval_results.csv` also holds a character error rate
for each photo; it is left out of this table because it is harder to
interpret.

**Quality gate.** It passed 7 of the 8 photos on the author's Windows machine
and rejected the cheese slices as too blurry (blur score 278 against a
threshold of 300). In the environment used to write this table, 6 of 8
passed: the grated parmesan photo has text 15 px high on the author's machine,
exactly at the 15 px limit, and just under it in the other environment. So
the gate result for a photo near a threshold can differ between machines. The
cheese slices were rejected and also scored poorly (12.5%), which fits the
rejection; the parmesan photo scored 100%, so a borderline rejection there
would have turned away a readable photo. Both are single examples, not a
test of the thresholds.

**How to read it.** With 8 photos and one or two shots per product, these
numbers say what happened on this set. They are not an estimate of accuracy on
real-world labels. The spread is the useful part: a flat list printed clearly
(parmesan) was read almost perfectly, while curved or multilingual
packaging gave low recall, because one misspelled word makes that ingredient a
miss. Two shots of the same chili sauce bottle found 0% and 33% of the
ingredients exactly, so results can vary a lot between photos of the same
product (the pilot did not test why).

**Known weakness on photographed labels.** Curved or angled bottles with
labels in several languages gave unreliable OCR: Tesseract sometimes
returned lines out of order and with spelling errors, and the cleanup step,
which discards text before the "Ingredients:" marker, then dropped part of
the list. This was seen in the informal checks that preceded the pilot and is
consistent with its low-recall rows, but the pilot did not isolate the cause
photo by photo.

## Tech stack

| Layer | Tool |
|---|---|
| OCR | Tesseract, via `pytesseract` |
| Image handling | Pillow |
| NLP / NER | `bert-base-uncased` fine-tuned for token classification (Hugging Face `transformers`), hosted on Hugging Face Hub and downloaded automatically at runtime |
| NER baseline | BiLSTM-CRF in PyTorch (`pytorch-crf`), trained on the same splits |
| Classification | Rule-based engine over a tiered ingredient reference table (T1–T4 evidence provenance) plus a SANHA E-number lookup table |
| Interface | Streamlit, with custom CSS for a light, branded theme |
| Model hosting | Hugging Face Hub |
| Code hosting / deployment | GitHub, auto-deployed to Streamlit Community Cloud on every push |

## Engineering notes

A few decisions worth knowing about, since they shaped the final design:

**Word correction is exact-match only, not fuzzy.** An earlier version
used fuzzy string matching to recover more OCR typos automatically.
Testing against real photographed labels (not just synthetic test text)
found a genuinely serious problem: a garbled fragment of unrelated
manufacturer boilerplate ("ARED", from a mangled "PREPARED BY" label)
fuzzy-matched into the scripturally haram term "LARD," producing a false
HARAM verdict for text that was never an ingredient at all. Auditing
every correction made during development against *real* photos (rather
than hand-typed test strings) also showed the fuzzy version's real-world
benefit was much smaller than it first appeared: in those informal
checks it produced one correct fix and one dangerous one. This was a
development audit on a small set of photos, not a formal evaluation.
Given a false HARAM claim is about the most consequential
mistake this system could make, correction was deliberately simplified
to exact-match only: a token is corrected if and only if it matches a
known term precisely, after undoing a specific, well-known set of OCR
character confusions (0/o, 1/l, 8/b, etc.). This structurally cannot
reproduce the false-HARAM failure mode. The trade-off is real: OCR noise
that isn't an exact-match case surfaces as "unrecognized, needs review"
instead of being auto-corrected — the same conservative default the
classifier itself already uses whenever it lacks a certifier ruling.

**Orientation handling compares real OCR confidence, not a single
detector's guess.** An earlier version trusted Tesseract's own
orientation-detection output directly. That worked in initial testing
but caused a real regression on a different Tesseract build/platform,
where the same detector call wrongly flipped an already-correct photo
into unreadable garbage. The current version tries both the original
orientation and the detector's suggested rotation, and keeps whichever
produces higher actual OCR confidence — grounded in what the OCR engine
really produces, not a separate heuristic's guess.

**A specific ruling outranks a generic keyword.** Keyword matching on
substrings of chemical names contradicted the certifier: "Amidated
Pectin" (SANHA: mashbooh) matched the "pectin" halal keyword. The
classifier now consults SANHA's ruling on the exact named substance first.

**Whole-word matching, not substring matching.** A plain substring check
made `e100` (curcumin, halal) match inside `e1000` (cholic acid, an animal
bile acid SANHA files as doubtful), returning a confident false "halal".
Matching now uses word boundaries. The same audit found and fixed corrupted
CSV rows (unquoted commas splitting one field into three); see
[`data/halalguard_source_provenance.md`](data/halalguard_source_provenance.md).

**Every verdict states its own evidence.** Each reference-table term
declares the basis it rests on, and confidence is derived from the tier
rather than asserted:

| Tier | Meaning |
|---|---|
| `T1_CERTIFIER_RULING` | a named body published this exact ruling |
| `T2_SCRIPTURAL` | Quranic or hadith prohibition |
| `T3_PRINCIPLE` | original permissibility; no animal pathway |
| `T4_JUDGMENT` | engineering decision, no external source |

## Data and sources

| Data | What it is | Where |
|---|---|---|
| FINER | Recipe ingredient-line NER corpus (scraped from Allrecipes), 181,970 sentences split 145,576 / 18,197 / 18,197; labels ING, PRODUCT, QUANTITY, STATE, UNIT. CC BY 4.0, by Komariah, Purnomo and Sin ([DOI](https://doi.org/10.6084/m9.figshare.20222361)); converted and re-split here | `data/finer.conll`, `data/{train,val,test}.jsonl` |
| Ingredient reference table | 165 rows, each with `verdict`, `evidence_tier`, `source` and optional `notes` | `data/ingredient_reference.csv` |
| SANHA E-number table | 419 entries parsed from SANHA's published Halaal (354), Mashbooh (64) and Haraam (1) pages | `data/sanha_e_numbers_all.csv` |
| Keyword basis | The basis recorded for each hardcoded keyword, with its verification status | `data/keyword_sources.csv` |
| Open Food Facts sample | 60 real English product ingredient lists, for the domain-gap test | `data/off_sample.txt`, `data/off_results.json` |

SANHA is the source of the E-number rulings. MUIS's Annex A risk tiers feed
a separate `verification_risk` field that never changes the halal / haram /
syubhah verdict. JAKIM's Malaysian Halal Management System 2020 (MHMS 2020) is cited
   in 99 of the 165 reference-table rows, 12 of them as the certifier ruling
   itself. JAKIM publishes no E-number list, so the E-number rulings
   come from SANHA only. A fuller account, including which entries are still
   awaiting an exact citation, is in
[`data/halalguard_source_provenance.md`](data/halalguard_source_provenance.md).

## Known limitations

- **English-language labels only.** Labels in Bahasa Malaysia are not yet
  supported; ingredients read from non-English text will appear as
  "unrecognized, needs review"
- OCR occasionally drops or interleaves words on complex label layouts
  (e.g. a nutrition table or storage-instructions box positioned next to
  the ingredients text) — a structural OCR/layout limitation, not
  something a text-correction layer can fix
- Reference coverage is partial. Real-label coverage was 65.18% in the
  Open Food Facts test above. Generic terms (e.g. *emulsifier*,
  *flavourings*) and some specific ones (e.g. vanilla) are not in the
  reference table and are flagged as unrecognized rather than classified.
  Matching is whole-word, so a plural can miss a singular entry
  ("hazelnuts" is unrecognized while "hazelnut" is not). All of these are
  surfaced as "needs review", never silently misclassified
- Word-level correction is intentionally conservative (exact match
  only, see *Engineering notes*) — ingredients with OCR noise beyond a
  known character confusion will show as "unrecognized"
- The cleanup layer strips known boilerplate patterns only. Other
  manufacturer text (e.g. "PREPARED BY …") can be read as an ingredient
  and show as unrecognized, which raises the product verdict to
  "syubhah" rather than "halal"
- No end-to-end accuracy figure on photographed labels; only a small pilot
  (8 photos, see *Results*)
- The photo-quality thresholds were calibrated on a small number of real
  photos and are starting points, not validated numbers

## Project structure

```
├── src/
│   ├── app.py               Streamlit UI
│   ├── pipeline.py          CLI entry point: text -> classified report
│   ├── classifier.py        rule-based classification engine
│   ├── segmentation.py      ingredient-list segmentation
│   ├── ocr_bridge.py        OCR cleanup + exact-match correction
│   ├── photo_quality.py     blur / text-size pre-checks
│   ├── preprocess.py        FINER .conll -> train/val/test split
│   ├── run_ocr.py           standalone Tesseract OCR script
│   ├── run_easyocr.py       standalone EasyOCR comparison script
│   ├── train_bert.py        NER model training script
│   └── train_bilstm_crf.py  baseline model training script
├── data/                    reference tables, SANHA E-number list, NER
│                            splits, evaluation outputs, provenance notes
├── tools/
│   ├── evaluate_ner.py                 per-entity NER evaluation
│   ├── evaluate_classifier_holdout.py  classifier hold-out test
│   ├── get_open_food_facts.py          real-label sample download
│   ├── analyse_off_results.py          coverage / unrecognized analysis
│   ├── training_curve.py               extract training curve
│   ├── parse_sanha.py                  SANHA page dumps -> CSV
│   └── merge_descriptions.py           merge hand-collected descriptions
├── images/                  terminal captures of the two training runs
├── requirements.txt
├── packages.txt             system-level dependency (Tesseract) for
│                            Streamlit Cloud deployment
└── README.md
```

## Running it locally

```bash
pip install -r requirements.txt
```

The code uses `X | None` type syntax, so it needs Python 3.10 or newer.
Tesseract OCR also needs to be installed separately — it's a system
program, not a Python package:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

Then run, from the repository root:

```bash
streamlit run src/app.py
```

The training and evaluation scripts need extra packages that the app does
not (for example `datasets`, `seqeval` and `pytorch-crf`).

## Deployment

- The trained NER model is hosted on Hugging Face Hub and downloaded
  automatically at runtime — it's too large for a normal GitHub file, so
  it's never committed to this repo
- The app is deployed on Streamlit Community Cloud, connected directly to
  this GitHub repository — every push to `main` triggers an automatic
  redeploy
- `packages.txt` handles the one system-level dependency (Tesseract)
  that Streamlit Cloud needs installed alongside the Python packages in
  `requirements.txt`

## Project background

Originally developed as HalalGuard-NLP, an academic NLP project under
supervision, combining NLP-based ingredient classification with a
jurisprudentially-grounded rule engine. This repository extends that work
into ScanHalal: a computer-vision-enabled personal portfolio project,
adding real-time photo capture, OCR robustness engineering, and a
deployable web interface.

