# ScanHalal

Point a camera at an ingredient label, or upload a photo of one, and get an
instant **halal / haram / syubhah** verdict — with the actual reasoning
behind every decision, not a black-box yes/no.

**Live demo:** https://foodstatus-ingredient-ner-jiqrb4emmsyjcvs6x95xvx.streamlit.app

> **Disclaimer:** This is a personal/academic portfolio project, not an
> official halal certification tool. It should not be relied on as a
> substitute for recognized certification bodies (e.g. JAKIM) for any
> real dietary or religious decision.

---

## Table of contents

- [What it does](#what-it-does)
- [How it works — the full pipeline](#how-it-works--the-full-pipeline)
- [Tech stack](#tech-stack)
- [Engineering notes](#engineering-notes)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)
- [Running it locally](#running-it-locally)
- [Deployment](#deployment)
- [Project background](#project-background)

---

## What it does

1. Scan or upload a photo of an ingredient label
2. A photo-quality check rejects blurry photos, or ones where the text is
   too small to read reliably, *before* wasting time on OCR
3. OCR reads the raw text off the photo, automatically checking multiple
   candidate orientations (in case the photo is sideways or upside down)
   and keeping whichever produces the highest real OCR confidence, rather
   than trusting a single guess
4. A cleanup layer strips packaging boilerplate (nutrition tables,
   addresses, "best before" dates, storage instructions) and corrects OCR
   noise — but only when a token matches a known ingredient term *exactly*
   after undoing a small set of well-documented OCR character confusions.
   It does not guess at "close enough" matches (see *Engineering notes*
   below for why)
5. A fine-tuned BERT NER model and a rule-based classifier work together
   to identify each ingredient and classify it as halal, haram, or
   syubhah (doubtful/needs review), following JAKIM's dairy-source
   framework and the *al-asl fi al-ashya' al-ibahah* jurisprudential
   principle for plant-derived ingredients
6. The most severe verdict across all ingredients determines the overall
   product verdict — a conservative, "when in doubt, don't assume safe"
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
[4] Classification ────────── fine-tuned BERT NER extracts ingredient
     │  (pipeline.py +          spans; a rule-based engine classifies
     │   classifier.py)         each one against a tiered reference
     │                          table (T1-T4 evidence provenance) plus
     │                          an E-number lookup table
     ▼
[5] Aggregation ────────────── most severe verdict across all
     │                          ingredients wins (conservative default)
     ▼
   Verdict + per-ingredient reasoning, shown in the UI
```

Each stage is independently testable — `ocr_bridge.py` and
`pipeline.py` both have their own CLI entry points, so any stage can be
run and inspected in isolation without going through a browser.

## Tech stack

| Layer | Tool |
|---|---|
| OCR | Tesseract, via `pytesseract` |
| Image handling | Pillow |
| NLP / NER | Fine-tuned BERT, hosted on Hugging Face Hub, downloaded automatically at runtime |
| Classification | Rule-based engine over a tiered ingredient reference table (T1–T4 evidence provenance) plus an E-number lookup table |
| Interface | Streamlit, with custom CSS for a light, branded theme |
| Model hosting | Hugging Face Hub |
| Code hosting / CI | GitHub, auto-deployed to Streamlit Community Cloud on every push |

## Engineering notes

A few decisions worth knowing about, since they shaped the final design:

**Word correction is exact-match only, not fuzzy.** An earlier version
used fuzzy string matching to recover more OCR typos automatically.
Testing against real photographed labels (not just synthetic test text)
found a genuinely serious problem: a garbled fragment of unrelated
manufacturer boilerplate ("ARED", from a mangled "PREPARED BY" label)
fuzzy-matched into the scripturally haram term "LARD," producing a false
HARAM verdict for text that was never an ingredient at all. Auditing
every correction made during testing against *real* photos (rather than
hand-typed test strings) also showed the fuzzy version's real-world
benefit was much smaller than it first appeared — across every real
photo tested, fuzzy matching produced exactly one correct fix and one
dangerous one. Given a false HARAM claim is about the most consequential
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

## Known limitations

- OCR occasionally drops or interleaves words on complex label layouts
  (e.g. a nutrition table or storage-instructions box positioned next to
  the ingredients text) — a structural OCR/layout limitation, not
  something a text-correction layer can fix
- Some ingredients (e.g. gelatin, vanilla, hazelnuts) aren't in the
  current reference table and are flagged as unrecognized rather than
  classified — surfaced as "needs review," never silently misclassified
- Word-level correction is intentionally conservative (exact match
  only, see *Engineering notes*) — ingredients with OCR noise beyond a
  known character confusion will show as "unrecognized"

## Project structure

```
├── src/
│   ├── app.py               Streamlit UI
│   ├── pipeline.py          CLI entry point: text -> classified report
│   ├── classifier.py        rule-based classification engine
│   ├── segmentation.py      ingredient-list segmentation
│   ├── ocr_bridge.py        OCR cleanup + exact-match correction
│   ├── photo_quality.py     blur / text-size pre-checks
│   ├── run_ocr.py           standalone Tesseract OCR script
│   ├── run_easyocr.py       standalone EasyOCR comparison script
│   ├── train_bert.py        NER model training script
│   └── train_bilstm_crf.py  baseline model training script
├── data/                    ingredient reference table, E-number list,
│                            training/eval data
├── tools/                   evaluation and data-prep scripts
├── requirements.txt
├── packages.txt             system-level dependency (Tesseract) for
│                            Streamlit Cloud deployment
└── README.md
```

## Running it locally

```bash
pip install -r requirements.txt
```

Tesseract OCR also needs to be installed separately — it's a system
program, not a Python package:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

Then run:

```bash
streamlit run src/app.py
```

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

Originally developed as a conference paper (targeting PROCS_ICMLDE
format, under the name HalalGuard-NLP), combining NLP-based ingredient
classification with a jurisprudentially-grounded rule engine, developed
under academic supervision. This repository extends that work into
ScanHalal: a computer-vision-enabled personal portfolio project, adding
real-time photo capture, OCR robustness engineering, and a deployable
web interface.
