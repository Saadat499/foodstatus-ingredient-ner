# ScanHalal

Point a camera at an ingredient label and get a halal / haram / syubhah
verdict, backed by a fine-tuned NER model and a rule-based classifier
grounded in JAKIM's halal certification framework.

**Live demo:** https://foodstatus-ingredient-ner-jiqrb4emmsyjcvs6x95xvx.streamlit.app

> **Disclaimer:** This is a personal/academic portfolio project, not an
> official halal certification tool. It should not be relied on as a
> substitute for recognized certification bodies (e.g. JAKIM) for any
> real dietary or religious decision.

---

## What it does

1. Scan or upload a photo of an ingredient label
2. A photo-quality check rejects blurry photos or ones where the text is
   too small to read reliably, *before* wasting time on OCR
3. Tesseract OCR reads the raw text off the photo -- automatically
   comparing multiple candidate orientations (in case the photo was
   captured sideways or upside down) and keeping whichever produces the
   highest real OCR confidence, rather than trusting a single guess
4. A cleanup layer strips packaging boilerplate (nutrition tables,
   addresses, "best before" dates) and corrects OCR noise -- but only
   when a token matches a known ingredient *exactly* after undoing a
   small set of well-documented OCR character confusions (0/o, 1/l,
   8/b, etc.). It does not guess at "close enough" matches; see
   *Engineering notes* below for why
5. A fine-tuned BERT NER model and a rule-based classifier work together
   to identify each ingredient and classify it as halal, haram, or
   syubhah (doubtful/needs review), following JAKIM's dairy-source
   framework and the *al-asl fi al-ashya' al-ibahah* jurisprudential
   principle for plant-derived ingredients
6. The most severe verdict across all ingredients determines the overall
   product verdict (a conservative, "when in doubt, don't assume safe"
   aggregation)
7. Results are shown as a clear pass/fail-style verdict with a
   plain-language reason for each ingredient -- full jurisprudential
   detail (evidence tier, source citation) is available per ingredient
   on request, not hidden, just not shown by default. If an ingredient
   couldn't be read clearly, the app says so directly and suggests a
   clearer retake, rather than silently guessing at what it might say

## Tech stack

- **OCR:** Tesseract (via `pytesseract`)
- **NLP:** Fine-tuned BERT for named entity recognition (ingredient
  extraction), hosted on Hugging Face Hub and downloaded automatically at
  runtime
- **Classification:** Rule-based engine using a tiered ingredient
  reference table (T1-T4 evidence provenance) plus an E-number lookup
  table
- **Interface:** Streamlit

## Engineering notes: why OCR correction is exact-match only

An earlier version of the correction layer used fuzzy string matching to
recover more OCR typos automatically. Testing against real photographed
labels (not just synthetic test text) found real problems with that
approach, including one genuinely serious case: a garbled fragment of
unrelated manufacturer boilerplate ("ARED", from a mangled "PREPARED BY"
label) fuzzy-matched into the scripturally haram term "LARD," producing
a false HARAM verdict for text that was never an ingredient at all.

Auditing every correction made during testing against *real* photos
(rather than hand-typed test strings) also showed the fuzzy version's
real-world benefit was much smaller than it first appeared: across every
real photo tested, fuzzy matching produced exactly one correct fix and
one dangerous one.

Given that a false HARAM claim is about the most consequential mistake
this system could make, and the measured real-world benefit of fuzzy
matching was marginal, the correction layer was deliberately simplified
to exact-match only: a token is corrected if and only if it matches a
known ingredient term precisely, after undoing a specific, well-known set
of OCR character confusions. This structurally cannot reproduce the
false-HARAM failure mode. The trade-off is real: ingredients with OCR
noise that isn't an exact-match case now surface as "unrecognized,
needs review" instead of being auto-corrected -- which is the same
conservative default the classifier itself already uses whenever it
lacks a certifier ruling.

## Known limitations

- OCR occasionally drops or interleaves words entirely on complex label
  layouts (e.g. a nutrition table or storage-instructions box positioned
  next to the ingredients text) -- a structural OCR/layout limitation,
  not something a text-correction layer can fix
- Word-level correction is intentionally conservative (exact match only,
  see above) -- ingredients with OCR noise beyond a known character
  confusion will show as "unrecognized" rather than being auto-corrected
- Some ingredients (e.g. gelatin, vanilla, hazelnuts) aren't in the
  current reference table and will be flagged as unrecognized rather
  than classified -- surfaced as "needs review," never silently
  misclassified
- Orientation handling compares OCR confidence across candidate
  rotations rather than trusting a single detector, specifically because
  Tesseract's orientation detector was found to behave inconsistently
  across platforms during testing

## Running locally

```bash
pip install -r requirements.txt
```

You'll also need Tesseract OCR installed separately (it's a system
program, not a Python package):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

Then run:

```bash
streamlit run src/app.py
```

## Project background

Originally developed as a conference paper (targeting PROCS_ICMLDE
format, under the name HalalGuard-NLP), combining NLP-based ingredient
classification with a jurisprudentially-grounded rule engine. This
repository extends that work into ScanHalal: a computer-vision-enabled
personal portfolio project, adding real-time photo capture, OCR
robustness engineering, and a deployable web interface.
