# HalalGuard-NLP

Point a camera at an ingredient label and get a halal / haram / syubhah
verdict, backed by a fine-tuned NER model and a rule-based classifier
grounded in JAKIM's halal certification framework.

**Live demo:** _add your Streamlit Cloud link here once deployed_

> **Disclaimer:** This is a personal/academic portfolio project, not an
> official halal certification tool. It should not be relied on as a
> substitute for recognized certification bodies (e.g. JAKIM) for any
> real dietary or religious decision.

---

## What it does

1. Upload a photo of an ingredient label (or use your camera)
2. A photo-quality check rejects blurry or too-small-text photos before
   wasting time on OCR
3. Tesseract OCR reads the raw text off the photo
4. A custom cleanup layer fixes common OCR noise (misread characters,
   dropped commas, boilerplate text mixed in with ingredients) while
   explicitly avoiding overcorrection -- it will not silently guess an
   ingredient into existence
5. A fine-tuned BERT NER model and a rule-based classifier work together
   to identify each ingredient and classify it as halal, haram, or
   syubhah (doubtful/needs review), following JAKIM's dairy-source
   framework and the *al-asl fi al-ashya' al-ibahah* jurisprudential
   principle for plant-derived ingredients
6. The most severe verdict across all ingredients determines the overall
   product verdict (a conservative, "when in doubt, don't assume safe"
   aggregation)

## Tech stack

- **OCR:** Tesseract (via `pytesseract`)
- **NLP:** Fine-tuned BERT for named entity recognition (ingredient
  extraction), hosted on Hugging Face Hub and downloaded automatically at
  runtime
- **Classification:** Rule-based engine using a tiered ingredient
  reference table (T1-T4 evidence provenance) plus an E-number lookup
  table
- **Interface:** Streamlit
- **Spell correction:** `pyspellchecker`, with a British-spelling
  supplement (JAKIM-region labels commonly use British spelling)

## A few technical challenges worth mentioning

This project went through substantial real-world testing against actual
photographed labels, not just clean/synthetic text, which surfaced (and
fixed) several non-obvious bugs:

- **Compounded OCR guesses**: an early version of the correction layer
  could confidently "fix" a garbled word into the *wrong* real word by
  chaining two uncertain guesses together. Fixed by requiring a much
  higher confidence bar for any correction built on an unconfirmed guess.
- **E-number ambiguity**: short alphanumeric codes (E-numbers) are too
  easy to fuzzy-match into a *different*, wrong additive when garbled.
  The system now refuses to guess at these and leaves them for manual
  review rather than risk a confident wrong answer.
- **Real-word protection**: correctly-spelled English words that simply
  aren't in the ingredient vocabulary (e.g. "flavour") were being
  mistaken for typos of unrelated vocabulary words. Fixed with an
  explicit dictionary check -- a word that isn't broken shouldn't be
  "corrected."
- **Photo quality gating**: blur and text-size are measured and checked
  *before* OCR runs, calibrated against real photos rather than assumed
  thresholds.

## Known limitations

- OCR occasionally drops words entirely (not just garbles them) on
  low-quality photos -- this is a fundamental OCR/image-quality
  limitation, not something a text-correction layer can fix
- Some ingredients (e.g. gelatin, vanilla) aren't in the current
  reference table and will be flagged as unrecognized rather than
  classified -- flagged as "needs review," not silently misclassified
- Complex label layouts (nutrition table and ingredients panel side by
  side) can still leak some cross-contamination noise into the parsed
  ingredient text

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
format), combining NLP-based ingredient classification with a
jurisprudentially-grounded rule engine. This repository extends that
work into a computer-vision-enabled personal portfolio project, adding
real-time photo capture, OCR robustness engineering, and a deployable
web interface.
