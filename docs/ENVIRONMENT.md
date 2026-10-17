# Tested environment

These are the versions the project is known to work with. They come from the
Streamlit Community Cloud build log of the deploy on 2026-10-16 and from local
test runs. Where a version could not be pinned, it says so.

| Component | Version | Where it comes from |
|---|---|---|
| Python | 3.14.8 | Streamlit Cloud build log (local test runs used 3.14.6) |
| streamlit | 1.66.0 | pinned in `requirements.txt` |
| torch | 2.14.1 | pinned in `requirements.txt` |
| transformers | 5.20.1 | pinned in `requirements.txt` |
| pillow | 12.3.0 | pinned in `requirements.txt` |
| pytesseract | 0.3.13 | pinned in `requirements.txt` |
| Tesseract OCR | 5.5.0 (Debian trixie package `5.5.0-1+b1`) | installed from `packages.txt` with apt; **not pinned**, so a later deploy may get a newer build |
| Tesseract language data | `tesseract-ocr-eng` 4.1.0-2 | installed as a dependency of the Tesseract package |

## Notes

- Only direct dependencies are pinned. Their dependencies (numpy, pandas,
  huggingface-hub and so on) are resolved by `uv` at deploy time. For a fully
  frozen environment, run `pip freeze` in a working environment and use that
  as a constraints file.
- Streamlit Cloud replaces `pyarrow` 26.0.0 with a 24.x release at build time
  because of a known crash in the newer one. That replacement is done by the
  platform, not by this repository.
- The NER model is not in this repository. The deployed app downloads it from
  Hugging Face Hub (`Saadat57/foodstatus-ingredient-ner`, set by the
  `MODEL_DIR` setting). The revision is pinned to commit
  `6182865bc94cc3ed4f53f2159d8c349a8337efae` (upload of 2026-08-30) in
  `src/app.py`. A `MODEL_REVISION` secret overrides it. A local model folder is
  never given a revision.
- To run the tests locally from the repository root:
  `pip install -r requirements-dev.txt` then `python -m pytest`.
