"""
HalalGuard-NLP -- static-photo demo.

Upload a photo of an ingredient label -> OCR reads it -> ocr_bridge.py
cleans it up -> your real classifier pipeline gives a verdict. This is the
photo-based version from the roadmap (step 6); live video comes after
this works reliably.

Run from anywhere with:
    streamlit run app.py

All paths below are computed relative to THIS file's location, not to
whatever folder you happen to launch the command from -- the same
cwd-dependent-path bug we already found and fixed in ocr_bridge.py would
otherwise resurface here too.
"""

import os
import subprocess
import sys

import pytesseract
import streamlit as st
from PIL import Image

from ocr_bridge import clean_ocr_text
from photo_quality import assess_photo_quality

# --- Paths, resolved relative to this file, not the launch directory ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # one level up from src/
PIPELINE_SCRIPT = os.path.join("src", "pipeline.py")  # path AS SEEN from PROJECT_ROOT

# Only needed on Windows, where Tesseract isn't on PATH on this machine.
# Harmless elsewhere -- pytesseract just uses PATH if this file doesn't exist.
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if sys.platform.startswith("win") and os.path.exists(_WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT


def run_ocr(image: Image.Image) -> str:
    """Raw, unmodified OCR text -- exactly what run_ocr.py produces."""
    return pytesseract.image_to_string(image)


def run_classifier(cleaned_text: str) -> tuple[str, str]:
    """
    Run the real pipeline.py exactly the way it's always been run from the
    command line, and just return its printed output as-is. Deliberately
    NOT parsing pipeline.py's internal JSON schema here -- the CLI's
    human-readable table output is the one interface already confirmed
    working across many real runs, and re-implementing a parser against a
    guessed field structure risks introducing exactly the kind of silent
    mismatch bug this project has already hit more than once.

    Returns (stdout, stderr, returncode). Use returncode == 0 to check
    success -- stderr can be non-empty on a successful run too (e.g. a
    model-loading progress bar), so its mere presence isn't a failure signal.
    """
    result = subprocess.run(
        [sys.executable, PIPELINE_SCRIPT, "--text", cleaned_text],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def process_photo(image: Image.Image) -> dict:
    """
    The whole photo -> verdict flow, kept separate from the Streamlit UI
    below so it can be tested with a plain script (Streamlit's own widgets
    can't be exercised outside a running app).
    """
    quality = assess_photo_quality(image)
    if quality["is_too_blurry"] or quality["is_text_too_small"]:
        return {
            "raw_text": "", "cleaned_text": "", "corrections": [],
            "classifier_stdout": "", "classifier_stderr": "",
            "classifier_returncode": None,
            "no_ingredients_found": False,
            "quality_check_failed": True, "quality_message": quality["message"],
        }

    raw_text = run_ocr(image)
    bridge_result = clean_ocr_text(raw_text)
    cleaned_text = bridge_result["cleaned_text"]
    corrections = bridge_result["corrections"]

    if not cleaned_text.strip():
        # Happens when OCR found little or nothing usable (e.g. the photo
        # mostly captured a nutrition table, which cleanup correctly
        # strips -- but if that's ALL there was, nothing is left to send
        # to the classifier). Fail here with a clear message instead of
        # calling pipeline.py with empty text, which it isn't written to
        # handle gracefully.
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "corrections": corrections,
            "classifier_stdout": "",
            "classifier_stderr": "",
            "classifier_returncode": None,
            "no_ingredients_found": True,
            "quality_check_failed": False,
        }

    stdout, stderr, returncode = run_classifier(cleaned_text)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "corrections": corrections,
        "classifier_stdout": stdout,
        "classifier_stderr": stderr,
        "classifier_returncode": returncode,
        "no_ingredients_found": False,
        "quality_check_failed": False,
    }


# --- Streamlit UI ---

st.set_page_config(page_title="HalalGuard-NLP", page_icon="\U0001F50E")
st.title("HalalGuard-NLP")
st.caption("Scan or upload a photo of an ingredient label to get a halal/haram/syubhah verdict.")

input_mode = st.radio(
    "How do you want to provide the photo?",
    ["Use camera", "Upload a file"],
    horizontal=True,
)

if input_mode == "Use camera":
    # A single snapshot per click, not continuous streaming -- true
    # frame-by-frame live video needs a different, heavier approach
    # (streamlit-webrtc), which is a separate step, not this one.
    captured = st.camera_input("Point your camera at the ingredients list, then click to capture")
    uploaded_file = captured
else:
    uploaded_file = st.file_uploader("Ingredient label photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if input_mode == "Upload a file":
        st.image(image, caption="Uploaded photo", width="stretch")
        # st.camera_input already shows its own captured-photo preview,
        # so showing it again here would just be a redundant duplicate image.

    with st.spinner("Reading the label..."):
        result = process_photo(image)

    if result["quality_check_failed"]:
        st.warning(result["quality_message"])
    elif result["no_ingredients_found"]:
        st.warning(
            "No ingredient text was found in this photo -- it may have mostly "
            "captured the nutrition table, packaging design, or something else "
            "instead of the ingredients list. Try a closer, straight-on shot of "
            "just the ingredients paragraph."
        )
    else:
        st.subheader("Verdict")
        if result["classifier_returncode"] != 0:
            st.error("The classifier reported an error -- see details below.")
        st.code(result["classifier_stdout"] or "(no output)")

    with st.expander("What the OCR bridge fixed"):
        if result["corrections"]:
            for c in result["corrections"]:
                st.write(f"`{c['original']}` \u2192 `{c['corrected']}`")
        else:
            st.write("No corrections were needed.")

    with st.expander("Raw OCR output (before cleanup)"):
        st.text(result["raw_text"])

    with st.expander("Cleaned text (what was sent to the classifier)"):
        st.text(result["cleaned_text"])

    if result["classifier_stderr"]:
        label = ("Classifier error details" if result["classifier_returncode"] != 0
                 else "Classifier log output (model loading, etc. -- not an error)")
        with st.expander(label):
            st.text(result["classifier_stderr"])
else:
    st.info("Upload a photo to get started.")
