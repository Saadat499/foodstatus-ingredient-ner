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

import json
import os
import subprocess
import sys
import tempfile

import pytesseract
import streamlit as st
from PIL import Image

from ocr_bridge import clean_ocr_text
from photo_quality import assess_photo_quality

# --- Paths, resolved relative to this file, not the launch directory ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # one level up from src/
PIPELINE_SCRIPT = os.path.join("src", "pipeline.py")  # path AS SEEN from PROJECT_ROOT

# Where to load the NER model from: a local folder for local development,
# or a Hugging Face repo ID when deployed (where the local model folder
# doesn't exist at all -- it's excluded from git via .gitignore).
#
# Deliberately reading this directly via st.secrets and passing it as an
# explicit --model_dir CLI argument below, rather than relying on
# Streamlit Cloud's secrets-to-environment-variable propagation reaching
# a separate subprocess -- that mechanism has a documented history of
# inconsistency (see e.g. streamlit/streamlit#4123), and this makes the
# actual value used fully explicit and controlled by this code instead.
try:
    MODEL_DIR = st.secrets.get("MODEL_DIR", "models/bert-ner")
except Exception:
    MODEL_DIR = "models/bert-ner"

# Only needed on Windows, where Tesseract isn't on PATH on this machine.
# Harmless elsewhere -- pytesseract just uses PATH if this file doesn't exist.
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if sys.platform.startswith("win") and os.path.exists(_WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT


def run_ocr(image: Image.Image) -> str:
    """
    Raw OCR text, with orientation handling: if the photo might be
    upside down or sideways, try both the original and Tesseract's
    suggested rotation, and keep whichever produces higher REAL OCR
    confidence (via image_to_data's per-word confidence scores) --
    not whichever orientation-detection alone guesses.

    This replaced a simpler version that trusted the orientation
    detector's own guess directly. That worked in initial testing but
    caused a real regression: on the user's actual Windows Tesseract
    build, the same detector call that correctly reported "no rotation
    needed" here reported something different there, wrongly flipping an
    already-correct photo into unreadable garbage. Orientation detection's
    own confidence score was low (0.13-0.15) even when it happened to be
    right, so it isn't a reliable signal to filter on either -- comparing
    actual OCR output quality between candidates is more robust because
    it's grounded in what the OCR engine actually produces, not a
    separate heuristic's guess, and isn't sensitive to
    build/platform-specific detector differences the way a single
    trusted guess is.
    """
    candidates = [image]
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)
        if rotation:
            candidates.append(image.rotate(-rotation, expand=True))
    except Exception:
        pass  # Orientation detection can fail on sparse/simple images --
              # just proceeds with only the original orientation as a candidate.

    best_text, best_confidence = "", -1.0
    for candidate in candidates:
        data = pytesseract.image_to_data(candidate, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if str(c) not in ("-1",)]
        avg_confidence = sum(confidences) / len(confidences) if confidences else -1.0
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_text = pytesseract.image_to_string(candidate)
    return best_text


def run_classifier(cleaned_text: str) -> dict:
    """
    Run the real pipeline.py with --json, so we get structured access to
    each ingredient's status AND its full reasoning -- not just a printed
    table. Confirmed against a real --json output the user shared, not
    guessed at: each ingredient item has at least "ingredient", "status",
    "confidence", and "reason"; "evidence_tier" and "qualifier" are present
    on some items but not all, so nothing here assumes they always exist.

    Returns:
        {
            "report": dict or None,   # parsed JSON report, or None if unavailable
            "stdout": str,            # kept for the (still hidden) technical log view
            "stderr": str,
            "returncode": int,
        }
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, PIPELINE_SCRIPT, "--text", cleaned_text,
             "--model_dir", MODEL_DIR, "--json", json_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        report = None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # pipeline.py writes a LIST of reports (one per --text call, or
            # per line of a --file input) -- we always pass one --text, so
            # take the first (and only) report, but don't assume the list
            # is non-empty.
            if isinstance(data, list) and data:
                report = data[0]
        except (FileNotFoundError, json.JSONDecodeError, IndexError):
            # pipeline.py crashed before writing valid JSON, or wrote
            # something unexpected -- report stays None, and the UI falls
            # back to showing the raw stdout instead of a broken display.
            report = None
        return {
            "report": report,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)


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
            "report": None, "classifier_stdout": "", "classifier_stderr": "",
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
            "report": None,
            "classifier_stdout": "",
            "classifier_stderr": "",
            "classifier_returncode": None,
            "no_ingredients_found": True,
            "quality_check_failed": False,
        }

    classifier_result = run_classifier(cleaned_text)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "corrections": corrections,
        "report": classifier_result["report"],
        "classifier_stdout": classifier_result["stdout"],
        "classifier_stderr": classifier_result["stderr"],
        "classifier_returncode": classifier_result["returncode"],
        "no_ingredients_found": False,
        "quality_check_failed": False,
    }


# --- Streamlit UI ---

st.set_page_config(page_title="HalalGuard-NLP", page_icon="\U0001F50E")
st.title("HalalGuard-NLP")
st.caption("Scan or upload a photo of an ingredient label to get a halal/haram/syubhah verdict.")

_STATUS_DISPLAY = {
    "halal": ("HALAL", "\u2705", "success"),
    "syubhah": ("SYUBHAH (needs review)", "\u26A0\uFE0F", "warning"),
    "haram": ("HARAM", "\u274C", "error"),
}

# One short, plain-language summary per (evidence tier, status) pair --
# NOT an automatic text-shortening of the full "reason" field. Keyed on
# BOTH tier and status, not tier alone: real data showed T1_CERTIFIER_RULING
# covers two genuinely different situations (an ingredient flatly banned on
# a certifier's list, vs. one needing certification that can't be verified
# from a photo) -- an earlier version of this mapping keyed on tier alone
# and would have shown the wrong explanation for banned E-numbers like
# E120. Caught by testing against real output before it shipped.
#
# T4 (engineering judgment calls, e.g. "salami") deliberately has no entry
# yet -- no real example has been confirmed, and guessing at its wording
# risks the same kind of mistake T1 just demonstrated.
_TIER_SHORT_SUMMARY = {
    ("T3_PRINCIPLE", "halal"): "No certifier needs to rule on this \u2014 presumed permissible by default (the standard starting point for plant, mineral, or synthetic-origin ingredients) unless shown otherwise.",
    ("T2_SCRIPTURAL", "haram"): "Directly named as prohibited in the Quran \u2014 the highest level of certainty.",
    ("T1_CERTIFIER_RULING", "haram"): "Confirmed on a recognized certifier's list of prohibited ingredients.",
    ("T1_CERTIFIER_RULING", "syubhah"): "Animal-derived \u2014 requires certification of the source, which can't be verified from a photo alone.",
    ("T4_JUDGMENT", "syubhah"): "Traditionally made with pork, but halal versions genuinely exist too \u2014 treated as doubtful, since a label alone can't tell which this is.",
}


def render_verdict(report: dict) -> None:
    """
    Friendly, non-technical display: a big colored banner for the overall
    verdict, a one-line summary of which ingredient(s) drove it, a short
    plain-language summary per ingredient where one is safely available,
    and the full detailed reasoning behind a "Detail" toggle -- not hidden
    as a technical afterthought, just collapsed, since some of these
    explanations are long (Quranic citations, JAKIM references) and would
    overwhelm a quick-glance view if all shown open at once.
    """
    overall = report.get("overall", "syubhah")
    label, icon, box_type = _STATUS_DISPLAY.get(overall, (overall.upper(), "", "warning"))
    ingredients = report.get("ingredients", [])

    box_fn = {"success": st.success, "warning": st.warning, "error": st.error}[box_type]
    box_fn(f"{icon} **{label}**")

    non_halal = [i for i in ingredients if i.get("status") != "halal"]
    if non_halal:
        names = ", ".join(i.get("ingredient", "?") for i in non_halal)
        st.caption(f"Flagged because of: {names}")
    elif ingredients:
        st.caption("All recognized ingredients are halal.")

    st.subheader("Ingredients")
    for item in ingredients:
        name = item.get("ingredient", "?")
        qualifier = item.get("qualifier")
        display_name = f"{name} ({qualifier})" if qualifier else name
        status = item.get("status", "syubhah")
        status_word, item_icon, _ = _STATUS_DISPLAY.get(status, (status.upper(), "", ""))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{item_icon} **{display_name}**")
        with col2:
            st.write(status_word)

        short = _TIER_SHORT_SUMMARY.get((item.get("evidence_tier"), status))
        if short:
            st.caption(short)

        reason = item.get("reason")
        if reason:
            with st.expander("Detail"):
                st.write(reason)

    # "confidence: none" is the classifier's own signal for "not found in
    # any reference list" (confirmed against real output throughout
    # testing) -- a genuinely different situation from "found, but
    # doubtful" (syubhah with a real evidence tier). Since ocr_bridge.py
    # no longer guesses at fixing unclear words (see its module
    # docstring), an unrecognized ingredient is a real signal the photo
    # itself may be the problem, not just an ingredient with no ruling.
    # Deliberately a GENERAL suggestion, not a guess at which specific
    # word was garbled -- naming a specific guess would just be fuzzy
    # matching again, wearing a different hat.
    unrecognized = [i for i in ingredients if i.get("confidence") == "none"]
    if unrecognized:
        st.info(
            "\U0001F4F7 Some ingredients weren't recognized clearly "
            f"({', '.join(i.get('ingredient', '?') for i in unrecognized)}). "
            "This can happen with a blurry, angled, or oddly-cropped photo. "
            "If any of this looks wrong, try retaking the photo closer up, "
            "in better lighting, and as straight-on as possible."
        )


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
    elif result["report"] is not None:
        render_verdict(result["report"])
    else:
        # Fallback only: the structured report wasn't available (e.g.
        # pipeline.py crashed before writing valid JSON) -- show the raw
        # output instead of a broken or empty friendly display.
        st.subheader("Verdict")
        if result["classifier_returncode"] != 0:
            st.error("The classifier reported an error -- see details below.")
        st.code(result["classifier_stdout"] or "(no output)")

    with st.expander("\U0001F527 Show technical details"):
        st.markdown("**OCR corrections made:**")
        if result["corrections"]:
            for c in result["corrections"]:
                st.write(f"`{c['original']}` \u2192 `{c['corrected']}`")
        else:
            st.write("No corrections were needed.")

        st.markdown("**Raw OCR output (before cleanup):**")
        st.text(result["raw_text"] or "(none)")

        st.markdown("**Cleaned text (what was sent to the classifier):**")
        st.text(result["cleaned_text"] or "(none)")

        if result["classifier_stderr"]:
            log_label = ("Classifier error details" if result["classifier_returncode"] not in (0, None)
                         else "Classifier log output (model loading, etc. -- not an error)")
            st.markdown(f"**{log_label}:**")
            st.text(result["classifier_stderr"])
else:
    st.info("Upload a photo to get started.")
