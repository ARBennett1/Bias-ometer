"""
caption_ocr.py
========================================================================
Local OCR-based caption extraction pipeline.

Provides the first tier in the tiered speaker identification approach:
  1. Crop a source-specific region from a video frame
  2. Pre-screen the crop by checking expected background colour (fast reject)
  3. Run Tesseract OCR to extract text lines
  4. Either return all caption text (ALL_CAPTIONS mode) or filter to frames
     where spaCy NER finds a PERSON entity (NAMES_ONLY mode)

Falls back gracefully when Tesseract or spaCy is unavailable —
caption_from_frame() returns None, which causes the caller to fall through
to the Vision API tier.

Dependencies:
  brew install tesseract
  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np

from sources import CaptionRegion, PreScreen, SourceConfig

log = logging.getLogger(__name__)

# ── Tesseract availability ────────────────────────────────────────────────────

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    TESSERACT_AVAILABLE = False
    log.warning(
        "pytesseract not installed — OCR tier unavailable; "
        "Vision API fallback will be used. "
        "Install with: brew install tesseract && pip install pytesseract"
    )


# ── Capture mode ──────────────────────────────────────────────────────────────

class CaptureMode(str, Enum):
    ALL_CAPTIONS = "all_captions"
    NAMES_ONLY   = "names_only"


# ── Core functions ────────────────────────────────────────────────────────────

def crop_region(frame_bgr: np.ndarray, region: CaptionRegion) -> np.ndarray:
    """
    Crop a normalised CaptionRegion from a BGR frame.

    Returns the sub-array (still BGR), or an empty zero-shape array if the
    resulting crop would be zero-sized.
    """
    h, w = frame_bgr.shape[:2]
    x1 = int(region.x * w)
    y1 = int(region.y * h)
    x2 = int((region.x + region.w) * w)
    y2 = int((region.y + region.h) * h)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame_bgr.dtype)

    return frame_bgr[y1:y2, x1:x2]


def pre_screen_passes(
    crop_bgr: np.ndarray, pre_screen: Optional[PreScreen]
) -> bool:
    """
    Check whether the top-left corner of the crop matches the expected
    background colour.

    Samples a 10×10 block starting at pixel (5, 5) to avoid edge artefacts,
    converts the mean to RGB, and tests each channel against the tolerance.
    Returns True (pass) when pre_screen is None.
    """
    if pre_screen is None:
        return True

    if crop_bgr is None or crop_bgr.size == 0:
        return False

    # Sample with a 5-pixel inset to skip edge compression artefacts
    sample = crop_bgr[5:15, 5:15]
    if sample.size == 0:
        sample = crop_bgr[:10, :10]
    if sample.size == 0:
        # Crop is too small to sample — pass through so OCR can try
        return True

    # Mean pixel in BGR order → convert to RGB for comparison with bg_colour
    mean_bgr = sample.mean(axis=(0, 1))
    mean_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

    bg = pre_screen.bg_colour
    tol = pre_screen.tolerance
    result = all(abs(mean_rgb[i] - bg[i]) <= tol for i in range(3))
    log.debug(
        f"Pre-screen: mean_rgb={tuple(round(v) for v in mean_rgb)} "
        f"bg={bg} tol={tol} → {'pass' if result else 'fail'}"
    )
    return result


def ocr_crop(crop_bgr: np.ndarray) -> list[str]:
    """
    Run Tesseract OCR on a BGR crop.

    Converts to RGB PIL Image, runs pytesseract with --psm 6 (assume a single
    uniform block of text), strips whitespace, and returns non-empty lines.
    """
    from PIL import Image

    # BGR → RGB for PIL
    rgb = crop_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb.astype(np.uint8))

    raw: str = pytesseract.image_to_string(pil_img, config="--psm 6")
    lines = [line.strip() for line in raw.split("\n")]
    return [line for line in lines if line]


def extract_caption_result(
    lines: list[str],
    name_line_index: int,
    mode: CaptureMode,
    nlp,
) -> Optional[dict]:
    """
    Build a caption result dict from OCR lines according to the capture mode.

    Parameters
    ----------
    lines : list[str]
        Non-empty OCR output lines.
    name_line_index : int
        Which line index (from the source config) is expected to hold the name.
    mode : CaptureMode
        ALL_CAPTIONS returns any non-empty result; NAMES_ONLY filters by NER.
    nlp : spacy Language | None
        Loaded spaCy model. If None in NAMES_ONLY mode, returns None.

    Returns
    -------
    dict with keys ``name``, ``raw_lines``, ``method``, or None.
    """
    if not lines:
        return None

    name_line = lines[name_line_index] if len(lines) > name_line_index else lines[0]

    if mode == CaptureMode.ALL_CAPTIONS:
        return {
            "name": name_line,
            "raw_lines": lines,
            "method": "ocr",
        }

    # NAMES_ONLY: require a PERSON entity from spaCy NER
    if nlp is None:
        log.debug("NAMES_ONLY mode but spaCy model unavailable — skipping OCR result")
        return None

    # Prefer the designated name line; fall back to joining all lines if OOB
    candidate = name_line if len(lines) > name_line_index else " ".join(lines)
    doc = nlp(candidate)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    if not persons:
        log.debug(f"NAMES_ONLY: no PERSON entity in {candidate!r}")
        return None

    return {
        "name": persons[0],
        "raw_lines": lines,
        "method": "ocr",
    }


def caption_from_frame(
    frame_bgr: np.ndarray,
    config: SourceConfig,
    mode: CaptureMode,
    nlp,
) -> Optional[dict]:
    """
    Top-level OCR caption extractor. Orchestrates the full pipeline:
      1. crop_region
      2. pre_screen_passes  — return None immediately on failure
      3. ocr_crop           — return None if no text found
      4. extract_caption_result

    Returns a dict ``{name, raw_lines, method}`` or None.
    Returns None immediately if Tesseract is unavailable (triggers Vision tier).
    """
    if not TESSERACT_AVAILABLE:
        return None

    if frame_bgr is None or frame_bgr.size == 0:
        return None

    crop = crop_region(frame_bgr, config.caption_region)
    if crop.size == 0:
        log.debug("caption_from_frame: zero-size crop — skipping")
        return None

    if not pre_screen_passes(crop, config.pre_screen):
        log.debug("caption_from_frame: pre-screen failed — skipping")
        return None

    lines = ocr_crop(crop)
    if not lines:
        log.debug("caption_from_frame: OCR returned no text")
        return None

    return extract_caption_result(lines, config.name_line_index, mode, nlp)
