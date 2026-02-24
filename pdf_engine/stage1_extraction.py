"""
pdf_engine/stage1_extraction.py
================================
Stage 1 — Text Layer Extraction + Phantom Detection

Responsibility
--------------
For every ``PageIR`` classified as "text_native" or "hybrid" by Stage 0,
this stage:

  1. Extracts the embedded text layer via PyMuPDF's ``rawdict`` (per-char
     bounding boxes — **not** ``"dict"``).
  2. Runs phantom-detection to score how trustworthy that text layer is.
  3. Builds ``TextBlock`` objects for every span that passes the quality bar.
  4. Tags each block with ``script_direction`` (ltr / rtl), ``language``
     (if detectable from metadata or heuristics), and
     ``extraction_method = ExtractionMethod.PYMUPDF_DIRECT``.

Design rules enforced here
--------------------------
* ``rawdict`` is used exclusively — never ``"dict"`` — so per-character
  bounding boxes are always available  (fixes W4).
* ``script_direction`` is detected explicitly from Unicode bidi properties;
  it is never silently defaulted  (design rule 4).
* The PyMuPDF page handle is passed in from the caller; this module never
  re-opens the PDF  (design rule 3).
* All returned objects are typed; no bare dicts cross a stage boundary
  (design rule 5).

Phantom detection — ``detect_phantom_text_layer(page) -> float``
----------------------------------------------------------------
Combines four signals into a quality score ∈ [0.0, 1.0]:

  S1  Zero-width bbox ratio           weight 0.35
  S2  Rendering-mode-3 ratio          weight 0.30
  S3  Text quality (language-aware)   weight 0.20
  S4  Sample-and-compare (OCR diff)   weight 0.15  — only in ambiguous zone

Signal 4 is only computed when the weighted S1+S2+S3 sub-score falls in the
ambiguous zone [0.30, 0.70].  If the OCR engine is unavailable, S4 returns
``None`` and its weight is redistributed proportionally to S1–S3.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Optional OCR import — Stage 5 owns the real OCR; we only need a lightweight
# check here when the score is in the ambiguous zone.
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

from pdf_engine.models import (
    BBox,
    BlockConfidence,
    BlockType,
    DocumentIR,
    ExtractionMethod,
    PageIR,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Characters narrower than this (in pts) are considered zero-width (W4 fix).
ZERO_WIDTH_THRESHOLD: float = 0.1

#: Score range in which Signal 4 (sample-and-compare) is triggered.
S4_AMBIGUOUS_LOW:  float = 0.30
S4_AMBIGUOUS_HIGH: float = 0.70

#: Size of the centre crop used for Signal 4 OCR comparison (PDF pts).
S4_CROP_SIZE: float = 200.0

#: Resolution used when rasterising the S4 crop (DPI).
S4_RENDER_DPI: int = 150

#: Blocks shorter than this (in chars) are skipped during extraction.
MIN_BLOCK_TEXT_LEN: int = 1

#: Quality threshold below which a page's text layer is treated as phantom.
PHANTOM_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Language / script family tables
# ---------------------------------------------------------------------------

#: Space-ratio bounds are language-aware  (fixes W5).
SPACE_RATIO_BOUNDS: dict[str, tuple[float, float]] = {
    "default": (0.08, 0.25),  # Latin scripts (English, French, German …)
    "cjk":     (0.00, 0.05),  # Chinese, Japanese, Korean
    "arabic":  (0.05, 0.15),  # Arabic
    "thai":    (0.00, 0.08),  # Thai — minimal inter-word spaces
}

# BCP-47 prefix → script family (checked before Unicode heuristic).
_LANG_TO_FAMILY: dict[str, str] = {
    "zh": "cjk", "ja": "cjk", "ko": "cjk",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "th": "thai",
}

# Unicode block ranges for CJK, Arabic, Thai used in the heuristic detector.
_CJK_RANGES   = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),
                 (0xAC00, 0xD7AF), (0x3040, 0x30FF)]
_ARABIC_RANGE = (0x0600, 0x06FF)
_THAI_RANGE   = (0x0E00, 0x0E7F)


# ---------------------------------------------------------------------------
# Script / language helpers
# ---------------------------------------------------------------------------


def get_script_family(language: Optional[str], sample_text: str = "") -> str:
    """
    Return the script family for *language* (BCP-47 tag) or, when
    *language* is ``None``, heuristically from the Unicode codepoints in
    *sample_text*.

    Returns one of: ``"default"``, ``"cjk"``, ``"arabic"``, ``"thai"``.
    """
    if language:
        prefix = language.split("-")[0].lower()
        if prefix in _LANG_TO_FAMILY:
            return _LANG_TO_FAMILY[prefix]
        return "default"

    # Heuristic: count characters in each script block.
    if not sample_text:
        return "default"

    counts: dict[str, int] = {"cjk": 0, "arabic": 0, "thai": 0}
    for ch in sample_text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            counts["cjk"] += 1
        elif _ARABIC_RANGE[0] <= cp <= _ARABIC_RANGE[1]:
            counts["arabic"] += 1
        elif _THAI_RANGE[0] <= cp <= _THAI_RANGE[1]:
            counts["thai"] += 1

    total = len(sample_text)
    if total == 0:
        return "default"
    for family, cnt in counts.items():
        if cnt / total > 0.30:
            return family
    return "default"


def _detect_script_direction(text: str) -> str:
    """
    Detect whether *text* is predominantly RTL or LTR using Python's
    ``unicodedata`` module (bidi category).

    Returns ``"rtl"`` if more than 30 % of strong-directional characters
    are RTL; otherwise ``"ltr"``.
    """
    ltr_count = 0
    rtl_count = 0
    for ch in text:
        bc = unicodedata.bidirectional(ch)
        if bc in ("L", "LRE", "LRO"):
            ltr_count += 1
        elif bc in ("R", "AL", "RLE", "RLO"):
            rtl_count += 1

    total_strong = ltr_count + rtl_count
    if total_strong == 0:
        return "ltr"
    return "rtl" if (rtl_count / total_strong) > 0.30 else "ltr"


# ---------------------------------------------------------------------------
# Text quality check (language-aware, W5 fix)
# ---------------------------------------------------------------------------


def quick_text_quality_check(text: str, language: Optional[str] = None) -> float:
    """
    Return a quality score ∈ [0.0, 1.0] for *text* using three sub-signals.

    Sub-signals
    -----------
    space_score         — Space ratio within expected bounds for the script.
    printable_score     — Fraction of printable non-whitespace characters.
    replacement_score   — Absence of Unicode replacement char U+FFFD.

    Weights: 0.40 / 0.40 / 0.20.

    Parameters
    ----------
    text : str
        The text to evaluate.
    language : str | None
        BCP-47 language tag.  When ``None``, the script family is inferred
        from the character distribution of *text* itself.
    """
    if not text:
        return 0.0

    family = get_script_family(language, sample_text=text)
    lo, hi = SPACE_RATIO_BOUNDS[family]

    total   = len(text)
    spaces  = text.count(" ")
    space_ratio = spaces / total

    # Space ratio score: 1.0 when within [lo, hi], decays outside.
    if lo <= space_ratio <= hi:
        space_score = 1.0
    elif space_ratio < lo:
        # Below minimum — penalise proportionally to how far below.
        space_score = max(0.0, space_ratio / lo) if lo > 0 else 1.0
    else:
        # Above maximum — penalise proportionally to excess.
        excess = space_ratio - hi
        space_score = max(0.0, 1.0 - excess / (1.0 - hi + 1e-9))

    # Printable-character score.
    printable_non_ws = sum(
        1 for ch in text if ch.isprintable() and not ch.isspace()
    )
    printable_score = printable_non_ws / total

    # Replacement-character score.
    replacement_chars = text.count("\ufffd")
    replacement_score = max(0.0, 1.0 - (replacement_chars / total) * 10)

    quality = (
        0.40 * space_score
        + 0.40 * printable_score
        + 0.20 * replacement_score
    )
    return min(1.0, max(0.0, quality))


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _s1_zero_width_ratio(rawdict: dict) -> float:
    """
    Signal 1: fraction of zero-width characters in the page's rawdict.

    A high ratio → likely phantom/invisible overlay.
    Returns a quality score: 1.0 = no zero-width chars, 0.0 = all zero-width.
    """
    total_chars = 0
    zero_width_chars = 0

    for block in rawdict.get("blocks", []):
        if block.get("type") != 0:   # 0 = text block in PyMuPDF rawdict
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    total_chars += 1
                    bbox = char.get("bbox", (0, 0, 0, 0))
                    width = bbox[2] - bbox[0]
                    if width < ZERO_WIDTH_THRESHOLD:
                        zero_width_chars += 1

    if total_chars == 0:
        return 1.0   # No characters at all — defer to other signals.

    zero_ratio = zero_width_chars / total_chars
    return max(0.0, 1.0 - zero_ratio)


def _s2_render_mode_ratio(rawdict: dict) -> tuple[float, float]:
    """
    Signal 2: fraction of characters that appear invisible.

    Detection strategy (HIGH-7 fix):
    * PyMuPDF's `span["flags"]` in rawdict encodes font properties
      (superscript=1, italic=2, bold=16, etc.) — NOT PDF text rendering modes.
    * The PDF rendering mode (mode 3 = invisible text) is not exposed via rawdict.
    * Therefore, we rely on heuristics:
      1. Char-level: bbox width < 0.1 pt (nearly zero-width characters)
      2. Span-level: font size < 0.5 pt (microscopic fonts)
      3. Font name patterns: fonts named "GlyphLessFont" or similar

    Returns
    -------
    tuple[float, float]
        ``(quality_score, invisible_ratio)`` where:
        - ``quality_score`` ∈ [0.0, 1.0]: 1.0 = no invisible chars, 0.0 = all.
        - ``invisible_ratio`` ∈ [0.0, 1.0]: raw fraction for the hard-cap logic.
    """
    total_chars = 0
    invisible_chars = 0

    for block in rawdict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_size = span.get("size", 12)
                font_name = span.get("font", "").lower()

                # Check for fonts commonly used for invisible text overlays
                span_has_invisible_font = (
                    "glyphless" in font_name or
                    "invisible" in font_name or
                    font_size < 0.5
                )

                for char in span.get("chars", []):
                    total_chars += 1
                    char_bbox = char.get("bbox", (0, 0, 0, 0))
                    char_width = abs(char_bbox[2] - char_bbox[0])

                    # Character is invisible if:
                    # - Nearly zero width (< 0.1pt), OR
                    # - Microscopic font (< 0.5pt), OR
                    # - Font name indicates invisible text
                    if char_width < 0.1 or span_has_invisible_font:
                        invisible_chars += 1

    if total_chars == 0:
        return 1.0, 0.0

    invisible_ratio = invisible_chars / total_chars
    quality_score   = max(0.0, 1.0 - invisible_ratio)
    return quality_score, invisible_ratio


def _s4_sample_and_compare(page, page_text: str) -> Optional[float]:
    """
    Signal 4: rasterise a 200×200pt centre crop and compare its OCR output
    against the extracted text.

    Returns a quality score ∈ [0.0, 1.0], or ``None`` if OCR is unavailable
    or the page is too small to sample.

    This is intentionally lightweight — just enough to detect the "scanned
    PDF with invisible text overlay" case.
    """
    if not _TESSERACT_AVAILABLE:
        logger.debug("S4: pytesseract not available — skipping sample-and-compare.")
        return None

    if not _FITZ_AVAILABLE:
        logger.debug("S4: PyMuPDF (fitz) not available — skipping sample-and-compare.")
        return None

    try:
        rect = page.rect
        pw, ph = rect.width, rect.height
        cx, cy = pw / 2, ph / 2
        half = S4_CROP_SIZE / 2

        crop_rect = fitz.Rect(
            max(0, cx - half),
            max(0, cy - half),
            min(pw, cx + half),
            min(ph, cy + half),
        )

        if crop_rect.width < 10 or crop_rect.height < 10:
            return None

        # Rasterise the crop.
        mat = fitz.Matrix(S4_RENDER_DPI / 72, S4_RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, clip=crop_rect, colorspace=fitz.csGRAY)
        img = PILImage.frombytes("L", (pix.width, pix.height), pix.samples)

        ocr_text = pytesseract.image_to_string(img, config="--psm 6").strip()

        # Build a comparable excerpt from the extracted text near the centre.
        # Simple approach: take 200 chars around the midpoint.
        mid = len(page_text) // 2
        excerpt = page_text[max(0, mid - 100): mid + 100].strip()

        if not excerpt or not ocr_text:
            return None

        score = _normalised_similarity(ocr_text, excerpt)
        logger.debug("S4 OCR similarity: %.3f", score)
        return score

    except Exception as exc:
        logger.debug("S4 failed: %s", exc)
        return None


def _normalised_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity between two strings, used as a proxy for
    edit-distance similarity.  Fast, language-agnostic, and good enough for
    a binary phantom/real decision.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Phantom detection (public API)
# ---------------------------------------------------------------------------


def detect_phantom_text_layer(page) -> float:
    """
    Score how trustworthy the embedded text layer of *page* is.

    Parameters
    ----------
    page
        A PyMuPDF ``Page`` object.

    Returns
    -------
    float
        Quality score ∈ [0.0, 1.0].
        ≥ 0.85 → high-quality real text layer.
        ≤ 0.30 → likely phantom / invisible overlay.
        In between → uncertain (Stage 5 may need to reconcile).
    """
    # ------------------------------------------------------------------ rawdict
    # Use rawdict — NOT "dict" — so per-character bboxes are available (W4 fix).
    # fitz.TEXT_PRESERVE_WHITESPACE = 1; use the constant if available, else
    # fall back to the raw integer so this function works when _FITZ_AVAILABLE
    # is True but fitz was imported via the try/except guard above.
    _preserve_ws_flag = fitz.TEXT_PRESERVE_WHITESPACE if _FITZ_AVAILABLE else 1
    rawdict = page.get_text("rawdict", flags=_preserve_ws_flag)

    plain_text = page.get_text("text").strip()

    # ------------------------------------------------------------------ S1
    s1_score = _s1_zero_width_ratio(rawdict)

    # ------------------------------------------------------------------ S2
    s2_score, s2_invisible_ratio = _s2_render_mode_ratio(rawdict)

    # ------------------------------------------------------------------ S3
    s3_score = quick_text_quality_check(plain_text)

    # ------------------------------------------------------------------ Weighted sub-score (S1–S3)
    sub_score = 0.35 * s1_score + 0.30 * s2_score + 0.20 * s3_score

    logger.debug(
        "Phantom signals — S1(zero-width): %.3f  S2(render-mode): %.3f  "
        "S3(text-quality): %.3f  sub_score: %.3f  invisible_ratio: %.3f",
        s1_score, s2_score, s3_score, sub_score, s2_invisible_ratio,
    )

    # ------------------------------------------------------------------ S4 (ambiguous zone only)
    s4_score: Optional[float] = None
    if S4_AMBIGUOUS_LOW <= sub_score <= S4_AMBIGUOUS_HIGH:
        s4_score = _s4_sample_and_compare(page, plain_text)

    if s4_score is not None:
        # Redistribute weights: S1+S2+S3 share = 0.85, S4 = 0.15.
        # sub_score already holds 0.35+0.30+0.20 = 0.85 worth of weight.
        final_score = sub_score + 0.15 * s4_score
    else:
        # S4 unavailable — renormalise S1–S3 to fill the full 1.0.
        final_score = sub_score / 0.85

    # ------------------------------------------------------------------ S2 hard cap
    # Invisible text (render mode 3) is a *definitive* phantom signal.
    # When more than half the characters are invisible, no amount of clean S1
    # or S3 signal should let the overall score exceed 1 − invisible_ratio.
    # Example: 100 % invisible → cap = 0.0; 80 % invisible → cap = 0.20.
    #
    # Without this cap, the weighted average can never fall below ~0.41 even
    # when every character is invisible but text content looks typographically
    # normal (S1 = 1.0, S3 = 1.0) — making the test "≤ 0.30" impossible to
    # satisfy through the additive path alone.
    if s2_invisible_ratio > 0.5:
        invisible_cap = 1.0 - s2_invisible_ratio
        final_score   = min(final_score, invisible_cap)

    final_score = min(1.0, max(0.0, final_score))

    logger.debug(
        "detect_phantom_text_layer: final=%.3f (s4=%s)",
        final_score,
        f"{s4_score:.3f}" if s4_score is not None else "skipped",
    )
    return final_score


# ---------------------------------------------------------------------------
# Block extraction helpers
# ---------------------------------------------------------------------------


def _bbox_from_rawdict_block(block: dict) -> Optional[BBox]:
    """Convert a PyMuPDF rawdict block bbox to our ``BBox`` type."""
    b = block.get("bbox")
    if b and len(b) == 4:
        return BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return None


def _collect_span_text(block: dict) -> str:
    """Concatenate all character text within a rawdict text block."""
    parts: list[str] = []
    for line in block.get("lines", []):
        line_parts: list[str] = []
        for span in line.get("spans", []):
            for char in span.get("chars", []):
                c = char.get("c", "")
                if c:
                    line_parts.append(c)
        if line_parts:
            parts.append("".join(line_parts))
    return "\n".join(parts)


def _make_text_block(
    raw_block: dict,
    page_num: int,
    method_score: float = 1.0,
) -> Optional[TextBlock]:
    """
    Convert one PyMuPDF rawdict text block into a ``TextBlock``.

    Returns ``None`` if the block contains no usable text.
    """
    if raw_block.get("type") != 0:   # Not a text block (could be image = 1).
        return None

    text = _collect_span_text(raw_block).strip()
    if len(text) < MIN_BLOCK_TEXT_LEN:
        return None

    bbox = _bbox_from_rawdict_block(raw_block)
    direction = _detect_script_direction(text)
    text_q    = quick_text_quality_check(text)

    confidence = BlockConfidence(
        text_quality  = text_q,
        order_quality = 0.8,    # Stage 3 will refine this.
        type_quality  = 0.7,    # Stage 2 will refine this.
        method_score  = method_score,
        final         = 0.0,    # Stage 9 computes this.
    )

    return TextBlock(
        text             = text,
        bbox             = bbox,
        block_type       = BlockType.BODY,  # Stage 2 will reclassify.
        extraction_method= ExtractionMethod.PYMUPDF_DIRECT,
        confidence       = confidence,
        page_num         = page_num,
        parent_id        = None,
        language         = None,    # Stage 7 / language detection will fill.
        script_direction = direction,
    )


# ---------------------------------------------------------------------------
# Stage 1 entry point
# ---------------------------------------------------------------------------


def extract_text_layer(page, page_ir: PageIR) -> PageIR:
    """
    Extract the embedded text layer from *page* and populate *page_ir.blocks*.

    Parameters
    ----------
    page
        A PyMuPDF ``Page`` object for the page to process.  The caller is
        responsible for keeping the parent ``Document`` open; this function
        never re-opens the PDF.
    page_ir : PageIR
        The ``PageIR`` produced by Stage 0 for this page.  Its
        ``triage_result`` determines whether extraction is skipped.

    Returns
    -------
    PageIR
        The same *page_ir* object, enriched with:
        - ``blocks``  populated with ``TextBlock`` objects.
        - A ``method_score`` on each block's confidence reflecting the
          phantom-detection result.
        - Warnings appended for low-quality or phantom pages.

    Notes
    -----
    Only pages with ``triage_result`` in ``{"text_native", "hybrid"}`` are
    processed.  ``"ocr_needed"`` pages are left for Stage 5.
    """
    if page_ir.triage_result not in ("text_native", "hybrid"):
        logger.debug(
            "Page %d triage_result=%r — skipping Stage 1 extraction.",
            page_ir.page_num, page_ir.triage_result,
        )
        return page_ir

    # ------------------------------------------------------------------ Phantom detection
    phantom_score = detect_phantom_text_layer(page)
    logger.info(
        "Page %d phantom_score=%.3f (triage=%r)",
        page_ir.page_num, phantom_score, page_ir.triage_result,
    )

    if phantom_score < PHANTOM_THRESHOLD:
        page_ir.add_warning(
            f"[Stage1] Page {page_ir.page_num}: phantom_score={phantom_score:.3f} "
            f"< {PHANTOM_THRESHOLD} — text layer may be unreliable; "
            f"OCR fallback is recommended."
        )
        if page_ir.triage_result == "text_native":
            # Downgrade so Stage 5 knows to OCR this page.
            page_ir.triage_result = "hybrid"
            page_ir.add_warning(
                f"[Stage1] Page {page_ir.page_num}: triage_result downgraded to "
                f"'hybrid' due to low phantom score."
            )

    # ------------------------------------------------------------------ Block extraction
    # Use rawdict to get per-character bboxes (W4 fix).
    _preserve_ws_flag = fitz.TEXT_PRESERVE_WHITESPACE if _FITZ_AVAILABLE else 1
    rawdict = page.get_text("rawdict", flags=_preserve_ws_flag)

    new_blocks: list[TextBlock] = []
    for raw_block in rawdict.get("blocks", []):
        block = _make_text_block(
            raw_block,
            page_num     = page_ir.page_num,
            method_score = phantom_score,   # Trust == phantom quality.
        )
        if block is not None:
            new_blocks.append(block)

    page_ir.blocks.extend(new_blocks)

    logger.info(
        "Page %d: extracted %d text blocks (phantom_score=%.3f).",
        page_ir.page_num, len(new_blocks), phantom_score,
    )
    return page_ir


# ---------------------------------------------------------------------------
# Document-level convenience wrapper
# ---------------------------------------------------------------------------


def extract_document_text(doc, doc_ir: DocumentIR) -> DocumentIR:
    """
    Apply Stage 1 to every page in *doc_ir*.

    Parameters
    ----------
    doc
        An open PyMuPDF ``Document`` handle.  Opened exactly once by the
        caller (design rule 3).
    doc_ir : DocumentIR
        The ``DocumentIR`` produced by Stage 0.

    Returns
    -------
    DocumentIR
        The same *doc_ir* with all eligible pages' blocks populated.
    """
    for page_ir in doc_ir.pages:
        page = doc.load_page(page_ir.page_num)
        extract_text_layer(page, page_ir)

    return doc_ir
