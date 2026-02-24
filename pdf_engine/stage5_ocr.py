"""
pdf_engine/stage5_ocr.py
=========================
Stage 5 — OCR Pipeline

Responsibility
--------------
For every ``PageIR`` whose ``triage_result`` is ``"ocr_needed"`` or
``"hybrid"``, this stage:

  1. Rasterises the page to a numpy RGB array at ``DEFAULT_OCR_DPI`` (300).
  2. Detects the script family from the page image (W16 fix).
  3. Preprocesses the image: grayscale → deskew → binarise (W18 fix).
  4. Routes to the correct OCR engine:
       - latin / cyrillic / unknown → Tesseract first; surya fallback if
         confidence < TESSERACT_CONFIDENCE_THRESHOLD.
       - cjk / arabic              → surya directly (Tesseract never called).
  5. Creates ``TextBlock`` objects from OCR results and appends them to
     ``page_ir.blocks``.

Design rules enforced here
--------------------------
* ``rasterize_page`` emits a warning if the estimated image size > 50 MB
  (likely 600 DPI on A3 or larger paper).
* Skew correction is skipped when |angle| > ``MAX_SKEW_DEGREES`` (15°) and
  a warning is added to ``PageIR.warnings`` instead (W18 fix).
* Black triangles produced by rotation are removed via the inscribed-rectangle
  crop formula (W18 fix).
* Surya and Tesseract are both optional — if either is absent the code falls
  back gracefully rather than crashing.
* Every produced ``TextBlock`` has its ``extraction_method`` set to
  ``TESSERACT_OCR`` or ``SURYA_OCR`` and ``confidence.method_score`` set
  proportionally to the engine's per-block confidence.
* ``text_native`` pages are skipped silently.
"""

from __future__ import annotations

import logging
import math
import uuid
import warnings
from typing import Optional

import numpy as np

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
# Tuneable constants
# ---------------------------------------------------------------------------

#: DPI used when rasterising pages for OCR.
DEFAULT_OCR_DPI: int = 300

#: Estimated memory threshold (bytes) at which we warn about large rasters.
#: 50 MB = 50 * 1024 * 1024 bytes
RASTER_MEMORY_WARN_BYTES: int = 50 * 1024 * 1024

#: Maximum skew angle (degrees) that will be corrected automatically.
#: Pages with detected skew beyond this threshold are left unrotated and a
#: warning is added to PageIR.warnings.
MAX_SKEW_DEGREES: float = 15.0

#: Tesseract confidence below which surya is tried as a fallback (0–100 scale).
TESSERACT_CONFIDENCE_THRESHOLD: float = 60.0

#: Minimum area (px²) for a Tesseract word-level detection to be kept.
MIN_WORD_AREA_PX: float = 25.0

# ---------------------------------------------------------------------------
# Optional dependency availability flags
# ---------------------------------------------------------------------------

try:
    import pytesseract
    from pytesseract import Output as TessOutput
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.debug("pytesseract not installed — Tesseract OCR unavailable")

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.debug("Pillow not installed — rasterisation / preprocessing unavailable")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.debug("opencv-python not installed — deskew unavailable")

try:
    from pdf_engine.surya_adapter import run_surya_layout, map_surya_label
    _SURYA_ADAPTER_AVAILABLE = True
except ImportError:
    _SURYA_ADAPTER_AVAILABLE = False
    logger.debug("surya_adapter unavailable — surya OCR path disabled")


# ---------------------------------------------------------------------------
# Script-family constants
# ---------------------------------------------------------------------------

SCRIPT_LATIN    = "latin"
SCRIPT_CJK      = "cjk"
SCRIPT_ARABIC   = "arabic"
SCRIPT_CYRILLIC = "cyrillic"
SCRIPT_UNKNOWN  = "unknown"

#: Scripts that should be routed to surya directly (no Tesseract attempt).
SURYA_FIRST_SCRIPTS: frozenset[str] = frozenset({SCRIPT_CJK, SCRIPT_ARABIC})

#: Tesseract OSD script name → our script family
_OSD_SCRIPT_MAP: dict[str, str] = {
    "Latin":         SCRIPT_LATIN,
    "Han":           SCRIPT_CJK,
    "Hangul":        SCRIPT_CJK,
    "Japanese":      SCRIPT_CJK,
    "HanS":          SCRIPT_CJK,
    "HanT":          SCRIPT_CJK,
    "Arabic":        SCRIPT_ARABIC,
    "Cyrillic":      SCRIPT_CYRILLIC,
    "Devanagari":    SCRIPT_UNKNOWN,
    "Bengali":       SCRIPT_UNKNOWN,
    "Tamil":         SCRIPT_UNKNOWN,
    "Hebrew":        SCRIPT_ARABIC,   # also right-to-left; surya handles it well
}


# ---------------------------------------------------------------------------
# Step 1 — Rasterisation
# ---------------------------------------------------------------------------


def rasterize_page(fitz_page, dpi: int = DEFAULT_OCR_DPI) -> np.ndarray:
    """
    Rasterise a PyMuPDF page to an RGB numpy array.

    Parameters
    ----------
    fitz_page:
        An open ``fitz.Page`` handle.
    dpi:
        Resolution in dots-per-inch.  300 DPI produces ≈8 MB/page (A4 RGB).
        600 DPI produces ≈33 MB/page — a warning is emitted if the
        estimated footprint exceeds ``RASTER_MEMORY_WARN_BYTES`` (50 MB).

    Returns
    -------
    np.ndarray
        Shape ``(height, width, 3)``, dtype ``uint8``, RGB colour order.

    Raises
    ------
    RuntimeError
        If PyMuPDF (fitz) is not importable.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for rasterisation.  "
            "Install it with:  pip install pymupdf"
        ) from exc

    # STAB-3 fix: auto-reduce DPI if the initial raster would exceed the memory
    # threshold.  We halve the DPI repeatedly (down to a floor of 72 DPI) until
    # the estimate fits, or warn and return a reduced-scale raster rather than
    # silently allocating a huge array and risk an OOM crash.
    _DPI_FLOOR = 72
    _REDUCTION_FACTOR = 0.5

    def _render(d: float):
        m = fitz.Matrix(d / 72, d / 72)
        return fitz_page.get_pixmap(matrix=m, colorspace=fitz.csRGB, alpha=False)

    pixmap = _render(dpi)
    estimated_bytes = pixmap.width * pixmap.height * 3

    if estimated_bytes > RASTER_MEMORY_WARN_BYTES:
        reduced_dpi = dpi
        while estimated_bytes > RASTER_MEMORY_WARN_BYTES and reduced_dpi > _DPI_FLOOR:
            reduced_dpi = max(_DPI_FLOOR, reduced_dpi * _REDUCTION_FACTOR)
            pixmap = _render(reduced_dpi)
            estimated_bytes = pixmap.width * pixmap.height * 3

        if reduced_dpi < dpi:
            logger.warning(
                "Large raster auto-reduced: DPI %d → %d "
                "(%.1f MB fits under %.0f MB threshold).  OCR quality may be lower.",
                dpi, reduced_dpi,
                estimated_bytes / 1_048_576,
                RASTER_MEMORY_WARN_BYTES / 1_048_576,
            )
        else:
            # Even at floor DPI it's still large — warn but proceed
            logger.warning(
                "Raster at minimum DPI (%d) is %.1f MB — proceeding but may be slow.",
                _DPI_FLOOR, estimated_bytes / 1_048_576,
            )

    arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return arr


# ---------------------------------------------------------------------------
# Step 2 — Script detection (W16 fix)
# ---------------------------------------------------------------------------


def detect_script_from_image(page_image: np.ndarray) -> str:
    """
    Estimate the script family of a page image before choosing an OCR engine.

    Strategy
    --------
    Run Tesseract OSD (orientation + script detection) on a centre 25 % crop
    of the rasterised image.  OSD is much faster than full OCR and provides
    the script name directly.

    Falls back to ``"unknown"`` if Tesseract is not installed, if OSD fails,
    or if the detected script is not in ``_OSD_SCRIPT_MAP``.

    Parameters
    ----------
    page_image:
        RGB numpy array, shape ``(H, W, 3)``.

    Returns
    -------
    str
        One of ``"latin"``, ``"cjk"``, ``"arabic"``, ``"cyrillic"``,
        ``"unknown"``.
    """
    if not _TESSERACT_AVAILABLE or not _PIL_AVAILABLE:
        logger.debug("Tesseract/Pillow unavailable — script detection skipped, returning 'unknown'")
        return SCRIPT_UNKNOWN

    try:
        h, w = page_image.shape[:2]
        # Centre 25 % crop to reduce noise from page borders.
        y0 = h // 4
        y1 = 3 * h // 4
        x0 = w // 4
        x1 = 3 * w // 4
        crop = page_image[y0:y1, x0:x1]

        pil_crop = PILImage.fromarray(crop, "RGB")
        osd = pytesseract.image_to_osd(
            pil_crop, output_type=TessOutput.DICT, config="--psm 0"
        )
        raw_script: str = osd.get("script", "")
        script_family = _OSD_SCRIPT_MAP.get(raw_script, SCRIPT_UNKNOWN)
        logger.debug("OSD detected script %r → family %r", raw_script, script_family)
        return script_family

    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("OSD script detection failed (%s) — returning 'unknown'", exc)
        return SCRIPT_UNKNOWN


# ---------------------------------------------------------------------------
# Step 3 — Preprocessing (W18 fix)
# ---------------------------------------------------------------------------


def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate the skew angle of a grayscale image using the Hough transform.

    Returns the estimated angle in degrees (positive = counter-clockwise).
    Returns 0.0 if OpenCV is unavailable or detection fails.
    """
    if not _CV2_AVAILABLE:
        return 0.0

    try:
        # Binarise first so lines are clear.
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate horizontally to merge characters into long line segments.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated = cv2.dilate(bw, kernel)

        # Find contours of text lines.
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Max area a text-line contour should occupy — anything larger is
        # almost certainly an image block or page border, not a text line.
        # Using 5 % of the total image area as the upper cap.
        total_area = gray.shape[0] * gray.shape[1]
        max_contour_area = total_area * 0.05

        angles: list[float] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Skip tiny noise AND large image/border regions.
            if area < 200 or area > max_contour_area:
                continue
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            # cv2.minAreaRect returns angles in [-90, 0).
            # Text lines that are roughly horizontal will be close to 0 or -90.
            if angle < -45:
                angle += 90
            angles.append(angle)

        if not angles:
            return 0.0

        # Median is more robust than mean against outliers (e.g. vertical text).
        median_angle = float(np.median(angles))
        return median_angle

    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Skew detection failed: %s", exc)
        return 0.0


def _largest_inscribed_rect(w: int, h: int, angle_rad: float) -> tuple[int, int]:
    """
    Compute the dimensions of the largest axis-aligned rectangle that fits
    inside a ``w × h`` image after it has been rotated by ``angle_rad`` radians.

    This eliminates the black triangles introduced by rotation.

    Formula source: https://stackoverflow.com/a/16778797
    """
    if w <= 0 or h <= 0:
        return w, h

    angle = abs(angle_rad) % math.pi
    if angle > math.pi / 2:
        angle = math.pi - angle

    if w == h:
        new_w = new_h = int(w * (math.cos(angle) - math.sin(angle)))
        return new_w, new_h

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Quadrant (0 < angle < pi/2)
    if h * sin_a <= w * cos_a:
        # Height-constrained solution
        new_h = int(h / (cos_a + sin_a * h / w))
        new_w = int(new_h * w / h)
    else:
        # Width-constrained solution
        new_w = int(w / (cos_a + sin_a * w / h))
        new_h = int(new_w * h / w)

    return new_w, new_h


def preprocess_for_ocr(
    image: np.ndarray,
    max_skew_degrees: float = MAX_SKEW_DEGREES,
    page_warnings: Optional[list] = None,
) -> np.ndarray:
    """
    Prepare an RGB image for OCR.

    Pipeline
    --------
    1. Convert to grayscale.
    2. Detect skew angle via Hough transform.
    3. If ``|angle| > max_skew_degrees``: skip deskew, append warning.
    4. Rotate and crop black border triangles (inscribed rectangle).
    5. Binarise with Otsu's threshold.

    Parameters
    ----------
    image:
        RGB numpy array ``(H, W, 3)``.
    max_skew_degrees:
        Rotation angles beyond this are treated as non-standard and skipped.
    page_warnings:
        If provided, skew-skip warnings are appended here (mutated in-place).

    Returns
    -------
    np.ndarray
        Grayscale binarised image ``(H', W')``, dtype ``uint8``.
        ``H'`` and ``W'`` may differ from ``H`` and ``W`` after cropping.
    """
    if not _CV2_AVAILABLE or not _PIL_AVAILABLE:
        # Minimal fallback: just grayscale + Otsu via numpy if cv2 absent.
        gray = np.mean(image, axis=2).astype(np.uint8)
        threshold = _otsu_numpy(gray)
        # Tesseract requires black-on-white (text=0, background=255).
        # Pixels >= threshold are background (white); pixels < threshold are text (black).
        return (gray >= threshold).astype(np.uint8) * 255

    # ---- 1. Grayscale -------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ---- 2. Detect skew angle -----------------------------------------------
    angle_deg = _detect_skew_angle(gray)

    # ---- 3. Skip if angle is too large --------------------------------------
    if abs(angle_deg) > max_skew_degrees:
        msg = (
            f"Skew angle {angle_deg:.1f}° exceeds limit {max_skew_degrees}°; "
            "deskew skipped to avoid aggressive cropping."
        )
        logger.debug(msg)
        if page_warnings is not None:
            page_warnings.append(msg)
        # Skip rotation; proceed directly to binarisation.
        _, binarised = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binarised

    # ---- 4. Rotate and crop -------------------------------------------------
    if abs(angle_deg) > 0.1:   # Only rotate if there is a meaningful angle.
        h, w = gray.shape
        centre = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            gray, rot_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Crop to the largest inscribed rectangle (removes black triangles).
        angle_rad = math.radians(abs(angle_deg))
        new_w, new_h = _largest_inscribed_rect(w, h, angle_rad)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        cx, cy = w // 2, h // 2
        x0 = cx - new_w // 2
        y0 = cy - new_h // 2
        gray = rotated[y0: y0 + new_h, x0: x0 + new_w]
    # else: angle is negligible; skip rotation entirely.

    # ---- 5. Binarise (Otsu) -------------------------------------------------
    _, binarised = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binarised


def _otsu_numpy(gray: np.ndarray) -> int:
    """Minimal Otsu threshold via numpy histogram (cv2-free fallback)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total   = gray.size
    sum_b   = 0
    w_b     = 0
    maximum = 0.0
    sum_all = np.dot(np.arange(256, dtype=np.float64), hist)
    threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > maximum:
            maximum   = between
            threshold = t

    return threshold


# ---------------------------------------------------------------------------
# Step 4a — Tesseract OCR
# ---------------------------------------------------------------------------


def _run_tesseract(binarised: np.ndarray, page_num: int) -> list[TextBlock]:
    """
    Run Tesseract OCR on a pre-processed binary image.

    Words are grouped into paragraph-level ``TextBlock`` objects using
    Tesseract's own block/paragraph numbering.  This produces coherent
    sentences and paragraphs rather than one block per word.

    Returns an empty list if Tesseract is unavailable or raises.
    """
    if not _TESSERACT_AVAILABLE or not _PIL_AVAILABLE:
        logger.warning("Tesseract unavailable — cannot run Tesseract OCR")
        return []

    try:
        pil_img = PILImage.fromarray(binarised, "L")
        data = pytesseract.image_to_data(
            pil_img, output_type=TessOutput.DICT, config="--oem 1 --psm 3"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Tesseract OCR failed: %s", exc, exc_info=True)
        return []

    # --- Group words by (block_num, par_num) into paragraph blocks ----------
    # Each entry in data has parallel arrays; zip them for easy iteration.
    n = len(data["text"])

    # para_key → {"words": [...], "confs": [...], "boxes": [...]}
    paragraphs: dict[tuple, dict] = {}

    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue

        conf_raw = float(data["conf"][i])
        if conf_raw < 0:
            continue  # Tesseract encodes non-word rows as conf=-1

        x  = int(data["left"][i])
        y  = int(data["top"][i])
        bw = int(data["width"][i])
        bh = int(data["height"][i])

        if bw * bh < MIN_WORD_AREA_PX:
            continue

        # Use (block_num, par_num) as the grouping key.
        key = (int(data["block_num"][i]), int(data["par_num"][i]))

        if key not in paragraphs:
            paragraphs[key] = {"words": [], "confs": [], "x0": x, "y0": y,
                                "x1": x + bw, "y1": y + bh}
        p = paragraphs[key]
        p["words"].append(word)
        p["confs"].append(conf_raw)
        p["x0"] = min(p["x0"], x)
        p["y0"] = min(p["y0"], y)
        p["x1"] = max(p["x1"], x + bw)
        p["y1"] = max(p["y1"], y + bh)

    # --- Convert grouped paragraphs → TextBlocks ----------------------------
    blocks: list[TextBlock] = []
    for p in paragraphs.values():
        if not p["words"]:
            continue

        text = " ".join(p["words"])
        mean_conf = sum(p["confs"]) / len(p["confs"])
        method_score = mean_conf / 100.0

        bbox = BBox(
            x0=float(p["x0"]), y0=float(p["y0"]),
            x1=float(p["x1"]), y1=float(p["y1"]),
        )
        block = TextBlock(
            id=str(uuid.uuid4()),
            text=text,
            bbox=bbox,
            block_type=BlockType.BODY,
            extraction_method=ExtractionMethod.TESSERACT_OCR,
            confidence=BlockConfidence(
                text_quality=method_score,
                method_score=method_score,
            ),
            page_num=page_num,
            language="en",  # BUG-4 fix: default OCR output to English so
                            # stage8 confusion-table corrections fire.
                            # Stage 7 will overwrite this when it detects the
                            # actual document language.
        )
        blocks.append(block)

    return blocks


def _tesseract_mean_confidence(blocks: list[TextBlock]) -> float:
    """Mean method_score across a list of TextBlock objects (0.0 if empty)."""
    if not blocks:
        return 0.0
    return sum(b.confidence.method_score for b in blocks) / len(blocks)


# ---------------------------------------------------------------------------
# Step 4b — Surya OCR (via adapter)
# ---------------------------------------------------------------------------


def _run_surya_ocr(
    original_image: np.ndarray,
    page_num: int,
) -> list[TextBlock]:
    """
    Run the surya layout model as an OCR fallback on the original RGB image.

    Note: surya expects a PIL RGB image.  We use the *original* (un-binarised)
    image here because surya's internal preprocessing performs better on
    the un-modified input.

    Returns an empty list if the adapter is unavailable.
    """
    if not _SURYA_ADAPTER_AVAILABLE:
        logger.warning("surya_adapter unavailable — cannot run surya OCR")
        return []

    if not _PIL_AVAILABLE:
        logger.warning("Pillow unavailable — cannot convert array for surya")
        return []

    try:
        pil_img = PILImage.fromarray(original_image, "RGB")
        surya_blocks = run_surya_layout(pil_img, dpi=DEFAULT_OCR_DPI)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("surya OCR call failed: %s", exc, exc_info=True)
        return []

    result: list[TextBlock] = []
    for sb in surya_blocks:
        block_type = map_surya_label(sb.label)
        block = TextBlock(
            id=str(uuid.uuid4()),
            text="",          # surya layout returns regions; text filled by Stage 6
            bbox=sb.bbox,
            block_type=block_type,
            extraction_method=ExtractionMethod.SURYA_OCR,
            confidence=BlockConfidence(
                type_quality=sb.confidence,
                method_score=sb.confidence,
            ),
            page_num=page_num,
            language="en",  # BUG-4 fix: default OCR output to English
        )
        result.append(block)

    return result


# ---------------------------------------------------------------------------
# Step 4 — Engine routing
# ---------------------------------------------------------------------------


def _ocr_page(
    original_image: np.ndarray,
    preprocessed: np.ndarray,
    script_family: str,
    page_ir: PageIR,
) -> list[TextBlock]:
    """
    Select and run the appropriate OCR engine(s) for *one* page.

    Routing rules (W16 fix)
    -----------------------
    * CJK or Arabic → surya only (Tesseract is unreliable for these scripts and
      adds 3–5 s of latency per page for no gain).
    * Latin / Cyrillic / Unknown → Tesseract first.  If mean word confidence
      is below ``TESSERACT_CONFIDENCE_THRESHOLD``, surya is tried as a
      fallback and its results replace Tesseract's.

    Parameters
    ----------
    original_image:
        RGB numpy array — passed to surya (which does its own preprocessing).
    preprocessed:
        Grayscale binarised numpy array — passed to Tesseract.
    script_family:
        One of the ``SCRIPT_*`` constants.
    page_ir:
        The current page's IR object (used only for page_num and warnings).

    Returns
    -------
    list[TextBlock]
        Combined OCR result blocks for this page.
    """
    page_num = page_ir.page_num

    if script_family in SURYA_FIRST_SCRIPTS:
        logger.debug(
            "Page %d: script=%r → surya-first routing (no Tesseract)",
            page_num, script_family,
        )
        return _run_surya_ocr(original_image, page_num)

    # Tesseract-first routing.
    logger.debug("Page %d: script=%r → Tesseract-first routing", page_num, script_family)
    tess_blocks = _run_tesseract(preprocessed, page_num)
    mean_conf   = _tesseract_mean_confidence(tess_blocks) * 100.0  # back to 0–100

    if mean_conf < TESSERACT_CONFIDENCE_THRESHOLD:
        logger.info(
            "Page %d: Tesseract mean confidence %.1f < %.1f — trying surya fallback",
            page_num, mean_conf, TESSERACT_CONFIDENCE_THRESHOLD,
        )
        surya_blocks = _run_surya_ocr(original_image, page_num)
        if surya_blocks:
            return surya_blocks
        # surya returned nothing — keep Tesseract's low-confidence output.
        page_ir.add_warning(
            f"Page {page_num}: Tesseract confidence low ({mean_conf:.1f}) "
            "and surya fallback returned no blocks; keeping Tesseract output."
        )

    return tess_blocks


# ---------------------------------------------------------------------------
# Per-page OCR
# ---------------------------------------------------------------------------


def process_ocr_page(
    page_ir: PageIR,
    fitz_page,
    dpi: int = DEFAULT_OCR_DPI,
) -> PageIR:
    """
    Run the full OCR pipeline on a single page and append OCR blocks to
    ``page_ir.blocks``.

    For ``"text_native"`` pages this is a no-op.

    Parameters
    ----------
    page_ir:
        The ``PageIR`` produced by earlier stages.
    fitz_page:
        Live ``fitz.Page`` handle.  May be ``None`` in unit tests that
        inject a pre-built image.
    dpi:
        Rasterisation resolution.

    Returns
    -------
    PageIR
        The same object, mutated in-place and returned for chaining.
    """
    if page_ir.triage_result == "text_native":
        logger.debug("Page %d: text_native — skipping OCR", page_ir.page_num)
        return page_ir

    # ---- Rasterise ----------------------------------------------------------
    if fitz_page is not None:
        try:
            original_image = rasterize_page(fitz_page, dpi=dpi)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Page %d: rasterisation failed: %s", page_ir.page_num, exc,
                exc_info=True,
            )
            page_ir.add_warning(f"Page {page_ir.page_num}: rasterisation failed: {exc}")
            return page_ir
    else:
        logger.warning(
            "Page %d: no fitz_page provided — OCR skipped", page_ir.page_num
        )
        page_ir.add_warning(
            f"Page {page_ir.page_num}: no fitz page handle — OCR skipped."
        )
        return page_ir

    # ---- Script detection (W16) --------------------------------------------
    script_family = detect_script_from_image(original_image)
    logger.debug(
        "Page %d: detected script family %r", page_ir.page_num, script_family
    )

    # ---- Preprocessing (W18) -----------------------------------------------
    preprocessed = preprocess_for_ocr(
        original_image,
        max_skew_degrees=MAX_SKEW_DEGREES,
        page_warnings=page_ir.warnings,
    )

    # ---- OCR engine routing ------------------------------------------------
    ocr_blocks = _ocr_page(original_image, preprocessed, script_family, page_ir)

    logger.info(
        "Page %d: OCR produced %d block(s) via script=%r",
        page_ir.page_num, len(ocr_blocks), script_family,
    )

    # ---- Merge blocks -------------------------------------------------------
    if page_ir.triage_result == "ocr_needed":
        # Replace whatever placeholder blocks exist.
        page_ir.blocks = ocr_blocks
    else:
        # "hybrid" — append OCR blocks alongside native-text blocks.
        page_ir.blocks.extend(ocr_blocks)

    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage5(
    doc_ir: DocumentIR,
    fitz_doc,
    dpi: int = DEFAULT_OCR_DPI,
) -> DocumentIR:
    """
    Run OCR across all applicable pages of the document.

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 4 (or Stage 2 if tables are
        skipped).
    fitz_doc:
        Open ``fitz.Document`` handle.  Must remain open for the duration
        of this call.
    dpi:
        Rasterisation DPI, forwarded to every ``process_ocr_page`` call.

    Returns
    -------
    DocumentIR
        The same object, with OCR blocks appended/replacing blocks on
        ``"ocr_needed"`` and ``"hybrid"`` pages.
    """
    ocr_page_nums   = [
        p.page_num
        for p in doc_ir.pages
        if p.triage_result in ("ocr_needed", "hybrid")
    ]
    logger.info(
        "Stage 5: %d page(s) require OCR: %s",
        len(ocr_page_nums), ocr_page_nums,
    )

    for page_ir in doc_ir.pages:
        if page_ir.triage_result not in ("ocr_needed", "hybrid"):
            continue

        try:
            fitz_page = fitz_doc[page_ir.page_num] if fitz_doc is not None else None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Page %d: could not retrieve fitz page: %s",
                page_ir.page_num, exc, exc_info=True,
            )
            page_ir.add_warning(
                f"Page {page_ir.page_num}: could not retrieve fitz page: {exc}"
            )
            continue

        process_ocr_page(page_ir, fitz_page, dpi=dpi)

    # Final parent_id integrity check.
    errors = doc_ir.validate_parent_refs()
    for err in errors:
        logger.error("Parent-ref integrity error after Stage 5: %s", err)
        doc_ir.add_warning(err)

    return doc_ir
