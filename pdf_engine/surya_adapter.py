"""
pdf_engine/surya_adapter.py
===========================
Surya Layout Model — Contract + Adapter

This module is the **single** integration point between the pipeline and
the surya library.  No other stage imports surya directly.

Design goals
------------
* If surya is not installed or raises at runtime, ``run_surya_layout``
  returns ``[]`` (never None, never raises).
* All coordinates are normalised to PyMuPDF top-left origin / points
  before leaving this module.
* Detections with confidence < CONFIDENCE_THRESHOLD are silently dropped.

Label mapping
-------------
surya label → BlockType
  "Text"         → BODY
  "Title"        → HEADING
  "Table"        → TABLE
  "Figure"       → FIGURE
  "Caption"      → CAPTION
  "Page-header"  → HEADER
  "Page-footer"  → FOOTER
  "Footnote"     → FOOTNOTE
  <anything else> → BODY  (with a logged warning)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pdf_engine.models import BBox, BlockType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# surya availability
# ---------------------------------------------------------------------------

try:
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.processor import load_processor as load_det_processor
    from surya.layout import batch_layout_detection
    _SURYA_AVAILABLE = True
except ImportError:
    _SURYA_AVAILABLE = False
    logger.debug("surya not installed — layout model unavailable")


# ---------------------------------------------------------------------------
# Public contract type
# ---------------------------------------------------------------------------


@dataclass
class SuryaBlock:
    """
    A single region detection from the surya layout model,
    normalised to PyMuPDF coordinate space (top-left origin, points).
    """

    bbox: BBox
    label: str           # surya's raw label string (before mapping)
    confidence: float    # detection confidence ∈ [0, 1]


# ---------------------------------------------------------------------------
# Label → BlockType mapping
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, BlockType] = {
    "Text":        BlockType.BODY,
    "Title":       BlockType.HEADING,
    "Table":       BlockType.TABLE,
    "Figure":      BlockType.FIGURE,
    "Caption":     BlockType.CAPTION,
    "Page-header": BlockType.HEADER,
    "Page-footer": BlockType.FOOTER,
    "Footnote":    BlockType.FOOTNOTE,
    # Common surya variant spellings
    "Header":      BlockType.HEADER,
    "Footer":      BlockType.FOOTER,
    "List-item":   BlockType.BODY,
    "Formula":     BlockType.BODY,
    "Equation":    BlockType.BODY,
}


def map_surya_label(raw_label: str) -> BlockType:
    """
    Convert a raw surya label string to the pipeline's ``BlockType``.

    Unknown labels fall back to ``BODY`` with a warning so the pipeline
    never crashes on a new surya category.
    """
    block_type = _LABEL_MAP.get(raw_label)
    if block_type is None:
        logger.warning(
            "Unknown surya label %r — defaulting to BODY", raw_label
        )
        return BlockType.BODY
    return block_type


# ---------------------------------------------------------------------------
# Coordinate normalisation helpers
# ---------------------------------------------------------------------------


def _surya_bbox_to_pymupdf(raw_bbox: list | tuple, dpi: float = 150.0) -> BBox:
    """
    Convert a surya bounding box from image-pixel coordinates to PDF user-space
    points.

    surya operates on rasterised images and returns bboxes in pixel coordinates
    (origin at top-left, same as PyMuPDF after normalisation).  PDF points use
    72 pt/inch, so the scale factor is ``72 / dpi``.

    Parameters
    ----------
    raw_bbox:
        [x0, y0, x1, y1] in image pixel coordinates.
    dpi:
        Resolution at which the page was rasterised before passing to surya.
        Defaults to 150, matching ``stage2_layout.RASTER_DPI``.

    Returns
    -------
    BBox
        Coordinates in PDF user-space points.
    """
    x0, y0, x1, y1 = float(raw_bbox[0]), float(raw_bbox[1]), \
                      float(raw_bbox[2]), float(raw_bbox[3])
    scale = 72.0 / max(dpi, 1.0)
    return BBox(x0=x0 * scale, y0=y0 * scale, x1=x1 * scale, y1=y1 * scale)


# ---------------------------------------------------------------------------
# Lazy model cache (loaded once per process)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


#: Minimum free RAM required before loading the surya model (~5 GB on-disk,
#: ~8 GB peak RSS).  If the system has less free memory we skip loading and
#: log a clear warning rather than silently OOM.
_SURYA_MIN_FREE_RAM_BYTES: int = 6 * 1024 ** 3  # 6 GB


def _check_available_ram() -> tuple:
    """Return (enough_ram, free_bytes).  Falls back to (True, 0) if psutil unavailable."""
    try:
        import psutil  # optional dep
        free = psutil.virtual_memory().available
        return free >= _SURYA_MIN_FREE_RAM_BYTES, free
    except ImportError:
        return True, 0  # can't check; proceed optimistically


def _get_surya_model():
    """Load and cache the surya layout model + processor.

    STAB-2 fix: checks available RAM before attempting to load the ~5 GB
    model.  If the system reports < 6 GB free, logs a clear warning and
    returns (None, None) so the caller falls back gracefully.
    """
    if "model" not in _model_cache:
        enough, free_bytes = _check_available_ram()
        if not enough:
            free_gb = free_bytes / 1024 ** 3
            logger.warning(
                "Skipping surya model load: only %.1f GB RAM free "
                "(need %.0f GB).  Stage 2 will use fallback blocks.  "
                "Close other applications or add swap to enable layout analysis.",
                free_gb,
                _SURYA_MIN_FREE_RAM_BYTES / 1024 ** 3,
            )
            _model_cache["model"] = None
            _model_cache["processor"] = None
        else:
            logger.debug("Loading surya layout model (%.1f GB free RAM)...", free_bytes / 1024 ** 3)
            _model_cache["model"] = load_det_model()
            _model_cache["processor"] = load_det_processor()
    return _model_cache["model"], _model_cache["processor"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_surya_layout(page_image, dpi: float = 150.0) -> list[SuryaBlock]:
    """
    Run the surya layout model on a single page image and return
    normalised ``SuryaBlock`` objects.

    Parameters
    ----------
    page_image:
        A ``PIL.Image.Image`` (RGB) of the page, rasterised at whatever
        DPI the caller prefers (150–300 dpi recommended).
    dpi:
        The resolution at which ``page_image`` was rasterised.  Used to
        convert surya's pixel-space bboxes to PDF user-space points.
        Must match the DPI used when rendering the image (e.g. 150 for
        Stage 2, 300 for Stage 5 OCR).

    Returns
    -------
    list[SuryaBlock]
        Possibly empty list.  Never returns ``None``.  Detections whose
        ``confidence < CONFIDENCE_THRESHOLD`` are excluded.
    """
    if not _SURYA_AVAILABLE:
        logger.debug("surya not available — run_surya_layout returning []")
        return []

    try:
        model, processor = _get_surya_model()
        if model is None:
            logger.debug("surya model unavailable (RAM check failed) — returning []")
            return []
        results = batch_layout_detection([page_image], model, processor)

        blocks: list[SuryaBlock] = []
        if not results:
            return blocks

        # batch_layout_detection returns one result per image
        page_result = results[0]

        for detection in page_result.bboxes:
            conf = float(getattr(detection, "confidence", 1.0))
            if conf < CONFIDENCE_THRESHOLD:
                continue

            raw_label = getattr(detection, "label", "Text")
            raw_bbox  = getattr(detection, "bbox", [0, 0, 0, 0])

            bbox = _surya_bbox_to_pymupdf(raw_bbox, dpi=dpi)

            blocks.append(SuryaBlock(
                bbox=bbox,
                label=raw_label,
                confidence=conf,
            ))

        return blocks

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("surya layout detection failed: %s", exc, exc_info=True)
        return []
