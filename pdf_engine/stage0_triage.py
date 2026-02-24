"""
pdf_engine/stage0_triage.py
============================
Stage 0 — PDF Ingestion + Page Triage

Responsibility
--------------
Open the PDF **once**, iterate its pages, and classify each page into:

    "text_native"  — Reliable embedded text layer; no OCR path needed.
    "ocr_needed"   — Image-dominated page; must be fully rasterised + OCR'd.
    "hybrid"       — Mix of embedded text and images; both paths required.

Classification is based on two signals computed per page:

    image_coverage  — fraction of the page area covered by images
                      (union of bounding boxes, NOT a simple sum — W1 fix)
    text_density    — text character count normalised by page area (W2 fix)

Decision table
--------------
    image_coverage >= 0.80
        OR (text_density < TEXT_DENSITY_SPARSE and image_coverage > 0.10)
                                        →  "ocr_needed"
    text_density >= TEXT_DENSITY_RICH
        AND image_coverage <= 0.20      →  "text_native"
    (everything else)                   →  "hybrid"

Design fixes enforced here
--------------------------
W1  Image coverage uses the UNION area of all image bounding boxes,
    computed via coordinate-compressed sweep-line.  Summing raw rectangle
    areas double-counts overlapping regions (e.g. watermarked PDFs).

W2  Raw text_len is divided by (page_area / AREA_NORM_PT2) to produce
    text_density.  This makes thresholds page-size–agnostic: an A4 page
    (≈ 50 000 pt²) and a thumbnail (≈ 5 000 pt²) are judged by the same
    density standard.

W29 Encrypted documents are detected immediately after opening.
    If decryption fails (no password supplied, or wrong password supplied),
    a structured warning is appended to DocumentIR.warnings and the
    function returns a DocumentIR with zero pages processed — never
    silently returning empty output.

Session management
------------------
The PDF backend (pypdfium2 / PyMuPDF) is opened ONCE at document level.
The open handle — not the file path — is passed to ``_triage_page()``.
This satisfies design rule 3: "Never open pdfplumber or pymupdf more than
once per document."

Backend compatibility note
--------------------------
This module is written for pypdfium2, which is API-compatible enough with
PyMuPDF for Stage 0 purposes.  The shim ``_PageAdapter`` normalises the two
APIs so that the triage logic itself is backend-agnostic.  Switching to
PyMuPDF requires only replacing the import and the ``_open_document()``
helper.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    import pypdfium2 as pdfium
    _BACKEND = "pypdfium2"
except ImportError:
    try:
        import fitz as pdfium  # type: ignore[no-redef]
        _BACKEND = "pymupdf"
    except ImportError as exc:
        raise ImportError(
            "Stage 0 requires either pypdfium2 or PyMuPDF (fitz).\n"
            "Install one of them:  pip install pypdfium2  OR  pip install pymupdf"
        ) from exc

from pdf_engine.models import DocumentIR, PageIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold constants
# All are module-level so tests can monkeypatch them without subclassing.
# ---------------------------------------------------------------------------

#: Normalisation constant for text-density calculation (pt²).
#: text_density = text_len / (page_area / AREA_NORM_PT2)
AREA_NORM_PT2: float = 10_000.0

#: image_coverage ≥ this  →  always ocr_needed
IMAGE_COV_HIGH: float = 0.80

#: image_coverage > this AND text sparse  →  ocr_needed
IMAGE_COV_THRESHOLD: float = 0.10

#: image_coverage ≤ this AND text rich  →  text_native
IMAGE_COV_CLEAN: float = 0.20

#: text_density below this is considered "sparse"
#: Pages with density below this are considered sparse (likely OCR-needed).
#: Set lower than TEXT_DENSITY_RICH to create a buffer zone for hybrid pages.
TEXT_DENSITY_SPARSE: float = 5.0

#: Pages with density at or above this are considered text-rich (text_native).
#: The gap between SPARSE(5.0) and RICH(10.0) catches borderline pages as
#: hybrid instead of knife-edging between ocr_needed and text_native.
TEXT_DENSITY_RICH: float = 10.0


# ---------------------------------------------------------------------------
# W1 fix — Union area via coordinate-compressed sweep-line
# ---------------------------------------------------------------------------


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """
    Compute the area of the **union** of a set of axis-aligned rectangles.

    Each box is a 4-tuple ``(x0, y0, x1, y1)`` in any consistent coordinate
    space.  Overlapping rectangles are counted only once.

    Algorithm
    ---------
    1. Collect all unique y-coordinates and sort them.
    2. For each horizontal slab between consecutive y-values, determine which
       rectangles are active (they span that slab).
    3. Compute the 1-D union of the active rectangles' x-intervals for that
       slab.
    4. Accumulate: slab_height × 1D_union_length.

    Complexity: O(n² ) in the number of rectangles, which is perfectly
    acceptable here because a PDF page rarely contains more than a few dozen
    image objects.

    Returns 0.0 for an empty input.
    """
    if not boxes:
        return 0.0

    # Normalise so that y0 < y1 and x0 < x1 (handle inverted coords).
    normalised: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in boxes:
        normalised.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    # Collect and sort all unique y boundaries.
    y_coords: list[float] = sorted(
        {y for box in normalised for y in (box[1], box[3])}
    )

    total_area = 0.0

    for i in range(len(y_coords) - 1):
        slab_y0 = y_coords[i]
        slab_y1 = y_coords[i + 1]
        slab_h  = slab_y1 - slab_y0
        if slab_h <= 0:
            continue

        # Rectangles whose y-range fully covers this slab.
        active_x_intervals: list[tuple[float, float]] = [
            (box[0], box[2])
            for box in normalised
            if box[1] <= slab_y0 and box[3] >= slab_y1
        ]

        # Merge overlapping x-intervals and sum their lengths.
        if not active_x_intervals:
            continue

        active_x_intervals.sort()
        merged_x0, merged_x1 = active_x_intervals[0]
        x_union_length = 0.0

        for ix0, ix1 in active_x_intervals[1:]:
            if ix0 <= merged_x1:
                # Overlapping or adjacent — extend the current interval.
                merged_x1 = max(merged_x1, ix1)
            else:
                # Gap found — finalise the previous interval.
                x_union_length += merged_x1 - merged_x0
                merged_x0, merged_x1 = ix0, ix1

        x_union_length += merged_x1 - merged_x0
        total_area += slab_h * x_union_length

    return total_area


# ---------------------------------------------------------------------------
# Backend-agnostic page adapter
# ---------------------------------------------------------------------------


class _PageAdapter:
    """
    Thin normalisation layer over pypdfium2 / PyMuPDF page objects.

    Exposes exactly what Stage 0 needs:
        .width, .height
        .get_text() → str
        .get_image_boxes() → list[(x0,y0,x1,y1)]

    This keeps the triage logic free of any backend-specific conditionals.
    """

    def __init__(self, raw_page, backend: str) -> None:
        self._page   = raw_page
        self._backend = backend

        if backend == "pypdfium2":
            w, h = raw_page.get_size()
        else:  # pymupdf
            rect = raw_page.rect
            w, h = rect.width, rect.height

        self.width  = float(w)
        self.height = float(h)

    def get_text(self) -> str:
        """Return all embedded text on the page as a plain string."""
        if self._backend == "pypdfium2":
            textpage = self._page.get_textpage()
            return textpage.get_text_bounded()
        else:
            return self._page.get_text("text")

    def get_image_boxes(self) -> list:
        """
        Return a list of (x0, y0, x1, y1) tuples for every image object
        on the page, clipped to the page boundary.
        """
        boxes: list[tuple[float, float, float, float]] = []

        if self._backend == "pypdfium2":
            # FPDF_PAGEOBJ_IMAGE == 3
            image_type = pdfium.raw.FPDF_PAGEOBJ_IMAGE
            for obj in self._page.get_objects():
                if obj.type == image_type:
                    try:
                        b = obj.get_bounds()
                        # get_bounds() returns (left, bottom, right, top) in PDF
                        # user-space (origin bottom-left); normalise to x0<x1, y0<y1.
                        raw_x0, raw_y0 = min(b[0], b[2]), min(b[1], b[3])
                        raw_x1, raw_y1 = max(b[0], b[2]), max(b[1], b[3])
                    except AttributeError:
                        # Fallback for pypdfium2 versions where get_bounds() is
                        # not available on PdfImage: derive the bounding box from
                        # the 2-D affine transformation matrix.
                        # Matrix (a, b, c, d, e, f) maps the unit square
                        # [0,1]×[0,1] (object space) to page space:
                        #   x' = a*x + c*y + e
                        #   y' = b*x + d*y + f
                        m = obj.get_matrix()
                        corners_x = [m.e, m.e + m.a, m.e + m.c, m.e + m.a + m.c]
                        corners_y = [m.f, m.f + m.b, m.f + m.d, m.f + m.b + m.d]
                        raw_x0, raw_y0 = min(corners_x), min(corners_y)
                        raw_x1, raw_y1 = max(corners_x), max(corners_y)

                    x0 = max(0.0, raw_x0)
                    x1 = min(self.width, raw_x1)
                    y0 = max(0.0, raw_y0)
                    y1 = min(self.height, raw_y1)
                    if x1 > x0 and y1 > y0:
                        boxes.append((x0, y0, x1, y1))

        else:  # pymupdf
            for img_info in self._page.get_images(full=True):
                # img_info: (xref, smask, width, height, bpc, cs, alt, name, filter, referencer)
                xref = img_info[0]
                rects = self._page.get_image_rects(xref)
                for r in rects:
                    x0 = max(0.0, float(r.x0))
                    y0 = max(0.0, float(r.y0))
                    x1 = min(self.width,  float(r.x1))
                    y1 = min(self.height, float(r.y1))
                    if x1 > x0 and y1 > y0:
                        boxes.append((x0, y0, x1, y1))

        return boxes


# ---------------------------------------------------------------------------
# Per-page triage
# ---------------------------------------------------------------------------


def _triage_page(adapter: _PageAdapter, page_num: int) -> PageIR:
    """
    Classify a single page and return a PageIR with triage_result set.

    No blocks are populated here — Stage 0 only classifies; extraction
    begins in Stage 1.
    """
    page_ir = PageIR(
        page_num=page_num,
        width=adapter.width,
        height=adapter.height,
    )

    page_area = adapter.width * adapter.height
    if page_area <= 0.0:
        page_ir.triage_result = "ocr_needed"
        page_ir.add_warning("Page has zero area; defaulting to ocr_needed.")
        return page_ir

    # ------------------------------------------------------------------ W2
    # Normalised text density instead of raw character count.
    raw_text = adapter.get_text().strip()
    text_len = len(raw_text)
    normalisation_factor = page_area / AREA_NORM_PT2
    text_density = text_len / normalisation_factor if normalisation_factor > 0 else 0.0

    # ------------------------------------------------------------------ W1
    # Image coverage via union bounding box area.
    image_boxes = adapter.get_image_boxes()
    image_union  = _union_area(image_boxes)
    image_coverage = min(1.0, image_union / page_area)

    logger.debug(
        "Page %d: text_density=%.2f, image_coverage=%.3f, "
        "text_len=%d, images=%d",
        page_num, text_density, image_coverage, text_len, len(image_boxes),
    )

    # ------------------------------------------------------------------ Decision
    if image_coverage >= IMAGE_COV_HIGH or (
        text_density < TEXT_DENSITY_SPARSE and image_coverage > IMAGE_COV_THRESHOLD
    ):
        triage_result = "ocr_needed"

    elif text_density >= TEXT_DENSITY_RICH and image_coverage <= IMAGE_COV_CLEAN:
        triage_result = "text_native"

    else:
        triage_result = "hybrid"

    page_ir.triage_result = triage_result
    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def triage_document(
    pdf_path: str,
    password: Optional[str] = None,
) -> DocumentIR:
    """
    Open *pdf_path* once and classify every page.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file on disk.
    password : str | None
        Decryption password.  Pass ``None`` if the document is not
        encrypted (or to detect encrypted documents — see W29 fix below).

    Returns
    -------
    DocumentIR
        A document IR whose pages have ``triage_result`` set.
        If the document could not be opened (e.g. encryption), the
        ``pages`` list is empty and ``warnings`` explains why.

    W29 fix — Encrypted PDF handling
    ----------------------------------
    pypdfium2 raises ``PdfiumError`` with the message containing
    "Incorrect password" when opening a password-protected PDF without
    a valid password.  We catch that specific error, record a structured
    warning in ``DocumentIR.warnings``, and return immediately with zero
    pages — never producing silent empty output.
    """
    doc_ir = DocumentIR()

    if not os.path.exists(pdf_path):
        doc_ir.add_warning(f"[W29] File not found: {pdf_path}")
        return doc_ir

    # ------------------------------------------------------------------ W29
    # Open the document ONCE.  Detect encryption failure immediately.
    try:
        if _BACKEND == "pypdfium2":
            doc = pdfium.PdfDocument(pdf_path, password=password)
        else:
            # PyMuPDF (fitz): open then authenticate separately — the
            # `password` keyword is not accepted by fitz.open() in older
            # versions, and passing it positionally can cause silent failures.
            doc = pdfium.open(pdf_path)
            if doc.is_encrypted:
                if not doc.authenticate(password or ""):
                    raise ValueError(
                        "PDF is encrypted and the supplied password is incorrect."
                    )

    except Exception as exc:
        exc_str = str(exc).lower()
        if "password" in exc_str or "encrypted" in exc_str or "incorrect" in exc_str:
            doc_ir.add_warning(
                f"[W29] PDF is encrypted and could not be decrypted "
                f"(password={'<none>' if password is None else '<provided>'}).  "
                f"No pages were processed.  Original error: {exc}"
            )
        else:
            doc_ir.add_warning(
                f"[W29] Failed to open PDF '{pdf_path}': {exc}"
            )
        return doc_ir

    # ------------------------------------------------------------------ Metadata
    try:
        if _BACKEND == "pypdfium2":
            meta = doc.get_metadata_dict()
            doc_ir.title = meta.get("Title") or None
        else:
            doc_ir.title = doc.metadata.get("title") or None
    except Exception:
        pass  # Metadata is best-effort; never block triage.

    # ------------------------------------------------------------------ Per-page loop
    # The open handle `doc` is reused across all pages — never re-opened.
    try:
        page_count = len(doc)
        for page_num in range(page_count):
            try:
                if _BACKEND == "pypdfium2":
                    raw_page = doc[page_num]
                else:
                    raw_page = doc.load_page(page_num)

                adapter  = _PageAdapter(raw_page, _BACKEND)
                page_ir  = _triage_page(adapter, page_num)
                doc_ir.pages.append(page_ir)

                if _BACKEND == "pypdfium2":
                    raw_page.close()

            except Exception as exc:
                logger.warning("Error triaging page %d: %s", page_num, exc)
                fallback = PageIR(page_num=page_num)
                fallback.triage_result = "ocr_needed"
                fallback.add_warning(
                    f"Triage failed; defaulted to ocr_needed. Error: {exc}"
                )
                doc_ir.pages.append(fallback)
                doc_ir.add_warning(
                    f"[Stage0] Page {page_num} triage error: {exc}"
                )

    finally:
        doc.close()

    logger.info(
        "Stage 0 complete: %d pages classified (%s)",
        len(doc_ir.pages),
        _summarise_triage(doc_ir),
    )
    return doc_ir


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _summarise_triage(doc_ir: DocumentIR) -> str:
    """Return a human-readable tally of triage results."""
    counts: dict[str, int] = {}
    for p in doc_ir.pages:
        counts[p.triage_result] = counts.get(p.triage_result, 0) + 1
    return ", ".join(f"{v}×{k}" for k, v in sorted(counts.items()))
