"""
pdf_engine/api.py
==================
Public API for the PDF extraction pipeline.

Usage
-----
::

    from pdf_engine import extract

    result = extract(
        "document.pdf",
        output_format="markdown",
        ocr_engine="auto",
        strip_headers=True,
        confidence_threshold=0.0,
        max_workers=4,
        reading_order_strategy="xy_cut",
        ocr_dpi=300,
        password=None,
    )

    print(result.text)        # formatted output string
    print(result.confidence)  # document-level confidence (mean of block finals)
    print(result.warnings)    # list of per-page warning strings
    result.ir                 # full DocumentIR
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Container returned by :func:`extract`."""

    text: str
    """Formatted output string (markdown, plain text, or empty on failure)."""

    ir: "DocumentIR"  # noqa: F821 — forward-ref avoids circular import at module level
    """The full ``DocumentIR`` produced by the pipeline."""

    confidence: float
    """Document-level confidence — arithmetic mean of all block ``final`` scores."""

    warnings: list[str] = field(default_factory=list)
    """Aggregated warning strings from every page and stage."""


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _heading_depth(block, block_by_id: dict, all_headings: list = None) -> int:
    """
    Compute the depth of a heading in the parent_id hierarchy.

    H1 (no parent) → depth 1, H2 → depth 2, etc., capped at 6.

    HIGH-6 fix: If parent_id chains aren't available (e.g., Stage 7 didn't run),
    fall back to font-size-based levels using bbox height.
    """
    depth = 1
    current = block
    for _ in range(5):  # max depth 6
        if not current.parent_id:
            break
        parent = block_by_id.get(current.parent_id)
        if parent is None:
            break
        depth += 1
        current = parent

    # HIGH-6 fix: If we're still at depth 1 and have no parent_id,
    # try to infer depth from bbox height relative to other headings
    if depth == 1 and not block.parent_id and all_headings and block.bbox:
        # Collect bbox heights of all headings with bboxes
        heights = [h.bbox.height for h in all_headings if h.bbox and h.bbox.height > 0]
        if heights and len(heights) > 1:
            heights_sorted = sorted(set(heights), reverse=True)  # larger = higher priority
            # Find which height tier this block belongs to
            block_height = block.bbox.height
            for i, h in enumerate(heights_sorted):
                if abs(block_height - h) < 2.0:  # within 2pt tolerance
                    depth = min(i + 1, 6)  # cap at H6
                    break

    return depth


def _format_markdown(doc_ir, strip_headers: bool, threshold: float) -> str:
    from pdf_engine.models import BlockType

    # Build a fast id→block index for heading depth traversal (BUG-13 fix).
    block_by_id: dict = {
        b.id: b
        for page in doc_ir.pages
        for b in page.blocks
    }

    # HIGH-6 fix: Collect all headings for font-size-based fallback
    all_headings = [
        b for page in doc_ir.pages
        for b in page.blocks
        if b.block_type == BlockType.HEADING
    ]

    lines: list[str] = []
    for page in doc_ir.pages:
        for block in page.blocks:
            if block.confidence.final < threshold:
                continue
            if strip_headers and block.block_type in (
                BlockType.HEADER, BlockType.FOOTER
            ):
                continue

            text = block.text.strip()
            if not text:
                continue

            if block.block_type == BlockType.HEADING:
                depth = _heading_depth(block, block_by_id, all_headings)
                hashes = "#" * depth
                lines.append(f"{hashes} {text}\n")
            elif block.block_type == BlockType.TABLE:
                lines.append(f"```\n{text}\n```\n")
            elif block.block_type == BlockType.CAPTION:
                lines.append(f"*{text}*\n")
            elif block.block_type == BlockType.FOOTNOTE:
                lines.append(f"> {text}\n")
            else:
                lines.append(f"{text}\n")

    return "\n".join(lines)


def _format_plain(doc_ir, strip_headers: bool, threshold: float) -> str:
    from pdf_engine.models import BlockType

    lines: list[str] = []
    for page in doc_ir.pages:
        for block in page.blocks:
            if block.confidence.final < threshold:
                continue
            if strip_headers and block.block_type in (
                BlockType.HEADER, BlockType.FOOTER
            ):
                continue

            text = block.text.strip()
            if text:
                lines.append(text)

    return "\n\n".join(lines)


def _collect_warnings(doc_ir) -> list[str]:
    warnings: list[str] = list(doc_ir.warnings)
    for page in doc_ir.pages:
        warnings.extend(page.warnings)
    return warnings


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline(
    pdf_path: str,
    ocr_engine: str,
    reading_order_strategy: str,
    ocr_dpi: int,
    password: Optional[str],
) -> object:  # returns DocumentIR
    """
    Execute Stages 0-9 in sequence and return the scored ``DocumentIR``.

    Actual entry-point names (verified against source)
    ---------------------------------------------------
    Stage 0  ``triage_document(pdf_path, password)``         stage0_triage.py
             Opens its own pypdfium2 handle; returns DocumentIR.
             Encrypted-PDF failures → empty doc_ir + warnings (no raise).

    Stage 1  ``extract_document_text(fitz_doc, doc_ir)``     stage1_extraction.py
             Needs a PyMuPDF fitz handle opened by the caller.

    Stage 2  ``run_stage2(doc_ir, fitz_doc)``                stage2_layout.py

    Stage 3  ``run_stage3(doc_ir, strategy)``                stage3_reading_order.py

    Stage 4  ``detect_tables(plumber_page, page_ir)``        stage4_tables.py
             Per-page; called inside a ``pdfplumber.open()`` context.

    Stage 5  ``run_stage5(doc_ir, fitz_doc, dpi)``           stage5_ocr.py

    Stage 6  ``run_stage6(doc_ir)``                          stage6_assembly.py

    Stage 7  ``run_stage7(doc_ir)``                          stage7_crosspage.py

    Stage 8  ``run_stage8(doc_ir)``                          stage8_postprocessing.py

    Stage 9  ``run_stage9(doc_ir)``                          stage9_confidence.py
    """

    # ------------------------------------------------------------------
    # Stage 0 — Triage
    # Opens & closes its own pypdfium2 handle internally.
    # Returns an empty doc_ir (with warnings) for encrypted/missing files.
    # ------------------------------------------------------------------
    from pdf_engine.stage0_triage import triage_document

    doc_ir = triage_document(pdf_path, password=password)

    if not doc_ir.pages:
        # Encrypted PDF or unreadable — skip remaining stages.
        logger.warning(
            "Stage 0 returned no pages for %r — skipping remaining stages.",
            pdf_path,
        )
        return doc_ir

    # ------------------------------------------------------------------
    # Open a fitz (PyMuPDF) handle — shared by Stages 1, 2, and 5.
    # ------------------------------------------------------------------
    import fitz  # PyMuPDF

    try:
        fitz_doc = fitz.open(pdf_path)
        if fitz_doc.is_encrypted:
            if not fitz_doc.authenticate(password or ""):
                doc_ir.warnings.append(
                    f"[api] fitz could not authenticate PDF with supplied password."
                )
                fitz_doc.close()
                return doc_ir
    except Exception as exc:
        logger.error("fitz.open failed for %r: %s", pdf_path, exc)
        doc_ir.warnings.append(f"[api] fitz could not open PDF: {exc}")
        return doc_ir

    try:
        # --------------------------------------------------------------
        # Stage 1 — Text layer extraction
        # --------------------------------------------------------------
        from pdf_engine.stage1_extraction import extract_document_text

        doc_ir = extract_document_text(fitz_doc, doc_ir)

        # --------------------------------------------------------------
        # Stage 2 — Layout analysis
        # --------------------------------------------------------------
        from pdf_engine.stage2_layout import run_stage2

        doc_ir = run_stage2(doc_ir, fitz_doc)

        # --------------------------------------------------------------
        # Stage 3 — Reading order
        # --------------------------------------------------------------
        from pdf_engine.stage3_reading_order import run_stage3

        doc_ir = run_stage3(doc_ir, strategy=reading_order_strategy)

        # --------------------------------------------------------------
        # Stage 4 — Table detection  (per-page, pdfplumber)
        # pdfplumber is optional; skip gracefully if not installed.
        # --------------------------------------------------------------
        try:
            import pdfplumber
            from pdf_engine.stage4_tables import detect_tables

            with pdfplumber.open(pdf_path, password=password) as plumber_doc:
                for page_ir in doc_ir.pages:
                    try:
                        plumber_page = plumber_doc.pages[page_ir.page_num]
                        detect_tables(plumber_page, page_ir)
                    except Exception as exc:
                        logger.warning(
                            "Stage 4 failed on page %d: %s", page_ir.page_num, exc
                        )
                        page_ir.warnings.append(f"[Stage4] {exc}")

        except ImportError:
            logger.warning(
                "pdfplumber not installed — skipping explicit table detection."
            )
            doc_ir.warnings.append(
                "[api] pdfplumber unavailable; explicit table detection skipped."
            )
        except Exception as exc:
            logger.warning("Stage 4 failed: %s", exc)
            doc_ir.warnings.append(f"[Stage4] {exc}")

        # --------------------------------------------------------------
        # Stage 5 — OCR
        # --------------------------------------------------------------
        from pdf_engine.stage5_ocr import run_stage5

        doc_ir = run_stage5(doc_ir, fitz_doc, dpi=ocr_dpi)

    except KeyboardInterrupt:
        # STAB-4 fix: ensure fitz handle is closed cleanly on Ctrl+C, then
        # re-raise so the caller / CLI can handle the interruption gracefully.
        logger.warning("Extraction interrupted by user (Ctrl+C) — cleaning up.")
        fitz_doc.close()
        raise
    finally:
        # Runs for both normal completion and unhandled exceptions (but NOT
        # after the KeyboardInterrupt branch above, which already closed it).
        try:
            fitz_doc.close()
        except Exception:
            pass  # already closed in the except KeyboardInterrupt branch

    # ------------------------------------------------------------------
    # Stage 6 — Block assembly  (no fitz required)
    # ------------------------------------------------------------------
    from pdf_engine.stage6_assembly import run_stage6

    doc_ir = run_stage6(doc_ir)

    # ------------------------------------------------------------------
    # Stage 7 — Cross-page analysis
    # ------------------------------------------------------------------
    from pdf_engine.stage7_crosspage import run_stage7

    doc_ir = run_stage7(doc_ir)

    # ------------------------------------------------------------------
    # Stage 8 — Post-processing
    # ------------------------------------------------------------------
    from pdf_engine.stage8_postprocessing import run_stage8

    doc_ir = run_stage8(doc_ir)

    # ------------------------------------------------------------------
    # Stage 9 — Confidence scoring
    # ------------------------------------------------------------------
    from pdf_engine.stage9_confidence import run_stage9

    doc_ir = run_stage9(doc_ir)

    return doc_ir


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract(
    pdf_path: str,
    output_format: str = "markdown",
    ocr_engine: str = "auto",
    strip_headers: bool = True,
    confidence_threshold: float = 0.0,
    max_workers: int = 4,
    reading_order_strategy: str = "xy_cut",
    ocr_dpi: int = 300,
    password: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract text from a PDF document using the full 9-stage pipeline.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the PDF file.
    output_format:
        One of ``"markdown"``, ``"plain"``, or ``"ir"``.
        Defaults to ``"markdown"`` for structured output.
        ``"ir"`` returns an empty ``text`` string; inspect ``result.ir``
        directly.
    ocr_engine:
        ``"auto"`` selects the best available engine.  Pass ``"tesseract"``
        or ``"surya"`` to force a specific backend.  Note: Stage 5 manages
        engine selection internally; this parameter is reserved for future
        forwarding.
    strip_headers:
        When ``True``, ``HEADER`` and ``FOOTER`` blocks are excluded from
        formatted output (but remain in ``result.ir``).
    confidence_threshold:
        Blocks with ``final`` confidence below this value are excluded from
        formatted output.  ``0.0`` includes everything.
    max_workers:
        Kept for API stability; Stage 5 manages its own concurrency.
    reading_order_strategy:
        ``"xy_cut"`` (default) or ``"band_based"``.
    ocr_dpi:
        Rasterisation DPI forwarded to Stage 5.
    password:
        Decryption password for encrypted PDFs.  ``None`` for unprotected files.

    Returns
    -------
    ExtractionResult

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist on disk.
    ValueError
        If *output_format* is not one of the three supported values.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    supported_formats = {"markdown", "plain", "ir"}
    if output_format not in supported_formats:
        raise ValueError(
            f"output_format must be one of {sorted(supported_formats)!r}, "
            f"got {output_format!r}"
        )

    doc_ir = _run_pipeline(
        pdf_path=pdf_path,
        ocr_engine=ocr_engine,
        reading_order_strategy=reading_order_strategy,
        ocr_dpi=ocr_dpi,
        password=password,
    )

    if output_format == "markdown":
        text = _format_markdown(doc_ir, strip_headers, confidence_threshold)
    elif output_format == "plain":
        text = _format_plain(doc_ir, strip_headers, confidence_threshold)
    else:  # "ir"
        text = ""

    warnings = _collect_warnings(doc_ir)

    return ExtractionResult(
        text=text,
        ir=doc_ir,
        confidence=doc_ir.confidence,
        warnings=warnings,
    )
