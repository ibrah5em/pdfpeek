"""
pdf_engine/stage9_confidence.py
================================
Stage 9 — Confidence Scoring

Responsibility
--------------
Compute a meaningful final confidence score for every TextBlock and an
aggregate document-level confidence score.

Design rules enforced here
--------------------------
* text_quality and method_score are SEPARATE dimensions — never multiplied
  before entering the geometric mean (fixes W26).
* order_quality and type_quality are READ from each block as set by
  Stages 3 and 2 respectively — never replaced with hard-coded defaults
  (fixes W25).
* Final score is the geometric mean of all four dimensions.
* A perfect Tesseract block scores (1.0 × 0.70 × 0.8 × 0.7)^0.25 ≈ 0.84,
  safely above the 0.75 RAG threshold (W26 validation).
* This stage never re-opens the PDF.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from pdf_engine.models import (
    BlockConfidence,
    DocumentIR,
    ExtractionMethod,
    PageIR,
    TextBlock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method trust table
# ---------------------------------------------------------------------------

METHOD_TRUST: dict[ExtractionMethod, float] = {
    ExtractionMethod.PYMUPDF_DIRECT: 1.00,
    ExtractionMethod.SURYA_LAYOUT:   0.90,
    ExtractionMethod.SURYA_OCR:      0.80,
    ExtractionMethod.TESSERACT_OCR:  0.70,
}

_DEFAULT_METHOD_TRUST = 0.70


# ---------------------------------------------------------------------------
# Block-level confidence
# ---------------------------------------------------------------------------


def compute_block_confidence(block: TextBlock) -> BlockConfidence:
    """
    Compute and return a fully-populated ``BlockConfidence`` for *block*.

    The four dimensions are:

    text_quality
        Fraction of printable characters, discounted by Unicode replacement
        characters (U+FFFD).  Computed purely from character-level signals —
        **never** pre-multiplied by method trust (fixes W26).

    method_score
        Lookup from ``METHOD_TRUST``; falls back to 0.70 for unknown methods.

    order_quality
        Read directly from ``block.confidence.order_quality`` as set by
        Stage 3.  Never overwritten with a hard-coded default (fixes W25).

    type_quality
        Read directly from ``block.confidence.type_quality`` as set by
        Stage 2.  Never overwritten with a hard-coded default (fixes W25).

    final
        Geometric mean of the four dimensions above:
        ``(text_quality × method_score × order_quality × type_quality) ^ 0.25``

    Parameters
    ----------
    block:
        A ``TextBlock`` whose ``confidence.order_quality`` and
        ``confidence.type_quality`` have already been set by earlier stages.

    Returns
    -------
    BlockConfidence
        A new ``BlockConfidence`` instance; the caller is responsible for
        assigning it back to ``block.confidence``.
    """
    text = block.text or ""

    # text_quality: reuse Stage 1's language-aware score when it was already
    # set (non-default value).  Stage 1 uses a sophisticated formula that
    # accounts for CJK space ratios and Unicode replacement chars (W5 fix).
    # Only recompute with the simple formula for blocks that never had their
    # text_quality set (default is 1.0, but a block with quality=1.0 from
    # Stage 1 is valid — so we only override when the block is from OCR stages
    # which may not have had language-aware scoring applied yet).
    if block.confidence.text_quality < 1.0:
        # Non-default value from an upstream stage — trust it.
        text_quality = block.confidence.text_quality
    elif not text:
        text_quality = 1.0
    else:
        # Compute a baseline quality score for blocks that still have the
        # default (1.0).  This acts as a safety net for OCR blocks.
        length = len(text)
        printable_count = sum(1 for c in text if c.isprintable())
        replacement_count = text.count("\ufffd")
        printable_ratio = printable_count / length
        replacement_ratio = replacement_count / length
        text_quality = printable_ratio * (1.0 - replacement_ratio)

    # method_score: extraction method trust
    method_score = METHOD_TRUST.get(block.extraction_method, _DEFAULT_METHOD_TRUST)

    # order_quality and type_quality: set by upstream stages — read, never override
    order_quality = block.confidence.order_quality
    type_quality = block.confidence.type_quality

    # final: geometric mean of the four independent dimensions (fixes W26)
    product = text_quality * method_score * order_quality * type_quality
    final = product ** 0.25 if product > 0 else 0.0

    return BlockConfidence(
        text_quality=text_quality,
        method_score=method_score,
        order_quality=order_quality,
        type_quality=type_quality,
        final=final,
    )


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------


def score_page(page_ir: PageIR) -> PageIR:
    """
    Apply ``compute_block_confidence`` to every block on *page_ir*.

    Mutates each block's ``confidence`` in-place and returns the page.
    """
    for block in page_ir.blocks:
        block.confidence = compute_block_confidence(block)
    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage9(doc_ir: DocumentIR) -> DocumentIR:
    """
    Run confidence scoring across the entire document.

    After scoring every block, the document-level ``confidence`` is set to
    the mean of all block ``final`` scores (or 0.0 for empty documents).

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 8.

    Returns
    -------
    DocumentIR
        The same object, with all block confidences populated and
        ``doc_ir.confidence`` set.
    """
    finals: list[float] = []

    for page_ir in doc_ir.pages:
        score_page(page_ir)
        for block in page_ir.blocks:
            finals.append(block.confidence.final)

    doc_ir.confidence = (sum(finals) / len(finals)) if finals else 0.0

    logger.info(
        "Stage 9 complete: scored %d blocks across %d pages; "
        "document confidence = %.4f",
        len(finals),
        len(doc_ir.pages),
        doc_ir.confidence,
    )
    return doc_ir
