"""
pdf_engine/stage6_assembly.py
==============================
Stage 6 — Block Assembly + De-duplication

Responsibility
--------------
Merge pymupdf text extraction output (Stages 1–5) with surya layout output
(Stage 2), resolving the granularity mismatch between fine-grained pymupdf
line blocks and coarse surya paragraph regions.

De-duplication strategy (center-containment merging)
------------------------------------------------------
For each surya block:
  1. Find all pymupdf blocks whose CENTER point falls inside the surya bbox.
  2. If avg pymupdf text_quality > quality_threshold (default 0.7):
       - Use pymupdf text content (accurate character-level text)
       - Use surya bbox + block_type (accurate layout understanding)
       - Set extraction_method = PYMUPDF_DIRECT, preserve type from surya
  3. If avg pymupdf quality <= threshold:
       - Use surya block text if available, else flag for Stage 5 re-OCR
       - Add warning to page_ir
  4. pymupdf blocks not contained in any surya block: keep as-is
  5. surya blocks with no pymupdf blocks inside: keep for OCR

Key insight: A pymupdf line bbox (height ≈ 12pt) inside a surya paragraph
bbox (height ≈ 150pt) has IoU ≈ 0.08, but its center is clearly inside the
surya block — containment correctly handles this mismatch.

Design rules
------------
* Never re-opens the PDF — operates purely on in-memory IR objects.
* All returned objects are typed TextBlock / PageIR instances.
* extraction_method reflects the actual text source (pymupdf vs surya).
* Blocks flagged for re-OCR carry a page-level warning.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

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

#: Minimum average text_quality for pymupdf content to be preferred over surya.
DEFAULT_QUALITY_THRESHOLD: float = 0.7

#: Sentinel extraction method to indicate a block needs Stage 5 re-OCR.
#: We reuse SURYA_LAYOUT to mark surya-sourced blocks awaiting OCR.
_NEEDS_OCR_WARNING_PREFIX = "stage6:needs_ocr"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _center(bbox: BBox) -> tuple[float, float]:
    """Return the (cx, cy) center point of a bounding box."""
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def _center_inside(point_bbox: BBox, container: BBox) -> bool:
    """
    Return True if the CENTER of ``point_bbox`` falls inside ``container``
    (inclusive boundaries).
    """
    cx, cy = _center(point_bbox)
    return (
        container.x0 <= cx <= container.x1
        and container.y0 <= cy <= container.y1
    )


# ---------------------------------------------------------------------------
# Core de-duplication logic
# ---------------------------------------------------------------------------


def _average_text_quality(blocks: list[TextBlock]) -> float:
    """Compute the mean text_quality across a list of blocks."""
    if not blocks:
        return 0.0
    return sum(b.confidence.text_quality for b in blocks) / len(blocks)


def _merge_pymupdf_into_surya(
    surya_block: TextBlock,
    contained_pymupdf: list[TextBlock],
    quality_threshold: float,
    page_ir: PageIR,
) -> TextBlock:
    """
    Produce a single merged TextBlock from a surya region and the pymupdf
    lines whose centers fall inside it.

    Strategy is determined by average pymupdf text quality:
      * High quality → use pymupdf text, surya geometry + block type.
      * Low quality  → use surya text (or flag for re-OCR).
    """
    avg_quality = _average_text_quality(contained_pymupdf)

    if avg_quality > quality_threshold:
        # Sort contained lines top-to-bottom, then left-to-right
        sorted_lines = sorted(
            contained_pymupdf,
            key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0),
        )
        merged_text = "\n".join(b.text for b in sorted_lines if b.text).strip()

        merged = TextBlock(
            id=str(uuid.uuid4()),
            text=merged_text,
            bbox=surya_block.bbox,                      # surya geometry
            block_type=surya_block.block_type,          # surya label
            extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
            confidence=BlockConfidence(
                text_quality=avg_quality,
                type_quality=surya_block.confidence.type_quality,
                order_quality=surya_block.confidence.order_quality,
                method_score=1.0,
            ),
            page_num=surya_block.page_num,
            language=contained_pymupdf[0].language if contained_pymupdf else None,
            script_direction=contained_pymupdf[0].script_direction
                             if contained_pymupdf else "ltr",
        )
    else:
        # Low-quality pymupdf coverage — prefer surya text or flag for re-OCR
        surya_text = surya_block.text.strip() if surya_block.text else ""

        if not surya_text:
            # No usable text at all — flag for Stage 5 re-OCR
            page_ir.add_warning(
                f"{_NEEDS_OCR_WARNING_PREFIX} page={surya_block.page_num} "
                f"block_id=<pending> bbox={surya_block.bbox}"
            )
            logger.info(
                "Page %d: surya block at %s has no text and low pymupdf quality "
                "(%.2f) — flagged for re-OCR",
                surya_block.page_num, surya_block.bbox, avg_quality,
            )

        merged = TextBlock(
            id=str(uuid.uuid4()),
            text=surya_text,
            bbox=surya_block.bbox,
            block_type=surya_block.block_type,
            extraction_method=ExtractionMethod.SURYA_LAYOUT,
            confidence=BlockConfidence(
                text_quality=avg_quality,
                type_quality=surya_block.confidence.type_quality,
                order_quality=surya_block.confidence.order_quality,
                method_score=0.6,  # lower trust — OCR path or low quality
            ),
            page_num=surya_block.page_num,
            script_direction="ltr",
        )

        # Patch the warning with the real block ID now we have it
        if not surya_text and page_ir.warnings:
            old = page_ir.warnings[-1]
            page_ir.warnings[-1] = old.replace("block_id=<pending>",
                                                f"block_id={merged.id}")

    return merged


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------


def _partition_blocks(
    blocks: list[TextBlock],
) -> tuple[list[TextBlock], list[TextBlock]]:
    """
    Split a page's block list into:
      (pymupdf_blocks, surya_blocks)

    pymupdf blocks: extraction_method == PYMUPDF_DIRECT or TESSERACT_OCR
    surya blocks:   extraction_method == SURYA_LAYOUT or SURYA_OCR
    """
    pymupdf_methods = {ExtractionMethod.PYMUPDF_DIRECT, ExtractionMethod.TESSERACT_OCR}
    surya_methods   = {ExtractionMethod.SURYA_LAYOUT,   ExtractionMethod.SURYA_OCR}

    pymupdf_blocks: list[TextBlock] = []
    surya_blocks:   list[TextBlock] = []

    for b in blocks:
        if b.extraction_method in pymupdf_methods:
            pymupdf_blocks.append(b)
        elif b.extraction_method in surya_methods:
            surya_blocks.append(b)
        else:
            # Unknown method — treat as pymupdf (keep as-is)
            pymupdf_blocks.append(b)

    return pymupdf_blocks, surya_blocks


# ---------------------------------------------------------------------------
# Main de-duplication function (public API)
# ---------------------------------------------------------------------------


def deduplicate_blocks(
    pymupdf_blocks: list[TextBlock],
    surya_blocks: list[TextBlock],
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    page_ir: Optional[PageIR] = None,
) -> list[TextBlock]:
    """
    Merge pymupdf and surya blocks using center-containment logic.

    For each surya block:
      1. Find all pymupdf blocks whose CENTER point falls inside the surya bbox.
      2. If avg pymupdf quality > threshold:
           - Use pymupdf text content (accurate)
           - Use surya bbox + type label (accurate layout)
      3. If avg pymupdf quality <= threshold:
           - Use surya OCR text (or flag for Stage 5 re-OCR)
      4. pymupdf blocks not contained in any surya block: keep as-is
      5. surya blocks with no pymupdf blocks inside: keep for OCR

    Parameters
    ----------
    pymupdf_blocks:
        Blocks from the native text layer (Stages 1/5, PYMUPDF_DIRECT or
        TESSERACT_OCR extraction methods).
    surya_blocks:
        Blocks from surya layout analysis (Stage 2, SURYA_LAYOUT method).
    quality_threshold:
        Minimum average text_quality for pymupdf content to win.
    page_ir:
        Optional PageIR used to record warnings about blocks needing re-OCR.
        A dummy PageIR is created internally if None is passed.

    Returns
    -------
    list[TextBlock]
        Merged block list with no duplicates.  Order: merged/surya blocks
        (sorted top-to-bottom) followed by unmatched pymupdf blocks.
    """
    if page_ir is None:
        page_num = pymupdf_blocks[0].page_num if pymupdf_blocks else (
                   surya_blocks[0].page_num if surya_blocks else 0)
        page_ir = PageIR(page_num=page_num)

    # Track which pymupdf blocks have been "consumed" by a surya region
    consumed_pymupdf_ids: set[str] = set()
    merged_blocks: list[TextBlock] = []

    for surya_blk in surya_blocks:
        if surya_blk.bbox is None:
            # Can't do containment without a bbox — keep surya block as-is
            merged_blocks.append(surya_blk)
            continue

        # --- 1. Find pymupdf blocks whose center is inside this surya bbox --
        # Skip blocks already claimed by a previously processed surya region.
        contained: list[TextBlock] = []
        for pm_blk in pymupdf_blocks:
            if pm_blk.id in consumed_pymupdf_ids:
                continue
            if pm_blk.bbox is None:
                continue
            if _center_inside(pm_blk.bbox, surya_blk.bbox):
                contained.append(pm_blk)

        if contained:
            # Mark these pymupdf blocks as consumed
            for pm_blk in contained:
                consumed_pymupdf_ids.add(pm_blk.id)

            # --- 2/3. Merge based on quality -----------------------------
            merged = _merge_pymupdf_into_surya(
                surya_blk, contained, quality_threshold, page_ir
            )
            merged_blocks.append(merged)

        else:
            # --- 5. surya block with no pymupdf coverage -----------------
            if not surya_blk.text:
                page_ir.add_warning(
                    f"{_NEEDS_OCR_WARNING_PREFIX} page={surya_blk.page_num} "
                    f"block_id={surya_blk.id} bbox={surya_blk.bbox}"
                )
                logger.info(
                    "Page %d: surya block at %s has no pymupdf coverage and "
                    "no text — flagged for re-OCR",
                    surya_blk.page_num, surya_blk.bbox,
                )
            merged_blocks.append(surya_blk)

    # --- 4. Keep unmatched pymupdf blocks (not inside any surya region) -----
    unmatched_pymupdf = [
        b for b in pymupdf_blocks if b.id not in consumed_pymupdf_ids
    ]
    if unmatched_pymupdf:
        logger.debug(
            "Page: %d pymupdf block(s) not contained in any surya region — "
            "keeping as-is",
            len(unmatched_pymupdf),
        )

    # Sort merged (surya-aligned) blocks top-to-bottom
    merged_blocks.sort(
        key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0)
    )

    return merged_blocks + unmatched_pymupdf


# ---------------------------------------------------------------------------
# Per-page assembly
# ---------------------------------------------------------------------------


def assemble_page(
    page_ir: PageIR,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> PageIR:
    """
    Run block assembly on a single page.

    Partitions ``page_ir.blocks`` into pymupdf and surya subsets, runs
    center-containment de-duplication, and replaces ``page_ir.blocks``
    with the merged result.

    Parameters
    ----------
    page_ir:
        PageIR whose blocks contain a mix of PYMUPDF_DIRECT and SURYA_LAYOUT
        blocks (as produced by Stages 1–5).
    quality_threshold:
        Forwarded to ``deduplicate_blocks``.

    Returns
    -------
    PageIR
        The same object, mutated in-place, then returned for chaining.
    """
    pymupdf_blocks, surya_blocks = _partition_blocks(page_ir.blocks)

    if not surya_blocks:
        # Nothing to merge against — keep all blocks as-is
        logger.debug(
            "Page %d: no surya blocks — skipping assembly", page_ir.page_num
        )
        return page_ir

    if not pymupdf_blocks:
        # No pymupdf text — flag all surya blocks without text for re-OCR
        for blk in surya_blocks:
            if not blk.text:
                page_ir.add_warning(
                    f"{_NEEDS_OCR_WARNING_PREFIX} page={blk.page_num} "
                    f"block_id={blk.id} bbox={blk.bbox}"
                )
        return page_ir

    merged = deduplicate_blocks(
        pymupdf_blocks=pymupdf_blocks,
        surya_blocks=surya_blocks,
        quality_threshold=quality_threshold,
        page_ir=page_ir,
    )
    page_ir.blocks = merged
    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage6(
    doc_ir: DocumentIR,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> DocumentIR:
    """
    Run block assembly and de-duplication across all pages.

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 5.  Each ``PageIR.blocks`` list
        should contain a mix of PYMUPDF_DIRECT (or TESSERACT_OCR) blocks from
        the text layer and SURYA_LAYOUT blocks from the layout model.
    quality_threshold:
        Minimum average text_quality for pymupdf content to take precedence.

    Returns
    -------
    DocumentIR
        The same object, with every ``PageIR.blocks`` list replaced by the
        merged, de-duplicated block set.
    """
    for page_ir in doc_ir.pages:
        try:
            assemble_page(page_ir, quality_threshold=quality_threshold)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Page %d: stage 6 assembly failed: %s",
                page_ir.page_num, exc, exc_info=True,
            )
            page_ir.add_warning(
                f"Page {page_ir.page_num}: stage6 assembly error — {exc}"
            )

    # Validate parent_id integrity
    errors = doc_ir.validate_parent_refs()
    for err in errors:
        logger.error("Parent-ref integrity error after stage 6: %s", err)
        doc_ir.add_warning(err)

    return doc_ir
