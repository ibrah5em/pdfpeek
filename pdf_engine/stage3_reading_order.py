"""
pdf_engine/stage3_reading_order.py
====================================
Stage 3 — Reading Order Resolution

Responsibility
--------------
Sort the ``TextBlock`` objects in every ``PageIR`` into the correct human
reading sequence using recursive XY-cut partitioning, with a band-based
fallback for pages where no clean cut can be found.

After ordering, every block's ``confidence.order_quality`` is set based on
the clarity of the cuts used to sort it.

Design rules enforced here
--------------------------
* ``order_quality`` is set here (Stage 3) and nowhere else upstream.
* ``final`` is left to Stage 9.
* ``script_direction`` is always passed explicitly — never assumed.
* Every block appears in exactly ONE output partition (no duplicates).
* Straddling blocks go to whichever side has the greater overlap fraction.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from pdf_engine.models import BBox, DocumentIR, PageIR, TextBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Number of y-axis histogram bins for finding horizontal cuts.
Y_BINS: int = 50

#: Number of x-axis histogram bins for finding vertical cuts.
X_BINS: int = 50

#: Minimum fraction of page width that a gap must span to count as a cut.
MIN_GAP_FRACTION: float = 0.01

#: Minimum overlap fraction for a block to be considered a "true straddler".
STRADDLE_THRESHOLD: float = 0.2

#: order_quality assigned when we fall back to top-to-bottom sort.
FALLBACK_ORDER_QUALITY: float = 0.5

#: order_quality when a clean gap cut is found (scales with gap clarity).
MAX_ORDER_QUALITY: float = 0.95


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _block_height(block: TextBlock) -> float:
    if block.bbox is None:
        return 0.0
    return max(0.0, block.bbox.y1 - block.bbox.y0)


def _block_width(block: TextBlock) -> float:
    if block.bbox is None:
        return 0.0
    return max(0.0, block.bbox.x1 - block.bbox.x0)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping [lo, hi] intervals and return sorted non-overlapping list."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = [sorted_ivs[0]]
    for lo, hi in sorted_ivs[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _union_length(intervals: list[tuple[float, float]]) -> float:
    return sum(hi - lo for lo, hi in _merge_intervals(intervals))


# ---------------------------------------------------------------------------
# Partitioning (exclusive — every block goes to exactly one side)
# ---------------------------------------------------------------------------


def partition_at_cut(
    blocks: list[TextBlock],
    cut_coord: float,
    axis: str,  # "x" | "y"
) -> tuple[list[TextBlock], list[TextBlock]]:
    """
    Split ``blocks`` at ``cut_coord`` along ``axis``.

    Rules
    -----
    * For each block compute ``overlap_before`` and ``overlap_after`` fractions
      relative to the block's own extent along the axis.
    * If both fractions ≥ STRADDLE_THRESHOLD → true straddler: assign to the
      side with the greater fraction.  Ties go to "before".
    * All other blocks → majority overlap (or whichever side has any overlap).
    * Every block appears in exactly one output list.

    Returns
    -------
    (before, after)  where *before* is the lower-coordinate partition.
    """
    before: list[TextBlock] = []
    after: list[TextBlock] = []

    for block in blocks:
        if block.bbox is None:
            before.append(block)
            continue

        if axis == "y":
            lo, hi = block.bbox.y0, block.bbox.y1
        else:
            lo, hi = block.bbox.x0, block.bbox.x1

        extent = hi - lo
        if extent <= 0:
            # Zero-size block: assign purely by position relative to cut
            (before if lo <= cut_coord else after).append(block)
            continue

        overlap_before = max(0.0, min(cut_coord, hi) - lo) / extent
        overlap_after  = max(0.0, hi - max(lo, cut_coord))  / extent

        if overlap_before >= STRADDLE_THRESHOLD and overlap_after >= STRADDLE_THRESHOLD:
            # True straddler — pick dominant side
            (before if overlap_before >= overlap_after else after).append(block)
        else:
            (before if overlap_before >= overlap_after else after).append(block)

    return before, after


# ---------------------------------------------------------------------------
# Cut finders
# ---------------------------------------------------------------------------


def find_best_horizontal_cut(
    blocks: list[TextBlock],
    region_y0: float,
    region_y1: float,
    region_width: float,
) -> Optional[float]:
    """
    Search for the y-coordinate of the best horizontal whitespace gap.

    Uses a union-based coverage model: at each bin, the coverage is the
    total length of merged [x0, x1] intervals of all blocks crossing that bin
    — so coverage never exceeds ``region_width``.

    Returns the cut y-coordinate, or None when no gap is found.
    """
    if not blocks or region_y1 <= region_y0 or region_width <= 0:
        return None

    height = region_y1 - region_y0
    bin_size = height / Y_BINS

    # Build per-bin interval lists
    bin_intervals: list[list[tuple[float, float]]] = [[] for _ in range(Y_BINS)]
    for block in blocks:
        if block.bbox is None:
            continue
        b_y0, b_y1 = block.bbox.y0, block.bbox.y1
        b_x0, b_x1 = block.bbox.x0, block.bbox.x1
        if b_x1 <= b_x0:
            continue
        # Find bins this block overlaps
        bin_lo = max(0, int((b_y0 - region_y0) / bin_size))
        bin_hi = min(Y_BINS - 1, int((b_y1 - region_y0) / bin_size))
        for b in range(bin_lo, bin_hi + 1):
            bin_intervals[b].append((b_x0, b_x1))

    # coverage per bin (union-based)
    coverage = [
        _union_length(bin_intervals[b]) / region_width
        for b in range(Y_BINS)
    ]

    # Find the best zero-coverage gap bin
    best_cut: Optional[float] = None
    best_clarity: float = 0.0

    for b in range(1, Y_BINS - 1):
        if coverage[b] > MIN_GAP_FRACTION:
            continue
        # Measure clarity as difference in coverage on either side
        above_cov = max(coverage[max(0, b - 1)], coverage[max(0, b - 2)])
        below_cov = max(coverage[min(Y_BINS - 1, b + 1)], coverage[min(Y_BINS - 1, b + 2)])
        clarity = (above_cov + below_cov) / 2.0
        if clarity > best_clarity:
            best_clarity = clarity
            best_cut = region_y0 + (b + 0.5) * bin_size

    return best_cut


def find_best_vertical_cut(
    blocks: list[TextBlock],
    region_x0: float,
    region_x1: float,
    region_height: float,
) -> Optional[float]:
    """
    Search for the x-coordinate of the best vertical whitespace gap.

    Returns the cut x-coordinate, or None when no gap is found.
    """
    if not blocks or region_x1 <= region_x0 or region_height <= 0:
        return None

    width = region_x1 - region_x0
    bin_size = width / X_BINS

    bin_intervals: list[list[tuple[float, float]]] = [[] for _ in range(X_BINS)]
    for block in blocks:
        if block.bbox is None:
            continue
        b_x0, b_x1 = block.bbox.x0, block.bbox.x1
        b_y0, b_y1 = block.bbox.y0, block.bbox.y1
        if b_y1 <= b_y0:
            continue
        bin_lo = max(0, int((b_x0 - region_x0) / bin_size))
        bin_hi = min(X_BINS - 1, int((b_x1 - region_x0) / bin_size))
        for b in range(bin_lo, bin_hi + 1):
            bin_intervals[b].append((b_y0, b_y1))

    coverage = [
        _union_length(bin_intervals[b]) / region_height
        for b in range(X_BINS)
    ]

    best_cut: Optional[float] = None
    best_clarity: float = 0.0

    for b in range(1, X_BINS - 1):
        if coverage[b] > MIN_GAP_FRACTION:
            continue
        above_cov = max(coverage[max(0, b - 1)], coverage[max(0, b - 2)])
        below_cov = max(coverage[min(X_BINS - 1, b + 1)], coverage[min(X_BINS - 1, b + 2)])
        clarity = (above_cov + below_cov) / 2.0
        if clarity > best_clarity:
            best_clarity = clarity
            best_cut = region_x0 + (b + 0.5) * bin_size

    return best_cut


def _find_vertical_cut_with_clarity(
    blocks: list[TextBlock],
    region_x0: float,
    region_x1: float,
    region_height: float,
) -> tuple[Optional[float], float]:
    """Wrapper returning (cut_x, clarity) pair."""
    best_cut: Optional[float] = None
    best_clarity: float = 0.0

    if not blocks or region_x1 <= region_x0 or region_height <= 0:
        return best_cut, best_clarity

    width = region_x1 - region_x0
    bin_size = width / X_BINS

    bin_intervals: list[list[tuple[float, float]]] = [[] for _ in range(X_BINS)]
    for block in blocks:
        if block.bbox is None:
            continue
        b_x0, b_x1 = block.bbox.x0, block.bbox.x1
        b_y0, b_y1 = block.bbox.y0, block.bbox.y1
        if b_y1 <= b_y0:
            continue
        bin_lo = max(0, int((b_x0 - region_x0) / bin_size))
        bin_hi = min(X_BINS - 1, int((b_x1 - region_x0) / bin_size))
        for b in range(bin_lo, bin_hi + 1):
            bin_intervals[b].append((b_y0, b_y1))

    coverage = [
        _union_length(bin_intervals[b]) / region_height
        for b in range(X_BINS)
    ]

    for b in range(1, X_BINS - 1):
        if coverage[b] > MIN_GAP_FRACTION:
            continue
        above_cov = max(coverage[max(0, b - 1)], coverage[max(0, b - 2)])
        below_cov = max(coverage[min(X_BINS - 1, b + 1)], coverage[min(X_BINS - 1, b + 2)])
        clarity = (above_cov + below_cov) / 2.0
        if clarity > best_clarity:
            best_clarity = clarity
            best_cut = region_x0 + (b + 0.5) * bin_size

    return best_cut, best_clarity


# ---------------------------------------------------------------------------
# Recursive XY-cut
# ---------------------------------------------------------------------------


def _bounding_box(blocks: list[TextBlock]) -> Optional[BBox]:
    """Compute bounding box enclosing all blocks."""
    xs0 = [b.bbox.x0 for b in blocks if b.bbox]
    ys0 = [b.bbox.y0 for b in blocks if b.bbox]
    xs1 = [b.bbox.x1 for b in blocks if b.bbox]
    ys1 = [b.bbox.y1 for b in blocks if b.bbox]
    if not xs0:
        return None
    return BBox(min(xs0), min(ys0), max(xs1), max(ys1))


def _xy_cut_recursive(
    blocks: list[TextBlock],
    script_direction: str,
    depth: int = 0,
    max_depth: int = 20,
) -> list[TextBlock]:
    """
    Recursively sort ``blocks`` using XY-cut partitioning.

    Returns blocks in reading order.
    """
    if len(blocks) <= 1 or depth >= max_depth:
        return list(blocks)

    bbox = _bounding_box(blocks)
    if bbox is None:
        return list(blocks)

    region_width  = bbox.x1 - bbox.x0
    region_height = bbox.y1 - bbox.y0

    # --- Try horizontal cut first (splits top/bottom) -----------------
    h_cut = find_best_horizontal_cut(
        blocks, bbox.y0, bbox.y1, region_width
    )
    if h_cut is not None:
        top, bottom = partition_at_cut(blocks, h_cut, axis="y")
        if top and bottom:
            ordered_top    = _xy_cut_recursive(top,    script_direction, depth + 1, max_depth)
            ordered_bottom = _xy_cut_recursive(bottom, script_direction, depth + 1, max_depth)
            return ordered_top + ordered_bottom

    # --- Try vertical cut (splits left/right columns) -----------------
    v_cut, clarity = _find_vertical_cut_with_clarity(
        blocks, bbox.x0, bbox.x1, region_height
    )
    if v_cut is not None:
        left, right = partition_at_cut(blocks, v_cut, axis="x")
        if left and right:
            # RTL: process right column first
            if script_direction == "rtl":
                col_a, col_b = right, left
            else:
                col_a, col_b = left, right

            ordered_a = _xy_cut_recursive(col_a, script_direction, depth + 1, max_depth)
            ordered_b = _xy_cut_recursive(col_b, script_direction, depth + 1, max_depth)
            return ordered_a + ordered_b

    # --- No cuts found — sort top-to-bottom (RTL: right-to-left within row) ---
    def _sort_key(b: TextBlock):
        y = b.bbox.y0 if b.bbox else 0.0
        x = b.bbox.x0 if b.bbox else 0.0
        # For RTL, invert x so right-most comes first within same row
        x_key = -x if script_direction == "rtl" else x
        return (y, x_key)

    return sorted(blocks, key=_sort_key)


# ---------------------------------------------------------------------------
# Confidence scoring helpers
# ---------------------------------------------------------------------------


def _compute_order_quality(
    ordered: list[TextBlock],
    page_width: float,
    page_height: float,
) -> float:
    """
    Estimate order_quality from the spatial coherence of the ordering.

    Heuristic: count pairs of consecutive blocks where reading order
    matches geometric order (top-to-bottom, left-to-right).  Quality
    is the fraction of "coherent" transitions.
    """
    if len(ordered) <= 1:
        return MAX_ORDER_QUALITY

    coherent = 0
    for a, b in zip(ordered, ordered[1:]):
        if a.bbox is None or b.bbox is None:
            continue
        # Same column: b should be below a
        if abs((a.bbox.x0 + a.bbox.x1) / 2 - (b.bbox.x0 + b.bbox.x1) / 2) < page_width * 0.1:
            if b.bbox.y0 >= a.bbox.y0:
                coherent += 1
        else:
            # Different columns: b may be anywhere
            coherent += 1

    total = len(ordered) - 1
    if total == 0:
        return MAX_ORDER_QUALITY

    return FALLBACK_ORDER_QUALITY + (MAX_ORDER_QUALITY - FALLBACK_ORDER_QUALITY) * (coherent / total)


# ---------------------------------------------------------------------------
# Band-based fallback strategy
# ---------------------------------------------------------------------------


def _band_based_order(
    blocks: list[TextBlock],
    page_height: float,
    script_direction: str,
    num_bands: int = 10,
) -> list[TextBlock]:
    """
    Divide the page into ``num_bands`` equal horizontal bands, sort blocks
    within each band by x-position (respecting RTL), concatenate top-to-bottom.
    """
    if not blocks:
        return []

    band_height = page_height / num_bands if page_height > 0 else 1.0

    bands: list[list[TextBlock]] = [[] for _ in range(num_bands)]
    for block in blocks:
        y_mid = (block.bbox.y0 + block.bbox.y1) / 2 if block.bbox else 0.0
        band_idx = min(num_bands - 1, int(y_mid / band_height))
        bands[band_idx].append(block)

    ordered: list[TextBlock] = []
    for band in bands:
        reverse = script_direction == "rtl"
        band.sort(key=lambda b: (b.bbox.x0 if b.bbox else 0.0), reverse=reverse)
        ordered.extend(band)

    return ordered


# ---------------------------------------------------------------------------
# Public entry point (page-level)
# ---------------------------------------------------------------------------


def resolve_reading_order(
    page_ir: PageIR,
    strategy: str = "xy_cut",
    script_direction: str = "ltr",
) -> PageIR:
    """
    Sort ``page_ir.blocks`` into reading order and set ``order_quality``.

    Parameters
    ----------
    page_ir:
        The ``PageIR`` produced by Stage 2.
    strategy:
        ``"xy_cut"``    — recursive XY-cut (default).
        ``"band_based"`` — horizontal band partitioning (simpler fallback).
    script_direction:
        ``"ltr"`` or ``"rtl"``.  Must be detected upstream; never assumed.

    Returns
    -------
    PageIR
        The same object with ``blocks`` replaced by the ordered list and
        ``order_strategy`` set.
    """
    blocks = list(page_ir.blocks)
    if not blocks:
        page_ir.order_strategy = strategy
        return page_ir

    # --- Determine ordering -------------------------------------------
    if strategy == "band_based":
        ordered = _band_based_order(
            blocks, page_ir.height, script_direction
        )
        quality = FALLBACK_ORDER_QUALITY
        page_ir.order_strategy = "band_based"
    else:
        ordered = _xy_cut_recursive(blocks, script_direction)
        quality = _compute_order_quality(ordered, page_ir.width, page_ir.height)
        page_ir.order_strategy = "xy_cut"

    # --- Set order_quality on every block -----------------------------
    for block in ordered:
        block.confidence.order_quality = quality

    page_ir.blocks = ordered
    logger.debug(
        "Page %d: %d blocks ordered via %s (quality=%.2f, dir=%s)",
        page_ir.page_num, len(ordered), page_ir.order_strategy,
        quality, script_direction,
    )
    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage3(
    doc_ir: DocumentIR,
    strategy: str = "xy_cut",
) -> DocumentIR:
    """
    Apply reading-order resolution to every page in ``doc_ir``.

    The ``script_direction`` for each page is inferred from the dominant
    direction among blocks on that page; it falls back to ``"ltr"`` when
    all blocks default to ``"ltr"`` or no blocks are present.

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 2.
    strategy:
        Default ordering strategy.  Individual pages may override via
        ``page_ir.order_strategy`` if set by an earlier stage.

    Returns
    -------
    DocumentIR
        The same object, with every ``PageIR.blocks`` list in reading order.
    """
    for page_ir in doc_ir.pages:
        # Infer dominant script direction from existing blocks
        rtl_count = sum(
            1 for b in page_ir.blocks if b.script_direction == "rtl"
        )
        total = len(page_ir.blocks)
        script_direction = "rtl" if total > 0 and rtl_count / total > 0.5 else "ltr"

        # Respect a pre-set strategy (e.g. from Stage 0 triage)
        page_strategy = page_ir.order_strategy if page_ir.order_strategy else strategy

        resolve_reading_order(page_ir, strategy=page_strategy, script_direction=script_direction)

    return doc_ir
