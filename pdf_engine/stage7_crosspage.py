"""
pdf_engine/stage7_crosspage.py
==============================
Stage 7 — Cross-Page Analysis

Responsibility
--------------
For the full ``DocumentIR`` produced by Stage 6, this stage:

  1. Detects repeating header and footer blocks across pages using a
     fingerprint-bucket approach — O(n × blocks_per_page), never O(n²).
  2. Promotes those blocks to ``HEADER`` / ``FOOTER`` type in-place.
  3. Builds a heading hierarchy by assigning ``parent_id`` relationships
     between nested heading levels.
  4. Falls back to a heuristic heading detector for pages whose blocks
     were produced entirely by OCR (no surya layout labels).

Design rules enforced here
--------------------------
* Complexity is O(n × blocks_per_page) — no pairwise Jaccard comparison.
* Normalization strips digit-runs AND section-number prefixes so that
  "3.1 Methods" and "4.2 Methods" fingerprint identically.
* ``order_quality`` and ``type_quality`` on existing blocks are not
  overwritten here; only ``block_type`` and ``parent_id`` are mutated.
* All returned objects remain typed ``TextBlock`` / ``PageIR`` / ``DocumentIR``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Optional

from pdf_engine.models import (
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

#: A fingerprint must appear on at least this many pages to be a
#: repeating header or footer.
MIN_PAGES_FOR_REPEAT: int = 3

#: How many blocks from the top of a page are candidates for HEADER detection.
TOP_N_BLOCKS: int = 3

#: How many blocks from the bottom of a page are candidates for FOOTER detection.
BOTTOM_N_BLOCKS: int = 3

#: Fraction of page height that defines the "header zone" (top fraction).
#: Blocks whose y-centre falls within the top 10 % of the page are header candidates.
HEADER_ZONE_FRACTION: float = 0.10

#: Fraction of page height that defines the "footer zone" (bottom fraction).
FOOTER_ZONE_FRACTION: float = 0.10

#: Maximum character length of a block to be considered a heading heuristically.
HEADING_MAX_CHARS: int = 80

#: Minimum number of pages for which heading hierarchy is meaningful.
MIN_PAGES_FOR_HIERARCHY: int = 2

#: Threshold: a heading level is "larger" if its median bbox height exceeds
#: body average by this factor.
HEADING_SIZE_RATIO: float = 1.15


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_for_header_detection(text: str) -> str:
    """
    Normalise a block's text into a fingerprint that is stable across pages.

    Steps:
      1. Strip leading section numbers ("3.1 Methods" → "Methods").
      2. Strip trailing page numbers ("Introduction  7" → "Introduction").
      3. Collapse all remaining digit sequences to a placeholder so that
         year numbers, figure numbers, etc. do not break fingerprint matching.
      4. Lowercase and strip whitespace.

    Examples
    --------
    >>> normalize_for_header_detection("3.1 Methods")
    'methods'
    >>> normalize_for_header_detection("4.2 Methods")
    'methods'
    >>> normalize_for_header_detection("Page 7")
    'page'
    >>> normalize_for_header_detection("Running Head: My Paper")
    'running head: my paper'
    """
    text = text.strip()
    # Strip leading section numbers: "3.1 ", "1.2.3 ", "10. " etc.
    text = re.sub(r'^\d[\d.]*\s+', '', text)
    # Strip trailing standalone page numbers
    text = re.sub(r'\s+\d+$', '', text)
    # Collapse remaining digit runs to a placeholder token
    text = re.sub(r'\d+', '#', text)
    return text.lower().strip()


def _block_fingerprint(block: TextBlock) -> str:
    """Return a stable fingerprint for a block's text content."""
    return normalize_for_header_detection(block.text)


# ---------------------------------------------------------------------------
# Header / footer detection  (O(n × blocks_per_page))
# ---------------------------------------------------------------------------


def find_repeating_patterns(
    doc_ir: DocumentIR,
    min_pages: int = MIN_PAGES_FOR_REPEAT,
) -> tuple[set[str], set[str]]:
    """
    Identify text fingerprints that repeat across >= ``min_pages`` pages,
    returning two sets: one for header candidates (top of page) and one
    for footer candidates (bottom of page).

    Algorithm
    ---------
    For each page, take the top-N and bottom-N blocks (by y0 position).
    Compute a normalised fingerprint for each.  Use two ``Counter`` objects
    (one per position bucket) to count how many *distinct* pages carry each
    fingerprint.  Any fingerprint appearing on >= min_pages pages is a repeat.

    Complexity: O(n × blocks_per_page) — one pass, no pairwise comparison.

    Returns
    -------
    (header_fingerprints, footer_fingerprints)
        Two sets of fingerprint strings.  Empty strings are excluded.
    """
    # page_count_header[fp] = set of page_nums that contained fp in top region
    page_set_header: dict[str, set[int]] = defaultdict(set)
    page_set_footer: dict[str, set[int]] = defaultdict(set)

    for page_ir in doc_ir.pages:
        blocks = [b for b in page_ir.blocks if b.bbox is not None and b.text.strip()]
        if not blocks:
            continue

        page_h = page_ir.height or 842.0  # fall back to A4 if unknown
        header_threshold = page_h * HEADER_ZONE_FRACTION
        footer_threshold = page_h * (1.0 - FOOTER_ZONE_FRACTION)

        # Sort by y0 (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y0)

        # Use positional zone first; fall back to top-N when no zone hits.
        # Exclude blocks already claimed by the opposing zone from fallbacks
        # to prevent the same fingerprint from appearing in both header_fps
        # and footer_fps when the page has no blocks in a zone.
        top_by_zone = [b for b in sorted_blocks
                       if (b.bbox.y0 + b.bbox.y1) / 2 < header_threshold]
        bottom_by_zone = [b for b in sorted_blocks
                          if (b.bbox.y0 + b.bbox.y1) / 2 > footer_threshold]

        bottom_zone_ids = {b.id for b in bottom_by_zone}
        top_zone_ids = {b.id for b in top_by_zone}

        top_blocks = top_by_zone if top_by_zone else [
            b for b in sorted_blocks[:TOP_N_BLOCKS] if b.id not in bottom_zone_ids
        ]
        bottom_blocks = bottom_by_zone if bottom_by_zone else [
            b for b in sorted_blocks[-BOTTOM_N_BLOCKS:] if b.id not in top_zone_ids
        ]

        for blk in top_blocks:
            fp = _block_fingerprint(blk)
            if fp:
                page_set_header[fp].add(page_ir.page_num)

        for blk in bottom_blocks:
            fp = _block_fingerprint(blk)
            if fp:
                page_set_footer[fp].add(page_ir.page_num)

    header_fps = {fp for fp, pages in page_set_header.items() if len(pages) >= min_pages}
    footer_fps = {fp for fp, pages in page_set_footer.items() if len(pages) >= min_pages}

    logger.debug(
        "Repeating header fingerprints (%d): %s", len(header_fps), header_fps
    )
    logger.debug(
        "Repeating footer fingerprints (%d): %d total", len(footer_fps), len(footer_fps)
    )

    return header_fps, footer_fps


def _promote_repeating_blocks(
    doc_ir: DocumentIR,
    header_fps: set[str],
    footer_fps: set[str],
) -> int:
    """
    Walk every page and promote blocks whose fingerprints appear in the
    repeating sets to ``HEADER`` or ``FOOTER``.

    A block is promoted to HEADER when:
      * Its fingerprint is in ``header_fps``.
      * It is in the top-N positional group of the page.

    A block is promoted to FOOTER when:
      * Its fingerprint is in ``footer_fps``.
      * It is in the bottom-N positional group.

    Blocks can be promoted to at most one type; HEADER takes precedence
    for blocks that appear in both sets (very rare in practice).

    Returns
    -------
    int
        Total number of blocks promoted.
    """
    promoted = 0

    for page_ir in doc_ir.pages:
        blocks = [b for b in page_ir.blocks if b.bbox is not None and b.text.strip()]
        if not blocks:
            continue

        page_h = page_ir.height or 842.0
        header_threshold = page_h * HEADER_ZONE_FRACTION
        footer_threshold = page_h * (1.0 - FOOTER_ZONE_FRACTION)

        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y0)

        top_by_zone = [b for b in sorted_blocks
                       if (b.bbox.y0 + b.bbox.y1) / 2 < header_threshold]
        bottom_by_zone = [b for b in sorted_blocks
                          if (b.bbox.y0 + b.bbox.y1) / 2 > footer_threshold]

        bottom_zone_ids = {b.id for b in bottom_by_zone}
        top_zone_ids = {b.id for b in top_by_zone}

        # When using fallback (no zone hits), exclude blocks that are
        # explicitly in the opposing zone to prevent cross-zone misclassification.
        top_ids = top_zone_ids if top_by_zone else {
            b.id for b in sorted_blocks[:TOP_N_BLOCKS] if b.id not in bottom_zone_ids
        }
        bottom_ids = bottom_zone_ids if bottom_by_zone else {
            b.id for b in sorted_blocks[-BOTTOM_N_BLOCKS:] if b.id not in top_zone_ids
        }

        for blk in page_ir.blocks:
            fp = _block_fingerprint(blk)
            if blk.id in top_ids and fp in header_fps:
                if blk.block_type != BlockType.HEADER:
                    logger.debug(
                        "Page %d: promoting block %s to HEADER (fp=%r)",
                        page_ir.page_num, blk.id, fp,
                    )
                    blk.block_type = BlockType.HEADER
                    promoted += 1
            elif blk.id in bottom_ids and fp in footer_fps:
                if blk.block_type != BlockType.FOOTER:
                    logger.debug(
                        "Page %d: promoting block %s to FOOTER (fp=%r)",
                        page_ir.page_num, blk.id, fp,
                    )
                    blk.block_type = BlockType.FOOTER
                    promoted += 1

    return promoted


# ---------------------------------------------------------------------------
# Heuristic heading detection (OCR-only pages)
# ---------------------------------------------------------------------------


def _is_ocr_only_page(page_ir: PageIR) -> bool:
    """
    Return True if every text block on the page was produced by an OCR
    method rather than by surya layout analysis or PyMuPDF direct.

    A page is considered OCR-only when it has at least one block and all
    blocks originate from Tesseract or Surya OCR extraction.
    """
    if not page_ir.blocks:
        return False
    ocr_methods = {ExtractionMethod.TESSERACT_OCR, ExtractionMethod.SURYA_OCR}
    return all(b.extraction_method in ocr_methods for b in page_ir.blocks)


def _estimate_body_bbox_height(page_ir: PageIR) -> float:
    """
    Return the median bbox height of BODY blocks on the page, or 0.0 if
    no BODY blocks with a bbox exist.  Used as a proxy for body font size.
    """
    heights = [
        b.bbox.height
        for b in page_ir.blocks
        if b.block_type == BlockType.BODY and b.bbox is not None and b.bbox.height > 0
    ]
    if not heights:
        return 0.0
    heights.sort()
    mid = len(heights) // 2
    return heights[mid]


_HEADING_PATTERN = re.compile(
    r'^(?:'
    r'[A-Z][A-Z\s]{2,}$'                        # ALL CAPS (≥ 3 chars)
    r'|(?:[A-Z][a-z]+\s*){1,6}$'               # Title Case with ≤ 6 words
    r'|\d[\d.]*\s+(?:\S+\s*){1,8}$'            # Section-numbered: 1-8 words max (HIGH-4 fix)
    r')'
)


def _looks_like_heading(text: str, bbox_height: float, body_height: float) -> bool:
    """
    Heuristic: return True if this block looks like a heading.

    Criteria (any one is sufficient):
    * Text is short (< HEADING_MAX_CHARS chars) AND bbox height exceeds body
      average by HEADING_SIZE_RATIO.
    * Text matches a known heading pattern (ALL CAPS, Title Case ≤ 6 words,
      or leading section number).
    """
    text = text.strip()
    if not text or len(text) > HEADING_MAX_CHARS:
        return False

    # Size-based heuristic
    if body_height > 0 and bbox_height >= body_height * HEADING_SIZE_RATIO:
        return True

    # Pattern-based heuristic
    if _HEADING_PATTERN.match(text):
        return True

    return False


def detect_headings_heuristic(page_ir: PageIR) -> PageIR:
    """
    Fallback heading detector for pages where no HEADING blocks were
    produced by surya (typically OCR-only pages).

    Mutates ``page_ir.blocks`` in-place: qualifying blocks have their
    ``block_type`` changed to ``HEADING``.  The page's ``order_strategy``
    and all confidence sub-scores are left unchanged.

    Returns
    -------
    PageIR
        The same object (for chaining), mutated in-place.
    """
    body_height = _estimate_body_bbox_height(page_ir)
    promoted = 0

    for blk in page_ir.blocks:
        if blk.block_type != BlockType.BODY:
            continue  # already typed by layout; don't override
        if blk.bbox is None:
            continue

        if _looks_like_heading(blk.text, blk.bbox.height, body_height):
            blk.block_type = BlockType.HEADING
            promoted += 1
            logger.debug(
                "Page %d: heuristic heading detected — block %s %r",
                page_ir.page_num, blk.id, blk.text[:60],
            )

    if promoted:
        page_ir.add_warning(
            f"Page {page_ir.page_num}: {promoted} heading(s) detected "
            "heuristically (OCR-only page, no surya layout)."
        )

    return page_ir


# ---------------------------------------------------------------------------
# Heading hierarchy builder
# ---------------------------------------------------------------------------


def _normalised_heading_height(block: TextBlock) -> float:
    """
    Return an estimated per-line height for a heading block.

    Using raw ``bbox.height`` as a proxy for font size fails for multi-line
    headings: a 2-line H1 has bbox.height ≈ 40pt but a 1-line H1 has ≈20pt,
    causing them to cluster at different levels.  Dividing by the estimated
    line count recovers the effective single-line height.
    """
    if block.bbox is None:
        return 0.0
    text = block.text or ""
    # Count explicit newlines; fall back to a single line if none present.
    num_lines = max(1, text.count('\n') + 1)
    return block.bbox.height / num_lines


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    return values[mid]


def build_heading_hierarchy(doc_ir: DocumentIR) -> None:
    """
    Assign ``parent_id`` relationships between heading levels across the
    document to express nesting (H1 → H2 → H3, etc.).

    Strategy
    --------
    1. Collect all HEADING blocks across all pages.
    2. Cluster them into levels by bbox height (proxy for font size):
       * Sort unique heights; merge heights within 2 pt of each other.
       * Level 0 = tallest (largest font = H1), Level 1 = next, etc.
    3. Walk headings in document order (page_num, then y0).  Track the
       most-recently-seen heading at each level.  When a heading of
       level L is encountered, set its ``parent_id`` to the most recent
       heading at level L-1.

    This is a pure in-place mutation; nothing is returned.
    """
    all_headings = [
        blk
        for page_ir in doc_ir.pages
        for blk in page_ir.blocks
        if blk.block_type == BlockType.HEADING and blk.bbox is not None
    ]

    if not all_headings:
        return

    # --- Cluster heights into levels ----------------------------------
    unique_heights = sorted(
        {
            # Normalise by estimated line count so a 2-line heading at 18pt
            # doesn't get clustered separately from a 1-line 18pt heading.
            round(_normalised_heading_height(b), 1)
            for b in all_headings
        },
        reverse=True,  # tallest first = H1
    )

    # Merge heights within 2 pt tolerance into the same level
    levels: list[float] = []
    for h in unique_heights:
        if not levels or abs(h - levels[-1]) > 2.0:
            levels.append(h)

    def _height_to_level(block: TextBlock) -> int:
        """Map a block to the nearest level index (0 = H1) using normalised height."""
        height = _normalised_heading_height(block)
        best_idx, best_dist = 0, float("inf")
        for i, lh in enumerate(levels):
            dist = abs(height - lh)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    # --- Walk headings in document order ------------------------------
    # Sort by (page_num, y0)
    ordered = sorted(
        all_headings,
        key=lambda b: (b.page_num, b.bbox.y0 if b.bbox else 0),
    )

    # recent_at_level[i] = id of most recent heading at level i
    recent_at_level: dict[int, Optional[str]] = {}

    for blk in ordered:
        level = _height_to_level(blk)

        if level > 0:
            parent_level = level - 1
            parent_id = recent_at_level.get(parent_level)
            if parent_id is not None:
                blk.parent_id = parent_id
                logger.debug(
                    "Page %d: heading %s (level %d) → parent %s (level %d)",
                    blk.page_num, blk.id, level, parent_id, parent_level,
                )

        recent_at_level[level] = blk.id

        # Invalidate all deeper levels when a shallower heading appears
        for deeper in list(recent_at_level.keys()):
            if deeper > level:
                del recent_at_level[deeper]


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage7(
    doc_ir: DocumentIR,
    min_pages: int = MIN_PAGES_FOR_REPEAT,
) -> DocumentIR:
    """
    Run cross-page analysis on the full document.

    Steps
    -----
    1. Find repeating header/footer fingerprints (O(n)).
    2. Promote matching blocks to HEADER / FOOTER.
    3. For OCR-only pages with no HEADING blocks, run heuristic detection.
    4. Build heading hierarchy (parent_id linkage).
    5. Validate parent_id integrity and log any broken references.

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 6.
    min_pages:
        Minimum number of pages a fingerprint must appear on to be
        considered a repeating pattern.  Default: 3.

    Returns
    -------
    DocumentIR
        The same object, mutated in-place, then returned for chaining.
    """
    n_pages = len(doc_ir.pages)
    logger.info("Stage 7: cross-page analysis on %d pages", n_pages)

    # --- 1 + 2. Repeating header / footer detection -------------------
    header_fps, footer_fps = find_repeating_patterns(doc_ir, min_pages=min_pages)
    promoted = _promote_repeating_blocks(doc_ir, header_fps, footer_fps)
    logger.info(
        "Stage 7: %d header/footer blocks promoted "
        "(%d header fingerprints, %d footer fingerprints)",
        promoted, len(header_fps), len(footer_fps),
    )
    if promoted:
        doc_ir.add_warning(
            f"Stage 7: {promoted} block(s) reclassified as HEADER/FOOTER "
            f"from {len(header_fps)} header and {len(footer_fps)} footer fingerprints."
        )

    # --- 3. Heuristic heading detection for OCR-only pages -----------
    ocr_heading_pages = 0
    for page_ir in doc_ir.pages:
        has_headings = any(b.block_type == BlockType.HEADING for b in page_ir.blocks)
        if not has_headings and _is_ocr_only_page(page_ir):
            detect_headings_heuristic(page_ir)
            ocr_heading_pages += 1

    if ocr_heading_pages:
        logger.info(
            "Stage 7: heuristic heading detection applied to %d OCR-only page(s)",
            ocr_heading_pages,
        )

    # --- 4. Heading hierarchy ----------------------------------------
    build_heading_hierarchy(doc_ir)

    # --- 5. Validate parent_id integrity -----------------------------
    errors = doc_ir.validate_parent_refs()
    for err in errors:
        logger.error("Parent-ref integrity error after Stage 7: %s", err)
        doc_ir.add_warning(err)

    logger.info("Stage 7: complete.")
    return doc_ir
