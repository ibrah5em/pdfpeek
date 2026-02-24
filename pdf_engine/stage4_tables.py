"""
pdf_engine/stage4_tables.py
============================
Stage 4 — Table Detection + Reconstruction

Responsibility
--------------
For every ``PageIR``, this stage:

  1. Detects **explicit** (ruled) tables via pdfplumber (handle passed in — never opened here).
  2. Detects **implicit** (whitespace-aligned) tables from word-level spans extracted
     from existing TextBlock content.
  3. Reconstructs cell structure for both explicit and implicit tables.
  4. Replaces or annotates the relevant blocks in ``page_ir`` with TABLE-typed blocks.

Design rules enforced here
--------------------------
* ``pdfplumber`` is NEVER opened here — the caller passes ``plumber_page``.
* ``type_quality`` is set to 0.9 for explicit tables, 0.6 for implicit tables.
* All returned objects are typed ``TextBlock`` / ``PageIR`` instances.
* Word-level span splitting uses character-level x-positions where possible.
"""

from __future__ import annotations

import logging
import uuid
from statistics import median
from typing import Any, Optional

from pdf_engine.models import (
    BBox,
    BlockConfidence,
    BlockType,
    ExtractionMethod,
    PageIR,
    TextBlock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Minimum number of rows for a region to be considered an implicit table.
MIN_IMPLICIT_TABLE_ROWS: int = 2

#: Minimum number of columns for a region to be considered an implicit table.
MIN_IMPLICIT_TABLE_COLS: int = 2

#: Gap threshold (points) for row clustering.
ROW_GAP_THRESHOLD: float = 3.0

#: Minimum whitespace gap (points) to split a span into sub-spans.
MIN_GAP_PTS: float = 4.0

#: type_quality for explicit tables detected by pdfplumber.
EXPLICIT_TABLE_TYPE_QUALITY: float = 0.9

#: type_quality for implicit tables detected by whitespace analysis.
IMPLICIT_TABLE_TYPE_QUALITY: float = 0.6


# ---------------------------------------------------------------------------
# Row clustering (fixes W14)
# ---------------------------------------------------------------------------


def cluster_by_y(spans: list[dict], gap_threshold: float = ROW_GAP_THRESHOLD) -> list[list[dict]]:
    """
    Group spans into rows by y-center using gap-based clustering.

    A new row starts when the gap between the next span's y_center and the
    **median** y_center of the current cluster exceeds ``gap_threshold``.

    Parameters
    ----------
    spans:
        List of dicts, each with at least a ``"y_center"`` key.
    gap_threshold:
        Maximum distance (points) from cluster median to include a span in
        the current row.

    Returns
    -------
    list[list[dict]]
        Each inner list is a row of spans, sorted top-to-bottom.
    """
    rows: list[list[dict]] = []
    current_row: list[dict] = []

    for span in sorted(spans, key=lambda s: s["y_center"]):
        if not current_row:
            current_row.append(span)
        else:
            cluster_median = median(s["y_center"] for s in current_row)
            if abs(span["y_center"] - cluster_median) <= gap_threshold:
                current_row.append(span)
            else:
                rows.append(current_row)
                current_row = [span]

    if current_row:
        rows.append(current_row)

    return rows


# ---------------------------------------------------------------------------
# Word-level span splitting (fixes W15)
# ---------------------------------------------------------------------------


def split_spans_on_gaps(spans: list[dict], min_gap_pts: float = MIN_GAP_PTS) -> list[dict]:
    """
    Split each span into sub-spans wherever internal whitespace exceeds
    ``min_gap_pts`` points.

    Character-level x-positions are used when available (``"char_positions"``
    key in span); otherwise, position is estimated linearly from character count
    and span width.

    Parameters
    ----------
    spans:
        List of span dicts with keys: ``text``, ``x0``, ``x1``, ``y0``, ``y1``,
        ``y_center``.  May optionally have ``"char_positions"`` as a list of
        (char, x) tuples.
    min_gap_pts:
        Whitespace gap (points) at which to cut a span.

    Returns
    -------
    list[dict]
        Possibly longer list of sub-span dicts with the same keys.
    """
    result: list[dict] = []

    for span in spans:
        text: str = span.get("text", "")
        x0: float = span["x0"]
        x1: float = span["x1"]
        y0: float = span["y0"]
        y1: float = span["y1"]
        y_center: float = span["y_center"]
        span_width: float = x1 - x0

        if not text or span_width <= 0:
            result.append(span)
            continue

        # --- Determine character x-positions ---
        char_positions: list[tuple[str, float]] = span.get("char_positions", [])

        if not char_positions:
            # Estimate uniformly
            char_width = span_width / max(len(text), 1)
            char_positions = [(ch, x0 + i * char_width) for i, ch in enumerate(text)]

        # --- Find gap cut points ---
        # Build list of word tokens with their x-extents
        # We look at gaps between consecutive non-space characters
        words: list[dict] = []
        current_chars: list[tuple[str, float]] = []

        def flush_word(chars: list[tuple[str, float]]) -> None:
            if not chars:
                return
            word_x0 = chars[0][1]
            word_x1 = chars[-1][1] + (span_width / max(len(text), 1))
            words.append({
                "text": "".join(c for c, _ in chars),
                "x0": word_x0,
                "x1": word_x1,
                "y0": y0,
                "y1": y1,
                "y_center": y_center,
            })

        prev_x: Optional[float] = None
        for ch, cx in char_positions:
            if ch == " ":
                if prev_x is not None and current_chars:
                    gap = cx - prev_x
                    if gap >= min_gap_pts:
                        flush_word(current_chars)
                        current_chars = []
            else:
                if prev_x is not None and current_chars and ch != " ":
                    gap = cx - prev_x
                    if gap >= min_gap_pts:
                        flush_word(current_chars)
                        current_chars = []
                current_chars.append((ch, cx))
                prev_x = cx

        flush_word(current_chars)

        if words:
            result.extend(words)
        else:
            result.append(span)

    return result


# ---------------------------------------------------------------------------
# Span extraction from existing TextBlocks
# ---------------------------------------------------------------------------


def _blocks_to_word_spans(blocks: list[TextBlock]) -> list[dict]:
    """
    Convert TextBlocks to word-level span dicts for implicit table detection.

    Each word in a block becomes its own span, with x-position estimated
    from the block bbox and character count.
    """
    spans: list[dict] = []

    for block in blocks:
        if block.bbox is None or not block.text:
            continue

        bbox = block.bbox
        words = block.text.split()
        if not words:
            continue

        # Estimate word widths proportional to character count
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            continue

        block_width = bbox.x1 - bbox.x0
        block_height = bbox.y1 - bbox.y0
        cursor = bbox.x0

        # Estimate number of text lines in this block from line breaks in text.
        # This prevents all words in a multi-line paragraph from landing on the
        # same y-coordinate (which triggers false-positive table detection).
        raw_lines = [l for l in block.text.split('\n') if l.strip()]
        num_lines = max(1, len(raw_lines)) if raw_lines else max(1, len(words) // 8 + 1)
        line_height = block_height / num_lines

        for line_idx, line_text in enumerate(raw_lines if raw_lines else [block.text]):
            line_words = line_text.split()
            if not line_words:
                continue
            line_chars = sum(len(w) for w in line_words)
            if line_chars == 0:
                continue
            line_y0 = bbox.y0 + line_idx * line_height
            line_y1 = line_y0 + line_height
            line_y_center = (line_y0 + line_y1) / 2
            line_cursor = bbox.x0
            for word in line_words:
                word_width = (len(word) / line_chars) * block_width
                spans.append({
                    "text": word,
                    "x0": line_cursor,
                    "x1": line_cursor + word_width,
                    "y0": line_y0,
                    "y1": line_y1,
                    "y_center": line_y_center,
                })
                line_cursor += word_width

    return spans


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _detect_columns(rows: list[list[dict]], page_width: float) -> list[tuple[float, float]]:
    """
    Detect column boundaries from a set of rows.

    Groups span x0 positions into clusters to find column start positions.
    Returns a list of (col_x0, col_x1) tuples.
    """
    all_x0s = [span["x0"] for row in rows for span in row]
    if not all_x0s:
        return []

    all_x0s_sorted = sorted(all_x0s)
    # Gap-based clustering on x0 positions
    col_starts: list[float] = []
    col_gap = max(page_width * 0.02, 5.0)  # 2% of page width or 5pt min

    current_group: list[float] = [all_x0s_sorted[0]]
    for x in all_x0s_sorted[1:]:
        if x - current_group[-1] > col_gap:
            col_starts.append(median(current_group))
            current_group = [x]
        else:
            current_group.append(x)
    col_starts.append(median(current_group))

    if len(col_starts) < MIN_IMPLICIT_TABLE_COLS:
        return []

    # Build column ranges
    all_x1s = sorted([span["x1"] for row in rows for span in row])
    columns: list[tuple[float, float]] = []
    for i, col_x0 in enumerate(col_starts):
        col_x1 = col_starts[i + 1] - 1.0 if i + 1 < len(col_starts) else max(all_x1s)
        columns.append((col_x0, col_x1))

    return columns


# ---------------------------------------------------------------------------
# Cell text assembly
# ---------------------------------------------------------------------------


def _cell_text(row: list[dict], col_x0: float, col_x1: float, tolerance: float = 5.0) -> str:
    """Extract and join text from spans that fall within a column range."""
    tokens = [
        span["text"]
        for span in sorted(row, key=lambda s: s["x0"])
        if span["x0"] >= col_x0 - tolerance and span["x1"] <= col_x1 + tolerance
    ]
    return " ".join(tokens).strip()


# ---------------------------------------------------------------------------
# Explicit table extraction (pdfplumber)
# ---------------------------------------------------------------------------


def _extract_explicit_tables(plumber_page, page_ir: PageIR) -> list[TextBlock]:
    """
    Extract ruled (explicit) tables from a pdfplumber page object.

    Returns a list of TABLE TextBlocks, one per detected table.
    The cell data is serialised as tab-separated rows in ``block.text``.
    """
    if plumber_page is None:
        return []

    table_blocks: list[TextBlock] = []

    try:
        tables = plumber_page.extract_tables()
        table_bboxes = plumber_page.find_tables()
    except Exception as exc:
        logger.error("pdfplumber table extraction failed: %s", exc, exc_info=True)
        return []

    for i, (table_data, table_obj) in enumerate(zip(tables, table_bboxes)):
        if not table_data:
            continue

        try:
            bbox_raw = table_obj.bbox  # (x0, top, x1, bottom) in pdfplumber coords
            bbox = BBox(
                x0=float(bbox_raw[0]),
                y0=float(bbox_raw[1]),
                x1=float(bbox_raw[2]),
                y1=float(bbox_raw[3]),
            )
        except Exception:
            bbox = BBox(x0=0, y0=0, x1=page_ir.width, y1=page_ir.height)

        # Serialise cell data as TSV
        rows_text = []
        for row in table_data:
            cells = [cell or "" for cell in row]
            rows_text.append("\t".join(cells))
        table_text = "\n".join(rows_text)

        num_rows = len(table_data)
        num_cols = max((len(row) for row in table_data), default=0)

        block = TextBlock(
            id=str(uuid.uuid4()),
            text=table_text,
            bbox=bbox,
            block_type=BlockType.TABLE,
            extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
            confidence=BlockConfidence(
                type_quality=EXPLICIT_TABLE_TYPE_QUALITY,
                method_score=1.0,
            ),
            page_num=page_ir.page_num,
        )

        logger.debug(
            "Page %d: explicit table %d — %d rows × %d cols",
            page_ir.page_num, i, num_rows, num_cols,
        )
        table_blocks.append(block)

    return table_blocks


# ---------------------------------------------------------------------------
# Implicit table detection (whitespace alignment)
# ---------------------------------------------------------------------------


def _extract_implicit_tables(page_ir: PageIR) -> list[TextBlock]:
    """
    Detect implicit (whitespace-aligned) tables from word-level spans
    derived from existing TextBlocks.

    Returns a list of TABLE TextBlocks for regions that look tabular.
    """
    # Get word-level spans from body blocks (skip headings, headers, footers)
    candidate_blocks = [
        b for b in page_ir.blocks
        if b.block_type in (BlockType.BODY, BlockType.TABLE)
        and b.text.strip()
    ]

    if not candidate_blocks:
        return []

    raw_spans = _blocks_to_word_spans(candidate_blocks)
    if not raw_spans:
        return []

    # Split spans on internal gaps (W15 fix)
    word_spans = split_spans_on_gaps(raw_spans, min_gap_pts=MIN_GAP_PTS)

    # Cluster into rows (W14 fix)
    rows = cluster_by_y(word_spans, gap_threshold=ROW_GAP_THRESHOLD)

    if len(rows) < MIN_IMPLICIT_TABLE_ROWS:
        return []

    # Detect columns
    columns = _detect_columns(rows, page_ir.width or 612.0)
    if len(columns) < MIN_IMPLICIT_TABLE_COLS:
        return []

    # Compute table bbox
    all_spans_flat = [s for row in rows for s in row]
    table_x0 = min(s["x0"] for s in all_spans_flat)
    table_y0 = min(s["y0"] for s in all_spans_flat)
    table_x1 = max(s["x1"] for s in all_spans_flat)
    table_y1 = max(s["y1"] for s in all_spans_flat)

    # Build TSV
    rows_text: list[str] = []
    for row in rows:
        cells = [_cell_text(row, col_x0, col_x1) for col_x0, col_x1 in columns]
        rows_text.append("\t".join(cells))
    table_text = "\n".join(rows_text)

    block = TextBlock(
        id=str(uuid.uuid4()),
        text=table_text,
        bbox=BBox(x0=table_x0, y0=table_y0, x1=table_x1, y1=table_y1),
        block_type=BlockType.TABLE,
        extraction_method=ExtractionMethod.SURYA_LAYOUT,
        confidence=BlockConfidence(
            type_quality=IMPLICIT_TABLE_TYPE_QUALITY,
            method_score=0.8,
        ),
        page_num=page_ir.page_num,
    )

    logger.debug(
        "Page %d: implicit table — %d rows × %d cols",
        page_ir.page_num, len(rows), len(columns),
    )
    return [block]


# ---------------------------------------------------------------------------
# Table region deduplication
# ---------------------------------------------------------------------------


def _deduplicate_tables(
    explicit: list[TextBlock],
    implicit: list[TextBlock],
    overlap_threshold: float = 0.5,
) -> list[TextBlock]:
    """
    Remove implicit tables that substantially overlap with explicit tables.
    Explicit tables take precedence.
    """
    result = list(explicit)

    for imp in implicit:
        if imp.bbox is None:
            continue
        dominated = False
        for exp in explicit:
            if exp.bbox is None:
                continue
            ix0 = max(imp.bbox.x0, exp.bbox.x0)
            iy0 = max(imp.bbox.y0, exp.bbox.y0)
            ix1 = min(imp.bbox.x1, exp.bbox.x1)
            iy1 = min(imp.bbox.y1, exp.bbox.y1)
            if ix0 < ix1 and iy0 < iy1:
                overlap = (ix1 - ix0) * (iy1 - iy0)
                if overlap / imp.bbox.area > overlap_threshold:
                    dominated = True
                    break
        if not dominated:
            result.append(imp)

    return result


# ---------------------------------------------------------------------------
# Non-table block filtering
# ---------------------------------------------------------------------------


def _remove_blocks_inside_tables(
    blocks: list[TextBlock],
    table_blocks: list[TextBlock],
    overlap_threshold: float = 0.5,
) -> list[TextBlock]:
    """
    Remove non-table blocks that are substantially contained within a table bbox.
    Table blocks themselves are not filtered.
    """
    kept: list[TextBlock] = []

    for block in blocks:
        if block.block_type == BlockType.TABLE:
            kept.append(block)
            continue

        if block.bbox is None:
            kept.append(block)
            continue

        absorbed = False
        for table in table_blocks:
            if table.bbox is None:
                continue
            ix0 = max(block.bbox.x0, table.bbox.x0)
            iy0 = max(block.bbox.y0, table.bbox.y0)
            ix1 = min(block.bbox.x1, table.bbox.x1)
            iy1 = min(block.bbox.y1, table.bbox.y1)
            if ix0 < ix1 and iy0 < iy1:
                overlap = (ix1 - ix0) * (iy1 - iy0)
                if block.bbox.area > 0 and overlap / block.bbox.area > overlap_threshold:
                    absorbed = True
                    break

        if not absorbed:
            kept.append(block)

    return kept


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_tables(plumber_page, page_ir: PageIR) -> PageIR:
    """
    Detect explicit and implicit tables on a single page.

    Parameters
    ----------
    plumber_page:
        A live ``pdfplumber.Page`` object for this page.
        Must be passed in — this function never calls ``pdfplumber.open()``.
        Pass ``None`` to skip explicit table detection (e.g. in tests).
    page_ir:
        The ``PageIR`` returned by Stage 3.

    Returns
    -------
    PageIR
        The same object with TABLE blocks inserted and overlapping BODY
        blocks removed.

    Usage (in pipeline runner, NOT here)
    -------------------------------------
    >>> with pdfplumber.open(pdf_path) as plumber_doc:
    ...     for page_num, fitz_page in enumerate(fitz_doc):
    ...         detect_tables(plumber_doc.pages[page_num], page_ir)
    """
    # 1. Explicit tables via pdfplumber
    explicit_tables = _extract_explicit_tables(plumber_page, page_ir)

    # 2. Implicit tables via whitespace analysis
    implicit_tables = _extract_implicit_tables(page_ir)

    # 3. Deduplicate (explicit wins over implicit)
    all_tables = _deduplicate_tables(explicit_tables, implicit_tables)

    if not all_tables:
        logger.debug("Page %d: no tables detected", page_ir.page_num)
        return page_ir

    # 4. Remove non-table blocks absorbed by table regions
    surviving_blocks = _remove_blocks_inside_tables(page_ir.blocks, all_tables)

    # 5. Merge: surviving non-table blocks + new table blocks, sorted top-to-bottom
    merged = surviving_blocks + all_tables
    merged.sort(key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0))
    page_ir.blocks = merged

    logger.info(
        "Page %d: %d explicit + %d implicit table(s) detected",
        page_ir.page_num, len(explicit_tables), len(implicit_tables),
    )
    return page_ir
