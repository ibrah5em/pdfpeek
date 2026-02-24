"""
tests/test_stage3.py
=====================
Tests for Stage 3 — Reading Order Resolution.

All surya / fitz imports are stubbed before any pdf_engine import so CI
works without those libraries installed.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Stub out surya + fitz so importing pdf_engine modules never raises
# ---------------------------------------------------------------------------

for mod_name in [
    "surya",
    "surya.model",
    "surya.model.detection",
    "surya.model.detection.model",
    "surya.model.detection.processor",
    "surya.layout",
    "fitz",
    "PIL",
    "PIL.Image",
]:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# Provide dummy callables so attribute access in surya_adapter doesn't fail
sys.modules["surya.model.detection.model"].load_model = lambda: None
sys.modules["surya.model.detection.processor"].load_processor = lambda: None
sys.modules["surya.layout"].batch_layout_detection = lambda *a, **kw: []

# ---------------------------------------------------------------------------

import pytest

from pdf_engine.models import (
    BBox,
    BlockConfidence,
    BlockType,
    DocumentIR,
    ExtractionMethod,
    PageIR,
    TextBlock,
)
from pdf_engine.stage3_reading_order import (
    FALLBACK_ORDER_QUALITY,
    _band_based_order,
    _xy_cut_recursive,
    find_best_horizontal_cut,
    partition_at_cut,
    resolve_reading_order,
    run_stage3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block(
    x0: float, y0: float, x1: float, y1: float,
    text: str = "",
    script_direction: str = "ltr",
    page_num: int = 0,
) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=BBox(x0, y0, x1, y1),
        block_type=BlockType.BODY,
        extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
        page_num=page_num,
        script_direction=script_direction,
    )


def make_page(
    blocks: list[TextBlock],
    width: float = 600.0,
    height: float = 800.0,
    page_num: int = 0,
) -> PageIR:
    return PageIR(
        page_num=page_num,
        blocks=blocks,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# partition_at_cut
# ---------------------------------------------------------------------------


class TestPartitionAtCut:
    def test_clean_horizontal_split(self):
        """Blocks clearly above or below the cut end up in the right partition."""
        top    = make_block(0, 0, 100, 100)
        bottom = make_block(0, 200, 100, 300)

        before, after = partition_at_cut([top, bottom], cut_coord=150, axis="y")

        assert top in before
        assert bottom in after

    def test_clean_vertical_split(self):
        left  = make_block(0,   0, 100, 100)
        right = make_block(300, 0, 400, 100)

        before, after = partition_at_cut([left, right], cut_coord=200, axis="x")

        assert left in before
        assert right in after

    def test_straddling_block_appears_exactly_once(self):
        """A block that straddles the cut must land in exactly one partition."""
        # Block spans y=90..110; cut at y=100 → 50/50 split → goes to 'before' (tie)
        straddle = make_block(0, 90, 100, 110)
        other    = make_block(0, 200, 100, 300)

        before, after = partition_at_cut([straddle, other], cut_coord=100, axis="y")

        total = (straddle in before) + (straddle in after)
        assert total == 1, "Straddling block must appear in exactly one partition"

    def test_dominant_side_wins_for_straddler(self):
        """A block that overlaps 80% above and 20% below should go above."""
        # Block spans y=0..100; cut at y=80 → 80% above, 20% below
        block = make_block(0, 0, 100, 100)
        before, after = partition_at_cut([block], cut_coord=80, axis="y")

        assert block in before
        assert block not in after

    def test_all_blocks_accounted_for(self):
        """Every input block appears in exactly one output partition."""
        blocks = [
            make_block(0, 0,   100, 50),
            make_block(0, 45,  100, 100),   # slight straddle
            make_block(0, 200, 100, 300),
        ]
        before, after = partition_at_cut(blocks, cut_coord=55, axis="y")
        combined = before + after
        assert len(combined) == len(blocks)
        for b in blocks:
            assert combined.count(b) == 1


# ---------------------------------------------------------------------------
# Two-column layout → left column precedes right
# ---------------------------------------------------------------------------


class TestTwoColumnOrdering:
    """Left column content must precede right column content in LTR mode."""

    def _make_two_column_page(self) -> tuple[PageIR, list[TextBlock]]:
        # Left column: x ∈ [0, 250]; Right column: x ∈ [350, 600]
        # Blocks are contiguous vertically (no inter-row gap) so XY-cut finds
        # the VERTICAL gap (x=250..350) first at the document level.
        left_top    = make_block(0,   0,   250, 100, text="L1")
        left_mid    = make_block(0,   100, 250, 200, text="L2")
        left_bot    = make_block(0,   200, 250, 300, text="L3")
        right_top   = make_block(350, 0,   600, 100, text="R1")
        right_mid   = make_block(350, 100, 600, 200, text="R2")
        right_bot   = make_block(350, 200, 600, 300, text="R3")

        blocks = [left_top, left_mid, left_bot, right_top, right_mid, right_bot]
        page = make_page(blocks, width=600, height=350)
        return page, blocks

    def test_left_column_precedes_right(self):
        page, _ = self._make_two_column_page()
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        # All L blocks before any R block
        last_l = max(texts.index(t) for t in ["L1", "L2", "L3"])
        first_r = min(texts.index(t) for t in ["R1", "R2", "R3"])
        assert last_l < first_r, f"Expected left before right, got order: {texts}"

    def test_within_column_top_to_bottom(self):
        page, _ = self._make_two_column_page()
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        assert texts.index("L1") < texts.index("L2") < texts.index("L3")
        assert texts.index("R1") < texts.index("R2") < texts.index("R3")


# ---------------------------------------------------------------------------
# Full-width title + two-column body (W9 fix: title appears exactly once)
# ---------------------------------------------------------------------------


class TestFullWidthTitle:
    """A full-width title straddling a two-column body must appear exactly once."""

    def test_title_appears_once(self):
        title  = make_block(0,   0,   600, 60, text="TITLE")   # full-width
        left1  = make_block(0,   80,  250, 180, text="L1")
        left2  = make_block(0,   190, 250, 290, text="L2")
        right1 = make_block(350, 80,  600, 180, text="R1")
        right2 = make_block(350, 190, 600, 290, text="R2")

        page = make_page([title, left1, left2, right1, right2], width=600, height=350)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        assert texts.count("TITLE") == 1, (
            f"Full-width title appeared {texts.count('TITLE')} times: {texts}"
        )

    def test_title_precedes_body_columns(self):
        title  = make_block(0,   0,   600, 60, text="TITLE")
        left1  = make_block(0,   80,  250, 180, text="L1")
        right1 = make_block(350, 80,  600, 180, text="R1")

        page = make_page([title, left1, right1], width=600, height=250)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        assert texts.index("TITLE") < texts.index("L1")
        assert texts.index("TITLE") < texts.index("R1")


# ---------------------------------------------------------------------------
# RTL ordering — right column precedes left
# ---------------------------------------------------------------------------


class TestRTLOrdering:
    def test_rtl_right_column_first(self):
        """For RTL (Arabic) pages the right column must come before the left."""
        left_block  = make_block(0,   0, 250, 300, text="LEFT",  script_direction="rtl")
        right_block = make_block(350, 0, 600, 300, text="RIGHT", script_direction="rtl")

        page = make_page([left_block, right_block], width=600, height=400)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="rtl")
        texts = [b.text for b in result.blocks]

        assert texts.index("RIGHT") < texts.index("LEFT"), (
            f"RTL: expected RIGHT before LEFT, got {texts}"
        )

    def test_rtl_within_column_top_to_bottom(self):
        """Even in RTL mode, blocks within a column should still read top-to-bottom."""
        top = make_block(350, 0,   600, 100, text="TOP",  script_direction="rtl")
        bot = make_block(350, 150, 600, 250, text="BOT",  script_direction="rtl")

        page = make_page([top, bot], width=600, height=300)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="rtl")
        texts = [b.text for b in result.blocks]

        assert texts.index("TOP") < texts.index("BOT")


# ---------------------------------------------------------------------------
# Dense page — no cuts found → top-to-bottom fallback
# ---------------------------------------------------------------------------


class TestNoCutFallback:
    def test_dense_page_sorted_top_to_bottom(self):
        """When no cut is found the blocks are sorted top-to-bottom."""
        # Blocks that completely fill the page width — no whitespace gap
        blocks = [
            make_block(0, i * 50, 600, (i + 1) * 50 - 1, text=str(i))
            for i in range(8)
        ]
        import random
        shuffled = blocks[:]
        random.shuffle(shuffled)
        page = make_page(shuffled, width=600, height=400)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        expected = [str(i) for i in range(8)]
        assert texts == expected, f"Expected top-to-bottom, got {texts}"

    def test_fallback_order_quality_set(self):
        """order_quality must be set on every block after ordering."""
        blocks = [make_block(0, i * 50, 600, (i + 1) * 50, text=str(i)) for i in range(4)]
        page = make_page(blocks, width=600, height=200)
        result = resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")

        for block in result.blocks:
            assert block.confidence.order_quality > 0.0, (
                "order_quality should be set by Stage 3"
            )


# ---------------------------------------------------------------------------
# order_quality is set and within valid range
# ---------------------------------------------------------------------------


class TestOrderQuality:
    def test_order_quality_in_range(self):
        blocks = [
            make_block(0,   0,   250, 100, text="L1"),
            make_block(350, 0,   600, 100, text="R1"),
            make_block(0,   150, 250, 250, text="L2"),
            make_block(350, 150, 600, 250, text="R2"),
        ]
        page = make_page(blocks, width=600, height=300)
        resolve_reading_order(page, strategy="xy_cut", script_direction="ltr")

        for block in page.blocks:
            assert 0.0 < block.confidence.order_quality <= 1.0

    def test_order_strategy_set_on_page(self):
        page = make_page([make_block(0, 0, 100, 100)])
        resolve_reading_order(page, strategy="xy_cut")
        assert page.order_strategy == "xy_cut"


# ---------------------------------------------------------------------------
# Band-based strategy
# ---------------------------------------------------------------------------


class TestBandBased:
    def test_band_based_ltr_order(self):
        left  = make_block(0,   100, 100, 200, text="L")
        right = make_block(400, 100, 500, 200, text="R")

        page = make_page([right, left], width=600, height=800)
        result = resolve_reading_order(page, strategy="band_based", script_direction="ltr")
        texts = [b.text for b in result.blocks]

        assert texts.index("L") < texts.index("R")

    def test_band_based_rtl_order(self):
        left  = make_block(0,   100, 100, 200, text="L", script_direction="rtl")
        right = make_block(400, 100, 500, 200, text="R", script_direction="rtl")

        page = make_page([left, right], width=600, height=800)
        result = resolve_reading_order(page, strategy="band_based", script_direction="rtl")
        texts = [b.text for b in result.blocks]

        assert texts.index("R") < texts.index("L")

    def test_band_based_sets_order_quality(self):
        blocks = [make_block(0, i * 80, 600, (i + 1) * 80, text=str(i)) for i in range(4)]
        page = make_page(blocks, width=600, height=800)
        resolve_reading_order(page, strategy="band_based")

        for block in page.blocks:
            assert block.confidence.order_quality == FALLBACK_ORDER_QUALITY


# ---------------------------------------------------------------------------
# Empty page
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_page(self):
        page = make_page([], width=600, height=800)
        result = resolve_reading_order(page)
        assert result.blocks == []

    def test_single_block(self):
        block = make_block(0, 0, 600, 800, text="only")
        page = make_page([block])
        result = resolve_reading_order(page)
        assert len(result.blocks) == 1
        assert result.blocks[0].text == "only"

    def test_blocks_without_bbox_dont_crash(self):
        block = TextBlock(text="no-bbox")
        page = make_page([block])
        result = resolve_reading_order(page)
        assert len(result.blocks) == 1


# ---------------------------------------------------------------------------
# Document-level run_stage3
# ---------------------------------------------------------------------------


class TestRunStage3:
    def test_processes_all_pages(self):
        pages = [
            make_page(
                [make_block(0, 0, 600, 100, text=f"p{i}")],
                page_num=i,
            )
            for i in range(3)
        ]
        doc = DocumentIR(pages=pages)
        result = run_stage3(doc)

        assert len(result.pages) == 3
        for p in result.pages:
            assert p.order_strategy in ("xy_cut", "band_based")
            for b in p.blocks:
                assert b.confidence.order_quality > 0.0

    def test_rtl_detection_from_block_direction(self):
        """run_stage3 should detect RTL from block script_direction."""
        rtl_block_a = make_block(350, 0,   600, 200, text="R", script_direction="rtl")
        rtl_block_b = make_block(0,   0,   250, 200, text="L", script_direction="rtl")
        page = make_page([rtl_block_b, rtl_block_a], width=600, height=300)
        doc = DocumentIR(pages=[page])
        run_stage3(doc)

        texts = [b.text for b in doc.pages[0].blocks]
        assert texts.index("R") < texts.index("L"), (
            f"RTL blocks should trigger right-first ordering; got {texts}"
        )
