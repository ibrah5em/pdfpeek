"""
tests/test_stage2.py
====================
Unit tests for Stage 2 — Layout Analysis.

All tests are pure-Python and do not require a real PDF or surya installation.
External calls (``run_surya_layout``, ``_rasterise_page``) are monkey-patched
so the test suite runs in any CI environment.

Test matrix
-----------
1.  Single-column page → blocks ordered top-to-bottom (y0 ascending).
2.  Figure with caption within 5 pt → caption.parent_id == figure.id.
3.  Caption 10 pt outside figure bbox → not linked (parent_id is None).
4.  surya returns [] → one BODY block for full page, warning added.
5.  Unknown surya label → falls back to BODY, no crash.
6.  Multiple figures → caption linked to the smallest enclosing figure.
7.  type_quality propagated correctly from surya confidence.
8.  Fallback block spans full page dimensions.
9.  All blocks have non-empty UUID ids after processing.
10. validate_parent_refs passes after stage 2 (no dangling parent_ids).
"""

from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the surya library before importing our modules so the import chain
# succeeds even without surya installed.
# ---------------------------------------------------------------------------

def _stub_surya() -> None:
    surya_pkg      = types.ModuleType("surya")
    surya_model    = types.ModuleType("surya.model")
    surya_det      = types.ModuleType("surya.model.detection")
    surya_det_model = types.ModuleType("surya.model.detection.model")
    surya_det_proc  = types.ModuleType("surya.model.detection.processor")
    surya_layout   = types.ModuleType("surya.layout")

    surya_det_model.load_model          = MagicMock(return_value=MagicMock())
    surya_det_proc.load_processor       = MagicMock(return_value=MagicMock())
    surya_layout.batch_layout_detection = MagicMock(return_value=[])

    sys.modules.setdefault("surya",                            surya_pkg)
    sys.modules.setdefault("surya.model",                      surya_model)
    sys.modules.setdefault("surya.model.detection",            surya_det)
    sys.modules.setdefault("surya.model.detection.model",      surya_det_model)
    sys.modules.setdefault("surya.model.detection.processor",  surya_det_proc)
    sys.modules.setdefault("surya.layout",                     surya_layout)


_stub_surya()

from pdf_engine.models import (
    BBox, BlockConfidence, BlockType, DocumentIR,
    ExtractionMethod, PageIR, TextBlock,
)
from pdf_engine.surya_adapter import SuryaBlock, map_surya_label
from pdf_engine.stage2_layout import (
    CAPTION_CONTAINMENT_PAD,
    FALLBACK_TYPE_QUALITY,
    _link_captions_to_figures,
    _make_fallback_block,
    _surya_block_to_text_block,
    analyse_page_layout,
    is_contained_with_padding,
    run_stage2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_surya_block(
    x0: float, y0: float, x1: float, y1: float,
    label: str = "Text",
    confidence: float = 0.9,
) -> SuryaBlock:
    return SuryaBlock(
        bbox=BBox(x0, y0, x1, y1),
        label=label,
        confidence=confidence,
    )


def make_page_ir(page_num: int = 0, width: float = 595.0, height: float = 842.0) -> PageIR:
    return PageIR(page_num=page_num, width=width, height=height)


def run_analyse_with_surya_blocks(
    surya_blocks: list[SuryaBlock],
    page_num: int = 0,
    existing_blocks: list[TextBlock] | None = None,
) -> PageIR:
    """
    Helper: run ``analyse_page_layout`` with a pre-determined list of
    SuryaBlocks injected via mock, without touching the filesystem.
    """
    page_ir = make_page_ir(page_num=page_num)
    if existing_blocks:
        page_ir.blocks = existing_blocks

    with patch("pdf_engine.stage2_layout._rasterise_page", return_value=MagicMock()):
        with patch("pdf_engine.stage2_layout.run_surya_layout", return_value=surya_blocks):
            analyse_page_layout(page_ir, fitz_page=MagicMock())

    return page_ir


# ===========================================================================
# Tests
# ===========================================================================


class TestIsContainedWithPadding:
    """Unit tests for the padded-containment predicate."""

    def test_inner_fully_inside(self):
        outer = BBox(0, 0, 100, 100)
        inner = BBox(10, 10, 90, 90)
        assert is_contained_with_padding(inner, outer) is True

    def test_inner_exactly_on_border(self):
        outer = BBox(0, 0, 100, 100)
        inner = BBox(0, 0, 100, 100)
        assert is_contained_with_padding(inner, outer) is True

    def test_inner_within_padding(self):
        """Inner extends 4 pt beyond outer — within 5 pt pad → True."""
        outer = BBox(0, 0, 100, 100)
        inner = BBox(-4, -4, 104, 104)
        assert is_contained_with_padding(inner, outer, pad=5.0) is True

    def test_inner_exceeds_padding(self):
        """Inner extends 6 pt beyond outer — outside 5 pt pad → False."""
        outer = BBox(0, 0, 100, 100)
        inner = BBox(-6, 0, 106, 100)
        assert is_contained_with_padding(inner, outer, pad=5.0) is False

    def test_inner_exactly_at_padding_boundary(self):
        """Exactly 5 pt beyond → True (boundary is inclusive via >=)."""
        outer = BBox(0, 0, 100, 100)
        inner = BBox(-5, -5, 105, 105)
        assert is_contained_with_padding(inner, outer, pad=5.0) is True


class TestLabelMapping:
    def test_known_labels(self):
        assert map_surya_label("Text") == BlockType.BODY
        assert map_surya_label("Title") == BlockType.HEADING
        assert map_surya_label("Table") == BlockType.TABLE
        assert map_surya_label("Figure") == BlockType.FIGURE
        assert map_surya_label("Caption") == BlockType.CAPTION
        assert map_surya_label("Page-header") == BlockType.HEADER
        assert map_surya_label("Page-footer") == BlockType.FOOTER
        assert map_surya_label("Footnote") == BlockType.FOOTNOTE

    def test_unknown_label_falls_back_to_body(self):
        """Unknown label → BODY, no exception."""
        result = map_surya_label("WeirdCategory")
        assert result == BlockType.BODY


class TestSingleColumnPage:
    """Test 1: blocks on a single-column page come out top-to-bottom."""

    def test_blocks_ordered_top_to_bottom(self):
        surya_blocks = [
            make_surya_block(50, 200, 500, 250, label="Text"),   # middle
            make_surya_block(50,  50, 500, 100, label="Title"),  # top
            make_surya_block(50, 400, 500, 450, label="Text"),   # bottom
        ]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)

        y0_values = [b.bbox.y0 for b in page_ir.blocks]
        assert y0_values == sorted(y0_values), (
            f"Blocks not in top-to-bottom order: y0 values = {y0_values}"
        )

    def test_block_types_match_labels(self):
        surya_blocks = [
            make_surya_block(50, 50, 500, 100, label="Title"),
            make_surya_block(50, 120, 500, 400, label="Text"),
        ]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)

        types = {b.block_type for b in page_ir.blocks}
        assert BlockType.HEADING in types
        assert BlockType.BODY in types


class TestFigureCaptionLinking:
    """Tests 2 & 3: padded containment links captions to figures."""

    def _make_figure_block(self, x0, y0, x1, y1, page_num=0) -> TextBlock:
        return TextBlock(
            id=str(uuid.uuid4()),
            bbox=BBox(x0, y0, x1, y1),
            block_type=BlockType.FIGURE,
            page_num=page_num,
        )

    def _make_caption_block(self, x0, y0, x1, y1, page_num=0) -> TextBlock:
        return TextBlock(
            id=str(uuid.uuid4()),
            bbox=BBox(x0, y0, x1, y1),
            block_type=BlockType.CAPTION,
            page_num=page_num,
        )

    def test_caption_within_5pt_is_linked(self):
        """Test 2: Caption inside figure bbox (within 5 pt) → linked."""
        fig = self._make_figure_block(100, 100, 400, 300)
        # Caption just inside: x0=102, y0=280, x1=398, y1=298 — all inside
        cap = self._make_caption_block(102, 280, 398, 298)
        blocks = [fig, cap]
        _link_captions_to_figures(blocks)
        assert cap.parent_id == fig.id

    def test_caption_at_padding_boundary_is_linked(self):
        """Caption extends exactly 5 pt beyond figure → still linked."""
        fig = self._make_figure_block(100, 100, 400, 300)
        cap = self._make_caption_block(95, 95, 405, 305)  # 5 pt outside
        blocks = [fig, cap]
        _link_captions_to_figures(blocks)
        assert cap.parent_id == fig.id

    def test_caption_10pt_outside_is_not_linked(self):
        """Test 3: Caption 10 pt outside figure → not linked."""
        fig = self._make_figure_block(100, 100, 400, 300)
        cap = self._make_caption_block(90, 90, 410, 310)  # 10 pt outside
        blocks = [fig, cap]
        _link_captions_to_figures(blocks)
        assert cap.parent_id is None

    def test_caption_linked_to_smallest_enclosing_figure(self):
        """Test 6: When two figures enclose the caption, pick the smaller one."""
        big_fig = self._make_figure_block(50, 50, 500, 500)
        small_fig = self._make_figure_block(100, 100, 400, 300)
        cap = self._make_caption_block(120, 270, 380, 295)
        blocks = [big_fig, small_fig, cap]
        _link_captions_to_figures(blocks)
        assert cap.parent_id == small_fig.id, (
            "Caption should be linked to the smallest enclosing figure"
        )

    def test_caption_without_matching_figure_stays_unlinked(self):
        """No figure → caption.parent_id stays None."""
        cap = self._make_caption_block(100, 100, 300, 120)
        blocks = [cap]
        _link_captions_to_figures(blocks)
        assert cap.parent_id is None

    def test_via_analyse_page_layout(self):
        """End-to-end: run analyse_page_layout with figure+caption surya blocks."""
        surya_blocks = [
            make_surya_block(100, 100, 400, 300, label="Figure", confidence=0.95),
            make_surya_block(110, 270, 390, 298, label="Caption", confidence=0.88),
        ]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)

        figures  = [b for b in page_ir.blocks if b.block_type == BlockType.FIGURE]
        captions = [b for b in page_ir.blocks if b.block_type == BlockType.CAPTION]

        assert len(figures) == 1
        assert len(captions) == 1
        assert captions[0].parent_id == figures[0].id


class TestSuryaReturnsEmpty:
    """Test 4: When surya returns [], use a single full-page BODY block."""

    def test_fallback_block_type(self):
        page_ir = run_analyse_with_surya_blocks([])
        assert len(page_ir.blocks) == 1
        assert page_ir.blocks[0].block_type == BlockType.BODY

    def test_fallback_type_quality(self):
        page_ir = run_analyse_with_surya_blocks([])
        assert page_ir.blocks[0].confidence.type_quality == pytest.approx(
            FALLBACK_TYPE_QUALITY
        )

    def test_fallback_warning_added(self):
        """Test 4: A warning must be recorded in PageIR.warnings."""
        page_ir = run_analyse_with_surya_blocks([])
        assert len(page_ir.warnings) >= 1, "Expected at least one warning when surya returns []"
        assert any("BODY" in w or "surya" in w.lower() for w in page_ir.warnings)

    def test_fallback_block_spans_full_page(self):
        """Test 8: Fallback block bbox matches page dimensions."""
        page_ir = make_page_ir(width=595.0, height=842.0)
        with patch("pdf_engine.stage2_layout._rasterise_page", return_value=MagicMock()):
            with patch("pdf_engine.stage2_layout.run_surya_layout", return_value=[]):
                analyse_page_layout(page_ir, fitz_page=MagicMock())

        blk = page_ir.blocks[0]
        assert blk.bbox.x0 == pytest.approx(0.0)
        assert blk.bbox.y0 == pytest.approx(0.0)
        assert blk.bbox.x1 == pytest.approx(595.0)
        assert blk.bbox.y1 == pytest.approx(842.0)


class TestTypeQualityPropagation:
    """Test 7: type_quality is set from surya's per-block confidence."""

    def test_type_quality_matches_surya_confidence(self):
        surya_blocks = [
            make_surya_block(50, 50, 500, 100, label="Title", confidence=0.92),
            make_surya_block(50, 120, 500, 400, label="Text", confidence=0.75),
        ]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)

        # Find heading block
        heading = next(b for b in page_ir.blocks if b.block_type == BlockType.HEADING)
        body    = next(b for b in page_ir.blocks if b.block_type == BlockType.BODY)

        assert heading.confidence.type_quality == pytest.approx(0.92)
        assert body.confidence.type_quality    == pytest.approx(0.75)


class TestBlockIds:
    """Test 9: All blocks have non-empty UUID ids after processing."""

    def test_ids_are_non_empty_uuids(self):
        surya_blocks = [
            make_surya_block(50, 50, 500, 100, label="Title"),
            make_surya_block(50, 120, 500, 400, label="Text"),
            make_surya_block(50, 420, 500, 600, label="Table"),
        ]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)

        for blk in page_ir.blocks:
            assert blk.id, "Block id must not be empty"
            # Validate it's a parseable UUID
            parsed = uuid.UUID(blk.id)
            assert str(parsed) == blk.id

    def test_fallback_block_has_uuid(self):
        page_ir = run_analyse_with_surya_blocks([])
        blk = page_ir.blocks[0]
        assert blk.id
        uuid.UUID(blk.id)  # raises ValueError if not valid


class TestParentRefIntegrity:
    """Test 10: validate_parent_refs passes after stage 2."""

    def test_no_dangling_parent_refs(self):
        surya_blocks = [
            make_surya_block(100, 100, 400, 300, label="Figure", confidence=0.95),
            make_surya_block(110, 270, 390, 298, label="Caption", confidence=0.88),
        ]
        doc_ir = DocumentIR()
        page_ir = make_page_ir()
        doc_ir.pages.append(page_ir)

        with patch("pdf_engine.stage2_layout._rasterise_page", return_value=MagicMock()):
            with patch("pdf_engine.stage2_layout.run_surya_layout", return_value=surya_blocks):
                run_stage2(doc_ir, fitz_doc=None)

        errors = doc_ir.validate_parent_refs()
        assert errors == [], f"Dangling parent_id references: {errors}"


class TestExtractionMethod:
    """Blocks produced by stage 2 carry SURYA_LAYOUT extraction_method."""

    def test_extraction_method_is_surya_layout(self):
        surya_blocks = [make_surya_block(50, 50, 500, 100, label="Text")]
        page_ir = run_analyse_with_surya_blocks(surya_blocks)
        for blk in page_ir.blocks:
            assert blk.extraction_method == ExtractionMethod.SURYA_LAYOUT


class TestTextHarvesting:
    """Stage-1 text is harvested into surya blocks when bboxes overlap."""

    def test_overlapping_stage1_text_is_harvested(self):
        existing = TextBlock(
            id=str(uuid.uuid4()),
            text="Hello world",
            bbox=BBox(50, 50, 500, 100),
            page_num=0,
        )
        surya_blocks = [make_surya_block(50, 50, 500, 100, label="Text")]

        page_ir = run_analyse_with_surya_blocks(
            surya_blocks, existing_blocks=[existing]
        )
        assert "Hello world" in page_ir.blocks[0].text

    def test_non_overlapping_stage1_text_not_harvested(self):
        existing = TextBlock(
            id=str(uuid.uuid4()),
            text="Invisible text",
            bbox=BBox(600, 600, 800, 700),  # completely outside
            page_num=0,
        )
        surya_blocks = [make_surya_block(50, 50, 500, 100, label="Text")]

        page_ir = run_analyse_with_surya_blocks(
            surya_blocks, existing_blocks=[existing]
        )
        assert "Invisible text" not in page_ir.blocks[0].text


class TestMakeFallbackBlock:
    """Unit tests for _make_fallback_block in isolation."""

    def test_type_quality(self):
        page_ir = make_page_ir(width=595, height=842)
        blk = _make_fallback_block(page_ir)
        assert blk.confidence.type_quality == pytest.approx(FALLBACK_TYPE_QUALITY)

    def test_block_type(self):
        blk = _make_fallback_block(make_page_ir())
        assert blk.block_type == BlockType.BODY

    def test_bbox_covers_page(self):
        page_ir = make_page_ir(width=612, height=792)
        blk = _make_fallback_block(page_ir)
        assert blk.bbox == BBox(0.0, 0.0, 612.0, 792.0)
