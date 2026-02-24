"""
tests/test_stage6.py
=====================
Unit tests for Stage 6 — Block Assembly + De-duplication.

Coverage
--------
1. 12 pymupdf line blocks inside 1 surya paragraph → 1 merged block, text joined
2. pymupdf block outside all surya blocks → retained as-is
3. surya block with no pymupdf coverage → flagged for re-OCR (warning added)
4. Zero duplicate text across output (same sentence never appears twice)
5. Low-quality pymupdf content → surya text wins
6. Multiple surya blocks, some matched, some not
7. Empty inputs handled gracefully
8. Block IDs are always valid UUIDs (never empty)
9. Merged block uses surya geometry + block_type, pymupdf text
10. Unmatched pymupdf block lands at end of output
"""

from __future__ import annotations

import uuid
from typing import Optional

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
from pdf_engine.stage6_assembly import (
    DEFAULT_QUALITY_THRESHOLD,
    _NEEDS_OCR_WARNING_PREFIX,
    _center_inside,
    _partition_blocks,
    assemble_page,
    deduplicate_blocks,
    run_stage6,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def make_pymupdf_block(
    text: str,
    bbox: BBox,
    page_num: int = 0,
    text_quality: float = 1.0,
    language: Optional[str] = "en",
    script_direction: str = "ltr",
) -> TextBlock:
    return TextBlock(
        id=str(uuid.uuid4()),
        text=text,
        bbox=bbox,
        block_type=BlockType.BODY,
        extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
        confidence=BlockConfidence(text_quality=text_quality),
        page_num=page_num,
        language=language,
        script_direction=script_direction,
    )


def make_surya_block(
    text: str,
    bbox: BBox,
    block_type: BlockType = BlockType.BODY,
    page_num: int = 0,
    type_quality: float = 0.9,
) -> TextBlock:
    return TextBlock(
        id=str(uuid.uuid4()),
        text=text,
        bbox=bbox,
        block_type=block_type,
        extraction_method=ExtractionMethod.SURYA_LAYOUT,
        confidence=BlockConfidence(type_quality=type_quality),
        page_num=page_num,
    )


def make_page_ir(page_num: int = 0, blocks: Optional[list] = None) -> PageIR:
    page = PageIR(page_num=page_num, width=595.0, height=842.0)
    page.blocks = blocks or []
    return page


# ---------------------------------------------------------------------------
# Geometry helper tests
# ---------------------------------------------------------------------------


class TestCenterInside:
    def test_center_clearly_inside(self):
        container = BBox(0, 0, 100, 150)
        inner = BBox(40, 60, 60, 75)   # center = (50, 67.5) — inside
        assert _center_inside(inner, container) is True

    def test_center_outside(self):
        container = BBox(0, 0, 100, 150)
        inner = BBox(110, 10, 130, 20)  # center x=120 — outside
        assert _center_inside(inner, container) is False

    def test_tiny_line_inside_tall_paragraph(self):
        """Simulates a 12pt line inside a 150pt surya paragraph block."""
        paragraph = BBox(50, 100, 500, 250)
        line = BBox(50, 130, 500, 142)  # height=12, center_y=136 — inside
        assert _center_inside(line, paragraph) is True

    def test_center_on_boundary_is_inside(self):
        container = BBox(0, 0, 100, 100)
        inner = BBox(0, 0, 200, 200)  # center = (100, 100) — on boundary
        assert _center_inside(inner, container) is True


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------


class TestPartitionBlocks:
    def test_separates_pymupdf_and_surya(self):
        pm = make_pymupdf_block("hello", BBox(0, 0, 100, 20))
        su = make_surya_block("hello", BBox(0, 0, 100, 200))
        pymupdf_out, surya_out = _partition_blocks([pm, su])
        assert pm in pymupdf_out
        assert su in surya_out
        assert pm not in surya_out
        assert su not in pymupdf_out

    def test_tesseract_treated_as_pymupdf(self):
        blk = TextBlock(
            id=str(uuid.uuid4()),
            text="ocr text",
            bbox=BBox(0, 0, 100, 20),
            extraction_method=ExtractionMethod.TESSERACT_OCR,
        )
        pymupdf_out, surya_out = _partition_blocks([blk])
        assert blk in pymupdf_out
        assert surya_out == []

    def test_empty_input(self):
        pymupdf_out, surya_out = _partition_blocks([])
        assert pymupdf_out == []
        assert surya_out == []


# ---------------------------------------------------------------------------
# Core: test 1 — 12 pymupdf lines inside 1 surya block → 1 merged block
# ---------------------------------------------------------------------------


class TestManyLinesToOneParagraph:
    def _make_12_lines(self) -> list[TextBlock]:
        """Create 12 pymupdf line blocks stacked inside BBox(50, 100, 500, 400)."""
        lines = []
        for i in range(12):
            y0 = 105 + i * 20
            y1 = y0 + 12
            lines.append(
                make_pymupdf_block(
                    text=f"Line {i + 1} content here.",
                    bbox=BBox(50, y0, 500, y1),
                    text_quality=1.0,
                )
            )
        return lines

    def test_produces_single_merged_block(self):
        lines = self._make_12_lines()
        surya = make_surya_block("", BBox(50, 100, 500, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)

        assert len(result) == 1, f"Expected 1 merged block, got {len(result)}"

    def test_merged_block_contains_all_12_lines(self):
        lines = self._make_12_lines()
        surya = make_surya_block("", BBox(50, 100, 500, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        merged = result[0]

        for i in range(12):
            assert f"Line {i + 1} content here." in merged.text, (
                f"Line {i + 1} missing from merged text"
            )

    def test_merged_block_uses_surya_bbox(self):
        lines = self._make_12_lines()
        surya_bbox = BBox(50, 100, 500, 400)
        surya = make_surya_block("", surya_bbox)
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        assert result[0].bbox == surya_bbox

    def test_merged_block_uses_pymupdf_extraction_method(self):
        lines = self._make_12_lines()
        surya = make_surya_block("", BBox(50, 100, 500, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        assert result[0].extraction_method == ExtractionMethod.PYMUPDF_DIRECT

    def test_merged_block_id_is_valid_uuid(self):
        lines = self._make_12_lines()
        surya = make_surya_block("", BBox(50, 100, 500, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        merged_id = result[0].id
        assert merged_id, "Block ID must not be empty"
        # Verify it is parseable as a UUID
        parsed = uuid.UUID(merged_id)
        assert str(parsed) == merged_id

    def test_merged_block_uses_surya_block_type(self):
        lines = self._make_12_lines()
        surya = make_surya_block("", BBox(50, 100, 500, 400), block_type=BlockType.HEADING)
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        assert result[0].block_type == BlockType.HEADING

    def test_language_propagated_from_pymupdf(self):
        lines = self._make_12_lines()
        lines[0] = make_pymupdf_block(
            lines[0].text, lines[0].bbox, language="fr"
        )
        surya = make_surya_block("", BBox(50, 100, 500, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(lines, [surya], page_ir=page_ir)
        assert result[0].language == "fr"


# ---------------------------------------------------------------------------
# Core: test 2 — pymupdf block outside all surya blocks → retained as-is
# ---------------------------------------------------------------------------


class TestPymupdfBlockOutsideSurya:
    def test_unmatched_pymupdf_block_retained(self):
        """A pymupdf block whose center is outside all surya regions must survive."""
        outside_block = make_pymupdf_block(
            "Standalone paragraph outside any layout region.",
            BBox(400, 700, 590, 720),  # bottom-right corner
        )
        surya = make_surya_block("Para text", BBox(50, 100, 300, 400))
        inside_block = make_pymupdf_block(
            "Inside text.", BBox(60, 150, 290, 162)
        )
        page_ir = make_page_ir()

        result = deduplicate_blocks(
            [outside_block, inside_block], [surya], page_ir=page_ir
        )

        # The outside block must appear in the output
        result_texts = [b.text for b in result]
        assert any("Standalone paragraph" in t for t in result_texts), (
            "Unmatched pymupdf block was dropped"
        )

    def test_unmatched_pymupdf_preserves_original_object_identity(self):
        """The exact same TextBlock object should appear in the output."""
        outside = make_pymupdf_block("Orphan block.", BBox(400, 700, 590, 720))
        surya = make_surya_block("", BBox(50, 100, 300, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks([outside], [surya], page_ir=page_ir)

        assert outside in result, "Original unmatched block object not in output"

    def test_multiple_unmatched_pymupdf_blocks_all_retained(self):
        outside_1 = make_pymupdf_block("Orphan 1", BBox(0, 750, 100, 762))
        outside_2 = make_pymupdf_block("Orphan 2", BBox(400, 750, 500, 762))
        surya = make_surya_block("", BBox(50, 100, 300, 400))
        page_ir = make_page_ir()

        result = deduplicate_blocks(
            [outside_1, outside_2], [surya], page_ir=page_ir
        )
        result_texts = [b.text for b in result]
        assert "Orphan 1" in result_texts
        assert "Orphan 2" in result_texts


# ---------------------------------------------------------------------------
# Core: test 3 — surya block with no pymupdf coverage → flagged for OCR
# ---------------------------------------------------------------------------


class TestSuryaBlockWithNoPymupdf:
    def test_warning_added_to_page_ir(self):
        """A surya block with no pymupdf blocks inside must trigger a warning."""
        surya = make_surya_block("", BBox(50, 600, 300, 700))  # no text
        page_ir = make_page_ir()

        deduplicate_blocks([], [surya], page_ir=page_ir)

        ocr_warnings = [w for w in page_ir.warnings
                        if _NEEDS_OCR_WARNING_PREFIX in w]
        assert ocr_warnings, (
            "No re-OCR warning added for surya block with empty text and no pymupdf coverage"
        )

    def test_warning_contains_block_id(self):
        surya = make_surya_block("", BBox(50, 600, 300, 700))
        page_ir = make_page_ir()

        result = deduplicate_blocks([], [surya], page_ir=page_ir)

        ocr_warnings = [w for w in page_ir.warnings
                        if _NEEDS_OCR_WARNING_PREFIX in w]
        assert ocr_warnings
        # The warning must reference an actual block ID
        block_ids = {b.id for b in result}
        assert any(any(bid in w for bid in block_ids) for w in ocr_warnings), (
            "OCR warning doesn't reference the actual block ID"
        )

    def test_surya_block_with_text_not_flagged_for_ocr(self):
        """If surya already has text, it should NOT be flagged for re-OCR."""
        surya = make_surya_block("Surya extracted this text.", BBox(50, 600, 300, 700))
        page_ir = make_page_ir()

        deduplicate_blocks([], [surya], page_ir=page_ir)

        ocr_warnings = [w for w in page_ir.warnings
                        if _NEEDS_OCR_WARNING_PREFIX in w]
        assert not ocr_warnings, (
            "Surya block with existing text should not be flagged for re-OCR"
        )

    def test_no_pymupdf_all_surya_empty_page_warning_count(self):
        """Three empty surya blocks → three re-OCR warnings."""
        surya_blocks = [
            make_surya_block("", BBox(50, y, 300, y + 80))
            for y in [100, 250, 400]
        ]
        page_ir = make_page_ir()

        deduplicate_blocks([], surya_blocks, page_ir=page_ir)

        ocr_warnings = [w for w in page_ir.warnings
                        if _NEEDS_OCR_WARNING_PREFIX in w]
        assert len(ocr_warnings) == 3


# ---------------------------------------------------------------------------
# Core: test 4 — no duplicate text across output
# ---------------------------------------------------------------------------


class TestNoDuplicateText:
    def test_same_sentence_not_duplicated(self):
        """
        If a pymupdf block is consumed by a surya region, its text must NOT
        also appear as a standalone block.
        """
        sentence = "The quick brown fox jumps over the lazy dog."
        pm_block = make_pymupdf_block(sentence, BBox(60, 110, 490, 122))
        surya = make_surya_block("", BBox(50, 100, 500, 250))
        page_ir = make_page_ir()

        result = deduplicate_blocks([pm_block], [surya], page_ir=page_ir)

        occurrences = sum(1 for b in result if sentence in b.text)
        assert occurrences == 1, (
            f"Sentence appeared {occurrences} times — expected exactly 1"
        )

    def test_no_duplicate_across_multiple_surya_blocks(self):
        """A pymupdf block can only be claimed by one surya region."""
        pm = make_pymupdf_block("Shared text.", BBox(60, 150, 490, 162))

        # Two overlapping surya blocks — the pymupdf center can only be in one
        surya_a = make_surya_block("", BBox(50, 100, 500, 200))
        surya_b = make_surya_block("", BBox(50, 140, 500, 300))  # overlaps a
        page_ir = make_page_ir()

        result = deduplicate_blocks([pm], [surya_a, surya_b], page_ir=page_ir)

        occurrences = sum(1 for b in result if "Shared text." in b.text)
        assert occurrences == 1, (
            f"'Shared text.' appeared {occurrences} times — must be exactly 1"
        )

    def test_full_page_no_duplicates(self):
        """Larger scenario: 20 pymupdf lines, 2 surya blocks — no text duplicated."""
        blocks_a = [
            make_pymupdf_block(f"Para A line {i}.", BBox(50, 100 + i * 12, 300, 112 + i * 12))
            for i in range(10)
        ]
        blocks_b = [
            make_pymupdf_block(f"Para B line {i}.", BBox(50, 400 + i * 12, 300, 412 + i * 12))
            for i in range(10)
        ]
        surya_a = make_surya_block("", BBox(50, 100, 300, 230))
        surya_b = make_surya_block("", BBox(50, 400, 300, 530))
        page_ir = make_page_ir()

        result = deduplicate_blocks(
            blocks_a + blocks_b, [surya_a, surya_b], page_ir=page_ir
        )

        all_text = " ".join(b.text for b in result)
        for i in range(10):
            assert all_text.count(f"Para A line {i}.") == 1
            assert all_text.count(f"Para B line {i}.") == 1


# ---------------------------------------------------------------------------
# Quality threshold tests
# ---------------------------------------------------------------------------


class TestQualityThreshold:
    def test_low_quality_pymupdf_uses_surya_text(self):
        """When pymupdf quality is low, surya text should win."""
        pm_block = make_pymupdf_block(
            "G4rb4g3 0CR t3xt",
            BBox(60, 110, 490, 122),
            text_quality=0.2,  # below default threshold of 0.7
        )
        surya = make_surya_block(
            "Good surya text here.", BBox(50, 100, 500, 250)
        )
        page_ir = make_page_ir()

        result = deduplicate_blocks(
            [pm_block], [surya], quality_threshold=DEFAULT_QUALITY_THRESHOLD,
            page_ir=page_ir,
        )

        assert len(result) == 1
        assert result[0].text == "Good surya text here."

    def test_high_quality_pymupdf_wins_over_surya(self):
        """When pymupdf quality is high, pymupdf text should win."""
        pm_block = make_pymupdf_block(
            "High quality text from PDF layer.",
            BBox(60, 110, 490, 122),
            text_quality=1.0,
        )
        surya = make_surya_block(
            "Surya version of the text.", BBox(50, 100, 500, 250)
        )
        page_ir = make_page_ir()

        result = deduplicate_blocks([pm_block], [surya], page_ir=page_ir)

        assert len(result) == 1
        assert "High quality text from PDF layer." in result[0].text

    def test_custom_threshold(self):
        """A custom threshold of 0.5 should change which source wins."""
        pm_block = make_pymupdf_block(
            "Medium quality text.",
            BBox(60, 110, 490, 122),
            text_quality=0.6,  # above 0.5, below 0.7
        )
        surya = make_surya_block("Surya text.", BBox(50, 100, 500, 250))
        page_ir = make_page_ir()

        result = deduplicate_blocks(
            [pm_block], [surya], quality_threshold=0.5, page_ir=page_ir
        )

        # quality 0.6 > threshold 0.5 → pymupdf should win
        assert "Medium quality text." in result[0].text


# ---------------------------------------------------------------------------
# assemble_page tests
# ---------------------------------------------------------------------------


class TestAssemblePage:
    def test_assembles_mixed_blocks_correctly(self):
        pm = make_pymupdf_block("Line of text.", BBox(60, 110, 490, 122))
        surya = make_surya_block("", BBox(50, 100, 500, 250))
        page_ir = make_page_ir(blocks=[pm, surya])

        result_page = assemble_page(page_ir)

        assert len(result_page.blocks) == 1
        assert "Line of text." in result_page.blocks[0].text

    def test_no_surya_blocks_leaves_page_unchanged(self):
        pm = make_pymupdf_block("Standalone.", BBox(60, 110, 490, 122))
        page_ir = make_page_ir(blocks=[pm])

        original_blocks = list(page_ir.blocks)
        assemble_page(page_ir)

        assert page_ir.blocks == original_blocks

    def test_all_block_ids_are_valid_uuids_after_assembly(self):
        lines = [
            make_pymupdf_block(f"Line {i}.", BBox(60, 100 + i * 15, 490, 112 + i * 15))
            for i in range(5)
        ]
        surya = make_surya_block("", BBox(50, 95, 500, 200))
        page_ir = make_page_ir(blocks=lines + [surya])

        assemble_page(page_ir)

        for block in page_ir.blocks:
            assert block.id, "Block ID is empty"
            parsed = uuid.UUID(block.id)
            assert str(parsed) == block.id


# ---------------------------------------------------------------------------
# run_stage6 (document-level) tests
# ---------------------------------------------------------------------------


class TestRunStage6:
    def _make_doc_ir_with_mixed_page(self) -> DocumentIR:
        pm1 = make_pymupdf_block("First paragraph line 1.", BBox(60, 110, 490, 122))
        pm2 = make_pymupdf_block("First paragraph line 2.", BBox(60, 125, 490, 137))
        surya1 = make_surya_block("", BBox(50, 100, 500, 200), block_type=BlockType.BODY)
        pm3 = make_pymupdf_block("Second paragraph.", BBox(60, 300, 490, 312))
        surya2 = make_surya_block("", BBox(50, 290, 500, 350), block_type=BlockType.BODY)

        page = make_page_ir(page_num=0, blocks=[pm1, pm2, surya1, pm3, surya2])
        doc_ir = DocumentIR(pages=[page])
        return doc_ir

    def test_run_stage6_returns_document_ir(self):
        doc_ir = self._make_doc_ir_with_mixed_page()
        result = run_stage6(doc_ir)
        assert isinstance(result, DocumentIR)

    def test_run_stage6_merges_correctly(self):
        doc_ir = self._make_doc_ir_with_mixed_page()
        result = run_stage6(doc_ir)

        all_text = " ".join(b.text for b in result.pages[0].blocks)
        assert "First paragraph line 1." in all_text
        assert "First paragraph line 2." in all_text
        assert "Second paragraph." in all_text

    def test_run_stage6_no_text_duplicates(self):
        doc_ir = self._make_doc_ir_with_mixed_page()
        result = run_stage6(doc_ir)

        all_text = " ".join(b.text for b in result.pages[0].blocks)
        assert all_text.count("First paragraph line 1.") == 1
        assert all_text.count("First paragraph line 2.") == 1
        assert all_text.count("Second paragraph.") == 1

    def test_run_stage6_parent_ref_integrity(self):
        doc_ir = self._make_doc_ir_with_mixed_page()
        result = run_stage6(doc_ir)
        errors = result.validate_parent_refs()
        assert errors == [], f"Parent ref integrity errors: {errors}"

    def test_run_stage6_multi_page(self):
        def _page(page_num):
            pm = make_pymupdf_block(f"Page {page_num} text.", BBox(60, 110, 490, 122), page_num=page_num)
            su = make_surya_block("", BBox(50, 100, 500, 200), page_num=page_num)
            return make_page_ir(page_num=page_num, blocks=[pm, su])

        doc_ir = DocumentIR(pages=[_page(0), _page(1), _page(2)])
        result = run_stage6(doc_ir)

        for i, page in enumerate(result.pages):
            all_text = " ".join(b.text for b in page.blocks)
            assert f"Page {i} text." in all_text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_pymupdf_and_surya_lists(self):
        page_ir = make_page_ir()
        result = deduplicate_blocks([], [], page_ir=page_ir)
        assert result == []

    def test_surya_block_without_bbox_kept_as_is(self):
        surya = TextBlock(
            id=str(uuid.uuid4()),
            text="No bbox block",
            bbox=None,
            extraction_method=ExtractionMethod.SURYA_LAYOUT,
        )
        page_ir = make_page_ir()
        result = deduplicate_blocks([], [surya], page_ir=page_ir)
        assert surya in result

    def test_pymupdf_block_without_bbox_not_matched(self):
        """A pymupdf block with no bbox cannot be containment-tested — stays unmatched."""
        pm = TextBlock(
            id=str(uuid.uuid4()),
            text="No bbox pymupdf block",
            bbox=None,
            extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
        )
        surya = make_surya_block("", BBox(0, 0, 500, 500))
        page_ir = make_page_ir()

        result = deduplicate_blocks([pm], [surya], page_ir=page_ir)

        # The pymupdf block should appear as unmatched (not consumed)
        assert pm in result

    def test_all_block_ids_non_empty_in_output(self):
        """No block should ever have an empty or None ID."""
        pm_blocks = [
            make_pymupdf_block(f"Text {i}", BBox(60, 100 + i * 20, 490, 112 + i * 20))
            for i in range(5)
        ]
        surya_blocks = [
            make_surya_block("", BBox(50, 95, 500, 210)),
            make_surya_block("", BBox(50, 500, 500, 600)),  # no pymupdf match
        ]
        page_ir = make_page_ir()

        result = deduplicate_blocks(pm_blocks, surya_blocks, page_ir=page_ir)

        for block in result:
            assert block.id, f"Block has empty ID: {block}"
            uuid.UUID(block.id)  # raises ValueError if invalid
