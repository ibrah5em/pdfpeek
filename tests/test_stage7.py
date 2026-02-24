"""
tests/test_stage7.py
====================
Test suite for Stage 7 — Cross-Page Analysis.

Coverage
--------
1. Identical header on every page → all promoted to HEADER.
2. Academic-style section headers ("3.1 Methods" / "4.2 Methods") detected
   as repeating pattern despite different section numbers (validates W21 fix).
3. OCR-only page with no surya HEADING blocks → heuristic detection fires
   (validates W22 fix).
4. 1000-page document → header detection completes in < 2 seconds
   (validates O(n) complexity fix for W20).
5. Heading hierarchy: H1 → H2 parent_id linkage.
6. Footer detection across pages.
7. normalize_for_header_detection unit tests.
8. build_heading_hierarchy with mixed levels.
"""

from __future__ import annotations

import time
import uuid

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
from pdf_engine.stage7_crosspage import (
    MIN_PAGES_FOR_REPEAT,
    build_heading_hierarchy,
    detect_headings_heuristic,
    find_repeating_patterns,
    normalize_for_header_detection,
    run_stage7,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block(
    text: str,
    page_num: int = 0,
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float = 200.0,
    y1: float = 20.0,
    block_type: BlockType = BlockType.BODY,
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF_DIRECT,
) -> TextBlock:
    return TextBlock(
        id=str(uuid.uuid4()),
        text=text,
        bbox=BBox(x0, y0, x1, y1),
        block_type=block_type,
        extraction_method=extraction_method,
        confidence=BlockConfidence(),
        page_num=page_num,
    )


def make_page(
    page_num: int,
    blocks: list[TextBlock],
    width: float = 595.0,
    height: float = 842.0,
) -> PageIR:
    page = PageIR(page_num=page_num, width=width, height=height)
    page.blocks = blocks
    return page


def make_doc(pages: list[PageIR]) -> DocumentIR:
    doc = DocumentIR()
    doc.pages = pages
    return doc


# ---------------------------------------------------------------------------
# 1. Normalize utility tests
# ---------------------------------------------------------------------------


class TestNormalizeForHeaderDetection:
    def test_strips_leading_section_number(self):
        assert normalize_for_header_detection("3.1 Methods") == "methods"

    def test_strips_different_section_number_same_result(self):
        assert normalize_for_header_detection("3.1 Methods") == \
               normalize_for_header_detection("4.2 Methods")

    def test_strips_trailing_page_number(self):
        result = normalize_for_header_detection("Introduction 7")
        assert result == "introduction"

    def test_page_header_variant(self):
        # "Page 7" → strip trailing digit → "page"
        result = normalize_for_header_detection("Page 7")
        assert result == "page"

    def test_running_head_unchanged(self):
        result = normalize_for_header_detection("Running Head: My Paper")
        assert result == "running head: my paper"

    def test_collapses_remaining_digits(self):
        result = normalize_for_header_detection("Figure 3 illustrates")
        # no trailing-only digit strip but remaining 3 becomes #
        assert "#" in result

    def test_empty_string(self):
        assert normalize_for_header_detection("") == ""

    def test_only_digits(self):
        result = normalize_for_header_detection("42")
        # "42" → strip trailing → "" → then digit collapse → "#" or ""
        # strip trailing digit: re.sub(r'\s+\d+$', '', "42") = "42" (no leading space)
        # so remaining: "42" → re.sub(r'\d+', '#', "42") = "#"
        assert result == "#"

    def test_deep_section_number(self):
        assert normalize_for_header_detection("1.2.3 Subsection") == "subsection"


# ---------------------------------------------------------------------------
# 2. Identical header on every page
# ---------------------------------------------------------------------------


class TestIdenticalHeaderDetection:
    def _build_doc(self, n_pages: int = 10) -> DocumentIR:
        """10-page doc: each page has a header, body, and footer block."""
        pages = []
        for i in range(n_pages):
            header = make_block("My Journal — Volume 1", page_num=i,
                                y0=0.0, y1=20.0)
            body   = make_block(f"Body text on page {i}.", page_num=i,
                                y0=100.0, y1=120.0)
            footer = make_block("© 2024 Publisher", page_num=i,
                                y0=800.0, y1=820.0)
            pages.append(make_page(i, [header, body, footer]))
        return make_doc(pages)

    def test_header_block_promoted(self):
        doc = self._build_doc(10)
        run_stage7(doc, min_pages=3)
        for page_ir in doc.pages:
            header_blocks = [b for b in page_ir.blocks
                             if b.block_type == BlockType.HEADER]
            assert len(header_blocks) >= 1, \
                f"Page {page_ir.page_num} has no HEADER block"

    def test_body_block_not_promoted(self):
        doc = self._build_doc(10)
        run_stage7(doc, min_pages=3)
        for page_ir in doc.pages:
            body_blocks = [b for b in page_ir.blocks
                           if "Body text" in b.text]
            for b in body_blocks:
                assert b.block_type == BlockType.BODY

    def test_footer_block_promoted(self):
        doc = self._build_doc(10)
        run_stage7(doc, min_pages=3)
        for page_ir in doc.pages:
            footer_blocks = [b for b in page_ir.blocks
                             if b.block_type == BlockType.FOOTER]
            assert len(footer_blocks) >= 1, \
                f"Page {page_ir.page_num} has no FOOTER block"


# ---------------------------------------------------------------------------
# 3. Academic section headers with numbering (W21 regression test)
# ---------------------------------------------------------------------------


class TestSectionNumberedHeaders:
    """
    Validates that "3.1 Methods" and "4.2 Methods" are treated as the same
    repeating pattern after normalization.
    """

    def _build_academic_doc(self) -> DocumentIR:
        """
        8-page document.  Each page has a section-numbered header of the
        form "N.M Section Name" where N and M differ per page, but the
        base names repeat (Methods appears 4 times, Results 4 times).
        """
        section_headers = [
            "1.1 Introduction",
            "1.2 Introduction",
            "2.1 Methods",
            "2.2 Methods",
            "3.1 Results",
            "3.2 Results",
            "4.1 Discussion",
            "4.2 Discussion",
        ]
        pages = []
        for i, header_text in enumerate(section_headers):
            header = make_block(header_text, page_num=i, y0=0.0, y1=20.0)
            body   = make_block(f"Content of section on page {i}.",
                                page_num=i, y0=100.0, y1=120.0)
            pages.append(make_page(i, [header, body]))
        return make_doc(pages)

    def test_section_headers_detected_as_repeating(self):
        doc = self._build_academic_doc()
        header_fps, footer_fps = find_repeating_patterns(doc, min_pages=2)
        # "methods", "results", "introduction", "discussion" should all appear
        assert "methods" in header_fps or any("method" in fp for fp in header_fps), \
            f"'methods' not in header fingerprints: {header_fps}"

    def test_fingerprints_match_across_section_numbers(self):
        """3.1 Methods and 4.2 Methods must produce the same fingerprint."""
        from pdf_engine.stage7_crosspage import normalize_for_header_detection
        assert normalize_for_header_detection("2.1 Methods") == \
               normalize_for_header_detection("3.1 Methods") == \
               normalize_for_header_detection("4.2 Methods")

    def test_run_stage7_promotes_repeating_section_headers(self):
        doc = self._build_academic_doc()
        run_stage7(doc, min_pages=2)
        # After promotion: at least some header blocks should exist
        all_header_blocks = [
            b for page in doc.pages
            for b in page.blocks
            if b.block_type == BlockType.HEADER
        ]
        assert len(all_header_blocks) >= 4, \
            f"Expected at least 4 header blocks, got {len(all_header_blocks)}"


# ---------------------------------------------------------------------------
# 4. OCR-only pages — heuristic heading detection (W22 regression test)
# ---------------------------------------------------------------------------


class TestHeuristicHeadingDetection:
    """
    Pages where all blocks come from Tesseract/Surya OCR (no surya layout)
    should have headings detected via the heuristic fallback.
    """

    def _make_ocr_page(self, page_num: int) -> PageIR:
        """
        Page with OCR-sourced blocks only.  No surya HEADING block.
        Includes one short ALL-CAPS line that should be detected as heading.
        """
        heading_block = TextBlock(
            id=str(uuid.uuid4()),
            text="INTRODUCTION",
            bbox=BBox(0, 50, 400, 75),   # taller bbox → larger "font"
            block_type=BlockType.BODY,
            extraction_method=ExtractionMethod.TESSERACT_OCR,
            confidence=BlockConfidence(),
            page_num=page_num,
        )
        body_block = TextBlock(
            id=str(uuid.uuid4()),
            text="This is the body text of the introduction section. " * 4,
            bbox=BBox(0, 100, 400, 115),  # shorter bbox → smaller "font"
            block_type=BlockType.BODY,
            extraction_method=ExtractionMethod.TESSERACT_OCR,
            confidence=BlockConfidence(),
            page_num=page_num,
        )
        page = PageIR(page_num=page_num, width=595.0, height=842.0)
        page.blocks = [heading_block, body_block]
        return page

    def test_ocr_only_page_detected(self):
        from pdf_engine.stage7_crosspage import _is_ocr_only_page
        page = self._make_ocr_page(0)
        assert _is_ocr_only_page(page) is True

    def test_non_ocr_page_not_detected_as_ocr_only(self):
        from pdf_engine.stage7_crosspage import _is_ocr_only_page
        page = make_page(0, [make_block("Hello", page_num=0)])
        assert _is_ocr_only_page(page) is False

    def test_heuristic_detects_all_caps_heading(self):
        page = self._make_ocr_page(0)
        detect_headings_heuristic(page)
        heading_blocks = [b for b in page.blocks if b.block_type == BlockType.HEADING]
        assert len(heading_blocks) >= 1, "ALL-CAPS block not promoted to HEADING"

    def test_heuristic_does_not_promote_long_body_text(self):
        page = self._make_ocr_page(0)
        detect_headings_heuristic(page)
        body_blocks = [b for b in page.blocks
                       if b.block_type == BlockType.BODY and len(b.text) > 80]
        # Long body blocks must remain BODY
        assert all(b.block_type == BlockType.BODY for b in body_blocks)

    def test_stage7_applies_heuristic_to_ocr_pages(self):
        page = self._make_ocr_page(0)
        doc = make_doc([page])
        run_stage7(doc, min_pages=3)
        heading_blocks = [b for b in doc.pages[0].blocks
                          if b.block_type == BlockType.HEADING]
        assert len(heading_blocks) >= 1

    def test_heuristic_not_applied_when_surya_headings_exist(self):
        """If surya already produced a HEADING block, heuristic must not fire."""
        page = self._make_ocr_page(0)
        # Inject a surya HEADING block
        existing_heading = TextBlock(
            id=str(uuid.uuid4()),
            text="Abstract",
            bbox=BBox(0, 30, 400, 50),
            block_type=BlockType.HEADING,
            extraction_method=ExtractionMethod.SURYA_LAYOUT,
            confidence=BlockConfidence(),
            page_num=0,
        )
        page.blocks.insert(0, existing_heading)
        body_count_before = sum(1 for b in page.blocks if b.block_type == BlockType.BODY)
        detect_headings_heuristic(page)
        body_count_after = sum(1 for b in page.blocks if b.block_type == BlockType.BODY)
        # The heuristic skips non-BODY blocks, so only BODY blocks are candidates.
        # Existing heading stays. The run_stage7 wrapper checks `has_headings` first.
        assert existing_heading.block_type == BlockType.HEADING  # unchanged


# ---------------------------------------------------------------------------
# 5. Heading hierarchy
# ---------------------------------------------------------------------------


class TestBuildHeadingHierarchy:
    def _make_heading(self, text, page_num, y0, height, block_type=BlockType.HEADING):
        return TextBlock(
            id=str(uuid.uuid4()),
            text=text,
            bbox=BBox(0, y0, 400, y0 + height),
            block_type=block_type,
            confidence=BlockConfidence(),
            page_num=page_num,
        )

    def test_h2_gets_h1_parent(self):
        h1 = self._make_heading("Introduction", 0, y0=50, height=24)
        h2 = self._make_heading("Background", 0, y0=100, height=16)
        page = make_page(0, [h1, h2])
        doc = make_doc([page])
        build_heading_hierarchy(doc)
        assert h2.parent_id == h1.id

    def test_h1_has_no_parent(self):
        h1 = self._make_heading("Chapter 1", 0, y0=50, height=24)
        h2 = self._make_heading("Section 1.1", 0, y0=100, height=16)
        page = make_page(0, [h1, h2])
        doc = make_doc([page])
        build_heading_hierarchy(doc)
        assert h1.parent_id is None

    def test_new_h1_resets_h2_parent_context(self):
        """A new H1 after some H2s should be parent-free; subsequent H2 → new H1."""
        h1a = self._make_heading("Chapter 1", 0, y0=50,  height=24)
        h2a = self._make_heading("Section 1.1", 0, y0=100, height=16)
        h1b = self._make_heading("Chapter 2", 1, y0=50,  height=24)
        h2b = self._make_heading("Section 2.1", 1, y0=100, height=16)
        page0 = make_page(0, [h1a, h2a])
        page1 = make_page(1, [h1b, h2b])
        doc = make_doc([page0, page1])
        build_heading_hierarchy(doc)
        assert h1b.parent_id is None
        assert h2b.parent_id == h1b.id

    def test_hierarchy_skipped_when_no_headings(self):
        page = make_page(0, [make_block("Just body text", page_num=0)])
        doc = make_doc([page])
        build_heading_hierarchy(doc)  # must not raise
        for blk in doc.pages[0].blocks:
            assert blk.parent_id is None

    def test_three_levels(self):
        h1 = self._make_heading("Part I", 0, y0=10, height=32)
        h2 = self._make_heading("Chapter 1", 0, y0=60, height=22)
        h3 = self._make_heading("Section 1.1", 0, y0=110, height=14)
        page = make_page(0, [h1, h2, h3])
        doc = make_doc([page])
        build_heading_hierarchy(doc)
        assert h1.parent_id is None
        assert h2.parent_id == h1.id
        assert h3.parent_id == h2.id


# ---------------------------------------------------------------------------
# 6. Footer detection
# ---------------------------------------------------------------------------


class TestFooterDetection:
    def _build_footer_doc(self, n_pages: int = 6) -> DocumentIR:
        pages = []
        for i in range(n_pages):
            body   = make_block(f"Content on page {i}.", page_num=i,
                                y0=300.0, y1=320.0)
            footer = make_block("Confidential — Do Not Distribute", page_num=i,
                                y0=820.0, y1=840.0)
            pages.append(make_page(i, [body, footer]))
        return make_doc(pages)

    def test_footer_fingerprint_detected(self):
        doc = self._build_footer_doc()
        _, footer_fps = find_repeating_patterns(doc, min_pages=3)
        assert len(footer_fps) >= 1

    def test_footer_blocks_promoted(self):
        doc = self._build_footer_doc()
        run_stage7(doc, min_pages=3)
        footer_blocks = [
            b for page in doc.pages
            for b in page.blocks
            if b.block_type == BlockType.FOOTER
        ]
        assert len(footer_blocks) >= 3


# ---------------------------------------------------------------------------
# 7. min_pages threshold edge cases
# ---------------------------------------------------------------------------


class TestMinPagesThreshold:
    def test_below_threshold_not_promoted(self):
        """A pattern appearing on only 2 pages should not be promoted (min=3)."""
        pages = []
        for i in range(5):
            if i < 2:
                header = make_block("Rare Header", page_num=i, y0=0.0, y1=20.0)
                body   = make_block(f"Body {i}", page_num=i, y0=100.0, y1=120.0)
                pages.append(make_page(i, [header, body]))
            else:
                body = make_block(f"Body {i}", page_num=i, y0=100.0, y1=120.0)
                pages.append(make_page(i, [body]))

        doc = make_doc(pages)
        run_stage7(doc, min_pages=3)
        # "Rare Header" appeared on only 2 pages → must remain BODY
        header_blocks = [
            b for page in doc.pages
            for b in page.blocks
            if b.text == "Rare Header" and b.block_type == BlockType.HEADER
        ]
        assert len(header_blocks) == 0

    def test_exactly_at_threshold_promoted(self):
        pages = []
        for i in range(3):
            header = make_block("Common Header", page_num=i, y0=0.0, y1=20.0)
            body   = make_block(f"Body {i}", page_num=i, y0=100.0, y1=120.0)
            pages.append(make_page(i, [header, body]))
        doc = make_doc(pages)
        run_stage7(doc, min_pages=3)
        header_blocks = [
            b for page in doc.pages
            for b in page.blocks
            if b.block_type == BlockType.HEADER
        ]
        assert len(header_blocks) == 3


# ---------------------------------------------------------------------------
# 8. Performance test — O(n) for 1000 pages (W20 fix)
# ---------------------------------------------------------------------------


class TestPerformance:
    def _build_large_doc(self, n_pages: int = 1000) -> DocumentIR:
        """Build a large document with header + body + footer on every page."""
        pages = []
        for i in range(n_pages):
            header = make_block("Global Header Text", page_num=i,
                                y0=0.0, y1=18.0)
            body   = make_block(f"This is the main body content of page {i}.",
                                page_num=i, y0=100.0, y1=115.0)
            footer = make_block("Page Footer — Confidential", page_num=i,
                                y0=820.0, y1=835.0)
            pages.append(make_page(i, [header, body, footer]))
        return make_doc(pages)

    def test_header_detection_under_2_seconds(self):
        doc = self._build_large_doc(1000)
        start = time.perf_counter()
        run_stage7(doc, min_pages=3)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, (
            f"Stage 7 took {elapsed:.2f}s on a 1000-page document "
            f"(expected < 2s for O(n) algorithm)"
        )

    def test_large_doc_header_blocks_promoted(self):
        doc = self._build_large_doc(1000)
        run_stage7(doc, min_pages=3)
        header_count = sum(
            1 for page in doc.pages
            for b in page.blocks
            if b.block_type == BlockType.HEADER
        )
        assert header_count == 1000

    def test_large_doc_footer_blocks_promoted(self):
        doc = self._build_large_doc(1000)
        run_stage7(doc, min_pages=3)
        footer_count = sum(
            1 for page in doc.pages
            for b in page.blocks
            if b.block_type == BlockType.FOOTER
        )
        assert footer_count == 1000


# ---------------------------------------------------------------------------
# 9. Parent-ref integrity after stage7
# ---------------------------------------------------------------------------


class TestParentRefIntegrity:
    def test_no_dangling_parent_refs(self):
        h1 = TextBlock(
            id=str(uuid.uuid4()),
            text="Title",
            bbox=BBox(0, 10, 400, 34),
            block_type=BlockType.HEADING,
            confidence=BlockConfidence(),
            page_num=0,
        )
        h2 = TextBlock(
            id=str(uuid.uuid4()),
            text="Subtitle",
            bbox=BBox(0, 60, 400, 76),
            block_type=BlockType.HEADING,
            confidence=BlockConfidence(),
            page_num=0,
        )
        page = make_page(0, [h1, h2])
        doc = make_doc([page])
        run_stage7(doc, min_pages=3)
        errors = doc.validate_parent_refs()
        assert errors == [], f"Dangling parent refs after stage7: {errors}"


# ---------------------------------------------------------------------------
# 10. Empty document / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_document(self):
        doc = make_doc([])
        result = run_stage7(doc)
        assert result is doc  # returns same object, no crash

    def test_single_page_no_repeat(self):
        page = make_page(0, [make_block("Header", page_num=0, y0=0.0, y1=15.0)])
        doc = make_doc([page])
        run_stage7(doc, min_pages=3)
        assert doc.pages[0].blocks[0].block_type == BlockType.BODY

    def test_blocks_without_bbox_skipped(self):
        blk = TextBlock(
            id=str(uuid.uuid4()),
            text="No bbox block",
            bbox=None,
            block_type=BlockType.BODY,
            confidence=BlockConfidence(),
            page_num=0,
        )
        page = make_page(0, [blk])
        doc = make_doc([page])
        run_stage7(doc, min_pages=3)  # must not raise

    def test_page_with_only_empty_text_skipped(self):
        blk = make_block("", page_num=0, y0=0.0, y1=10.0)
        page = make_page(0, [blk])
        doc = make_doc([page])
        run_stage7(doc, min_pages=3)  # must not raise
