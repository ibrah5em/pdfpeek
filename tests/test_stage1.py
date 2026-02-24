"""
tests/test_stage1.py
====================
Tests for Stage 1 — Text Layer Extraction + Phantom Detection.

Test strategy
-------------
Most tests use synthetic PyMuPDF page mocks constructed via ``unittest.mock``
so the suite runs without needing real PDF files or Tesseract installed.

Real-PDF integration tests (marked with ``pytest.mark.integration``) create
tiny PDFs in memory with ``fitz.open()`` and test the full call chain.
Run them with:  ``pytest -m integration``

Coverage goals
--------------
✓  Born-digital page with real text → phantom_score ≥ 0.85
✓  Page with invisible text (render mode 3) → phantom_score ≤ 0.30
✓  Chinese page (no spaces) → quick_text_quality_check ≥ 0.75  (W5 fix)
✓  rawdict is used, not dict — assert per-character bboxes are returned
✓  extract_text_layer skips "ocr_needed" pages
✓  extract_text_layer creates TextBlock objects with valid UUIDs
✓  script_direction detected correctly for RTL text
✓  Phantom downgrade: text_native page with low score → becomes hybrid
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

# --- Module under test ---
from pdf_engine.stage1_extraction import (
    PHANTOM_THRESHOLD,
    ZERO_WIDTH_THRESHOLD,
    _detect_script_direction,
    _normalised_similarity,
    _s1_zero_width_ratio,
    _s2_render_mode_ratio,
    detect_phantom_text_layer,
    extract_text_layer,
    get_script_family,
    quick_text_quality_check,
)
from pdf_engine.models import (
    ExtractionMethod,
    PageIR,
    TextBlock,
)


# ===========================================================================
# Helpers — build synthetic rawdict structures
# ===========================================================================


def _make_char(
    text: str = "a",
    x0: float = 0.0,
    x1: float = 10.0,
    render_mode: int = 0,
) -> dict:
    """
    Build a synthetic rawdict character entry.

    HIGH-7 fix: render_mode parameter is kept for backward compatibility
    but now affects char width instead of flags. render_mode=3 creates
    near-zero-width chars to simulate invisible text.
    """
    # For invisible text (render_mode=3), make char width < 0.1pt
    if render_mode == 3:
        x1 = x0 + 0.05  # near-zero width
    return {
        "c": text,
        "bbox": (x0, 0.0, x1, 10.0),
    }


def _make_span(chars: list[dict], render_mode: int = 0) -> dict:
    """
    Build a synthetic rawdict span.

    HIGH-7 fix: render_mode parameter is kept for backward compatibility
    but now affects font properties instead of flags. render_mode=3 creates
    microscopic font size to simulate invisible text.
    """
    # For invisible text (render_mode=3), use microscopic font or invisible font name
    font_size = 0.1 if render_mode == 3 else 12.0
    font_name = "GlyphLessFont" if render_mode == 3 else "Helvetica"

    return {
        "chars": chars,
        "flags": 0,  # No longer used for render mode detection
        "font": font_name,
        "size": font_size,
    }


def _make_line(spans: list[dict]) -> dict:
    return {"spans": spans}


def _make_block(lines: list[dict], bbox: tuple = (0, 0, 100, 20)) -> dict:
    return {"type": 0, "lines": lines, "bbox": bbox}


def _make_rawdict(blocks: list[dict]) -> dict:
    return {"blocks": blocks}


def _make_mock_page(
    rawdict: dict,
    plain_text: str = "",
    page_rect: Optional[tuple] = None,
) -> MagicMock:
    """
    Create a MagicMock that behaves like a PyMuPDF page for Stage 1 purposes.
    """
    page = MagicMock()
    page.get_text.side_effect = lambda mode, **kw: (
        rawdict if mode == "rawdict" else plain_text
    )
    if page_rect is None:
        page_rect = (0, 0, 595, 842)   # A4
    rect = SimpleNamespace(
        x0=page_rect[0], y0=page_rect[1],
        x1=page_rect[2], y1=page_rect[3],
        width=page_rect[2] - page_rect[0],
        height=page_rect[3] - page_rect[1],
    )
    page.rect = rect
    return page


# ===========================================================================
# quick_text_quality_check — language-aware W5 fix
# ===========================================================================


class TestQuickTextQualityCheck:

    def test_normal_english_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        score = quick_text_quality_check(text, language="en")
        assert score >= 0.85, f"Expected ≥0.85, got {score:.3f}"

    def test_chinese_text_no_spaces(self):
        """CJK text has no spaces — must not be penalised (W5 fix)."""
        text = "你好世界这是一段中文文本没有空格但是应该得到高分" * 10
        score = quick_text_quality_check(text, language="zh")
        assert score >= 0.75, f"Expected ≥0.75, got {score:.3f}"

    def test_chinese_text_language_inferred(self):
        """Score CJK text without explicit language tag — detection from codepoints."""
        text = "日本語のテキストです。スペースはほとんどありません。" * 5
        score = quick_text_quality_check(text, language=None)
        assert score >= 0.75, f"Expected ≥0.75, got {score:.3f}"

    def test_arabic_text(self):
        text = "مرحبا بالعالم هذا نص عربي للاختبار " * 5
        score = quick_text_quality_check(text, language="ar")
        assert score >= 0.70, f"Expected ≥0.70, got {score:.3f}"

    def test_replacement_chars_penalised(self):
        bad_text = "\ufffd" * 50 + "some normal text"
        score = quick_text_quality_check(bad_text)
        assert score < 0.6, f"Expected <0.6, got {score:.3f}"

    def test_empty_string(self):
        assert quick_text_quality_check("") == 0.0

    def test_all_spaces_penalised(self):
        score = quick_text_quality_check("   " * 50, language="en")
        assert score < 0.5


# ===========================================================================
# get_script_family
# ===========================================================================


class TestGetScriptFamily:

    def test_explicit_chinese(self):
        assert get_script_family("zh") == "cjk"

    def test_explicit_japanese(self):
        assert get_script_family("ja-JP") == "cjk"

    def test_explicit_arabic(self):
        assert get_script_family("ar") == "arabic"

    def test_explicit_thai(self):
        assert get_script_family("th") == "thai"

    def test_explicit_english_default(self):
        assert get_script_family("en") == "default"

    def test_infer_cjk_from_text(self):
        text = "汉字" * 20
        assert get_script_family(None, sample_text=text) == "cjk"

    def test_infer_arabic_from_text(self):
        text = "مرحبا" * 15
        assert get_script_family(None, sample_text=text) == "arabic"

    def test_infer_default_latin(self):
        text = "Hello world, this is English text."
        assert get_script_family(None, sample_text=text) == "default"


# ===========================================================================
# _detect_script_direction
# ===========================================================================


class TestDetectScriptDirection:

    def test_latin_ltr(self):
        assert _detect_script_direction("Hello world this is English.") == "ltr"

    def test_arabic_rtl(self):
        assert _detect_script_direction("مرحبا بالعالم هذا نص عربي") == "rtl"

    def test_hebrew_rtl(self):
        assert _detect_script_direction("שלום עולם") == "rtl"

    def test_empty_defaults_ltr(self):
        assert _detect_script_direction("") == "ltr"

    def test_mixed_mostly_latin(self):
        text = "Hello " * 10 + "مرحبا"
        assert _detect_script_direction(text) == "ltr"


# ===========================================================================
# Signal 1 — zero-width bbox ratio
# ===========================================================================


class TestS1ZeroWidthRatio:

    def test_all_normal_chars_score_one(self):
        chars = [_make_char("a", 0, 10), _make_char("b", 10, 20)]
        span  = _make_span(chars)
        rd    = _make_rawdict([_make_block([_make_line([span])])])
        assert _s1_zero_width_ratio(rd) == pytest.approx(1.0)

    def test_all_zero_width_chars_score_zero(self):
        chars = [_make_char("a", 0, 0.05), _make_char("b", 5, 5.05)]
        span  = _make_span(chars)
        rd    = _make_rawdict([_make_block([_make_line([span])])])
        score = _s1_zero_width_ratio(rd)
        assert score == pytest.approx(0.0)

    def test_half_zero_width_half_score(self):
        chars = [
            _make_char("a", 0, 10),      # normal
            _make_char("b", 10, 10.05),  # zero-width
        ]
        span  = _make_span(chars)
        rd    = _make_rawdict([_make_block([_make_line([span])])])
        score = _s1_zero_width_ratio(rd)
        assert score == pytest.approx(0.5)

    def test_empty_rawdict_returns_one(self):
        assert _s1_zero_width_ratio({"blocks": []}) == pytest.approx(1.0)

    def test_image_blocks_ignored(self):
        """Type-1 (image) blocks must not be counted."""
        image_block = {"type": 1, "lines": []}
        rd = _make_rawdict([image_block])
        assert _s1_zero_width_ratio(rd) == pytest.approx(1.0)


# ===========================================================================
# Signal 2 — render mode 3 (invisible text)
# ===========================================================================


class TestS2RenderModeRatio:

    def test_all_visible_score_one(self):
        chars = [_make_char("a"), _make_char("b")]
        span  = _make_span(chars, render_mode=0)
        rd    = _make_rawdict([_make_block([_make_line([span])])])
        score, ratio = _s2_render_mode_ratio(rd)
        assert score == pytest.approx(1.0)
        assert ratio == pytest.approx(0.0)

    def test_all_invisible_score_zero(self):
        chars = [_make_char("a"), _make_char("b")]
        span  = _make_span(chars, render_mode=3)
        rd    = _make_rawdict([_make_block([_make_line([span])])])
        score, ratio = _s2_render_mode_ratio(rd)
        assert score == pytest.approx(0.0)
        assert ratio == pytest.approx(1.0)

    def test_empty_rawdict_returns_one(self):
        score, ratio = _s2_render_mode_ratio({"blocks": []})
        assert score == pytest.approx(1.0)
        assert ratio == pytest.approx(0.0)


# ===========================================================================
# detect_phantom_text_layer — integration of S1, S2, S3, (S4)
# ===========================================================================


class TestDetectPhantomTextLayer:

    def _page_with_real_text(self) -> MagicMock:
        """Simulate a page with clean, visible, normal-width characters."""
        real_text = "The quick brown fox jumps over the lazy dog. " * 10
        chars = [_make_char(c, i * 6, i * 6 + 5.9, render_mode=0)
                 for i, c in enumerate(real_text[:80])]
        span  = _make_span(chars, render_mode=0)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        return _make_mock_page(rd, plain_text=real_text)

    def _page_with_invisible_text(self) -> MagicMock:
        """Simulate a page where all text has render mode 3 (invisible)."""
        invisible_text = "invisible text overlay " * 20
        chars = [_make_char(c, i * 5, i * 5 + 4.9, render_mode=3)
                 for i, c in enumerate(invisible_text[:80])]
        # All spans carry render_mode=3
        span  = _make_span(chars, render_mode=3)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        return _make_mock_page(rd, plain_text=invisible_text)

    def test_real_text_high_score(self):
        """Born-digital page with real text → quality ≥ 0.85."""
        page = self._page_with_real_text()
        score = detect_phantom_text_layer(page)
        assert score >= 0.85, f"Expected ≥0.85, got {score:.3f}"

    def test_invisible_text_low_score(self):
        """Page with all invisible text (mode 3) → quality ≤ 0.30."""
        page = self._page_with_invisible_text()
        score = detect_phantom_text_layer(page)
        assert score <= 0.30, f"Expected ≤0.30, got {score:.3f}"

    def test_invisible_text_hard_cap_dominates(self):
        """
        Even when S1 and S3 are perfect, a 100 % invisible-text page must score
        near 0.0 (cap = 1 − 1.0 = 0.0).

        This validates that S2 is a *multiplicative cap*, not just an additive
        weight — because with additive-only weights the minimum reachable score
        when S1=S3=1.0 and S2=0.0 is ≈0.65, which would never satisfy ≤0.30.
        """
        page  = self._page_with_invisible_text()
        score = detect_phantom_text_layer(page)
        assert score <= 0.10, (
            f"100 % invisible text must cap score near 0.0, got {score:.3f}"
        )

    def test_rawdict_is_used_not_dict(self):
        """
        Confirm that ``page.get_text`` is called with ``"rawdict"``,
        never with ``"dict"``.  This enforces the W4 fix.
        """
        page = self._page_with_real_text()
        detect_phantom_text_layer(page)

        calls = [c[0][0] for c in page.get_text.call_args_list]
        assert "rawdict" in calls, "rawdict must be called"
        assert "dict" not in calls, "'dict' must never be called — use rawdict (W4)"


# ===========================================================================
# extract_text_layer — stage entry point
# ===========================================================================


class TestExtractTextLayer:

    def _make_real_text_page(self, text: str = "") -> MagicMock:
        text = text or ("Hello world. " * 20)
        chars = [_make_char(c, i * 6, i * 6 + 5.9)
                 for i, c in enumerate(text[:80])]
        span  = _make_span(chars, render_mode=0)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        return _make_mock_page(rd, plain_text=text)

    def test_skips_ocr_needed_pages(self):
        """Stage 1 must not touch pages triaged as 'ocr_needed'."""
        page    = MagicMock()
        page_ir = PageIR(page_num=0, triage_result="ocr_needed")
        result  = extract_text_layer(page, page_ir)
        page.get_text.assert_not_called()
        assert result.blocks == []

    def test_processes_text_native_pages(self):
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)
        assert len(result.blocks) >= 1

    def test_processes_hybrid_pages(self):
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="hybrid")
        result  = extract_text_layer(page, page_ir)
        assert len(result.blocks) >= 1

    def test_blocks_have_valid_uuids(self):
        """Every TextBlock must carry a non-empty UUID (design rule 1)."""
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)
        for block in result.blocks:
            assert block.id, "TextBlock.id must not be empty"
            parsed = uuid.UUID(block.id)   # raises if not a valid UUID
            assert str(parsed) == block.id

    def test_blocks_use_pymupdf_direct_method(self):
        """All blocks from Stage 1 must have PYMUPDF_DIRECT extraction method."""
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)
        for block in result.blocks:
            assert block.extraction_method == ExtractionMethod.PYMUPDF_DIRECT

    def test_rawdict_called_not_dict(self):
        """
        extract_text_layer must call get_text("rawdict") for block extraction,
        never get_text("dict").
        """
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="text_native")
        extract_text_layer(page, page_ir)
        modes_called = [c[0][0] for c in page.get_text.call_args_list]
        assert "rawdict" in modes_called
        assert "dict" not in modes_called, "Must use rawdict, not dict (W4)"

    def test_phantom_page_adds_warning(self):
        """Low phantom score on text_native → warning is appended."""
        # Craft a page where all chars are zero-width AND invisible.
        phantom_text = "x " * 40
        chars = [_make_char(c, i * 0.01, i * 0.01 + 0.05, render_mode=3)
                 for i, c in enumerate(phantom_text[:40])]
        span  = _make_span(chars, render_mode=3)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        page  = _make_mock_page(rd, plain_text=phantom_text)

        page_ir = PageIR(page_num=2, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)

        assert any("phantom" in w.lower() for w in result.warnings), (
            "Expected a phantom-score warning in page_ir.warnings"
        )

    def test_phantom_downgrade_text_native_to_hybrid(self):
        """
        text_native page whose phantom score < PHANTOM_THRESHOLD must be
        downgraded to 'hybrid' so Stage 5 knows to OCR it.
        """
        phantom_text = "x " * 40
        chars = [_make_char(c, i * 0.01, i * 0.01 + 0.05, render_mode=3)
                 for i, c in enumerate(phantom_text[:40])]
        span  = _make_span(chars, render_mode=3)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        page  = _make_mock_page(rd, plain_text=phantom_text)

        page_ir = PageIR(page_num=2, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)

        assert result.triage_result == "hybrid", (
            "Phantom text_native page should be downgraded to hybrid"
        )

    def test_script_direction_rtl(self):
        """Arabic text must produce blocks with script_direction='rtl'."""
        arabic_text = "مرحبا بالعالم هذا نص عربي للاختبار " * 5
        chars = [_make_char(c, i * 5, i * 5 + 4.9)
                 for i, c in enumerate(arabic_text[:60])]
        span  = _make_span(chars, render_mode=0)
        block = _make_block([_make_line([span])])
        rd    = _make_rawdict([block])
        page  = _make_mock_page(rd, plain_text=arabic_text)

        page_ir = PageIR(page_num=0, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)

        rtl_blocks = [b for b in result.blocks if b.script_direction == "rtl"]
        assert rtl_blocks, "Expected at least one RTL block for Arabic text"

    def test_confidence_final_not_set(self):
        """Stage 9 sets final; Stage 1 must leave it at 0.0."""
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=0, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)
        for block in result.blocks:
            assert block.confidence.final == 0.0, (
                "confidence.final must remain 0.0 until Stage 9"
            )

    def test_page_num_propagated(self):
        """page_num from PageIR must appear on every block."""
        page    = self._make_real_text_page()
        page_ir = PageIR(page_num=7, triage_result="text_native")
        result  = extract_text_layer(page, page_ir)
        for block in result.blocks:
            assert block.page_num == 7


# ===========================================================================
# _normalised_similarity
# ===========================================================================


class TestNormalisedSimilarity:

    def test_identical_strings(self):
        s = "hello world foo bar"
        assert _normalised_similarity(s, s) == pytest.approx(1.0)

    def test_completely_different(self):
        a = "alpha beta gamma"
        b = "delta epsilon zeta"
        assert _normalised_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = "hello world foo"
        b = "hello world bar"
        score = _normalised_similarity(a, b)
        assert 0.3 < score < 0.8

    def test_empty_strings(self):
        assert _normalised_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert _normalised_similarity("hello", "") == pytest.approx(0.0)


# ===========================================================================
# Integration tests (require PyMuPDF — marked separately)
# ===========================================================================


@pytest.mark.integration
class TestIntegration:
    """
    These tests create real in-memory PDFs with PyMuPDF and exercise the full
    Stage 1 call chain.  Run with:  pytest -m integration
    """

    def _make_real_pdf_page(self):
        """Return an open PyMuPDF page with real embedded Latin text."""
        import fitz
        doc  = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (72, 100),
            "The quick brown fox jumps over the lazy dog.\n" * 20,
            fontsize=11,
        )
        return doc, page

    def test_real_pdf_phantom_score_high(self):
        import fitz
        doc, page = self._make_real_pdf_page()
        try:
            score = detect_phantom_text_layer(page)
            assert score >= 0.85, f"Expected ≥0.85 for real text, got {score:.3f}"
        finally:
            doc.close()

    def test_real_pdf_extract_text_layer(self):
        import fitz
        doc, page = self._make_real_pdf_page()
        page_ir = PageIR(page_num=0, triage_result="text_native",
                         width=595.0, height=842.0)
        try:
            result = extract_text_layer(page, page_ir)
        finally:
            doc.close()

        assert len(result.blocks) >= 1
        for block in result.blocks:
            assert block.id
            uuid.UUID(block.id)           # validates it's a proper UUID
            assert block.extraction_method == ExtractionMethod.PYMUPDF_DIRECT
            assert block.confidence.final == 0.0
            assert block.script_direction in ("ltr", "rtl")

    def test_real_pdf_rawdict_char_bboxes(self):
        """
        Verify that rawdict gives per-character bounding boxes (W4 fix).
        Specifically, chars must have a 4-element bbox with non-zero width.
        """
        import fitz
        doc  = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Hello world.", fontsize=11)
        try:
            rd = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        finally:
            doc.close()

        char_bboxes = []
        for block in rd.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        bbox = char.get("bbox")
                        assert bbox is not None, "Each char must have a bbox in rawdict"
                        assert len(bbox) == 4, "char bbox must be a 4-tuple"
                        char_bboxes.append(bbox)

        assert char_bboxes, "Expected at least one character with a bbox"
        # At least some chars must have non-zero width.
        has_nonzero = any((b[2] - b[0]) > ZERO_WIDTH_THRESHOLD for b in char_bboxes)
        assert has_nonzero, "Expected chars with non-zero width in a real PDF"

    def test_chinese_page_quality_score(self):
        """
        CJK page must pass quick_text_quality_check with score ≥ 0.75  (W5 fix).
        """
        import fitz
        doc  = fitz.open()
        page = doc.new_page(width=595, height=842)
        # PyMuPDF base14 fonts don't have CJK glyphs, so test via the
        # language-aware quality function directly with synthetic text.
        doc.close()

        cjk_text = "这是一段没有空格的中文文本用于测试语言感知的质量评分。" * 15
        score = quick_text_quality_check(cjk_text, language="zh")
        assert score >= 0.75, (
            f"CJK text without spaces must score ≥0.75 (W5 fix), got {score:.3f}"
        )
