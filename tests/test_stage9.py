"""
tests/test_stage9.py
=====================
Unit tests for Stage 9 — Confidence Scoring.

Coverage
--------
* compute_block_confidence: text_quality computation
* compute_block_confidence: method_score lookup for all known methods
* compute_block_confidence: upstream order_quality / type_quality are read,
  not overwritten (W25 fix)
* compute_block_confidence: text_quality and method_score are independent
  dimensions in the geometric mean (W26 fix)
* compute_block_confidence: perfect Tesseract block final > 0.75 (W26 validation)
* compute_block_confidence: replacement characters lower text_quality
* compute_block_confidence: empty text yields text_quality of 1.0 (length
  guard avoids ZeroDivisionError)
* score_page: mutates all blocks on a page in-place
* run_stage9: sets doc_ir.confidence to mean of block finals
* run_stage9: empty document yields confidence == 0.0
"""

from __future__ import annotations

import math
import uuid

import pytest

from pdf_engine.models import (
    BlockConfidence,
    BlockType,
    DocumentIR,
    ExtractionMethod,
    PageIR,
    TextBlock,
)
from pdf_engine.stage9_confidence import (
    METHOD_TRUST,
    compute_block_confidence,
    run_stage9,
    score_page,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    text: str = "Hello world",
    method: ExtractionMethod = ExtractionMethod.PYMUPDF_DIRECT,
    order_quality: float = 0.8,
    type_quality: float = 0.7,
) -> TextBlock:
    return TextBlock(
        id=str(uuid.uuid4()),
        text=text,
        extraction_method=method,
        confidence=BlockConfidence(
            order_quality=order_quality,
            type_quality=type_quality,
        ),
    )


def _geometric_mean(*values: float) -> float:
    product = math.prod(values)
    return product ** (1 / len(values))


# ---------------------------------------------------------------------------
# text_quality
# ---------------------------------------------------------------------------


class TestTextQuality:
    def test_clean_ascii_gives_one(self):
        block = _make_block(text="Hello world")
        conf = compute_block_confidence(block)
        assert conf.text_quality == pytest.approx(1.0)

    def test_all_replacement_chars_gives_zero(self):
        block = _make_block(text="\ufffd\ufffd\ufffd")
        conf = compute_block_confidence(block)
        # printable_ratio = 1.0 (replacement chars ARE printable),
        # replacement_ratio = 1.0 → text_quality = 1.0 * (1 - 1.0) = 0.0
        assert conf.text_quality == pytest.approx(0.0)

    def test_mixed_replacement_discounts_quality(self):
        # 5 normal + 5 replacement chars
        block = _make_block(text="abcde\ufffd\ufffd\ufffd\ufffd\ufffd")
        conf = compute_block_confidence(block)
        # printable_ratio = 1.0, replacement_ratio = 0.5
        assert conf.text_quality == pytest.approx(0.5)

    def test_empty_text_no_divide_by_zero(self):
        block = _make_block(text="")
        conf = compute_block_confidence(block)
        # length guard: max(0, 1) = 1; both counts are 0; quality = 1.0
        assert conf.text_quality == pytest.approx(1.0)

    def test_non_printable_chars_lower_quality(self):
        # Bell character (\x07) is not printable
        block = _make_block(text="ab\x07cd")
        conf = compute_block_confidence(block)
        # printable_ratio = 4/5 = 0.8, replacement_ratio = 0
        assert conf.text_quality == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# method_score
# ---------------------------------------------------------------------------


class TestMethodScore:
    @pytest.mark.parametrize(
        "method, expected",
        [
            (ExtractionMethod.PYMUPDF_DIRECT, 1.00),
            (ExtractionMethod.SURYA_LAYOUT,   0.90),
            (ExtractionMethod.SURYA_OCR,       0.80),
            (ExtractionMethod.TESSERACT_OCR,   0.70),
        ],
    )
    def test_known_methods(self, method, expected):
        block = _make_block(method=method)
        conf = compute_block_confidence(block)
        assert conf.method_score == pytest.approx(expected)

    def test_all_methods_in_trust_table(self):
        """Ensure every ExtractionMethod has an entry in METHOD_TRUST."""
        for method in ExtractionMethod:
            assert method in METHOD_TRUST, (
                f"{method} missing from METHOD_TRUST"
            )


# ---------------------------------------------------------------------------
# Upstream quality preservation (W25 fix)
# ---------------------------------------------------------------------------


class TestUpstreamQualityPreservation:
    def test_order_quality_not_overwritten(self):
        block = _make_block(order_quality=0.55)
        conf = compute_block_confidence(block)
        assert conf.order_quality == pytest.approx(0.55)

    def test_type_quality_not_overwritten(self):
        block = _make_block(type_quality=0.42)
        conf = compute_block_confidence(block)
        assert conf.type_quality == pytest.approx(0.42)

    def test_both_qualities_preserved_together(self):
        block = _make_block(order_quality=0.91, type_quality=0.33)
        conf = compute_block_confidence(block)
        assert conf.order_quality == pytest.approx(0.91)
        assert conf.type_quality == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# W26 fix: text_quality and method_score are separate dimensions
# ---------------------------------------------------------------------------


class TestW26SeparateDimensions:
    def test_perfect_tesseract_block_above_threshold(self):
        """
        Perfect Tesseract block: text_quality=1.0, method=0.70, order=0.8,
        type=0.7 → final = (1.0 * 0.70 * 0.8 * 0.7)^0.25 ≈ 0.84 > 0.75.
        """
        block = _make_block(
            text="clean text",
            method=ExtractionMethod.TESSERACT_OCR,
            order_quality=0.8,
            type_quality=0.7,
        )
        conf = compute_block_confidence(block)
        assert conf.final > 0.75, (
            f"Expected final > 0.75 for perfect Tesseract block, got {conf.final:.4f}"
        )

    def test_final_is_geometric_mean_of_four(self):
        """final == (text_quality * method_score * order_quality * type_quality)^0.25"""
        block = _make_block(
            text="Hello world",
            method=ExtractionMethod.SURYA_OCR,
            order_quality=0.9,
            type_quality=0.6,
        )
        conf = compute_block_confidence(block)
        expected = _geometric_mean(
            conf.text_quality,
            conf.method_score,
            conf.order_quality,
            conf.type_quality,
        )
        assert conf.final == pytest.approx(expected, abs=1e-9)

    def test_method_score_not_folded_into_text_quality(self):
        """
        Verify that text_quality is computed independently of method_score.
        Two blocks with identical text but different methods must have
        equal text_quality but different method_score.
        """
        b1 = _make_block(text="same text", method=ExtractionMethod.PYMUPDF_DIRECT)
        b2 = _make_block(text="same text", method=ExtractionMethod.TESSERACT_OCR)
        c1 = compute_block_confidence(b1)
        c2 = compute_block_confidence(b2)
        assert c1.text_quality == pytest.approx(c2.text_quality)
        assert c1.method_score != c2.method_score

    def test_lower_method_trust_lowers_final_only(self):
        """Switching from PYMUPDF to TESSERACT_OCR lowers final but not text_quality."""
        b_direct = _make_block(text="clean", method=ExtractionMethod.PYMUPDF_DIRECT)
        b_ocr    = _make_block(text="clean", method=ExtractionMethod.TESSERACT_OCR)
        c_direct = compute_block_confidence(b_direct)
        c_ocr    = compute_block_confidence(b_ocr)
        assert c_direct.text_quality == pytest.approx(c_ocr.text_quality)
        assert c_direct.final > c_ocr.final


# ---------------------------------------------------------------------------
# Zero-product edge case
# ---------------------------------------------------------------------------


class TestZeroProduct:
    def test_zero_text_quality_gives_zero_final(self):
        block = _make_block(
            text="\ufffd\ufffd",
            method=ExtractionMethod.PYMUPDF_DIRECT,
            order_quality=0.9,
            type_quality=0.9,
        )
        conf = compute_block_confidence(block)
        assert conf.final == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_page
# ---------------------------------------------------------------------------


class TestScorePage:
    def _make_page(self, n_blocks: int = 3) -> PageIR:
        page = PageIR(page_num=0)
        for _ in range(n_blocks):
            page.blocks.append(_make_block())
        return page

    def test_all_blocks_scored(self):
        page = self._make_page(4)
        score_page(page)
        for block in page.blocks:
            assert block.confidence.final > 0.0

    def test_returns_same_page_object(self):
        page = self._make_page(2)
        result = score_page(page)
        assert result is page

    def test_empty_page_no_error(self):
        page = PageIR(page_num=0)
        score_page(page)  # must not raise


# ---------------------------------------------------------------------------
# run_stage9 (document level)
# ---------------------------------------------------------------------------


class TestRunStage9:
    def _make_doc(self, n_pages: int = 2, blocks_per_page: int = 3) -> DocumentIR:
        doc = DocumentIR()
        for p in range(n_pages):
            page = PageIR(page_num=p)
            for _ in range(blocks_per_page):
                page.blocks.append(_make_block())
            doc.pages.append(page)
        return doc

    def test_document_confidence_is_mean_of_finals(self):
        doc = self._make_doc(n_pages=2, blocks_per_page=2)
        run_stage9(doc)
        all_finals = [b.confidence.final for p in doc.pages for b in p.blocks]
        expected_mean = sum(all_finals) / len(all_finals)
        assert doc.confidence == pytest.approx(expected_mean, abs=1e-9)

    def test_empty_document_gives_zero_confidence(self):
        doc = DocumentIR()
        run_stage9(doc)
        assert doc.confidence == pytest.approx(0.0)

    def test_returns_same_doc_ir(self):
        doc = self._make_doc()
        result = run_stage9(doc)
        assert result is doc

    def test_all_blocks_have_final_set(self):
        doc = self._make_doc(n_pages=3, blocks_per_page=4)
        run_stage9(doc)
        for page in doc.pages:
            for block in page.blocks:
                assert 0.0 <= block.confidence.final <= 1.0

    def test_single_page_single_block(self):
        doc = DocumentIR()
        page = PageIR(page_num=0)
        block = _make_block(
            text="Hello",
            method=ExtractionMethod.PYMUPDF_DIRECT,
            order_quality=1.0,
            type_quality=1.0,
        )
        page.blocks.append(block)
        doc.pages.append(page)
        run_stage9(doc)
        assert doc.confidence == pytest.approx(block.confidence.final, abs=1e-9)
        assert doc.confidence == pytest.approx(1.0)  # perfect block, perfect method
