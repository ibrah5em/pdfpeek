"""
tests/test_stage8.py
====================
Tests for pdf_engine/stage8_postprocessing.py

Covers:
  - Hyphenation rejoining: no content duplication (W23)
  - Hyphenation rejoining: cross-block suffix stripping
  - OCR corrections: skipped for non-English blocks (W24)
  - OCR corrections: applied for English OCR blocks
  - OCR corrections: skipped for native-text blocks
  - Edge cases: no hyphen, empty blocks, non-word trailing hyphen
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from pdf_engine.stage8_postprocessing import (
    fix_ocr_errors,
    rejoin_hyphenation,
    run_stage8,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block(
    text: str,
    page_num: int = 0,
    method: ExtractionMethod = ExtractionMethod.PYMUPDF_DIRECT,
    language: str | None = None,
) -> TextBlock:
    return TextBlock(
        text=text,
        page_num=page_num,
        extraction_method=method,
        language=language,
        bbox=BBox(0, 0, 100, 20),
    )


# ---------------------------------------------------------------------------
# Hyphenation rejoining tests
# ---------------------------------------------------------------------------


class TestRejoinHyphenation:
    def test_basic_rejoin_connection(self):
        """'connec-' + 'tion of the' → current='connection', next=' of the'"""
        b1 = make_block("connec-")
        b2 = make_block("tion of the")
        rejoin_hyphenation([b1, b2])

        assert b1.text == "connection"
        assert b2.text == " of the"

    def test_no_content_duplication(self):
        """W23 regression: next_block body must not appear in current."""
        b1 = make_block("connec-")
        b2 = make_block("tion of the pipeline")
        rejoin_hyphenation([b1, b2])

        # current must NOT contain anything from b2's tail
        assert "of the pipeline" not in b1.text
        assert b1.text == "connection"
        assert "of the pipeline" in b2.text

    def test_running_example(self):
        """'run-' + 'ning the tests' → current='running', next=' the tests'"""
        b1 = make_block("run-")
        b2 = make_block("ning the tests")
        rejoin_hyphenation([b1, b2])

        assert b1.text == "running"
        assert b2.text == " the tests"

    def test_no_hyphen_unchanged(self):
        """Block without trailing hyphen must be left alone."""
        b1 = make_block("hello world")
        b2 = make_block("another block")
        rejoin_hyphenation([b1, b2])

        assert b1.text == "hello world"
        assert b2.text == "another block"

    def test_hyphen_midword_not_rejoined(self):
        """A hyphen inside a word (not at line end) must not trigger a rejoin."""
        b1 = make_block("well-known term")
        b2 = make_block("following text")
        rejoin_hyphenation([b1, b2])

        assert b1.text == "well-known term"
        assert b2.text == "following text"

    def test_trailing_whitespace_after_hyphen(self):
        """Trailing spaces after the hyphen should still be handled."""
        b1 = make_block("connec-   ")
        b2 = make_block("tion and more")
        rejoin_hyphenation([b1, b2])

        assert b1.text == "connection"
        assert b2.text == " and more"

    def test_single_block_no_crash(self):
        """Single-block list must not raise."""
        b1 = make_block("only-")
        result = rejoin_hyphenation([b1])
        assert result[0].text == "only-"

    def test_multiple_hyphens_in_sequence(self):
        """Three consecutive hyphen-split blocks: only the first pair is rejoined.

        Step 1 (i=0): b1="pro-", b2="duc-"
          → prefix="pro", suffix="duc" → b1="produc", b2="-"
        Step 2 (i=1): b2="-", b3="tion pipeline"
          → re.search(r'(\\w+)-\\s*$', "-") finds NO word before the hyphen
          → no rejoin; b3 is unchanged.

        Final state: b1="produc", b2="-", b3="tion pipeline".
        A fully-merged "production" would require a smarter multi-pass or a
        different input format; the single-pass algorithm handles one pair at a time.
        """
        b1 = make_block("pro-")
        b2 = make_block("duc-")
        b3 = make_block("tion pipeline")
        rejoin_hyphenation([b1, b2, b3])

        # First pair is rejoined correctly
        assert b1.text == "produc"
        # b2 is left with only the trailing hyphen (the word "duc" was consumed)
        assert b2.text == "-"
        # b3 is untouched because b2 no longer has a leading word before its hyphen
        assert b3.text == "tion pipeline"

    def test_empty_next_block_no_crash(self):
        """next_block with empty text should not crash."""
        b1 = make_block("word-")
        b2 = make_block("")
        rejoin_hyphenation([b1, b2])
        # No suffix word found → no rejoin
        assert b1.text == "word-"
        assert b2.text == ""

    def test_returns_same_list(self):
        """rejoin_hyphenation must return the same list object (mutates in-place)."""
        blocks = [make_block("word-"), make_block("end")]
        result = rejoin_hyphenation(blocks)
        assert result is blocks


# ---------------------------------------------------------------------------
# OCR error correction tests
# ---------------------------------------------------------------------------


class TestFixOcrErrors:
    def test_english_ocr_rn_corrected(self):
        """English OCR block: 'rn' confusion corrected to 'm'."""
        block = make_block(
            "arnount of work",
            method=ExtractionMethod.TESSERACT_OCR,
            language="en",
        )
        fix_ocr_errors(block)
        assert "amount" in block.text

    def test_english_ocr_surya_corrected(self):
        """SURYA_OCR blocks with English language also get corrected."""
        block = make_block(
            "arnount",
            method=ExtractionMethod.SURYA_OCR,
            language="en",
        )
        fix_ocr_errors(block)
        assert "amount" in block.text

    def test_german_block_not_corrected(self):
        """W24 fix: German OCR block must NOT receive English corrections."""
        original = "arnount Wasser"
        block = make_block(
            original,
            method=ExtractionMethod.TESSERACT_OCR,
            language="de",
        )
        fix_ocr_errors(block)
        # Text must be unchanged — German corrections not applied
        assert block.text == original

    def test_no_language_block_not_corrected(self):
        """Block with language=None is treated as non-English → not corrected."""
        original = "arnount"
        block = make_block(
            original,
            method=ExtractionMethod.TESSERACT_OCR,
            language=None,
        )
        fix_ocr_errors(block)
        assert block.text == original

    def test_native_text_block_not_corrected(self):
        """PYMUPDF_DIRECT blocks must never be corrected, even for English."""
        original = "arnount of data"
        block = make_block(
            original,
            method=ExtractionMethod.PYMUPDF_DIRECT,
            language="en",
        )
        fix_ocr_errors(block)
        assert block.text == original

    def test_english_locale_variant(self):
        """language='en-GB' should still be corrected (startswith 'en')."""
        block = make_block(
            "arnount",
            method=ExtractionMethod.TESSERACT_OCR,
            language="en-GB",
        )
        fix_ocr_errors(block)
        assert "amount" in block.text

    def test_returns_same_block(self):
        """fix_ocr_errors must return the same block object."""
        block = make_block("hello", method=ExtractionMethod.TESSERACT_OCR, language="en")
        result = fix_ocr_errors(block)
        assert result is block

    def test_vv_corrected_to_w(self):
        """'vv' → 'w' correction applied to English OCR block."""
        block = make_block(
            "vvater quality",
            method=ExtractionMethod.TESSERACT_OCR,
            language="en",
        )
        fix_ocr_errors(block)
        assert "water" in block.text

    def test_empty_text_no_crash(self):
        """Empty text block must not raise."""
        block = make_block("", method=ExtractionMethod.TESSERACT_OCR, language="en")
        fix_ocr_errors(block)
        assert block.text == ""


# ---------------------------------------------------------------------------
# Document-level integration tests
# ---------------------------------------------------------------------------


class TestRunStage8:
    def _make_doc(self, page_blocks: list[list[TextBlock]]) -> DocumentIR:
        pages = []
        for i, blocks in enumerate(page_blocks):
            page = PageIR(page_num=i, blocks=blocks)
            pages.append(page)
        doc = DocumentIR(pages=pages)
        return doc

    def test_cross_page_hyphenation(self):
        """Hyphen at end of page 0 last block rejoined with page 1 first block."""
        b1 = make_block("connec-", page_num=0)
        b2 = make_block("tion layer", page_num=1)
        doc = self._make_doc([[b1], [b2]])
        run_stage8(doc)

        assert b1.text == "connection"
        assert b2.text == " layer"

    def test_ocr_correction_applied_in_doc(self):
        """run_stage8 applies OCR corrections to English OCR blocks."""
        b1 = make_block(
            "arnount",
            page_num=0,
            method=ExtractionMethod.TESSERACT_OCR,
            language="en",
        )
        doc = self._make_doc([[b1]])
        run_stage8(doc)
        assert "amount" in b1.text

    def test_german_block_untouched_in_doc(self):
        """run_stage8 does not apply corrections to German blocks."""
        original = "arnount Wasser"
        b1 = make_block(
            original,
            page_num=0,
            method=ExtractionMethod.TESSERACT_OCR,
            language="de",
        )
        doc = self._make_doc([[b1]])
        run_stage8(doc)
        assert b1.text == original

    def test_returns_same_doc_ir(self):
        """run_stage8 must return the same DocumentIR object."""
        doc = self._make_doc([[make_block("hello")]])
        result = run_stage8(doc)
        assert result is doc

    def test_empty_document_no_crash(self):
        """Empty document with no pages must not raise."""
        doc = DocumentIR()
        result = run_stage8(doc)
        assert result is doc


# ---------------------------------------------------------------------------
# Edge-case / regression tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_hyphen_only_block(self):
        """A block containing only '-' should not trigger a rejoin (no prefix word)."""
        b1 = make_block("-")
        b2 = make_block("word")
        rejoin_hyphenation([b1, b2])
        assert b1.text == "-"
        assert b2.text == "word"

    def test_next_block_starts_with_space_then_word(self):
        """Leading spaces in next_block are stripped before matching suffix."""
        b1 = make_block("re-")
        b2 = make_block("   sult of query")
        rejoin_hyphenation([b1, b2])
        assert b1.text == "result"
        assert "of query" in b2.text

    def test_no_double_application(self):
        """Calling rejoin_hyphenation twice must be idempotent after first pass."""
        b1 = make_block("con-")
        b2 = make_block("nect")
        rejoin_hyphenation([b1, b2])
        state_after_first = (b1.text, b2.text)
        rejoin_hyphenation([b1, b2])
        assert (b1.text, b2.text) == state_after_first
