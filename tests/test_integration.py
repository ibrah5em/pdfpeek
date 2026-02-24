"""
tests/test_integration.py
===========================
Integration tests for the end-to-end PDF extraction pipeline.

These tests exercise the public ``extract()`` API defined in
``pdf_engine/api.py`` and validate the following requirements:

1. Born-digital academic PDF → correct text, no duplicate blocks,
   confidence > 0.8.
2. Scanned single-column PDF → OCR text extracted, confidence reported.
3. Encrypted PDF without password → warnings non-empty, text is "".
4. Perfect OCR block → final confidence > 0.75 (W26 fix validation).
5. All block IDs are non-empty UUIDs (W27 fix validation).

Test isolation strategy
-----------------------
Tests that require real PDF files on disk are guarded by pytest ``skipif``
markers that check for fixtures in the ``tests/fixtures/`` directory.
This makes the test suite pass in CI environments that ship only the
fixture-less skeleton while still being runnable end-to-end when the
fixtures are present.

Unit-style integration tests (W26, W27) do NOT require fixture PDFs and
always run.
"""

from __future__ import annotations

import os
import uuid

import pytest

# ---------------------------------------------------------------------------
# Fixture paths (edit to match your fixture directory)
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

BORN_DIGITAL_PDF = os.path.join(FIXTURE_DIR, "born_digital_academic.pdf")
SCANNED_PDF      = os.path.join(FIXTURE_DIR, "scanned_single_column.pdf")
ENCRYPTED_PDF    = os.path.join(FIXTURE_DIR, "encrypted_no_password.pdf")

_have_born_digital = pytest.mark.skipif(
    not os.path.isfile(BORN_DIGITAL_PDF),
    reason=f"Fixture not found: {BORN_DIGITAL_PDF}",
)
_have_scanned = pytest.mark.skipif(
    not os.path.isfile(SCANNED_PDF),
    reason=f"Fixture not found: {SCANNED_PDF}",
)
_have_encrypted = pytest.mark.skipif(
    not os.path.isfile(ENCRYPTED_PDF),
    reason=f"Fixture not found: {ENCRYPTED_PDF}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


def _all_blocks(result_ir) -> list:
    blocks = []
    for page in result_ir.pages:
        blocks.extend(page.blocks)
    return blocks


# ---------------------------------------------------------------------------
# 1. Born-digital academic PDF
# ---------------------------------------------------------------------------


@_have_born_digital
class TestBornDigitalPDF:
    """End-to-end: text-native PDF → text, no duplicates, high confidence."""

    @pytest.fixture(scope="class")
    def result(self):
        from pdf_engine.api import extract
        return extract(BORN_DIGITAL_PDF, output_format="markdown")

    def test_text_is_non_empty(self, result):
        assert result.text.strip(), "Expected non-empty text from born-digital PDF"

    def test_confidence_above_threshold(self, result):
        assert result.confidence > 0.8, (
            f"Expected document confidence > 0.8, got {result.confidence:.4f}"
        )

    def test_no_duplicate_blocks(self, result):
        """Block IDs must all be unique (no accidental duplication in assembly)."""
        blocks = _all_blocks(result.ir)
        ids = [b.id for b in blocks]
        assert len(ids) == len(set(ids)), "Duplicate block IDs found"

    def test_no_duplicate_text_segments(self, result):
        """
        No two consecutive blocks in the same page should have identical
        non-empty text (catches naive de-duplication failures).
        """
        for page in result.ir.pages:
            texts = [b.text.strip() for b in page.blocks if b.text.strip()]
            for i in range(len(texts) - 1):
                assert texts[i] != texts[i + 1], (
                    f"Page {page.page_num}: consecutive duplicate text: {texts[i]!r}"
                )

    def test_all_block_ids_are_valid_uuids(self, result):
        """W27 fix: every block ID must be a valid, non-empty UUID string."""
        blocks = _all_blocks(result.ir)
        for block in blocks:
            assert block.id, "Found a block with an empty ID"
            assert _is_valid_uuid(block.id), (
                f"Block ID is not a valid UUID: {block.id!r}"
            )

    def test_output_format_ir_returns_empty_text(self):
        from pdf_engine.api import extract
        result = extract(BORN_DIGITAL_PDF, output_format="ir")
        assert result.text == ""
        assert result.ir is not None


# ---------------------------------------------------------------------------
# 2. Scanned PDF
# ---------------------------------------------------------------------------


@_have_scanned
class TestScannedPDF:
    """End-to-end: scanned single-column PDF → OCR text, confidence reported."""

    @pytest.fixture(scope="class")
    def result(self):
        from pdf_engine.api import extract
        return extract(SCANNED_PDF, output_format="plain", ocr_engine="auto")

    def test_text_is_non_empty(self, result):
        assert result.text.strip(), "Expected OCR text from scanned PDF"

    def test_confidence_is_positive(self, result):
        assert result.confidence > 0.0, "Expected positive confidence from OCR"

    def test_confidence_is_reported(self, result):
        assert 0.0 < result.confidence <= 1.0

    def test_at_least_one_ocr_block(self, result):
        from pdf_engine.models import ExtractionMethod
        blocks = _all_blocks(result.ir)
        ocr_methods = {ExtractionMethod.TESSERACT_OCR, ExtractionMethod.SURYA_OCR}
        ocr_blocks = [b for b in blocks if b.extraction_method in ocr_methods]
        assert ocr_blocks, "Expected at least one OCR block in scanned PDF"

    def test_all_block_ids_are_valid_uuids(self, result):
        blocks = _all_blocks(result.ir)
        for block in blocks:
            assert block.id, "Found a block with an empty ID"
            assert _is_valid_uuid(block.id), (
                f"Block ID is not a valid UUID: {block.id!r}"
            )


# ---------------------------------------------------------------------------
# 3. Encrypted PDF (no password supplied)
# ---------------------------------------------------------------------------


@_have_encrypted
class TestEncryptedPDF:
    """End-to-end: encrypted PDF → empty text, non-empty warnings."""

    @pytest.fixture(scope="class")
    def result(self):
        from pdf_engine.api import extract
        # Intentionally supply no password
        return extract(ENCRYPTED_PDF, password=None)

    def test_text_is_empty(self, result):
        assert result.text == "", (
            f"Expected empty text for encrypted PDF, got: {result.text[:80]!r}"
        )

    def test_warnings_non_empty(self, result):
        assert result.warnings, (
            "Expected at least one warning for unreadable encrypted PDF"
        )

    def test_confidence_is_zero(self, result):
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 4. W26 fix: perfect OCR block → final confidence > 0.75
# ---------------------------------------------------------------------------


class TestW26PerfectOCRConfidence:
    """
    A TextBlock with perfect text, Tesseract OCR method, default upstream
    qualities must score above 0.75.

    Expected: (1.0 × 0.70 × 0.8 × 0.7)^0.25 ≈ 0.8393 > 0.75
    """

    def test_perfect_tesseract_block_above_threshold(self):
        from pdf_engine.models import (
            BlockConfidence, ExtractionMethod, TextBlock
        )
        from pdf_engine.stage9_confidence import compute_block_confidence

        block = TextBlock(
            id=str(uuid.uuid4()),
            text="This is perfectly legible OCR output.",
            extraction_method=ExtractionMethod.TESSERACT_OCR,
            confidence=BlockConfidence(
                order_quality=0.8,   # default Stage 3 quality
                type_quality=0.7,    # default Stage 2 quality
            ),
        )
        conf = compute_block_confidence(block)
        assert conf.final > 0.75, (
            f"W26 regression: perfect Tesseract block final={conf.final:.4f}, "
            f"expected > 0.75"
        )

    def test_perfect_surya_ocr_block_above_threshold(self):
        from pdf_engine.models import (
            BlockConfidence, ExtractionMethod, TextBlock
        )
        from pdf_engine.stage9_confidence import compute_block_confidence

        block = TextBlock(
            id=str(uuid.uuid4()),
            text="Clean Surya OCR output.",
            extraction_method=ExtractionMethod.SURYA_OCR,
            confidence=BlockConfidence(
                order_quality=0.8,
                type_quality=0.7,
            ),
        )
        conf = compute_block_confidence(block)
        # (1.0 * 0.80 * 0.8 * 0.7)^0.25 ≈ 0.8627
        assert conf.final > 0.75

    def test_perfect_pymupdf_block_gives_one(self):
        from pdf_engine.models import (
            BlockConfidence, ExtractionMethod, TextBlock
        )
        from pdf_engine.stage9_confidence import compute_block_confidence
        import math

        block = TextBlock(
            id=str(uuid.uuid4()),
            text="Native text, no OCR needed.",
            extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
            confidence=BlockConfidence(
                order_quality=1.0,
                type_quality=1.0,
            ),
        )
        conf = compute_block_confidence(block)
        # (1.0 * 1.0 * 1.0 * 1.0)^0.25 = 1.0
        assert conf.final == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. W27 fix: all block IDs are non-empty UUIDs
# ---------------------------------------------------------------------------


class TestW27BlockIDs:
    """
    Validate that every TextBlock produced by the pipeline carries a
    non-empty, valid UUID string as its ``id``.
    """

    def _make_populated_doc_ir(self):
        """Build a small synthetic DocumentIR with blocks from each stage."""
        from pdf_engine.models import (
            BlockConfidence, BlockType, DocumentIR, ExtractionMethod,
            PageIR, TextBlock, BBox,
        )
        doc = DocumentIR()
        page = PageIR(page_num=0, width=595.0, height=842.0)
        methods = list(ExtractionMethod)
        block_types = list(BlockType)
        for i, (method, btype) in enumerate(
            zip(methods, block_types * len(methods))
        ):
            block = TextBlock(
                # id is auto-generated by the dataclass default_factory
                text=f"Sample text block {i}",
                extraction_method=method,
                block_type=btype,
                bbox=BBox(x0=0, y0=i * 50, x1=400, y1=(i + 1) * 50),
                confidence=BlockConfidence(order_quality=0.9, type_quality=0.8),
                page_num=0,
            )
            page.blocks.append(block)
        doc.pages.append(page)
        return doc

    def test_auto_generated_ids_are_valid_uuids(self):
        doc = self._make_populated_doc_ir()
        for page in doc.pages:
            for block in page.blocks:
                assert block.id, f"Block has empty ID: {block!r}"
                assert _is_valid_uuid(block.id), (
                    f"Block ID is not a valid UUID: {block.id!r}"
                )

    def test_no_two_blocks_share_an_id(self):
        doc = self._make_populated_doc_ir()
        blocks = _all_blocks(doc)
        ids = [b.id for b in blocks]
        assert len(ids) == len(set(ids)), "Two or more blocks share the same ID"

    def test_scored_blocks_retain_valid_ids(self):
        """Stage 9 must not overwrite or clear block IDs."""
        from pdf_engine.stage9_confidence import run_stage9

        doc = self._make_populated_doc_ir()
        original_ids = {b.id for page in doc.pages for b in page.blocks}
        run_stage9(doc)
        post_ids = {b.id for page in doc.pages for b in page.blocks}
        assert original_ids == post_ids, (
            "Block IDs changed after running Stage 9"
        )


# ---------------------------------------------------------------------------
# API contract tests (no PDF fixture required)
# ---------------------------------------------------------------------------


class TestAPIContract:
    """Light-weight tests of the extract() API's input validation."""

    def test_missing_file_raises_file_not_found(self):
        from pdf_engine.api import extract
        with pytest.raises(FileNotFoundError):
            extract("/nonexistent/path/to/file.pdf")

    def test_invalid_output_format_raises_value_error(self):
        # We can't easily get past the file-existence check without a real PDF,
        # so we create a temporary empty file and expect ValueError before the
        # pipeline starts.
        import tempfile

        from pdf_engine.api import extract

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            tmp_path = fh.name

        try:
            with pytest.raises(ValueError, match="output_format"):
                extract(tmp_path, output_format="html")
        finally:
            os.unlink(tmp_path)
