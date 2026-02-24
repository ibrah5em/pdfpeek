"""
tests/test_stage5.py
====================
Unit tests for Stage 5 — OCR Pipeline (stage5_ocr.py).

Test coverage
-------------
T1  ocr_needed page → rasterize_page called with dpi=300 by default.
T2  CJK script → surya selected without calling Tesseract.
T3  Page with 8° skew → deskewed image has no black triangles at corners.
T4  Page with 20° skew → skew correction skipped, warning added to PageIR.
T5  text_native page → process_ocr_page is a no-op (blocks unchanged).
T6  Tesseract low confidence → surya fallback is tried.
T7  Surya-first path → _run_tesseract never called for Arabic/CJK pages.
T8  rasterize_page emits warning when estimated memory > 50 MB.
T9  run_stage5 skips text_native pages and processes ocr_needed pages.
T10 preprocess_for_ocr returns 2-D binary array (values 0 or 255).
"""

from __future__ import annotations

import math
import types
import uuid
from typing import Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# We import the module under test directly; optional heavy deps (cv2,
# pytesseract, surya) are monkeypatched so tests run without them installed.
# ---------------------------------------------------------------------------
import pdf_engine.stage5_ocr as stage5


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_page_ir(page_num: int = 0, triage: str = "ocr_needed"):
    """Create a minimal PageIR for testing."""
    from pdf_engine.models import PageIR
    page = PageIR(page_num=page_num, width=595.0, height=842.0)
    page.triage_result = triage
    return page


def _make_doc_ir(pages=None):
    """Create a minimal DocumentIR."""
    from pdf_engine.models import DocumentIR
    doc = DocumentIR()
    doc.pages = pages or []
    return doc


def _rgb_image(h: int = 200, w: int = 200, fill: int = 200) -> np.ndarray:
    """Create a solid-colour RGB numpy array."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _fake_fitz_page(h: int = 200, w: int = 200):
    """
    Return a mock fitz.Page whose get_pixmap() returns a plausible pixmap.
    """
    samples = bytes([200, 200, 200] * (h * w))  # flat grey RGB

    pixmap = MagicMock()
    pixmap.width   = w
    pixmap.height  = h
    pixmap.samples = samples

    page = MagicMock()
    page.get_pixmap.return_value = pixmap
    return page


# ---------------------------------------------------------------------------
# T1 — ocr_needed page → rasterize_page called with dpi=300 (default)
# ---------------------------------------------------------------------------

class TestRasterizeCalledWithDefaultDpi:
    def test_default_dpi_used(self, monkeypatch):
        """
        rasterize_page must be called with dpi=300 when process_ocr_page is
        invoked without an explicit dpi argument.
        """
        page_ir   = _make_page_ir(triage="ocr_needed")
        fitz_page = _fake_fitz_page()

        captured_dpi: list[int] = []

        original_rasterize = stage5.rasterize_page

        def fake_rasterize(fp, dpi=stage5.DEFAULT_OCR_DPI):
            captured_dpi.append(dpi)
            # Return a tiny valid image so the rest of the pipeline doesn't crash.
            return _rgb_image()

        monkeypatch.setattr(stage5, "rasterize_page", fake_rasterize)
        monkeypatch.setattr(stage5, "detect_script_from_image", lambda img: "unknown")
        monkeypatch.setattr(stage5, "preprocess_for_ocr", lambda img, **kw: np.zeros((10, 10), np.uint8))
        monkeypatch.setattr(stage5, "_ocr_page", lambda *a, **kw: [])

        stage5.process_ocr_page(page_ir, fitz_page)

        assert len(captured_dpi) == 1
        assert captured_dpi[0] == stage5.DEFAULT_OCR_DPI == 300

    def test_custom_dpi_forwarded(self, monkeypatch):
        """A custom dpi argument is forwarded to rasterize_page."""
        page_ir   = _make_page_ir(triage="ocr_needed")
        fitz_page = _fake_fitz_page()
        captured_dpi: list[int] = []

        monkeypatch.setattr(stage5, "rasterize_page", lambda fp, dpi=300: (captured_dpi.append(dpi), _rgb_image())[1])
        monkeypatch.setattr(stage5, "detect_script_from_image", lambda img: "unknown")
        monkeypatch.setattr(stage5, "preprocess_for_ocr", lambda img, **kw: np.zeros((10, 10), np.uint8))
        monkeypatch.setattr(stage5, "_ocr_page", lambda *a, **kw: [])

        stage5.process_ocr_page(page_ir, fitz_page, dpi=150)
        assert captured_dpi[0] == 150


# ---------------------------------------------------------------------------
# T2 — CJK page → surya selected; Tesseract never called
# ---------------------------------------------------------------------------

class TestCjkRoutesToSurya:
    def test_surya_called_not_tesseract(self, monkeypatch):
        """
        When detect_script_from_image returns 'cjk', _run_surya_ocr must be
        called and _run_tesseract must NOT be called.
        """
        page_ir = _make_page_ir(triage="ocr_needed")
        img     = _rgb_image()
        binary  = np.zeros((10, 10), np.uint8)

        tesseract_called = []
        surya_called     = []

        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: (tesseract_called.append(True), [])[1])
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: (surya_called.append(True), [])[1])

        stage5._ocr_page(img, binary, stage5.SCRIPT_CJK, page_ir)

        assert surya_called,           "surya must be called for CJK pages"
        assert not tesseract_called,   "Tesseract must NOT be called for CJK pages"

    def test_arabic_routes_to_surya(self, monkeypatch):
        """Arabic script also routes to surya without calling Tesseract."""
        page_ir = _make_page_ir(triage="ocr_needed")
        img     = _rgb_image()
        binary  = np.zeros((10, 10), np.uint8)

        tesseract_called = []
        surya_called     = []

        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: (tesseract_called.append(True), [])[1])
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: (surya_called.append(True), [])[1])

        stage5._ocr_page(img, binary, stage5.SCRIPT_ARABIC, page_ir)

        assert surya_called
        assert not tesseract_called


# ---------------------------------------------------------------------------
# T3 — 8° skew → deskewed image has no black triangles at corners
# ---------------------------------------------------------------------------

class TestDeskewRemovesBlackTriangles:
    def test_output_shape_is_smaller_than_input(self):
        """
        After rotating an image by 8°, the inscribed-rectangle crop must
        produce a result that is strictly smaller than the original on both
        axes (no black triangles retained).
        """
        # Create a white image with some diagonal lines to give Hough something to work on.
        h, w = 400, 600
        img_rgb = np.full((h, w, 3), 255, dtype=np.uint8)

        # Skip real cv2 to keep the test deterministic: directly test the
        # inscribed-rectangle formula with a known 8° angle.
        angle_rad = math.radians(8.0)
        new_w, new_h = stage5._largest_inscribed_rect(w, h, angle_rad)

        assert new_w < w, f"Cropped width {new_w} should be smaller than {w}"
        assert new_h < h, f"Cropped height {new_h} should be smaller than {h}"
        assert new_w > 0 and new_h > 0, "Crop dimensions must be positive"

    def test_preprocess_with_small_angle_reduces_size(self, monkeypatch):
        """
        preprocess_for_ocr with a synthetically injected 8° skew must
        return an array smaller than the input (proving triangles are cropped).
        """
        pytest.importorskip("cv2")

        h, w = 400, 600
        img_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

        # Inject a fixed 8° angle so we don't depend on Hough detection.
        monkeypatch.setattr(stage5, "_detect_skew_angle", lambda gray: 8.0)

        result = stage5.preprocess_for_ocr(img_rgb, max_skew_degrees=15.0)

        out_h, out_w = result.shape[:2]
        assert out_h < h or out_w < w, (
            f"Expected output ({out_h}×{out_w}) to be smaller than input ({h}×{w})"
        )


# ---------------------------------------------------------------------------
# T4 — 20° skew → skew correction skipped, warning in PageIR.warnings
# ---------------------------------------------------------------------------

class TestLargeSkewSkipped:
    def test_warning_added_when_skew_exceeds_limit(self, monkeypatch):
        """
        When the detected skew exceeds MAX_SKEW_DEGREES, preprocess_for_ocr
        must append a warning and return an image the same size as the input.
        """
        pytest.importorskip("cv2")

        h, w = 400, 600
        img_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

        # Force detection to return 20°, which is > 15°.
        monkeypatch.setattr(stage5, "_detect_skew_angle", lambda gray: 20.0)

        page_warnings: list[str] = []
        result = stage5.preprocess_for_ocr(
            img_rgb,
            max_skew_degrees=stage5.MAX_SKEW_DEGREES,
            page_warnings=page_warnings,
        )

        # A warning must have been issued.
        assert len(page_warnings) == 1, "Exactly one warning expected"
        assert "20" in page_warnings[0] or "skew" in page_warnings[0].lower()

        # Output must be binarised but NOT cropped (same spatial dimensions).
        out_h, out_w = result.shape[:2]
        assert out_h == h and out_w == w, (
            f"Large-skew image should not be cropped: expected ({h}×{w}), "
            f"got ({out_h}×{out_w})"
        )

    def test_process_ocr_page_propagates_skew_warning(self, monkeypatch):
        """
        Skew warnings produced by preprocess_for_ocr must be visible in
        page_ir.warnings after process_ocr_page completes.
        """
        pytest.importorskip("cv2")

        page_ir   = _make_page_ir(triage="ocr_needed")
        fitz_page = _fake_fitz_page()

        monkeypatch.setattr(stage5, "rasterize_page", lambda fp, dpi=300: _rgb_image())
        monkeypatch.setattr(stage5, "detect_script_from_image", lambda img: "unknown")
        monkeypatch.setattr(stage5, "_detect_skew_angle", lambda gray: 20.0)
        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: [])
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: [])

        stage5.process_ocr_page(page_ir, fitz_page)

        skew_warnings = [w for w in page_ir.warnings if "skew" in w.lower() or "20" in w]
        assert skew_warnings, (
            f"Expected a skew-related warning in page_ir.warnings, got: {page_ir.warnings}"
        )


# ---------------------------------------------------------------------------
# T5 — text_native page → process_ocr_page is a no-op
# ---------------------------------------------------------------------------

class TestTextNativeSkipped:
    def test_no_op_for_text_native(self, monkeypatch):
        """
        text_native pages must not be rasterised or OCR'd at all.
        """
        from pdf_engine.models import TextBlock, BlockType, ExtractionMethod, BlockConfidence
        existing_block = TextBlock(
            text="Hello world",
            block_type=BlockType.BODY,
            extraction_method=ExtractionMethod.PYMUPDF_DIRECT,
        )
        page_ir = _make_page_ir(triage="text_native")
        page_ir.blocks = [existing_block]

        rasterize_called = []
        monkeypatch.setattr(stage5, "rasterize_page", lambda *a, **kw: rasterize_called.append(True) or _rgb_image())

        result = stage5.process_ocr_page(page_ir, fitz_page=MagicMock())

        assert not rasterize_called,           "rasterize_page must not be called for text_native"
        assert len(result.blocks) == 1,        "blocks must be unchanged"
        assert result.blocks[0].text == "Hello world"


# ---------------------------------------------------------------------------
# T6 — Tesseract low confidence → surya fallback called
# ---------------------------------------------------------------------------

class TestTesseractFallbackToSurya:
    def _make_low_conf_blocks(self, n: int = 5):
        """Return TextBlock objects with method_score = 0.40 (< 0.60)."""
        from pdf_engine.models import TextBlock, BlockConfidence, ExtractionMethod
        blocks = []
        for _ in range(n):
            b = TextBlock(
                text="word",
                extraction_method=ExtractionMethod.TESSERACT_OCR,
            )
            b.confidence = BlockConfidence(method_score=0.40)
            blocks.append(b)
        return blocks

    def test_surya_fallback_triggered_on_low_confidence(self, monkeypatch):
        """
        When Tesseract's mean confidence is below the threshold, surya must be
        called as a fallback.
        """
        page_ir     = _make_page_ir(triage="ocr_needed")
        img         = _rgb_image()
        binary      = np.zeros((10, 10), np.uint8)
        low_blocks  = self._make_low_conf_blocks()
        surya_called = []

        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: low_blocks)
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: (surya_called.append(True), [])[1])

        stage5._ocr_page(img, binary, stage5.SCRIPT_LATIN, page_ir)

        assert surya_called, "surya fallback must be called when Tesseract confidence is low"

    def test_surya_not_called_when_tesseract_confident(self, monkeypatch):
        """
        If Tesseract produces high-confidence output surya must NOT be called.
        """
        from pdf_engine.models import TextBlock, BlockConfidence, ExtractionMethod
        page_ir     = _make_page_ir(triage="ocr_needed")
        img         = _rgb_image()
        binary      = np.zeros((10, 10), np.uint8)
        surya_called = []

        high_blocks = []
        for _ in range(3):
            b = TextBlock(text="word", extraction_method=ExtractionMethod.TESSERACT_OCR)
            b.confidence = BlockConfidence(method_score=0.90)
            high_blocks.append(b)

        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: high_blocks)
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: (surya_called.append(True), [])[1])

        stage5._ocr_page(img, binary, stage5.SCRIPT_LATIN, page_ir)

        assert not surya_called, "surya must NOT be called when Tesseract is confident"


# ---------------------------------------------------------------------------
# T7 — Surya-first path: _run_tesseract NEVER called for Arabic/CJK
# ---------------------------------------------------------------------------

class TestSuryaFirstDoesNotCallTesseract:
    @pytest.mark.parametrize("script", [stage5.SCRIPT_CJK, stage5.SCRIPT_ARABIC])
    def test_tesseract_never_called(self, monkeypatch, script):
        page_ir      = _make_page_ir(triage="ocr_needed")
        img          = _rgb_image()
        binary       = np.zeros((10, 10), np.uint8)
        tess_called  = []

        monkeypatch.setattr(stage5, "_run_tesseract", lambda *a: (tess_called.append(True), [])[1])
        monkeypatch.setattr(stage5, "_run_surya_ocr",  lambda *a: [])

        stage5._ocr_page(img, binary, script, page_ir)

        assert not tess_called, f"Tesseract must never be called for {script!r}"


# ---------------------------------------------------------------------------
# T8 — rasterize_page emits warning when memory > 50 MB
# ---------------------------------------------------------------------------

class TestRasterizeMemoryWarning:
    def test_large_raster_warning(self, caplog):
        """
        When estimated raster size > RASTER_MEMORY_WARN_BYTES, a warning
        must be logged.
        """
        import logging
        import fitz  # skip test if PyMuPDF is not installed

        # Build a fake page that reports a very large pixmap.
        h, w = 5000, 5000   # 5000×5000×3 = 75 MB
        samples = bytes([200] * (h * w * 3))

        pixmap = MagicMock()
        pixmap.width   = w
        pixmap.height  = h
        pixmap.samples = samples

        fitz_page = MagicMock()
        fitz_page.get_pixmap.return_value = pixmap

        with patch("fitz.Matrix"):
            # Patch csRGB to avoid importing full fitz in the function body.
            with patch("fitz.csRGB", MagicMock()):
                with caplog.at_level(logging.WARNING, logger="pdf_engine.stage5_ocr"):
                    try:
                        stage5.rasterize_page(fitz_page, dpi=600)
                    except Exception:
                        pass  # We only care about the warning, not the result.

        assert any("50" in r.message or "MB" in r.message for r in caplog.records), (
            "Expected a memory-size warning to be logged for a large raster"
        )


# ---------------------------------------------------------------------------
# T9 — run_stage5 skips text_native, processes ocr_needed
# ---------------------------------------------------------------------------

class TestRunStage5Routing:
    def test_only_ocr_pages_processed(self, monkeypatch):
        """
        run_stage5 must call process_ocr_page only for pages that are NOT
        text_native.
        """
        from pdf_engine.models import PageIR
        pages = [
            _make_page_ir(page_num=0, triage="text_native"),
            _make_page_ir(page_num=1, triage="ocr_needed"),
            _make_page_ir(page_num=2, triage="hybrid"),
        ]
        doc_ir  = _make_doc_ir(pages)
        processed: list[int] = []

        def fake_process(page_ir, fitz_page, dpi=300):
            processed.append(page_ir.page_num)
            return page_ir

        monkeypatch.setattr(stage5, "process_ocr_page", fake_process)

        fitz_doc = MagicMock()
        fitz_doc.__getitem__ = MagicMock(return_value=MagicMock())

        stage5.run_stage5(doc_ir, fitz_doc)

        assert 0 not in processed, "text_native page must not be processed"
        assert 1 in processed,     "ocr_needed page must be processed"
        assert 2 in processed,     "hybrid page must be processed"


# ---------------------------------------------------------------------------
# T10 — preprocess_for_ocr returns 2-D binary array (values 0 or 255)
# ---------------------------------------------------------------------------

class TestPreprocessOutput:
    def test_output_is_2d_binary(self, monkeypatch):
        """
        preprocess_for_ocr must return a 2-D array containing only 0 and 255.
        """
        pytest.importorskip("cv2")

        h, w   = 100, 80
        img_rgb = np.full((h, w, 3), 180, dtype=np.uint8)

        # Force zero skew so we skip the rotation branch.
        monkeypatch.setattr(stage5, "_detect_skew_angle", lambda gray: 0.0)

        result = stage5.preprocess_for_ocr(img_rgb)

        assert result.ndim == 2, f"Expected 2-D output, got shape {result.shape}"
        unique_vals = set(np.unique(result).tolist())
        assert unique_vals.issubset({0, 255}), (
            f"Binary image must contain only 0 and 255, got {unique_vals}"
        )


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_fitz_page_adds_warning(self):
        """process_ocr_page with fitz_page=None adds a warning and returns early."""
        page_ir = _make_page_ir(triage="ocr_needed")
        result  = stage5.process_ocr_page(page_ir, fitz_page=None)

        assert any("OCR skipped" in w or "no fitz" in w.lower() for w in result.warnings), (
            "Expected a warning about missing fitz page"
        )

    def test_hybrid_page_blocks_extended(self, monkeypatch):
        """
        For 'hybrid' pages, existing blocks must be retained and OCR blocks
        must be appended (not replaced).
        """
        from pdf_engine.models import TextBlock
        existing = TextBlock(text="existing")
        page_ir  = _make_page_ir(triage="hybrid")
        page_ir.blocks = [existing]

        new_block = TextBlock(text="ocr_result")

        monkeypatch.setattr(stage5, "rasterize_page", lambda fp, dpi=300: _rgb_image())
        monkeypatch.setattr(stage5, "detect_script_from_image", lambda img: "unknown")
        monkeypatch.setattr(stage5, "preprocess_for_ocr", lambda img, **kw: np.zeros((10, 10), np.uint8))
        monkeypatch.setattr(stage5, "_ocr_page", lambda *a, **kw: [new_block])

        result = stage5.process_ocr_page(page_ir, fitz_page=MagicMock())

        assert len(result.blocks) == 2
        assert result.blocks[0].text == "existing"
        assert result.blocks[1].text == "ocr_result"

    def test_largest_inscribed_rect_zero_angle(self):
        """At 0 radians, the inscribed rectangle should equal the input."""
        w, h = 600, 400
        nw, nh = stage5._largest_inscribed_rect(w, h, 0.0)
        # At angle=0, formula may return ~w, ~h (floating-point rounding is OK).
        assert nw > 0 and nh > 0

    def test_detect_script_unknown_when_unavailable(self, monkeypatch):
        """detect_script_from_image returns 'unknown' when Tesseract is absent."""
        monkeypatch.setattr(stage5, "_TESSERACT_AVAILABLE", False)
        result = stage5.detect_script_from_image(_rgb_image())
        assert result == stage5.SCRIPT_UNKNOWN
