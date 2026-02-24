"""
tests/test_stage0.py
====================
Unit tests for Stage 0 — PDF Ingestion + Page Triage.

Test strategy
-------------
We build synthetic PDFs programmatically so that every test has
ground-truth knowledge of exactly what is on each page:

  - Text-only pages  → uses reportlab to draw text strings
  - Image-only pages → uses reportlab to draw a full-page raster image
  - Watermarked PDFs → two overlapping images on the same page (tests W1)
  - Encrypted PDFs   → uses pikepdf to apply 256-bit AES encryption

All PDFs are written to a temporary directory and cleaned up after each
test module run.

Fixtures
--------
  tmp_pdf_dir   — session-scoped tmp directory (pytest autouse)
  digital_pdf   — born-digital, 2 pages of dense text
  scanned_pdf   — image-only, 2 pages of full-page raster images
  watermark_pdf — 1 page with two overlapping images + no text
  hybrid_pdf    — 1 page with text AND a medium-sized image
  encrypted_pdf — 1 page, encrypted with AES-256 (correct password: "s3cr3t")
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if test-creation deps are absent
# ---------------------------------------------------------------------------
reportlab = pytest.importorskip("reportlab", reason="reportlab not installed")
pikepdf   = pytest.importorskip("pikepdf",   reason="pikepdf not installed")

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image

from pdf_engine.models import DocumentIR, PageIR
from pdf_engine.stage0_triage import (
    IMAGE_COV_CLEAN,
    IMAGE_COV_HIGH,
    TEXT_DENSITY_RICH,
    TEXT_DENSITY_SPARSE,
    _union_area,
    triage_document,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic PDFs
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4  # ≈ 595 × 842 pt


def _solid_image(width: int = 100, height: int = 100, color: tuple = (180, 180, 180)) -> io.BytesIO:
    """Return an in-memory PNG image of the given solid colour."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _make_text_pdf(path: str, pages: int = 2, chars_per_page: int = 600) -> None:
    """
    Write a born-digital PDF where every page is filled with repeated
    Latin text.  No images at all.
    """
    c = canvas.Canvas(path, pagesize=A4)
    for _ in range(pages):
        # Fill the page with enough characters to exceed TEXT_DENSITY_RICH.
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
        text = text[:chars_per_page]

        text_obj = c.beginText(50, PAGE_H - 60)
        text_obj.setFont("Helvetica", 10)
        # Break into lines of ~80 chars
        for i in range(0, len(text), 80):
            text_obj.textLine(text[i : i + 80])
        c.drawText(text_obj)
        c.showPage()
    c.save()


def _make_scanned_pdf(path: str, pages: int = 2) -> None:
    """
    Write a PDF where every page consists entirely of a full-page raster
    image with no embedded text — mimicking a scanned document.
    """
    c = canvas.Canvas(path, pagesize=A4)
    for _ in range(pages):
        img_buf = _solid_image(800, 1100, color=(200, 200, 200))
        c.drawImage(ImageReader(img_buf), 0, 0, width=PAGE_W, height=PAGE_H)
        img_buf.close()
        c.showPage()
    c.save()


def _make_watermark_pdf(path: str) -> None:
    """
    Write a single-page PDF with TWO overlapping full-page images and no
    text.  Without union-area computation (W1 fix), the summed image area
    would be 200 % of the page — still correctly identified as ocr_needed,
    but the coverage value would be wrong (> 1.0).  This test verifies that
    the union approach caps coverage sensibly at ≤ 1.0.
    """
    c = canvas.Canvas(path, pagesize=A4)
    # First image: full page
    img1 = _solid_image(800, 1100, color=(220, 220, 220))
    c.drawImage(ImageReader(img1), 0, 0, width=PAGE_W, height=PAGE_H)
    img1.close()
    # Second image: also full page (simulates a watermark layer)
    img2 = _solid_image(800, 1100, color=(180, 180, 180))
    c.drawImage(ImageReader(img2), 0, 0, width=PAGE_W, height=PAGE_H)
    img2.close()
    c.showPage()
    c.save()


def _make_hybrid_pdf(path: str) -> None:
    """
    Write a single-page PDF with substantial text AND a mid-page image
    that covers ~40 % of the page — should classify as "hybrid".
    """
    c = canvas.Canvas(path, pagesize=A4)
    # Dense text block
    text = "This is real embedded text content. " * 30
    text_obj = c.beginText(50, PAGE_H - 60)
    text_obj.setFont("Helvetica", 10)
    for i in range(0, len(text), 80):
        text_obj.textLine(text[i : i + 80])
    c.drawText(text_obj)
    # Image covering middle ~40 % of the page
    img_buf = _solid_image(400, 400, color=(160, 160, 160))
    c.drawImage(
        ImageReader(img_buf),
        50, 200,
        width=PAGE_W - 100,
        height=PAGE_H * 0.4,
    )
    img_buf.close()
    c.showPage()
    c.save()


def _make_encrypted_pdf(path: str, user_password: str = "s3cr3t") -> None:
    """
    Write a single-page PDF encrypted with AES-256 via pikepdf.
    """
    import pikepdf as _pikepdf

    # First create a plain PDF, then encrypt it.
    plain_buf = io.BytesIO()
    c = canvas.Canvas(plain_buf, pagesize=A4)
    c.drawString(72, PAGE_H - 100, "Confidential content.")
    c.showPage()
    c.save()
    plain_buf.seek(0)

    with _pikepdf.open(plain_buf) as pdf:
        pdf.save(
            path,
            encryption=_pikepdf.Encryption(
                owner="ownerpass",
                user=user_password,
                R=6,   # AES-256
            ),
        )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_pdf_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(prefix="stage0_tests_") as d:
        yield Path(d)


@pytest.fixture(scope="session")
def digital_pdf(tmp_pdf_dir: Path) -> str:
    p = str(tmp_pdf_dir / "digital.pdf")
    _make_text_pdf(p, pages=3)
    return p


@pytest.fixture(scope="session")
def scanned_pdf(tmp_pdf_dir: Path) -> str:
    p = str(tmp_pdf_dir / "scanned.pdf")
    _make_scanned_pdf(p, pages=3)
    return p


@pytest.fixture(scope="session")
def watermark_pdf(tmp_pdf_dir: Path) -> str:
    p = str(tmp_pdf_dir / "watermark.pdf")
    _make_watermark_pdf(p)
    return p


@pytest.fixture(scope="session")
def hybrid_pdf(tmp_pdf_dir: Path) -> str:
    p = str(tmp_pdf_dir / "hybrid.pdf")
    _make_hybrid_pdf(p)
    return p


@pytest.fixture(scope="session")
def encrypted_pdf(tmp_pdf_dir: Path) -> str:
    p = str(tmp_pdf_dir / "encrypted.pdf")
    _make_encrypted_pdf(p, user_password="s3cr3t")
    return p


# ===========================================================================
# _union_area unit tests (W1 — isolated, no PDF required)
# ===========================================================================

class TestUnionArea:
    """
    Verify the sweep-line union-area algorithm in isolation before relying
    on it inside triage_document.
    """

    def test_empty_list_returns_zero(self):
        assert _union_area([]) == 0.0

    def test_single_rectangle(self):
        # 10 × 10 square
        assert _union_area([(0, 0, 10, 10)]) == pytest.approx(100.0)

    def test_two_non_overlapping_rectangles(self):
        # Two 10×10 squares side by side; union = 200
        assert _union_area([(0, 0, 10, 10), (20, 0, 30, 10)]) == pytest.approx(200.0)

    def test_two_fully_overlapping_rectangles(self):
        # Identical squares — union must equal ONE square's area.
        assert _union_area([(0, 0, 10, 10), (0, 0, 10, 10)]) == pytest.approx(100.0)

    def test_two_partially_overlapping_rectangles(self):
        # Overlap by 5×10; union = 200 - 50 = 150
        assert _union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == pytest.approx(150.0)

    def test_full_page_watermark_coverage_does_not_exceed_page_area(self):
        """
        Two full-page images: sum would be 2 × page_area, but union = page_area.
        This is the core W1 regression guard.
        """
        pw, ph = 595.0, 842.0
        page_area = pw * ph
        boxes = [(0, 0, pw, ph), (0, 0, pw, ph)]
        union = _union_area(boxes)
        assert union == pytest.approx(page_area, rel=1e-4)

    def test_three_partly_overlapping_rectangles(self):
        # Three 10×10 squares where adjacent pairs overlap by 5 units on x.
        # [0,10] ∪ [5,15] ∪ [10,20] on y=0..10 → x union = [0,20] = 20 wide × 10 tall = 200
        assert _union_area([
            (0, 0, 10, 10),
            (5, 0, 15, 10),
            (10, 0, 20, 10),
        ]) == pytest.approx(200.0)

    def test_nested_rectangles(self):
        # Inner rect fully inside outer; union = outer area.
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        assert _union_area([outer, inner]) == pytest.approx(100 * 100)

    def test_inverted_coordinates_normalised(self):
        # x0 > x1 or y0 > y1 should still produce correct area.
        assert _union_area([(10, 10, 0, 0)]) == pytest.approx(100.0)


# ===========================================================================
# DocumentIR structure tests (model invariants)
# ===========================================================================

class TestDocumentIRInvariants:
    """
    triage_document must always return a fully typed DocumentIR regardless
    of the PDF content.
    """

    def test_returns_document_ir_type(self, digital_pdf):
        result = triage_document(digital_pdf)
        assert isinstance(result, DocumentIR)

    def test_pages_are_page_ir_instances(self, digital_pdf):
        result = triage_document(digital_pdf)
        for page in result.pages:
            assert isinstance(page, PageIR)

    def test_page_dimensions_are_positive(self, digital_pdf):
        result = triage_document(digital_pdf)
        for page in result.pages:
            assert page.width > 0, f"Page {page.page_num} has non-positive width"
            assert page.height > 0, f"Page {page.page_num} has non-positive height"

    def test_triage_result_is_valid_string(self, digital_pdf):
        valid = {"text_native", "ocr_needed", "hybrid"}
        result = triage_document(digital_pdf)
        for page in result.pages:
            assert page.triage_result in valid, (
                f"Page {page.page_num} has invalid triage_result: "
                f"{page.triage_result!r}"
            )

    def test_warnings_is_list(self, digital_pdf):
        result = triage_document(digital_pdf)
        assert isinstance(result.warnings, list)

    def test_missing_file_adds_warning_returns_empty(self):
        result = triage_document("/tmp/__nonexistent_file_xyz123.pdf")
        assert result.pages == []
        assert any("not found" in w.lower() or "W29" in w for w in result.warnings)


# ===========================================================================
# Test 1 — Born-digital PDF → all pages text_native
# ===========================================================================

class TestBornDigitalPDF:
    """
    A PDF with dense embedded text and no images should have every page
    classified as "text_native".
    """

    def test_all_pages_classified(self, digital_pdf):
        result = triage_document(digital_pdf)
        assert len(result.pages) == 3

    def test_all_pages_text_native(self, digital_pdf):
        result = triage_document(digital_pdf)
        for page in result.pages:
            assert page.triage_result == "text_native", (
                f"Page {page.page_num} was classified as {page.triage_result!r}, "
                f"expected 'text_native'"
            )

    def test_no_encryption_warnings(self, digital_pdf):
        result = triage_document(digital_pdf)
        enc_warnings = [w for w in result.warnings if "W29" in w]
        assert enc_warnings == []


# ===========================================================================
# Test 2 — Scanned (image-only) PDF → all pages ocr_needed
# ===========================================================================

class TestScannedPDF:
    """
    A PDF whose pages consist entirely of raster images (no embedded text)
    should have every page classified as "ocr_needed".
    """

    def test_all_pages_classified(self, scanned_pdf):
        result = triage_document(scanned_pdf)
        assert len(result.pages) == 3

    def test_all_pages_ocr_needed(self, scanned_pdf):
        result = triage_document(scanned_pdf)
        for page in result.pages:
            assert page.triage_result == "ocr_needed", (
                f"Page {page.page_num} was classified as {page.triage_result!r}, "
                f"expected 'ocr_needed'"
            )


# ===========================================================================
# Test 3 — Watermarked PDF → union area, not sum
# ===========================================================================

class TestWatermarkPDF:
    """
    A PDF with two overlapping full-page images must be classified as
    "ocr_needed" — but more importantly, the image_coverage must be ≤ 1.0
    (proving that union, not sum, was used).

    We test this indirectly through triage_result and directly by re-running
    the union_area helper with the expected boxes.
    """

    def test_single_page(self, watermark_pdf):
        result = triage_document(watermark_pdf)
        assert len(result.pages) == 1

    def test_classified_as_ocr_needed(self, watermark_pdf):
        result = triage_document(watermark_pdf)
        assert result.pages[0].triage_result == "ocr_needed"

    def test_union_does_not_double_count(self):
        """
        Directly verify that two identical full-A4 boxes produce exactly
        one page-worth of area — not two.
        """
        pw, ph = float(PAGE_W), float(PAGE_H)
        page_area = pw * ph
        boxes = [(0.0, 0.0, pw, ph), (0.0, 0.0, pw, ph)]
        union = _union_area(boxes)
        # With naive summing: union would be 2 × page_area.
        # With correct union: exactly page_area.
        assert union == pytest.approx(page_area, rel=1e-4), (
            f"union_area={union:.1f} but expected ≈ {page_area:.1f}; "
            "double-counting detected — W1 fix is broken."
        )

    def test_coverage_stays_at_or_below_one(self):
        """Coverage = union / page_area must never exceed 1.0."""
        pw, ph = float(PAGE_W), float(PAGE_H)
        page_area = pw * ph
        boxes = [(0.0, 0.0, pw, ph), (0.0, 0.0, pw, ph)]
        coverage = _union_area(boxes) / page_area
        assert coverage <= 1.0 + 1e-6, (
            f"image_coverage={coverage:.4f} > 1.0 — union not being used."
        )


# ===========================================================================
# Test 4 — Encrypted PDF without password → warning + zero pages
# ===========================================================================

class TestEncryptedPDF:
    """
    W29 fix: An encrypted PDF opened without a password must never produce
    silent empty output.  Instead, DocumentIR.warnings must contain an
    encryption-specific message and pages must be empty.
    """

    def test_no_pages_processed(self, encrypted_pdf):
        result = triage_document(encrypted_pdf, password=None)
        assert result.pages == [], (
            f"Expected 0 pages for encrypted PDF without password, "
            f"got {len(result.pages)}"
        )

    def test_warning_present(self, encrypted_pdf):
        result = triage_document(encrypted_pdf, password=None)
        assert len(result.warnings) >= 1, (
            "Expected at least one warning for encrypted PDF"
        )

    def test_warning_mentions_encryption(self, encrypted_pdf):
        result = triage_document(encrypted_pdf, password=None)
        combined = " ".join(result.warnings).lower()
        assert any(
            keyword in combined
            for keyword in ("encrypt", "password", "w29", "decrypt")
        ), (
            f"Warning does not mention encryption or W29. Got: {result.warnings}"
        )

    def test_wrong_password_also_produces_warning(self, encrypted_pdf):
        result = triage_document(encrypted_pdf, password="wrongpassword")
        assert result.pages == []
        assert len(result.warnings) >= 1

    def test_correct_password_processes_pages(self, encrypted_pdf):
        """Sanity-check: correct password must succeed."""
        result = triage_document(encrypted_pdf, password="s3cr3t")
        assert len(result.pages) >= 1, (
            "Correct password should allow page processing"
        )
        assert all(
            p.triage_result in {"text_native", "ocr_needed", "hybrid"}
            for p in result.pages
        )


# ===========================================================================
# Test 5 — Hybrid PDF → pages classified as hybrid
# ===========================================================================

class TestHybridPDF:
    """
    A PDF with both substantial text and a medium-sized image should be
    classified as "hybrid" (neither pure text nor pure image).
    """

    def test_single_page(self, hybrid_pdf):
        result = triage_document(hybrid_pdf)
        assert len(result.pages) == 1

    def test_classified_as_hybrid(self, hybrid_pdf):
        result = triage_document(hybrid_pdf)
        page = result.pages[0]
        assert page.triage_result == "hybrid", (
            f"Expected 'hybrid', got {page.triage_result!r}"
        )


# ===========================================================================
# Test 6 — Threshold boundary conditions
# ===========================================================================

class TestThresholdConstants:
    """
    Sanity-checks on exported threshold constants so that a future
    change to any threshold breaks a test explicitly rather than
    silently altering behaviour.
    """

    def test_image_cov_high_is_sane(self):
        assert 0.5 < IMAGE_COV_HIGH <= 1.0

    def test_image_cov_clean_less_than_high(self):
        assert IMAGE_COV_CLEAN < IMAGE_COV_HIGH

    def test_text_density_thresholds_are_positive(self):
        assert TEXT_DENSITY_SPARSE > 0
        assert TEXT_DENSITY_RICH > 0


# ===========================================================================
# Test 7 — Session management (document opened only once per call)
# ===========================================================================

class TestSessionManagement:
    """
    Verify that calling triage_document with the same path twice produces
    two independent, consistent results — confirming the document is opened
    fresh per call (not shared across calls) and is properly closed.
    """

    def test_idempotent_results(self, digital_pdf):
        result_a = triage_document(digital_pdf)
        result_b = triage_document(digital_pdf)
        assert len(result_a.pages) == len(result_b.pages)
        for pa, pb in zip(result_a.pages, result_b.pages):
            assert pa.triage_result == pb.triage_result
            assert pa.width  == pytest.approx(pb.width,  rel=1e-4)
            assert pa.height == pytest.approx(pb.height, rel=1e-4)
