"""
pdf_engine/models.py
====================
Core data structures for the PDF text extraction pipeline.

All ten stages consume and produce these typed objects exclusively —
no bare dicts, no untyped lists ever cross a stage boundary.

Design invariants (enforced here and in every stage):
  1. TextBlock.id is always a non-empty UUID string.
  2. parent_id, when set, MUST reference an existing block's id in the
     same document.
  3. script_direction must be explicitly detected; never assumed to be "ltr".
  4. BlockConfidence.final is set only by Stage 9; upstream stages only touch
     order_quality (Stage 3) and type_quality (Stage 2).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BlockType(Enum):
    """Semantic category of an extracted text block."""

    HEADING  = "heading"
    BODY     = "body"
    TABLE    = "table"
    FIGURE   = "figure"
    CAPTION  = "caption"
    HEADER   = "header"   # repeating page header
    FOOTER   = "footer"   # repeating page footer
    FOOTNOTE = "footnote"


class ExtractionMethod(Enum):
    """Which extraction path produced this block's text."""

    PYMUPDF_DIRECT = "pymupdf_direct"   # embedded text layer (Stage 1)
    TESSERACT_OCR  = "tesseract_ocr"    # Tesseract engine   (Stage 5)
    SURYA_OCR      = "surya_ocr"        # Surya OCR engine   (Stage 5)
    SURYA_LAYOUT   = "surya_layout"     # Surya layout model (Stage 2)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    """
    Axis-aligned bounding box in PDF user-space points.

    PDF coordinate origin is bottom-left; y increases upward.
    PyMuPDF / pypdfium2 both normalise to this convention internally,
    so callers should treat (x0, y0) as top-left after any normalisation
    the backend performs.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    # ------------------------------------------------------------------
    # Set operations (used by the sweep-line union in Stage 0)
    # ------------------------------------------------------------------

    def intersects(self, other: "BBox") -> bool:
        return (
            self.x0 < other.x1
            and self.x1 > other.x0
            and self.y0 < other.y1
            and self.y1 > other.y0
        )

    def intersection(self, other: "BBox") -> Optional["BBox"]:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix0 < ix1 and iy0 < iy1:
            return BBox(ix0, iy0, ix1, iy1)
        return None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tuple(cls, t: tuple) -> "BBox":
        return cls(*t)

    def __repr__(self) -> str:
        return (
            f"BBox({self.x0:.1f}, {self.y0:.1f}, "
            f"{self.x1:.1f}, {self.y1:.1f})"
        )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


@dataclass
class BlockConfidence:
    """
    Four independent quality signals per block, merged into ``final``
    by Stage 9 via geometric mean.

    Upstream stage responsibilities
    --------------------------------
    Stage 2 (Layout Analysis)   → set ``type_quality``
    Stage 3 (Reading Order)     → set ``order_quality``
    Stage 9 (Confidence)        → compute and write ``final``

    Default values reflect conservative priors before any stage has
    had a chance to refine them.
    """

    text_quality:  float = 1.0   # Character-level signal quality
    order_quality: float = 0.8   # Reading-order confidence (Stage 3)
    type_quality:  float = 0.7   # Block-type confidence   (Stage 2)
    method_score:  float = 1.0   # Extraction-method trust (Stage 1 / 5)
    final:         float = 0.0   # Geometric mean — written only by Stage 9


# ---------------------------------------------------------------------------
# Core IR nodes
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """
    Atomic unit of extracted content flowing through the pipeline.

    Invariants
    ----------
    * ``id`` is always a UUID4 string — it is generated automatically and
      must never be overwritten with an empty string or None.
    * ``parent_id``, when set, must reference an existing ``TextBlock.id``
      within the same ``DocumentIR``.
    * ``script_direction`` must be detected explicitly (see Stage 1);
      defaulting silently to "ltr" is only acceptable before detection runs.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    bbox: Optional[BBox] = None
    block_type: BlockType = BlockType.BODY
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF_DIRECT
    confidence: BlockConfidence = field(default_factory=BlockConfidence)
    page_num: int = 0
    parent_id: Optional[str] = None
    language: Optional[str] = None
    script_direction: str = "ltr"   # "ltr" | "rtl"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError(
                "TextBlock.id must be a non-empty UUID string. "
                "Never assign an empty string or None to this field."
            )

    @property
    def is_rtl(self) -> bool:
        return self.script_direction == "rtl"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class PageIR:
    """
    Intermediate representation for a single PDF page.

    ``triage_result`` is written by Stage 0; every subsequent stage
    enriches ``blocks``, ``order_strategy``, and ``warnings`` in place
    before passing the object downstream.
    """

    page_num: int
    blocks: list = field(default_factory=list)   # list[TextBlock]
    width: float = 0.0
    height: float = 0.0
    triage_result: str = ""     # "text_native" | "ocr_needed" | "hybrid"
    order_strategy: str = ""    # "xy_cut" | "band_based"
    warnings: list = field(default_factory=list)  # list[str]

    @property
    def area(self) -> float:
        return self.width * self.height

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class DocumentIR:
    """
    Top-level intermediate representation for an entire PDF.

    ``confidence`` is the document-level aggregate set by Stage 9.
    ``warnings`` accumulates non-fatal issues from all stages so that
    callers can audit what happened without crashing the pipeline.
    """

    pages: list = field(default_factory=list)    # list[PageIR]
    title: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0
    warnings: list = field(default_factory=list)  # list[str]

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def all_blocks(self) -> list:
        """Flatten every block across all pages into a single list."""
        return [block for page in self.pages for block in page.blocks]

    def block_by_id(self, block_id: str) -> Optional[TextBlock]:
        """Look up a block by its UUID; returns None if not found.

        Uses a freshly built dict for O(1) lookup.  For repeated calls in a
        tight loop, build the dict once yourself via
        ``{b.id: b for b in doc_ir.all_blocks()}``.
        """
        return {b.id: b for b in self.all_blocks()}.get(block_id)

    def validate_parent_refs(self) -> list:
        """
        Return error strings for any parent_id that does not resolve to a
        real block.  An empty list means the document is internally consistent.
        """
        # MED-5 fix: True single pass - collect known_ids and pending checks in one loop
        known_ids: set[str] = set()
        pending_checks: list[TextBlock] = []

        for block in self.all_blocks():
            known_ids.add(block.id)
            if block.parent_id:
                pending_checks.append(block)

        # Second pass only over blocks with parent_id (much smaller list)
        errors = []
        for block in pending_checks:
            if block.parent_id not in known_ids:
                errors.append(
                    f"Block {block.id} on page {block.page_num} references "
                    f"unknown parent_id={block.parent_id}"
                )
        return errors
