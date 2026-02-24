"""
pdfpeek — PDF to text extraction with confidence scoring.

Quick start
-----------
    from pdf_engine import extract

    result = extract("document.pdf")
    print(result.text)
    print(f"Confidence: {result.confidence:.3f}")
"""

from pdf_engine.api import extract, ExtractionResult
from pdf_engine.models import (
    DocumentIR,
    PageIR,
    TextBlock,
    BlockConfidence,
    BBox,
    BlockType,
    ExtractionMethod,
)

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("pdfpeek")
except Exception:
    __version__ = "0.1.0"  # fallback for editable / pre-install runs
__all__ = [
    "extract",
    "ExtractionResult",
    "DocumentIR",
    "PageIR",
    "TextBlock",
    "BlockConfidence",
    "BBox",
    "BlockType",
    "ExtractionMethod",
]