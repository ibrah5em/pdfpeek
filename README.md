# pdfpeek

**PDF to text — with a confidence score.**

[![PyPI](https://img.shields.io/pypi/v/pdfpeek?style=flat-square&color=blue)](https://pypi.org/project/pdfpeek/)
[![Python](https://img.shields.io/pypi/pyversions/pdfpeek?style=flat-square&color=blue)](https://pypi.org/project/pdfpeek/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

Most PDF tools dump text and leave you guessing whether it worked.
**pdfpeek** runs a 10-stage pipeline and tells you exactly how confident it is in every extraction.

![input](https://github.com/user-attachments/assets/0e592a62-90b1-42d7-90d6-c93b8f80722f)


## Install

```bash
# Born-digital PDFs only (~50 MB)
pip install pdfpeek

# + Scanned PDF support via Tesseract (~100 MB)
pip install pdfpeek[ocr]

# + AI layout detection via surya (~5 GB)
pip install pdfpeek[layout]
```

if you want to use `[ocr]` make sure you have tesseract-ocr

---

## Usage

### CLI

```bash
# Extract to plain text (default)
pdfpeek extract document.pdf

# Extract to markdown
pdfpeek extract document.pdf --format markdown --out result.md

# Batch a whole folder
pdfpeek extract ./pdfs/ --out ./txts/

# Password-protected PDF
pdfpeek extract encrypted.pdf --password secret

# Inspect a PDF before extracting
pdfpeek info document.pdf
```

### Python API

```python
from pdf_engine import extract

result = extract("document.pdf")

print(result.text)             # extracted text
print(result.confidence)       # 0.0 – 1.0 document-level confidence
print(result.warnings)         # any issues found during extraction

# Full structured output
for page in result.ir.pages:
    for block in page.blocks:
        print(block.text, block.confidence.final, block.block_type)
```

---

## How it works

pdfpeek runs every PDF through a 10-stage pipeline:

| Stage | Name | What it does |
|-------|------|-------------|
| 0 | Triage | Classifies each page: text-native, scanned, or hybrid |
| 1 | Extraction | Pulls embedded text with phantom-layer detection |
| 2 | Layout | Detects block types (heading, body, table, figure) via surya |
| 3 | Reading Order | XY-cut partitioning, RTL-aware |
| 4 | Tables | Explicit (ruled) and implicit (whitespace) table detection |
| 5 | OCR | Tesseract + surya for scanned/hybrid pages |
| 6 | Assembly | Merges pymupdf and surya outputs; de-duplicates |
| 7 | Cross-page | Strips headers/footers; builds heading hierarchy |
| 8 | Post-processing | Rejoins hyphenation; corrects OCR errors |
| 9 | Confidence | Scores every block on text quality, method trust, order, and type |

Each block gets a confidence score from 0 to 1. The document score is the mean of all block scores.

---

## Comparison

| | pdfpeek | pdfplumber | pypdf | unstructured |
|---|---|---|---|---|
| Born-digital | ✅ | ✅ | ✅ | ✅ |
| Scanned (OCR) | ✅ `[ocr]` | ❌ | ❌ | ✅ |
| Hybrid pages | ✅ | ⚠️ partial | ❌ | ✅ |
| Confidence score | ✅ | ❌ | ❌ | ❌ |
| Install size | **50 MB** | ~20 MB | ~5 MB | **5 GB+** |
| Python API | ✅ | ✅ | ✅ | ✅ |
| CLI | ✅ | ❌ | ❌ | ✅ |

---

## Confidence scoring

Every `TextBlock` has a `BlockConfidence` with four dimensions:

| Dimension | Set by | Meaning |
|-----------|--------|---------|
| `text_quality` | Stage 9 | Fraction of printable, non-garbled characters |
| `method_score` | Stage 9 | Trust in the extraction method (pymupdf=1.0, tesseract=0.7) |
| `order_quality` | Stage 3 | Confidence in reading-order placement |
| `type_quality` | Stage 2 | Confidence in block-type classification |
| `final` | Stage 9 | Geometric mean of the four above |

A document-level score above **0.8** means reliable extraction. Below **0.6** means you should check the output manually or try `[layout]`.

---

## Known limitations (v0.1)

- Very complex multi-column layouts (magazines, newspapers) may have reading-order issues
- Non-English OCR error correction is not yet implemented
- Documents > 1000 pages will be slow (surya processes ~2–3 pages/sec)
- Equations, handwriting, and deeply nested table-in-sidebar structures are best-effort

These are on the roadmap for v0.2.

---

## Contributing

```bash
git clone https://github.com/ibrah5em/pdfpeek
cd pdfpeek
pip install -e ".[ocr,dev]"
pytest tests/
```

PRs welcome. Please add a test for any bug fix.

---

## License

MIT — see [LICENSE](LICENSE).
