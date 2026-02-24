# Changelog

All notable changes to pdfpeek will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025

### Added
- 10-stage PDF extraction pipeline (triage → extraction → layout → reading order
  → tables → OCR → assembly → cross-page → post-processing → confidence scoring)
- Confidence scoring on every extracted block and at document level
- Three-tier install: core (~50 MB), `[ocr]` (~100 MB), `[layout]` (~5 GB)
- CLI: `pdfpeek extract` and `pdfpeek info`
- Born-digital, scanned, and hybrid PDF support
- Union-based image coverage for accurate page triage (W1)
- Page-size-normalised text density thresholds (W2)
- Center-containment block merging — no IoU false negatives (W19)
- Encrypted PDF detection with structured warnings (W29)
- RTL script direction support throughout (W11)
- O(n) header/footer detection via text fingerprinting (W20)
- Graceful surya fallback when layout model is not installed
- 337 passing unit tests

### Known limitations (V2)
- Multi-column reading order: XY-cut works; very complex layouts may scramble
- Hybrid page sub-region isolation is best-effort only
- Signal 4 phantom detection (borderline 0.3–0.7 scores) deferred
- Non-English OCR error correction not yet implemented
- Documents > 1000 pages: memory warning only, no streaming
