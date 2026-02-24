"""
pdf_engine/cli.py
==================
Command-line interface for pdfpeek.

Usage
-----
    pdfpeek extract document.pdf
    pdfpeek extract document.pdf --format markdown --out result.md
    pdfpeek extract document.pdf --format json --out result.json
    pdfpeek extract ./pdfs/ --out ./txts/
    pdfpeek info document.pdf
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import itertools
import logging
from pathlib import Path
from typing import Optional

import click

# BUG-22 fix: initialize colorama so Unicode symbols (✔ ✖ █ ⚠) and ANSI
# colours render correctly on Windows cmd.exe and older PowerShell versions.
try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    pass  # colorama is in base deps; this is a safety net only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stage(name: str, detail: str = "", ok: bool = True) -> None:
    symbol = click.style("✔", fg="green") if ok else click.style("⚠", fg="yellow")
    line = f"  {symbol}  {name:<22} {detail}"
    click.echo(line)


def _confidence_colour(score: float) -> str:
    colour = "green" if score >= 0.8 else ("yellow" if score >= 0.6 else "red")
    return click.style(f"{score:.3f}", fg=colour, bold=True)


def _triage_bars(pages: list) -> None:
    """Print coloured per-type bar breakdown inside the extract command (CLI-3 fix)."""
    counts: dict[str, int] = {}
    for p in pages:
        counts[p.triage_result] = counts.get(p.triage_result, 0) + 1

    labels = {
        "text_native": ("Text (native)", "green"),
        "hybrid":      ("Hybrid",        "yellow"),
        "ocr_needed":  ("Scanned (OCR)", "red"),
    }
    for key, (label, colour) in labels.items():
        n = counts.get(key, 0)
        if n:
            bar = click.style("█" * min(n, 40), fg=colour)
            click.echo(f"    {label:<18} {n:>3}  {bar}")


# ---------------------------------------------------------------------------
# CLI-1: Spinner — shows a live elapsed-time indicator while extraction runs
# ---------------------------------------------------------------------------

class _Spinner:
    """Runs a spinner in a background thread while the pipeline executes."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Processing") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._is_tty = sys.stdout.isatty()

    def start(self) -> None:
        self._start_time = time.time()
        if not self._is_tty:
            click.echo(f"  ⏳  {self._label}…")
            return
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            # Clear the spinner line
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()

    def elapsed(self) -> float:
        return time.time() - self._start_time

    def _spin(self) -> None:
        # CLI-1 fix: Use generic label instead of time-based stage guessing
        # Time-based labels were misleading - a fast PDF would show "Triaging"
        # the whole time, while a slow one might show "Running OCR" before OCR starts
        stage_label = self._label if self._label else "Processing…"

        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.time() - self._start_time
            colour_frame = click.style(frame, fg="cyan", bold=True)
            line = f"\r  {colour_frame}  {stage_label:<36} {click.style(f'{elapsed:.1f}s', fg='bright_black')}  "
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.08)


# ---------------------------------------------------------------------------
# Core extraction runner
# ---------------------------------------------------------------------------

def _run_and_report(
    pdf_path: str,
    fmt: str,
    out: str | None,
    password: str | None,
    ocr_dpi: int,
    strip_headers: bool,
    verbose: bool = False,
    as_json: bool = False,
) -> dict | None:
    """Run the pipeline with a live spinner and stage-by-stage progress output.

    Returns a JSON-serialisable dict when ``as_json=True``, otherwise None.
    """
    from pdf_engine.api import extract

    click.echo()
    click.echo(click.style(f"  pdfpeek  →  {pdf_path}", bold=True))
    click.echo()

    # --- CLI-1: spinner while extraction runs in a background thread ---
    spinner = _Spinner(label="Extracting")
    result = None
    exc_holder: list[BaseException] = []

    def _worker() -> None:
        nonlocal result
        try:
            result = extract(
                pdf_path,
                output_format=fmt if not as_json else "plain",
                ocr_dpi=ocr_dpi,
                strip_headers=strip_headers,
                password=password,
            )
        except BaseException as e:
            exc_holder.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    spinner.start()
    t.start()
    t.join()
    spinner.stop(success=not exc_holder)
    elapsed = spinner.elapsed()

    if exc_holder:
        exc = exc_holder[0]
        if isinstance(exc, FileNotFoundError):
            click.echo(click.style(f"  ✖  {exc}", fg="red"))
        else:
            click.echo(click.style(f"  ✖  Extraction failed: {exc}", fg="red"))
        sys.exit(1)

    assert result is not None

    # --- Triage summary with coloured bars (CLI-3 fix) ---
    pages = result.ir.pages if result.ir else []
    triage_counts: dict[str, int] = {}
    for p in pages:
        triage_counts[p.triage_result] = triage_counts.get(p.triage_result, 0) + 1
    triage_str = "  ".join(f"{k}: {v}" for k, v in sorted(triage_counts.items()))
    _print_stage("Triage", f"{len(pages)} page(s)  {triage_str}")
    _triage_bars(pages)

    # --- Block count ---
    total_blocks = sum(len(p.blocks) for p in pages)
    _print_stage("Extraction", f"{total_blocks} blocks")

    # --- Warnings summary ---
    warn_count = len(result.warnings)
    if warn_count:
        _print_stage("Warnings", f"{warn_count} warning(s)", ok=False)

    # --- Confidence + timing ---
    click.echo()
    click.echo(f"  Confidence   {_confidence_colour(result.confidence)}")
    click.echo(f"  Time         {click.style(f'{elapsed:.1f}s', fg='bright_black')}")
    click.echo()

    # --- CLI-4: JSON output ---
    if as_json:
        payload: dict = {
            "source": pdf_path,
            "confidence": result.confidence,
            "elapsed_seconds": round(elapsed, 2),
            "warnings": result.warnings,
            "triage": triage_counts,
            "total_blocks": total_blocks,
            "pages": [
                {
                    "page_num": p.page_num,
                    "triage": p.triage_result,
                    "blocks": len(p.blocks),
                    "warnings": p.warnings,
                }
                for p in pages
            ],
            "text": result.text,
        }
        json_str = json.dumps(payload, ensure_ascii=False, indent=2)
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
            click.echo(f"  Output  →  {out_path}")
        else:
            stem = Path(pdf_path).stem
            default_out = Path(pdf_path).parent / (stem + ".json")
            if default_out.exists():
                click.echo(click.style(f"  ⚠  Overwriting existing file: {default_out}", fg="yellow"))
            default_out.write_text(json_str, encoding="utf-8")
            click.echo(f"  Output  →  {default_out}")
        click.echo()
        return payload

    # --- Write text/markdown output ---
    if result.text:
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.text, encoding="utf-8")
            click.echo(f"  Output  →  {out_path}")
        else:
            stem = Path(pdf_path).stem
            suffix = ".md" if fmt == "markdown" else ".txt"
            default_out = Path(pdf_path).parent / (stem + suffix)
            # BUG-19 fix: warn the user when overwriting an existing file.
            if default_out.exists():
                click.echo(
                    click.style(
                        f"  ⚠  Overwriting existing file: {default_out}",
                        fg="yellow",
                    )
                )
            default_out.write_text(result.text, encoding="utf-8")
            click.echo(f"  Output  →  {default_out}")
    else:
        click.echo(click.style("  ⚠  No text extracted.", fg="yellow"))

    click.echo()

    # --- CLI-2: Print warnings when --verbose is set (not just env var) ---
    if warn_count and (verbose or os.environ.get("PDFPEEK_VERBOSE")):
        click.echo(click.style("  Warnings:", fg="yellow"))
        for w in result.warnings:
            click.echo(f"    {w}")
        click.echo()

    # MED-2 fix: Always return at least confidence for batch mode tracking
    return {"confidence": result.confidence}


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="pdfpeek", message="%(version)s")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed pipeline logs and warnings.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """pdfpeek — PDF to text with confidence scoring."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@main.command(name="extract")
@click.argument("input_path", metavar="PDF_OR_DIR")
@click.option("--format", "fmt", default="plain",
              type=click.Choice(["plain", "markdown", "json"]),
              show_default=True,
              help="Output format. 'json' emits structured data (CI-friendly).")
@click.option("--out", "-o", default=None,
              help="Output file (or directory when input is a folder). "
                   "Defaults to same location as input.")
@click.option("--password", "-p", default=None,
              help="Decryption password for encrypted PDFs.")
@click.option("--ocr-dpi", default=300, show_default=True,
              help="DPI for page rasterisation during OCR.")
@click.option("--no-strip-headers", is_flag=True, default=False,
              help="Keep detected header/footer blocks in output.")
def pdf_extract(input_path: str, fmt: str, out: str | None, password: str | None,
                ocr_dpi: int, no_strip_headers: bool) -> None:
    """Extract text from a PDF file or a folder of PDFs.

    \b
    Examples:
      pdfpeek extract document.pdf
      pdfpeek extract document.pdf --format markdown --out result.md
      pdfpeek extract document.pdf --format json --out result.json
      pdfpeek extract ./pdfs/ --out ./txts/
      pdfpeek extract scan.pdf --ocr-dpi 400
      pdfpeek extract encrypted.pdf --password secret
    """
    path = Path(input_path)
    ctx = click.get_current_context()
    verbose_flag: bool = ctx.find_root().params.get("verbose", False)
    as_json = fmt == "json"

    if path.is_dir():
        # Batch mode
        pdfs = sorted(path.glob("**/*.pdf"))
        if not pdfs:
            click.echo(click.style(f"  No PDFs found in {path}", fg="yellow"))
            sys.exit(1)

        out_dir = Path(out) if out else path
        out_dir.mkdir(parents=True, exist_ok=True)

        click.echo(click.style(f"\n  Batch: {len(pdfs)} PDF(s) in {path}\n", bold=True))
        successes = 0
        failures = 0
        confidences: list[float] = []
        batch_results: list[dict] = []

        for pdf in pdfs:
            suffix = ".json" if as_json else (".md" if fmt == "markdown" else ".txt")
            out_file = str(out_dir / (pdf.stem + suffix))
            try:
                payload = _run_and_report(
                    str(pdf), fmt, out_file, password, ocr_dpi,
                    strip_headers=not no_strip_headers,
                    verbose=verbose_flag,
                    as_json=as_json,
                )
                successes += 1
                # MED-2 fix: Track confidence in all modes, not just JSON
                if payload and "confidence" in payload:
                    confidences.append(payload["confidence"])
                if as_json and payload:
                    batch_results.append(payload)
            except SystemExit:
                failures += 1

        # CLI-5: batch summary with average confidence
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        click.echo()
        click.echo(
            f"  {click.style('Batch complete', bold=True)}  "
            f"{click.style(str(successes), fg='green')} succeeded  "
            f"{click.style(str(failures), fg='red' if failures else 'bright_black')} failed  "
            f"of {len(pdfs)} total"
            + (f"  |  avg confidence {_confidence_colour(avg_conf)}" if confidences else "")
        )
        click.echo()

        if as_json and batch_results:
            batch_summary = {
                "total": len(pdfs),
                "succeeded": successes,
                "failed": failures,
                "avg_confidence": round(avg_conf, 4),
                "results": batch_results,
            }
            summary_path = out_dir / "_batch_summary.json"
            summary_path.write_text(
                json.dumps(batch_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            click.echo(f"  Batch summary  →  {summary_path}\n")

    elif path.is_file():
        _run_and_report(
            str(path), fmt, out, password, ocr_dpi,
            strip_headers=not no_strip_headers,
            verbose=verbose_flag,
            as_json=as_json,
        )
    else:
        click.echo(click.style(f"  ✖  Not found: {input_path}", fg="red"))
        sys.exit(1)


@main.command()
@click.argument("pdf_path", metavar="PDF")
@click.option("--password", "-p", default=None,
              help="Decryption password for encrypted PDFs.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output info as JSON (for scripting).")
def info(pdf_path: str, password: str | None, as_json: bool) -> None:
    """Show metadata and page classification for a PDF without extracting text.

    \b
    Example:
      pdfpeek info document.pdf
      pdfpeek info document.pdf --json
    """
    from pdf_engine.stage0_triage import triage_document

    if not Path(pdf_path).is_file():
        click.echo(click.style(f"  ✖  Not found: {pdf_path}", fg="red"))
        sys.exit(1)

    # Show a brief spinner while triage runs
    spinner = _Spinner(label="Analysing")
    doc_ir = None
    exc_holder: list[BaseException] = []

    def _worker() -> None:
        nonlocal doc_ir
        try:
            doc_ir = triage_document(pdf_path, password=password)
        except BaseException as e:
            exc_holder.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    spinner.start()
    t.start()
    t.join()
    spinner.stop()

    if exc_holder:
        click.echo(click.style(f"  ✖  {exc_holder[0]}", fg="red"))
        sys.exit(1)

    assert doc_ir is not None

    if not doc_ir.pages:
        click.echo(click.style("  ✖  Could not open PDF (encrypted or corrupted).", fg="red"))
        for w in doc_ir.warnings:
            click.echo(f"     {w}")
        sys.exit(1)

    counts: dict[str, int] = {}
    for p in doc_ir.pages:
        counts[p.triage_result] = counts.get(p.triage_result, 0) + 1

    # --- JSON mode (for scripting) ---
    if as_json:
        payload = {
            "path": pdf_path,
            "title": doc_ir.title,
            "page_count": len(doc_ir.pages),
            "triage": counts,
            "warnings": doc_ir.warnings,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # --- Human-readable mode ---
    click.echo()
    click.echo(click.style(f"  {pdf_path}", bold=True))
    click.echo()

    if doc_ir.title:
        click.echo(f"  Title       {doc_ir.title}")

    click.echo(f"  Pages       {len(doc_ir.pages)}")
    click.echo()

    labels = {
        "text_native": ("Text (native)", "green"),
        "hybrid":      ("Hybrid",        "yellow"),
        "ocr_needed":  ("Scanned (OCR)", "red"),
    }
    for key, (label, colour) in labels.items():
        n = counts.get(key, 0)
        if n:
            # CLI-2 fix: Cap bar length to prevent terminal wrapping on large PDFs
            bar = click.style("█" * min(n, 40), fg=colour)
            click.echo(f"  {label:<18} {n:>3}  {bar}")

    click.echo()

    if doc_ir.warnings:
        click.echo(click.style("  Warnings:", fg="yellow"))
        for w in doc_ir.warnings:
            click.echo(f"    {w}")
        click.echo()


# ---------------------------------------------------------------------------
# CLI-3: Enhanced version command
# ---------------------------------------------------------------------------


@main.command(name="version")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show Python and dependency versions.")
def version_cmd(verbose: bool) -> None:
    """Show version information."""
    import sys
    try:
        from importlib.metadata import version
        pkg_version = version("pdfpeek")
    except Exception:
        pkg_version = "0.1.0 (dev)"

    click.echo(f"pdfpeek {pkg_version}")

    if verbose:
        click.echo(f"Python {sys.version.split()[0]}")

        # Show key dependencies
        deps = ["pymupdf", "torch", "pillow", "click", "numpy"]
        click.echo("\nKey dependencies:")
        for dep in deps:
            try:
                from importlib.metadata import version as get_version
                v = get_version(dep)
                click.echo(f"  {dep:<15} {v}")
            except Exception:
                click.echo(f"  {dep:<15} not installed")
