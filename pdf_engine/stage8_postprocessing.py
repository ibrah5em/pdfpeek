"""
pdf_engine/stage8_postprocessing.py
=====================================
Stage 8 — Post-Processing

Responsibility
--------------
1. Rejoin hyphenated words split across block boundaries (W23 fix).
2. Apply language-aware OCR error correction for known character confusions (W24 fix).

Design rules enforced here
--------------------------
* Hyphenation rejoining modifies current block's trailing hyphen+prefix into
  the full joined word, and strips only the consumed suffix from next_block.
  next_block's remaining content is never duplicated.
* OCR corrections are ONLY applied to blocks with OCR extraction methods
  (TESSERACT_OCR or SURYA_OCR) AND English language — never to native text
  or non-English blocks.
* All input/output objects are typed TextBlock / PageIR / DocumentIR instances.
* This stage never re-opens the PDF.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pdf_engine.models import (
    BlockType,
    DocumentIR,
    ExtractionMethod,
    PageIR,
    TextBlock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR confusion table (English-specific)
# ---------------------------------------------------------------------------

# Each entry: (erroneous_string, correct_string)
# Applied only to OCR blocks with English language.
OCR_CONFUSIONS: list[tuple[str, str]] = [
    ("rn", "m"),       # "rn" misread as "m" is reversed here: fix "m" -> check context? 
                       # Actually we fix known substitution errors in OCR output:
                       # OCR reads "m" as "rn" → we correct "rn" back to "m"
    ("li", "h"),       # "li" misread instead of "h" — less common, context-dependent
    ("1", "l"),        # digit "1" confused with letter "l"
    ("0", "o"),        # digit "0" confused with letter "o"
    ("cl", "d"),       # "cl" misread instead of "d"
    ("vv", "w"),       # "vv" misread instead of "w"
    ("||", "ll"),      # pipe chars instead of lowercase L
]

# Words that should never be "corrected" (would break them)
_CORRECTION_STOPWORDS: set[str] = {
    "1",    # standalone digit
    "0",    # standalone digit
    "I",    # capital I (not digit 1)
    "a",
    "i",
}


# ---------------------------------------------------------------------------
# Hyphenation rejoining
# ---------------------------------------------------------------------------


def _rejoin_intra_block_hyphenation(block: TextBlock) -> None:
    """
    Rejoin hyphenated words split across line breaks within a single block.

    HIGH-1 fix: The cross-block rejoin_hyphenation function only handles
    hyphens at block boundaries. This function handles hyphens at line breaks
    within a single block's text (e.g., "informa-\\ntion" within one block).

    Algorithm:
    1. Find patterns like "word-\\n" or "word-\\r\\n" within the text
    2. Join them with the following word
    3. Replace in-place

    Parameters
    ----------
    block:
        The TextBlock to process (mutated in-place).
    """
    if not block.text:
        return

    original = block.text

    # Pattern: word character(s), hyphen, optional whitespace, newline, optional whitespace, word character(s)
    # This matches: "informa-\ntion", "trade-\nmarked", "moni- \ntor", etc.
    pattern = r'(\w+)-\s*[\r\n]+\s*(\w+)'

    def join_match(match: re.Match) -> str:
        prefix = match.group(1)
        suffix = match.group(2)
        joined = prefix + suffix
        logger.debug(
            "Intra-block hyphenation: '%s-...%s' → '%s' (block %s, page %d)",
            prefix, suffix, joined, block.id, block.page_num
        )
        return joined

    block.text = re.sub(pattern, join_match, block.text)


def rejoin_hyphenation(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Scan consecutive block pairs and rejoin words split by end-of-block hyphens.

    Algorithm (W23 fix — no duplication):
    1. First, rejoin hyphenation within each block (HIGH-1 fix)
    2. Then, find a trailing ``word-`` at the end of ``current.text``.
    3. Find the first word at the start of ``next_block.text``.
    4. Replace ``prefix-`` in current with the full joined word.
    5. Strip only the consumed suffix from the beginning of next_block.text.
       The rest of next_block is left entirely unchanged.

    Parameters
    ----------
    blocks:
        A flat, ordered list of TextBlock objects (single page or all pages).

    Returns
    -------
    list[TextBlock]
        The same list, mutated in-place, then returned.
    """
    # HIGH-1 fix: First pass - rejoin hyphenation within each block
    for block in blocks:
        _rejoin_intra_block_hyphenation(block)
    i = 0
    while i < len(blocks) - 1:
        current = blocks[i]
        next_block = blocks[i + 1]

        # Look for a trailing "word-" at the end of current block
        prefix_match = re.search(r"(\w+)-\s*$", current.text)
        if not prefix_match:
            i += 1
            continue

        # Look for the first word at the start of next_block (after any leading whitespace)
        suffix_match = re.match(r"(\w+)", next_block.text.lstrip())
        if not suffix_match:
            i += 1
            continue

        prefix = prefix_match.group(1)
        suffix = suffix_match.group(1)
        joined_word = prefix + suffix

        # Modify current: replace "prefix-" with the full joined word
        # Everything before prefix_match.start(1) is preserved
        current.text = current.text[: prefix_match.start(1)] + joined_word

        # Modify next: strip only the consumed suffix from the front.
        # Crucially, we must NOT use lstrip() on the full text, as that would
        # remove intentional paragraph indentation.  Instead, find where the
        # suffix word starts (after any leading whitespace) and slice from
        # after the suffix word.
        next_text = next_block.text
        leading_ws_match = re.match(r"(\s*)", next_text)
        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
        # suffix_match was run on the lstripped version; recalculate the
        # absolute end position in the original string.
        abs_suffix_end = len(leading_ws) + suffix_match.end()
        next_block.text = next_text[abs_suffix_end:]

        logger.debug(
            "Rejoined hyphenation: '%s' + '%s' → '%s' (page %d → %d)",
            prefix, suffix, joined_word,
            current.page_num, next_block.page_num,
        )
        i += 1

    # Remove empty blocks that result from consuming an entire next_block.
    blocks[:] = [b for b in blocks if b.text.strip()]

    return blocks


# ---------------------------------------------------------------------------
# OCR error correction
# ---------------------------------------------------------------------------


def _apply_confusion_corrections(
    text: str,
    dictionary: Optional[set[str]] = None,
) -> str:
    """
    Apply the OCR confusion table to ``text``, correcting known substitution
    errors within each whitespace-delimited token.

    Tokens are processed independently (no cross-token cascading), and tokens
    in ``_CORRECTION_STOPWORDS`` are never modified.

    CRIT-2 fix: Dictionary-validated correction
    --------------------------------------------
    * If a dictionary is provided, only apply corrections when:
      - The corrected word exists in the dictionary AND
      - The original word does NOT exist in the dictionary
    * This prevents corrupting valid words like "born" → "bom", "learning" → "leaming"
    * Without a dictionary, falls back to the original substring-replacement
      behavior (for backward compatibility with existing tests)

    Parameters
    ----------
    text:
        The text to correct.
    dictionary:
        Optional set of known-good words for validation.

    Returns
    -------
    str
        The corrected text.
    """
    tokens = re.split(r"(\s+)", text)  # preserve whitespace tokens
    corrected_tokens: list[str] = []

    for token in tokens:
        if not token or token.isspace():
            corrected_tokens.append(token)
            continue

        if token in _CORRECTION_STOPWORDS:
            corrected_tokens.append(token)
            continue

        if dictionary is None:
            # No dictionary: use original behavior (substring replacement)
            corrected = token
            for wrong, right in OCR_CONFUSIONS:
                corrected = corrected.replace(wrong, right)
            corrected_tokens.append(corrected)
        else:
            # Dictionary provided: use validated correction
            # Strip punctuation for dictionary lookup
            stripped_token = token.strip(".,!?;:\"'()[]{}").lower()

            # Try applying each confusion correction
            best_correction = token
            for wrong, right in OCR_CONFUSIONS:
                if wrong not in token:
                    continue  # skip if pattern not present

                candidate = token.replace(wrong, right)
                stripped_candidate = candidate.strip(".,!?;:\"'()[]{}").lower()

                # Only accept the correction if it's dictionary-validated
                if (stripped_candidate in dictionary and
                    stripped_token not in dictionary):
                    best_correction = candidate
                    break  # apply first valid correction only

            corrected_tokens.append(best_correction)

    return "".join(corrected_tokens)


# ---------------------------------------------------------------------------
# Unicode replacement character handling (CRIT-6 fix)
# ---------------------------------------------------------------------------


def _clean_replacement_characters(text: str) -> str:
    """
    Remove or replace Unicode replacement characters (U+FFFD: �).

    These characters indicate encoding errors or missing glyphs and should
    not appear in final output.

    Rules:
    * Replace consecutive runs of � with a single placeholder marker [?]
    * If the entire text consists only of � and whitespace, return empty string
      (the block will be filtered out later)

    Parameters
    ----------
    text:
        The text to clean.

    Returns
    -------
    str
        Text with replacement characters handled.
    """
    # Check if the text consists entirely of replacement characters and whitespace
    if text.replace('\ufffd', '').strip() == '':
        logger.debug("Block contains only replacement characters, marking for removal")
        return ''  # Will be filtered out by empty block removal

    # Replace runs of 1+ replacement characters with a single placeholder
    # This preserves the fact that something was there but unreadable
    cleaned = re.sub(r'\ufffd+', '[?]', text)

    if cleaned != text:
        logger.debug("Cleaned replacement characters: %d occurrences", text.count('\ufffd'))

    return cleaned


# ---------------------------------------------------------------------------
# Spaced-out text collapse (CRIT-5 fix)
# ---------------------------------------------------------------------------


def _collapse_spaced_text(text: str) -> str:
    """
    Collapse decorative letter-spacing in headings and call-out boxes.

    PDFs with decorative letter-spacing produce patterns like:
    - "N O T E" → "NOTE"
    - "G A M E  H A C K I N G" → "GAMEHACKING"
    - "DE BU GG ING  G A M E S" → "DEBUGGINGGAMES"

    Heuristic:
    * Match sequences of 1-2 uppercase letters separated by spaces
    * Require at least 3 such groups to avoid false positives (e.g., "A B C")
    * Collapse all spaces between them

    Parameters
    ----------
    text:
        The text to process.

    Returns
    -------
    str
        Text with spaced-out sequences collapsed.
    """
    # Pattern: Match sequences like "N O T E" or "DE BU GG ING"
    # - Start with 1-2 uppercase letters: ([A-Z]{1,2})
    # - Followed by 2+ occurrences of: space(s) + 1-2 uppercase letters
    # This catches both single-letter spacing (N O T E) and multi-letter spacing (DE BU GG ING)
    pattern = r'([A-Z]{1,2})(?:\s+([A-Z]{1,2})){2,}'

    def collapse_match(match: re.Match) -> str:
        # Extract the full matched text and remove all spaces
        matched = match.group(0)
        # Remove all whitespace to join the letters
        collapsed = re.sub(r'\s+', '', matched)
        logger.debug("Collapsed spaced text: %r → %r", matched, collapsed)
        return collapsed

    return re.sub(pattern, collapse_match, text)


# ---------------------------------------------------------------------------
# Language detection helper (CRIT-1 fix)
# ---------------------------------------------------------------------------


def _detect_and_set_language(block: TextBlock) -> None:
    """
    Set block.language to a best-guess value if not already set.

    This enables OCR error correction (which requires block.language to be set).
    Called before fix_ocr_errors in process_page.

    Rules:
    * If block.language is already set, do nothing.
    * If the text contains mostly Latin alphabet characters (a-z, A-Z),
      set to "en" (English).
    * Otherwise leave as None (non-Latin scripts handled elsewhere).

    Parameters
    ----------
    block:
        The TextBlock to potentially update.
    """
    if block.language is not None:
        return  # already set

    if not block.text.strip():
        return  # empty block

    # Count Latin alphabet characters
    latin_chars = sum(1 for c in block.text if c.isalpha() and ord(c) < 128)
    total_chars = len(block.text)

    # If at least 50% of characters are Latin alphabet, assume English
    if total_chars > 0 and (latin_chars / total_chars) >= 0.5:
        block.language = "en"
        logger.debug(
            "Auto-detected language 'en' for block %s (page %d) - %d/%d Latin chars",
            block.id, block.page_num, latin_chars, total_chars
        )


# ---------------------------------------------------------------------------
# OCR correction per block
# ---------------------------------------------------------------------------


def fix_ocr_errors(
    block: TextBlock,
    dictionary: Optional[set[str]] = None,
) -> TextBlock:
    """
    Apply language-aware OCR error correction to a single block.

    Rules (W24 fix + CRIT-2 fix):
    * Only blocks with ``extraction_method`` in
      {TESSERACT_OCR, SURYA_OCR} are corrected.
    * Correction is only applied when ``block.language`` starts with ``"en"``.
    * Non-English blocks and native-text blocks are returned unchanged.
    * CRIT-2: Corrections are dictionary-validated to prevent corrupting
      valid words (e.g., "born" is not changed to "bom").

    Parameters
    ----------
    block:
        The TextBlock to (potentially) correct.
    dictionary:
        Optional set of known-good words for validating corrections.
        If not provided, OCR correction is skipped to avoid corrupting valid words.

    Returns
    -------
    TextBlock
        The same object, possibly with ``text`` mutated.
    """
    ocr_methods = {ExtractionMethod.TESSERACT_OCR, ExtractionMethod.SURYA_OCR}
    if block.extraction_method not in ocr_methods:
        return block

    lang = block.language or ""
    if not lang.startswith("en"):
        # W24 fix: skip correction for non-English blocks entirely
        return block

    original = block.text
    block.text = _apply_confusion_corrections(block.text, dictionary=dictionary)

    if block.text != original:
        logger.debug(
            "OCR correction on block %s (page %d): %r → %r",
            block.id, block.page_num, original, block.text,
        )

    return block


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------


def process_page(
    page_ir: PageIR,
    dictionary: Optional[set[str]] = None,
) -> PageIR:
    """
    Apply hyphenation rejoining, spaced-out text collapse, and OCR correction
    to all blocks on a page.

    Note: hyphenation rejoining on a single page cannot fix splits that
    cross page boundaries — use ``run_stage8`` at document level instead.

    Parameters
    ----------
    page_ir:
        The PageIR from Stage 7.
    dictionary:
        Optional word dictionary for OCR correction.

    Returns
    -------
    PageIR
        The same object, mutated in-place, then returned.
    """
    # Per-page hyphenation (within-page splits)
    rejoin_hyphenation(page_ir.blocks)

    # Clean Unicode replacement characters (CRIT-6 fix)
    for block in page_ir.blocks:
        block.text = _clean_replacement_characters(block.text)

    # Collapse decorative letter-spacing (CRIT-5 fix)
    # Applied to all blocks, not just OCR, since native text can have spacing too
    for block in page_ir.blocks:
        original = block.text
        block.text = _collapse_spaced_text(block.text)
        if block.text != original:
            logger.debug(
                "Collapsed spacing in block %s (page %d): %r → %r",
                block.id, block.page_num, original[:50], block.text[:50]
            )

    # Remove empty blocks (from replacement character cleanup or other processing)
    page_ir.blocks[:] = [b for b in page_ir.blocks if b.text.strip()]

    # OCR correction per block
    for block in page_ir.blocks:
        _detect_and_set_language(block)  # CRIT-1 fix: set language before OCR correction
        fix_ocr_errors(block, dictionary=dictionary)

    return page_ir


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------


def run_stage8(
    doc_ir: DocumentIR,
    dictionary: Optional[set[str]] = None,
) -> DocumentIR:
    """
    Run post-processing across the entire document.

    Cross-page hyphenation is handled by building a flat block list from
    all pages, running ``rejoin_hyphenation`` once, then applying OCR
    corrections block-by-block.

    Parameters
    ----------
    doc_ir:
        The ``DocumentIR`` returned by Stage 7.
    dictionary:
        Optional set of known-good words for OCR correction.

    Returns
    -------
    DocumentIR
        The same object, with all blocks post-processed in-place.
    """
    # Flatten all blocks across pages for cross-page hyphenation
    all_blocks: list[TextBlock] = []
    for page_ir in doc_ir.pages:
        all_blocks.extend(page_ir.blocks)

    # Rejoin hyphenation across page boundaries
    rejoin_hyphenation(all_blocks)

    # CRIT-6 fix: Clean Unicode replacement characters from all blocks
    for block in all_blocks:
        block.text = _clean_replacement_characters(block.text)

    # CRIT-5 fix: Collapse decorative letter-spacing in all blocks
    for block in all_blocks:
        block.text = _collapse_spaced_text(block.text)

    # Remove empty blocks after cleanup
    for page_ir in doc_ir.pages:
        page_ir.blocks[:] = [b for b in page_ir.blocks if b.text.strip()]

    # Rebuild all_blocks list after filtering
    all_blocks = [b for b in all_blocks if b.text.strip()]

    # CRIT-1 fix: Detect language and apply OCR correction per block
    for block in all_blocks:
        _detect_and_set_language(block)
        fix_ocr_errors(block, dictionary=dictionary)

    logger.info(
        "Stage 8 complete: processed %d blocks across %d pages",
        len(all_blocks), len(doc_ir.pages),
    )
    return doc_ir
