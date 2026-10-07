"""
sanitize_xlsx.py
Sanitize Excel files by masking sensitive information using spaCy NER and regex patterns.
"""

import re
import openpyxl
import spacy
from pathlib import Path

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Install with:\n"
        "  python -m spacy download en_core_web_sm"
    )

# Regex patterns for PII
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b\d{10}\b",
    "phone_formatted": r"\b\d{3}[-\s]\d{3}[-\s]\d{4}\b",
    "phone_india": r"\+91\s?\d{10}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    # Pattern for full names (capitalized first and last name)
    "full_name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
}

# Entity types to mask
SENSITIVE_ENTITIES = {"PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY"}


def mask_regex_pii(text: str) -> str:
    """Mask PII patterns using regex."""
    if not text:
        return text

    for name, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, "[MASKED]", text, flags=re.IGNORECASE)

    return text


def mask_spacy_entities(text: str) -> str:
    """Mask named entities detected by spaCy."""
    if not text:
        return text

    doc = nlp(text)

    # Sort entities by position in reverse to avoid offset issues
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

    for ent in entities:
        if ent.label_ in SENSITIVE_ENTITIES:
            text = text[:ent.start_char] + "[MASKED]" + text[ent.end_char:]

    return text


def contains_likely_name(text: str) -> bool:
    """
    Check if text likely contains a person's name using heuristics.
    Returns True if it matches common name patterns.
    """
    if not text or len(text) < 3:
        return False

    # Pattern: Two capitalized words (likely First Last name)
    name_pattern = r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b'
    if re.search(name_pattern, text):
        return True

    # Pattern: Three capitalized words (likely First Middle Last)
    full_name_pattern = r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b'
    if re.search(full_name_pattern, text):
        return True

    return False


def mask_text(text: str) -> str:
    """Full masking pipeline: regex -> spaCy NER -> name heuristics"""
    if not text:
        return text

    original = text

    # First mask regex patterns (including the full_name pattern)
    masked = mask_regex_pii(text)

    # If already masked by regex, return
    if masked != original:
        return masked

    # Then mask named entities with spaCy
    masked = mask_spacy_entities(masked)

    # If already masked by spaCy, return
    if masked != original:
        return masked

    # Finally, check for name-like patterns as fallback
    if contains_likely_name(masked):
        # Mask the entire cell if it looks like it contains a name
        return "[MASKED]"

    return masked


def sanitize_xlsx(input_path: str, output_path: str):
    """
    Sanitize Excel file by masking sensitive information in all sheets.

    Args:
        input_path: Path to input Excel file
        output_path: Path to save sanitized Excel file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    wb = openpyxl.load_workbook(input_path)

    masked_count = 0
    total_cells = 0

    for sheetname in wb.sheetnames:
        ws = wb[sheetname]
        for row in ws.iter_rows():
            for cell in row:
                if cell.value and isinstance(cell.value, str):
                    total_cells += 1
                    original = cell.value
                    masked = mask_text(original)
                    if masked != original:
                        cell.value = masked
                        masked_count += 1

    # Save with error handling
    try:
        wb.save(output_path)
        print(f"✓ Excel sanitized: {output_path}")
        print(f"  Masked {masked_count} cells out of {total_cells} text cells")
    except PermissionError:
        alt = output_path.with_name(output_path.stem + "_sanitized" + output_path.suffix)
        wb.save(alt)
        print(f"⚠ Could not write to {output_path}; saved as {alt}")
        print(f"  Masked {masked_count} cells out of {total_cells} text cells")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python sanitize_xlsx.py <input_xlsx> <output_xlsx>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Sanitizing {input_file}...")
    sanitize_xlsx(input_file, output_file)