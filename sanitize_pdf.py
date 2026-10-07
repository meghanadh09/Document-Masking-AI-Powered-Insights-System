"""
sanitize_pdf.py
Sanitize PDF files by redacting sensitive information using spaCy NER and regex patterns.
Works with both text-based and image-based (OCR) PDFs.
"""

import re
import spacy
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Install with:\n"
        "  python -m spacy download en_core_web_sm"
    )

# Regex patterns for sensitive information
REGEX_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL]"),
    (re.compile(r"\+91\s?\d{10}"), "[PHONE]"),
    (re.compile(r"\b\d{10}\b"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
]

# Entity types to redact
SENSITIVE_ENTITIES = {"PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY"}


def redact_with_ocr(page, pix):
    """
    Use OCR to detect and redact sensitive text with precise locations.

    Args:
        page: PyMuPDF page object
        pix: Pixmap of the page
    """
    try:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Scale factors (OCR coordinates to PDF coordinates)
        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            # Skip low confidence or empty text
            if conf < 30 or not text:
                continue

            # Get bounding box coordinates
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            # Convert to PDF coordinates
            pdf_rect = fitz.Rect(
                x * scale_x,
                y * scale_y,
                (x + w) * scale_x,
                (y + h) * scale_y
            )

            should_redact = False
            replacement_text = "[MASKED]"

            # Check regex patterns
            for pattern, replacement in REGEX_PATTERNS:
                if pattern.search(text):
                    should_redact = True
                    replacement_text = replacement
                    break

            # Check spaCy entities if not already matched
            if not should_redact:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in SENSITIVE_ENTITIES:
                        should_redact = True
                        replacement_text = f"[{ent.label_}]"
                        break

            # Apply redaction
            if should_redact:
                page.add_redact_annot(pdf_rect, text=replacement_text, fill=(0, 0, 0))

    except Exception as e:
        print(f"Warning: OCR processing failed: {e}")


def sanitize_pdf(input_path: str, output_path: str):
    """
    Sanitize a PDF by redacting sensitive fields/entities in-place.
    Works on both text-based PDFs and scanned/image-based PDFs (via OCR).

    Args:
        input_path: Path to input PDF file
        output_path: Path to save sanitized PDF
    """
    doc = fitz.open(input_path)

    for page_num, page in enumerate(doc):
        # Extract text layer
        text = page.get_text("text") or ""

        has_text_layer = bool(text.strip())

        if has_text_layer:
            # Process text-based PDF

            # Regex redaction on text layer
            for pattern, replacement in REGEX_PATTERNS:
                for match in pattern.finditer(text):
                    areas = page.search_for(match.group())
                    for inst in areas:
                        page.add_redact_annot(inst, text=replacement, fill=(0, 0, 0))

            # spaCy NER redaction on text layer
            spacy_doc = nlp(text)
            for ent in spacy_doc.ents:
                if ent.label_ in SENSITIVE_ENTITIES:
                    areas = page.search_for(ent.text)
                    for inst in areas:
                        page.add_redact_annot(
                            inst, text=f"[{ent.label_}]", fill=(0, 0, 0)
                        )
        else:
            # Process image-based PDF with OCR
            print(f"Page {page_num + 1}: Using OCR (no text layer detected)")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution for better OCR
            redact_with_ocr(page, pix)

        # Apply redactions for this page
        page.apply_redactions()

    # Save sanitized PDF
    doc.save(output_path, deflate=True)
    doc.close()
    print(f"✓ PDF sanitized: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python sanitize_pdf.py <input_pdf> <output_pdf>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Sanitizing {input_file}...")
    sanitize_pdf(input_file, output_file)