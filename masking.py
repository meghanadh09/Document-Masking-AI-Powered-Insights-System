import re
import spacy
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Regex patterns for sensitive info (keep your existing ones)
REGEX_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL_REDACTED]"),
    (re.compile(r"\+91\s?\d{10}"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{10}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE_REDACTED]"),
]

# Entity types to redact (keep your existing ones)
SENSITIVE_ENTITIES = {"PERSON", "ORG", "GPE", "DATE", "CARDINAL"}


def sanitize_pdf(input_path: str, output_path: str):
    """Sanitize a PDF by redacting sensitive fields/entities in-place (format preserved).
    Works on both text-based PDFs and scanned/image-based PDFs (via OCR).
    """
    doc = fitz.open(input_path)

    for page in doc:
        # --- Extract text layer ---
        text = page.get_text("text") or ""

        # --- Regex redaction on text layer ---
        for pattern, replacement in REGEX_PATTERNS:
            for match in pattern.finditer(text):
                areas = page.search_for(match.group())
                for inst in areas:
                    page.add_redact_annot(inst, text=replacement, fill=(0, 0, 0))

        # --- spaCy NER redaction on text layer ---
        spacy_doc = nlp(text)
        for ent in spacy_doc.ents:
            if ent.label_ in SENSITIVE_ENTITIES:
                areas = page.search_for(ent.text)
                for inst in areas:
                    page.add_redact_annot(
                        inst, text=f"[{ent.label_}_REDACTED]", fill=(0, 0, 0)
                    )

        # --- OCR fallback if no text found ---
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)

            # Regex on OCR text
            for pattern, replacement in REGEX_PATTERNS:
                for match in pattern.finditer(ocr_text):
                    rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                    page.add_redact_annot(rect, text=replacement, fill=(0, 0, 0))

            # spaCy NER on OCR text
            spacy_doc = nlp(ocr_text)
            for ent in spacy_doc.ents:
                if ent.label_ in SENSITIVE_ENTITIES:
                    rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                    page.add_redact_annot(
                        rect, text=f"[{ent.label_}_REDACTED]", fill=(0, 0, 0)
                    )

        # --- Apply redactions for this page ---
        page.apply_redactions()

    # Save sanitized PDF
    doc.save(output_path, deflate=True)
    print(f"Sanitized PDF saved to {output_path}")


if __name__ == "__main__":
    input_file = input("Enter the path to the input PDF file: ").strip().strip('"')
    output_file = input("Enter the path for the sanitized PDF file: ").strip().strip('"')

    print(f"\nSanitizing {input_file} ...")
    sanitize_pdf(input_file, output_file)
