# src/read_pdf.py
from pathlib import Path
import sys
from pypdf import PdfReader

from config import DEFAULT_PDF_PATH


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def main():
    # Allow: python src/read_pdf.py data/another.pdf
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF_PATH

    # If user passed a relative path, make it relative to current working directory (fine for now).
    # If you want it relative to project root, tell me and I’ll adjust.
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_pages(pdf_path)
    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(pages)}")

    if pages:
        print("\n--- Page 1 preview ---")
        print(pages[0]["text"][:1000] or "[No extractable text on page 1]")


if __name__ == "__main__":
    main()
