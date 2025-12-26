# src/test_chunking.py
from pypdf import PdfReader

from config import DEFAULT_PDF_PATH
from chunk_text import chunk_pages


def extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def main():
    pages = extract_pages(DEFAULT_PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"PDF: {DEFAULT_PDF_PATH}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    if chunks:
        print("\n--- Chunk 1 meta ---")
        print(chunks[0]["meta"])
        print("\n--- Chunk 1 preview ---")
        print(chunks[0]["text"][:500])

        if len(chunks) > 1:
            print("\n--- Chunk 2 meta ---")
            print(chunks[1]["meta"])
            print("\n--- Chunk 2 preview ---")
            print(chunks[1]["text"][:500])


if __name__ == "__main__":
    main()
