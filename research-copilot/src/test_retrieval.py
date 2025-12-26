# src/test_retrieval.py
from pypdf import PdfReader

from config import DEFAULT_PDF_PATH, TOP_K
from chunk_text import chunk_pages
from vector_store import reset_collection, get_collection, add_chunks, query


def extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def main():
    # For MVP, reset so reruns are clean
    reset_collection()
    _, collection = get_collection()

    pages = extract_pages(DEFAULT_PDF_PATH)
    chunks = chunk_pages(pages)

    n = add_chunks(collection, chunks, source_name=DEFAULT_PDF_PATH.name)
    print(f"Ingested chunks: {n}")

    # Try a couple of questions
    questions = [
        "What is this document about?",
        "Which department is mentioned?",
        "What website is included?",
    ]

    for q in questions:
        hits = query(collection, q, k=TOP_K)
        print("\n====================")
        print("Q:", q)
        print("Top hit:", f"{hits[0]['meta']['source']} page {hits[0]['meta']['page']} dist={hits[0]['distance']:.4f}")
        print("Snippet:", hits[0]["text"][:220], "...")


if __name__ == "__main__":
    main()
