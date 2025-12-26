# src/test_rag_answer.py
from pypdf import PdfReader

from config import DEFAULT_PDF_PATH, TOP_K
from chunk_text import chunk_pages
from vector_store import reset_collection, get_collection, add_chunks, query
from qa_ollama import answer


def extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def main():
    reset_collection()
    _, collection = get_collection()

    pages = extract_pages(DEFAULT_PDF_PATH)
    chunks = chunk_pages(pages)

    n = add_chunks(collection, chunks, source_name=DEFAULT_PDF_PATH.name)
    print(f"Ingested chunks: {n}")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        hits = query(collection, q, k=TOP_K)

        if not hits:
            print("No evidence retrieved.")
            continue

        print("\n--- Retrieved evidence blocks ---")
        for i, h in enumerate(hits, start=1):
            src = h["meta"].get("source")
            page = h["meta"].get("page")
            dist = h["distance"]
            print(f"\n[{i}] {src} | page {page} | dist={dist:.4f}")
            print(h["text"])

        out = answer(q, hits)

        print("\n====================")
        print("ANSWER (GROUNDED)")
        print("====================")
        print(out)


if __name__ == "__main__":
    main()
