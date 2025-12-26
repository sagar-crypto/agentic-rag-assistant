from typing import List, Dict
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Chunk each page into overlapping character windows.
    Keeps page number so we can cite sources later.
    """
    chunks = []

    for p in pages:
        page_num = p["page"]
        text = p["text"]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "meta": {
                            "page": page_num,
                            "start": start,
                            "end": end,
                        },
                    }
                )

            if end == len(text):
                break

            start = max(0, end - overlap)

    return chunks