# src/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

from src.chunk_text import chunk_pages


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append({"page": i + 1, "text": text})
    return pages


def pdf_to_chunks(pdf_path: Path) -> List[Dict]:
    pages = extract_pages_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    return chunks


def text_to_chunks(text: str, source_name: str = "pasted_text") -> List[Dict]:
    """
    Convert raw text into the same chunk format we use for PDFs.
    We treat it as "page 1" for citations.
    """
    cleaned = " ".join((text or "").split())
    pages = [{"page": 1, "text": cleaned}]
    return chunk_pages(pages)

