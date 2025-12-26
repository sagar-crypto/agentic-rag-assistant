# src/api/schemas.py
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None  # default will be used if None


class SourceItem(BaseModel):
    source: str
    page: int
    distance: float
    chunk_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class IngestTextRequest(BaseModel):
    text: str
    source_name: Optional[str] = "pasted_text"


class IngestTextResponse(BaseModel):
    message: str
    doc_id: str
    source_name: str
    chunks_added: int