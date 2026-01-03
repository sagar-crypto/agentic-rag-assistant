# src/api/schemas.py
from __future__ import annotations

from typing import Optional, List, Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    chat_history: List[ChatMessage] = []


class SourceItem(BaseModel):
    source: str
    page: int
    distance: float
    chunk_preview: str
    chunk_id: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


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


