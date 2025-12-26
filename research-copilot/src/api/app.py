# src/api/app.py
from __future__ import annotations

import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.config import UPLOADS_DIR, TOP_K
from src.ingest import pdf_to_chunks
from src.vector_store import get_collection, add_chunks, query
from src.qa_ollama import answer as grounded_answer
from src.api.schemas import AskRequest, AskResponse, SourceItem
from src.ingest import pdf_to_chunks, text_to_chunks
from src.api.schemas import AskRequest, AskResponse, SourceItem, IngestTextRequest, IngestTextResponse


app = FastAPI(title="Research Copilot API", version="0.3.0")

@app.on_event("startup")
def _startup():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # Save upload
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path = UPLOADS_DIR / file.filename
    save_path.write_bytes(raw)

    # Create doc_id (stable-ish per upload content)
    doc_id = hashlib.sha1(raw).hexdigest()[:12]

    # Extract + chunk
    chunks = pdf_to_chunks(save_path)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found. (Scanned PDFs need OCR; we can add later.)",
        )

    # Store in Chroma
    _, collection = get_collection()
    n_added = add_chunks(collection, chunks, source_name=file.filename)

    return {
        "message": "ingested",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_added": n_added,
        "uploads_path": str(save_path),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    k = req.top_k if req.top_k is not None else TOP_K
    k = max(1, min(int(k), 20))  # safety cap

    _, collection = get_collection()
    hits = query(collection, q, k=k)

    if not hits:
        return AskResponse(answer="INSUFFICIENT_EVIDENCE: No relevant chunks retrieved.", sources=[])

    # Call Ollama with strict citations
    ans = grounded_answer(q, hits)

    # Return sources for UI
    sources = []
    for h in hits:
        meta = h["meta"]
        txt = h["text"]
        sources.append(
            SourceItem(
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0) or 0),
                distance=float(h["distance"]),
                chunk_preview=(txt[:240] + ("..." if len(txt) > 240 else "")),
            )
        )

    return AskResponse(answer=ans, sources=sources)



@app.post("/ingest/text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest):
    raw_text = (req.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # doc_id based on content (stable-ish)
    doc_id = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]
    source_name = (req.source_name or "pasted_text").strip() or "pasted_text"

    chunks = text_to_chunks(raw_text, source_name=source_name)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text produced no chunks.")

    _, collection = get_collection()
    n_added = add_chunks(collection, chunks, source_name=source_name)

    return IngestTextResponse(
        message="ingested",
        doc_id=doc_id,
        source_name=source_name,
        chunks_added=n_added,
    )
