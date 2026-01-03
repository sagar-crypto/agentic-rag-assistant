# src/api/app.py
from __future__ import annotations

import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
import json
from collections import Counter
from fastapi.responses import StreamingResponse
from src.vector_store import reset_collection
from src.config import UPLOADS_DIR, TOP_K
from src.vector_store import get_collection, add_chunks, query
from src.qa_ollama import answer as grounded_answer, answer_stream
from src.ingest import pdf_to_chunks, text_to_chunks
from src.api.schemas import AskRequest, AskResponse, IngestTextRequest, IngestTextResponse


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
    history = [m.model_dump() for m in (req.chat_history or [])]
    ans = grounded_answer(q, hits, chat_history=history)

    # Return sources for UI
    sources = []
    for h in hits:
        meta = h["meta"] or {}
        txt = h["text"] or ""
        sources.append(
            {
                "source": str(meta.get("source", "unknown")),
                "page": int(meta.get("page", 0) or 0),
                "distance": float(h.get("distance", 0.0)),
                "chunk_preview": (txt[:240] + ("..." if len(txt) > 240 else "")),
            }
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




@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    k = req.top_k if req.top_k is not None else TOP_K
    k = max(1, min(int(k), 20))

    _, collection = get_collection()
    hits = query(collection, q, k=k)

    # Build sources payload once (same as /ask, plus optional chunk_id/start/end)
    sources_payload = []
    for h in hits:
        meta = h.get("meta") or {}
        txt = h.get("text") or ""
        sources_payload.append(
            {
                "source": str(meta.get("source", "unknown")),
                "page": int(meta.get("page", 0) or 0),
                "distance": float(h.get("distance", 0.0)),
                "chunk_id": h.get("id"),
                "start": meta.get("start"),
                "end": meta.get("end"),
                "chunk_preview": (txt[:240] + ("..." if len(txt) > 240 else "")),
            }
        )

    def gen():
        if not hits:
            yield json.dumps({"type": "token", "data": "INSUFFICIENT_EVIDENCE: No relevant chunks retrieved."}) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # stream tokens
        history = [m.model_dump() for m in (req.chat_history or [])]
        for tok in answer_stream(q, hits, chat_history=history):
            yield json.dumps({"type": "token", "data": tok}) + "\n"

        # send sources at end
        yield json.dumps({"type": "sources", "data": sources_payload}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/documents")
def list_documents():
    _, collection = get_collection()

    # Get all metadatas (may be large; okay for now)
    data = collection.get(include=["metadatas"])
    metas = data.get("metadatas", []) or []

    sources = [m.get("source", "unknown") for m in metas if isinstance(m, dict)]
    counts = Counter(sources)

    docs = [{"source": s, "chunks": int(c)} for s, c in sorted(counts.items())]
    return {"documents": docs, "total_sources": len(docs)}


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    source_name = (source_name or "").strip()
    if not source_name:
        raise HTTPException(status_code=400, detail="source_name cannot be empty")

    _, collection = get_collection()

    # Chroma supports where filters on metadata
    collection.delete(where={"source": source_name})

    return {"message": "deleted", "source": source_name}


@app.post("/documents/reset")
def reset_documents():
    reset_collection()
    return {"message": "reset_done"}