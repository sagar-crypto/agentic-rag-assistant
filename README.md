# Research Copilot
A Citation-Grounded RAG Chat System for PDFs and Text

Research Copilot is a Retrieval-Augmented Generation (RAG) application that allows users to chat with their documents (PDFs or pasted text) while ensuring that every answer is grounded in retrieved evidence with explicit citations.

--- 

## Key Features

- PDF & text ingestion
- Semantic retrieval with ChromaDB
- Strict citation enforcement
- Streaming answers
- Multi-turn chat memory
- Document management (list / delete / reset)
- Local-first LLM inference via Ollama

--- 

## Architecture Overview

1. Ingest documents (PDF or text)
2. Chunk and embed content
3. Store embeddings in ChromaDB
4. Retrieve Top-K relevant chunks per query
5. Generate answers strictly from evidence
6. Stream responses to the UI

--- 

## Project Structure

src/
  api/
    app.py
    schemas.py
  ingest.py
  chunk_text.py
  vector_store.py
  qa_ollama.py
  config.py
ui/
  app.py

--- 

## Requirements

- Python 3.10+
- Ollama installed and running locally
- A supported LLM model (e.g. llama3.1)

--- 

## Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

--- 

## Running the App

Backend:
uvicorn src.api.app:app --reload

UI:
streamlit run ui/app.py

--- 

## API Endpoints

GET /health  
POST /ingest/pdf  
POST /ingest/text  
POST /ask  
POST /ask/stream  
GET /documents  
DELETE /documents/{source_name}  
POST /documents/reset  

--- 

## Limitations

- No OCR for scanned PDFs
- Streaming answers cannot be citation-repaired post hoc
- Semantic-only retrieval (hybrid planned)

--- 

## License

MIT
