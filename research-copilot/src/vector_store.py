# src/vector_store.py
import hashlib
from typing import List, Dict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL


def make_id(source: str, page: int, start: int, end: int, text: str) -> str:
    key = f"{source}|{page}|{start}|{end}|{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def reset_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass


def add_chunks(collection, chunks: List[Dict], source_name: str):
    texts = []
    metadatas = []
    ids = []

    for c in chunks:
        text = c["text"]
        meta = c["meta"].copy()
        meta["source"] = source_name

        _id = make_id(
            source=source_name,
            page=meta["page"],
            start=meta["start"],
            end=meta["end"],
            text=text,
        )

        texts.append(text)
        metadatas.append(meta)
        ids.append(_id)

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return len(texts)


def query(collection, question: str, k: int = 5):
    res = collection.query(query_texts=[question], n_results=k)
    hits = []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "meta": meta, "distance": dist})

    return hits
