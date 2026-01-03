# ui/app.py
import json
from typing import Dict, Any, List

import pandas as pd
import requests
import streamlit as st
import uuid

API_BASE = "http://127.0.0.1:8000"


# -----------------------------
# Helpers
# -----------------------------
def ask_stream(api_base: str, payload: dict):
    """
    Yields events from /ask/stream (NDJSON):
      {"type":"token","data":"..."}
      {"type":"sources","data":[...]}
      {"type":"done"}
    """
    with requests.post(
        f"{api_base}/ask/stream",
        json=payload,
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)


def get_active_chat():
    cid = st.session_state["active_conversation"]
    return st.session_state["conversations"][cid]["chat"]


def render_chat():
    for m in get_active_chat():
        role = m.get("role", "user")
        content = m.get("content", "")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)


def render_sources(sources: List[Dict[str, Any]]):
    """Render sources in a compact, readable way."""
    if not sources:
        st.info("No sources returned.")
        return

    # Compact table first (scan-friendly)
    rows = []
    for i, s in enumerate(sources, start=1):
        rows.append(
            {
                "#": i,
                "source": s.get("source", "unknown"),
                "page": s.get("page", 0),
                "distance": float(s.get("distance", 0.0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("distance", ascending=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption("Tip: lower distance generally means the chunk is more relevant.")

    # Details below
    for i, s in enumerate(sources, start=1):
        src = s.get("source", "unknown")
        page = s.get("page", 0)
        dist = float(s.get("distance", 0.0))
        preview = s.get("chunk_preview", "")

        st.markdown(f"**[{i}] {src} — page {page} — dist {dist:.4f}**")
        st.write(preview)
        st.divider()


# -----------------------------
# Session state init
# -----------------------------
if "conversations" not in st.session_state:
    # conversations: {conv_id: {"title": str, "chat": list[{"role","content"}]}}
    first_id = uuid.uuid4().hex[:8]
    st.session_state["conversations"] = {
        first_id: {"title": "Conversation 1", "chat": []}
    }
    st.session_state["active_conversation"] = first_id

if "active_conversation" not in st.session_state:
    # fallback safety
    st.session_state["active_conversation"] = next(iter(st.session_state["conversations"].keys()))  # list of {"role": "user"/"assistant", "content": str}

if "documents" not in st.session_state:
    st.session_state["documents"] = []  # list of {"source": str, "chunks": int}


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Research Copilot", layout="wide")
st.title("📄 Research Copilot (RAG)")
st.caption("Upload PDFs or paste text, then chat. Answers are grounded in retrieved evidence with citations.")


# -----------------------------
# Sidebar: backend + docs + ingest
# -----------------------------
with st.sidebar:
    st.header("Backend")

    # Health check
    if st.button("✅ Check API health", width='stretch'):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            r.raise_for_status()
            st.success(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.write("API Base:", API_BASE)
    st.divider()

    # Documents manager
    st.subheader("📚 Documents in Vector DB")

    c1, c2 = st.columns(2)
    with c1:
        refresh = st.button("🔄 Refresh", width='stretch')
    with c2:
        wipe = st.button("🧹 Reset all", width='stretch')

    if refresh:
        try:
            r = requests.get(f"{API_BASE}/documents", timeout=30)
            r.raise_for_status()
            st.session_state["documents"] = r.json().get("documents", [])
        except Exception as e:
            st.error(f"Failed to fetch documents: {e}")

    if wipe:
        try:
            r = requests.post(f"{API_BASE}/documents/reset", timeout=60)
            r.raise_for_status()
            st.session_state["documents"] = []
            st.success("Vector DB reset.")
        except Exception as e:
            st.error(f"Reset failed: {e}")

    docs = st.session_state.get("documents", [])

    if docs:
        df = pd.DataFrame(docs).sort_values("chunks", ascending=False)
        st.dataframe(df, width='stretch', hide_index=True)

        options = [d["source"] for d in docs]
        selected = st.selectbox("Select a source to delete", options)

        if st.button("🗑️ Delete selected", type="primary", width='stretch'):
            try:
                r = requests.delete(f"{API_BASE}/documents/{selected}", timeout=30)
                r.raise_for_status()
                st.success(f"Deleted: {selected}")

                # refresh immediately
                r2 = requests.get(f"{API_BASE}/documents", timeout=30)
                r2.raise_for_status()
                st.session_state["documents"] = r2.json().get("documents", [])
            except Exception as e:
                st.error(f"Delete failed: {e}")
    else:
        st.info("No documents listed yet. Click Refresh after ingesting.")

    st.divider()

    # Ingest PDF
    st.subheader("⬆️ Ingest PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_uploader")

    if uploaded is not None:
        if st.button("Ingest PDF", width='stretch'):
            with st.spinner("Uploading and ingesting..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                    r = requests.post(f"{API_BASE}/ingest/pdf", files=files, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Ingest failed ({r.status_code}): {r.text}")
                    else:
                        data = r.json()
                        st.success(f"Ingested {data.get('filename')} (chunks: {data.get('chunks_added')})")
                        # auto refresh docs
                        try:
                            r2 = requests.get(f"{API_BASE}/documents", timeout=30)
                            r2.raise_for_status()
                            st.session_state["documents"] = r2.json().get("documents", [])
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Error: {e}")

    # Ingest Text
    st.subheader("📝 Ingest Text")
    source_name = st.text_input("Source name", value="pasted_text", key="source_name")
    pasted_text = st.text_area("Paste text", height=140, key="pasted_text")

    if st.button("Ingest Text", width='stretch'):
        if not pasted_text.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Ingesting text..."):
                try:
                    payload = {"text": pasted_text, "source_name": source_name.strip() or "pasted_text"}
                    r = requests.post(f"{API_BASE}/ingest/text", json=payload, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Ingest failed ({r.status_code}): {r.text}")
                    else:
                        data = r.json()
                        st.success(f"Ingested '{data.get('source_name')}' (chunks: {data.get('chunks_added')})")
                        # auto refresh docs
                        try:
                            r2 = requests.get(f"{API_BASE}/documents", timeout=30)
                            r2.raise_for_status()
                            st.session_state["documents"] = r2.json().get("documents", [])
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Error: {e}")


# -----------------------------
# Main: Chat UI
# -----------------------------
st.subheader("💬 Chat with your documents")
top_bar = st.columns([2, 1])
with top_bar[0]:
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
    )
with top_bar[1]:
    if st.button("🧹 Clear chat", use_container_width=True):
        get_active_chat().clear()
        st.session_state.pop("last_answer", None)
        st.rerun()

# Conversation controls
conv_ids = list(st.session_state["conversations"].keys())
conv_titles = [st.session_state["conversations"][cid]["title"] for cid in conv_ids]

cA, cB, cC = st.columns([3, 1, 1])

with cA:
    selected_title = st.selectbox(
        "Conversation",
        conv_titles,
        index=conv_ids.index(st.session_state["active_conversation"]),
    )
    # map title back to id
    st.session_state["active_conversation"] = conv_ids[conv_titles.index(selected_title)]

with cB:
    if st.button("➕ New", use_container_width=True):
        new_id = uuid.uuid4().hex[:8]
        n = len(st.session_state["conversations"]) + 1
        st.session_state["conversations"][new_id] = {"title": f"Conversation {n}", "chat": []}
        st.session_state["active_conversation"] = new_id
        st.rerun()

with cC:
    if st.button("🧹 Clear", use_container_width=True):
        get_active_chat().clear()
        st.session_state.pop("last_answer", None)
        st.rerun()
# Render existing conversation
render_chat()

# Chat input
user_msg = st.chat_input("Ask a question about your documents...")

if user_msg:
    user_msg = user_msg.strip()
    if user_msg:
        # Add user message
        get_active_chat().append({"role": "user", "content": user_msg})

        # Build payload with history
        payload = {
            "question": user_msg,
            "top_k": int(top_k),
            "chat_history": get_active_chat()[-12:],  # last N messages
        }

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        # Stream assistant answer
        with st.chat_message("assistant"):
            answer_box = st.empty()
            with st.spinner("Streaming answer..."):
                try:
                    for evt in ask_stream(API_BASE, payload):
                        t = evt.get("type")
                        if t == "token":
                            answer_text += evt.get("data", "")
                            answer_box.markdown(answer_text)
                        elif t == "sources":
                            sources = evt.get("data", []) or []
                        elif t == "done":
                            break
                except Exception as e:
                    st.error(f"Streaming error: {e}")

        # Store assistant message
        st.session_state["chat"].append({"role": "assistant", "content": answer_text})
        st.session_state["last_answer"] = {"answer": answer_text, "sources": sources}

        # Render sources below the streamed answer
        with st.expander("📌 Sources used"):
            render_sources(sources)
