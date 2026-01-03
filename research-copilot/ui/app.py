# ui/app.py
import streamlit as st
import requests

import json
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

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

if "chat" not in st.session_state:
    st.session_state["chat"] = []


st.set_page_config(page_title="Research Copilot", layout="wide")

st.title("📄 Research Copilot (PDF RAG)")
st.caption("Upload a PDF, then ask questions. Answers are grounded in retrieved evidence with citations.")

# -------- Sidebar: API status --------
with st.sidebar:
    st.header("Backend")
    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            st.success(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.divider()
    st.write("API Base:", API_BASE)

    st.subheader("📚 Documents in Vector DB")

    c1, c2 = st.columns([1, 1])
    with c1:
        refresh = st.button("🔄 Refresh", use_container_width=True)
    with c2:
        wipe = st.button("🧹 Reset all", use_container_width=True)  # only if you add /documents/reset

    if refresh or ("documents" not in st.session_state):
        try:
            r = requests.get(f"{API_BASE}/documents", timeout=30)
            r.raise_for_status()
            st.session_state["documents"] = r.json().get("documents", [])
        except Exception as e:
            st.error(f"Failed to fetch documents: {e}")

    docs = st.session_state.get("documents", [])

    if docs:
        df = pd.DataFrame(docs).sort_values("chunks", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.caption("Delete a document (removes all its chunks from Chroma).")

        options = [d["source"] for d in docs]
        selected = st.selectbox("Select source", options)

        del_clicked = st.button("🗑️ Delete selected", type="primary", use_container_width=True)
        if del_clicked:
            try:
                r = requests.delete(f"{API_BASE}/documents/{selected}", timeout=30)
                r.raise_for_status()
                st.success(f"Deleted: {selected}")
                # refresh list immediately
                r2 = requests.get(f"{API_BASE}/documents", timeout=30)
                r2.raise_for_status()
                st.session_state["documents"] = r2.json().get("documents", [])
            except Exception as e:
                st.error(f"Delete failed: {e}")
    else:
        st.info("No documents yet. Ingest a PDF/text and refresh.")

st.subheader("Chat")
for m in st.session_state["chat"]:
    if m["role"] == "user":
        st.markdown(f"**You:** {m['content']}")
    else:
        st.markdown(f"**Assistant:** {m['content']}")

# -------- Upload section --------
st.subheader("1) Upload a PDF")
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Ingest PDF"):
            with st.spinner("Uploading and ingesting..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                    r = requests.post(f"{API_BASE}/ingest/pdf", files=files, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Upload failed ({r.status_code}): {r.text}")
                    else:
                        data = r.json()
                        st.session_state["last_ingest"] = data
                        st.success(f"Ingested {data.get('filename')} (chunks: {data.get('chunks_added')})")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "last_ingest" in st.session_state:
            st.write("Last ingest:")
            st.json(st.session_state["last_ingest"])
        else:
            st.info("Upload a PDF and click **Ingest PDF**.")

st.subheader("1b) Paste text (optional)")
pasted_text = st.text_area(
    "Paste text here (e.g., abstract, notes, or an entire document)",
    height=180,
    placeholder="Paste your text here...",
)

source_name = st.text_input("Source name (optional)", value="pasted_text")

if st.button("Ingest Text"):
    if not pasted_text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Ingesting text..."):
            try:
                payload = {"text": pasted_text, "source_name": source_name.strip() or "pasted_text"}
                r = requests.post(f"{API_BASE}/ingest/text", json=payload, timeout=300)
                if r.status_code != 200:
                    st.error(f"Ingest text failed ({r.status_code}): {r.text}")
                else:
                    data = r.json()
                    st.session_state["last_ingest_text"] = data
                    st.success(f"Ingested text source '{data.get('source_name')}' (chunks: {data.get('chunks_added')})")
            except Exception as e:
                st.error(f"Error: {e}")

if "last_ingest_text" in st.session_state:
    st.write("Last text ingest:")
    st.json(st.session_state["last_ingest_text"])

st.divider()

# -------- Ask section --------
st.subheader("2) Ask a question")
question = st.text_input("Your question", placeholder="e.g., What is this document about?")

ask_col1, ask_col2 = st.columns([1, 2])

with ask_col1:
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5)

    b1, b2 = st.columns(2)

    # Non-streaming (existing)
    with b1:
        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                # 1️⃣ ADD USER MESSAGE TO CHAT
                st.session_state["chat"].append(
                    {"role": "user", "content": question.strip()}
                )

                # 2️⃣ BUILD PAYLOAD WITH CHAT HISTORY
                payload = {
                    "question": question.strip(),
                    "top_k": int(top_k),
                    "chat_history": st.session_state["chat"][-12:],  # last 12 messages
                }

                with st.spinner("Retrieving evidence and generating answer..."):
                    try:
                        r = requests.post(f"{API_BASE}/ask", json=payload, timeout=300)
                        r.raise_for_status()
                        data = r.json()
                        st.session_state["last_answer"] = data

                        # 3️⃣ ADD ASSISTANT MESSAGE TO CHAT
                        st.session_state["chat"].append(
                            {"role": "assistant", "content": data.get("answer", "")}
                        )

                    except Exception as e:
                        st.error(f"Error: {e}")
    # Streaming
    with b2:
        if st.button("Ask (stream)"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                # 1️⃣ ADD USER MESSAGE
                st.session_state["chat"].append(
                    {"role": "user", "content": question.strip()}
                )

                payload = {
                    "question": question.strip(),
                    "top_k": int(top_k),
                    "chat_history": st.session_state["chat"][-12:],  # last 12
                }

                answer_text = ""
                sources = []

                answer_box = st.empty()
                sources_box = st.empty()

                with st.spinner("Streaming answer..."):
                    try:
                        for evt in ask_stream(API_BASE, payload):
                            if evt["type"] == "token":
                                answer_text += evt["data"]
                                answer_box.markdown("### ✅ Answer")
                                answer_box.write(answer_text)

                            elif evt["type"] == "sources":
                                sources = evt["data"]

                            elif evt["type"] == "done":
                                break

                        # 2️⃣ ADD ASSISTANT MESSAGE AFTER STREAM ENDS
                        st.session_state["chat"].append(
                            {"role": "assistant", "content": answer_text}
                        )

                        st.session_state["last_answer"] = {
                            "answer": answer_text,
                            "sources": sources,
                        }

                    except Exception as e:
                        st.error(f"Streaming error: {e}")

with ask_col2:
    if "last_answer" in st.session_state:
        data = st.session_state["last_answer"]
        st.markdown("### ✅ Answer")
        st.write(data.get("answer", ""))

        st.markdown("### 📌 Sources")
        sources = data.get("sources", [])
        if not sources:
            st.info("No sources returned.")
        else:
            for i, s in enumerate(sources, start=1):
                with st.expander(f"[{i}] {s['source']} — page {s['page']} — dist {s['distance']:.4f}"):
                    st.write(s["chunk_preview"])
    else:
        st.info("Ask a question to see the answer and sources.")

st.subheader("Chat History")

for m in st.session_state["chat"]:
    if m["role"] == "user":
        st.markdown(f"**You:** {m['content']}")
    else:
        st.markdown(f"**Assistant:** {m['content']}")
