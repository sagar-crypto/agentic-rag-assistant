# ui/app.py
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

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

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving evidence and generating answer..."):
                try:
                    payload = {"question": question.strip(), "top_k": int(top_k)}
                    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Ask failed ({r.status_code}): {r.text}")
                    else:
                        st.session_state["last_answer"] = r.json()
                except Exception as e:
                    st.error(f"Error: {e}")

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
