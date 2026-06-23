"""
frontend/streamlit_app.py
Streamlit frontend for the Enterprise RAG Engine.

Run:
  streamlit run frontend/streamlit_app.py

Connects to the FastAPI backend at localhost:8000 (or BACKEND_URL env var).
"""
import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Engine")
    st.caption("Document Q&A with Evaluation")

    # ── Connection status
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"✓ Backend connected")
        st.caption(f"Model: {health.get('llm_provider', '?')}/{health.get('llm_model', '?')}")
        chunks = health.get("vector_store", {}).get("total_chunks", 0)
        st.metric("Indexed Chunks", chunks)
    except Exception:
        st.error("✗ Backend unreachable — start the API server")

    st.divider()

    # ── Document upload
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest Documents", type="primary"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}..."):
                resp = requests.post(
                    f"{BACKEND_URL}/ingest/file",
                    files={"file": (f.name, f.getvalue(), f.type)},
                )
                if resp.ok:
                    result = resp.json()["result"]
                    if result["status"] == "ingested":
                        st.success(f"✓ {f.name} — {result['chunks']} chunks")
                    else:
                        st.info(f"↩ {f.name} already ingested")
                else:
                    st.error(f"✗ {f.name}: {resp.json().get('detail', 'error')}")
        st.rerun()

    st.divider()

    # ── Navigation
    page = st.radio(
        "Navigate",
        ["💬 Ask Questions", "📊 Evaluation Dashboard"],
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("🗑 Clear Vector Store", type="secondary", use_container_width=True):
        if st.session_state.get("confirm_clear"):
            r = requests.delete(f"{BACKEND_URL}/collection")
            st.success("Vector store cleared") if r.ok else st.error(r.text)
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm")


# ── Chat page ─────────────────────────────────────────────────────────────────
if "💬" in page:
    st.title("💬 Ask Your Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['file']}** (chunk {s.get('chunk_index', '?')})")
                        st.caption(s.get("preview", ""))

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Build history for API (last 5 turns)
        history = []
        msgs = st.session_state.messages[:-1]
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "user" and i + 1 < len(msgs):
                history.append([msgs[i]["content"], msgs[i + 1]["content"]])

        # Query the RAG backend
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"question": question, "chat_history": history},
                        timeout=60,
                    )
                    if resp.ok:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📎 {len(sources)} source(s) — {data['chunk_count']} chunks retrieved"):
                                for s in sources:
                                    st.markdown(f"**{s['file']}** (chunk {s.get('chunk_index', '?')})")
                                    st.caption(s.get("preview", ""))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        err = resp.json().get("detail", "Unknown error")
                        st.error(f"Error: {err}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend API. Is it running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Evaluation dashboard ──────────────────────────────────────────────────────
elif "📊" in page:
    st.title("📊 Evaluation Dashboard")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Run Evaluation")
        dataset_path = st.text_input(
            "Dataset path",
            value="tests/eval/eval_dataset.json",
            help="Path to your RAGAS eval dataset JSON",
        )
        set_baseline = st.checkbox("Set as baseline after run")
        if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running RAGAS evaluation (this takes a few minutes)..."):
                resp = requests.post(
                    f"{BACKEND_URL}/eval/run",
                    json={"dataset_path": dataset_path, "set_as_baseline": set_baseline},
                )
                if resp.ok:
                    st.success("Evaluation started in background. Refresh in ~2 minutes.")
                else:
                    st.error(resp.text)

    with col1:
        st.subheader("Latest Results")
        try:
            eval_data = requests.get(f"{BACKEND_URL}/eval/latest", timeout=5).json()

            scores = eval_data.get("scores", {})
            gates = eval_data.get("gates_passed", False)

            gate_col, _ = st.columns([1, 2])
            with gate_col:
                if gates:
                    st.success("✓ CI Gates Passed")
                else:
                    st.error("✗ CI Gates Failed")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.1%}")
            m2.metric("Hallucination Rate", f"{scores.get('hallucination_rate', 0):.1%}",
                      delta=None, delta_color="inverse")
            m3.metric("Answer Relevancy", f"{scores.get('answer_relevancy', 0):.1%}")
            m4.metric("Context Precision", f"{scores.get('context_precision', 0):.1%}")

            st.caption(f"Run at: {eval_data.get('timestamp', 'unknown')} UTC | Dataset size: {eval_data.get('dataset_size', '?')}")

            config = eval_data.get("chunk_config", {})
            st.json(config)

        except requests.exceptions.HTTPError:
            st.info("No evaluation results yet. Run an evaluation above.")
        except Exception as e:
            st.warning(f"Could not load results: {e}")

    st.divider()

    # ── Trend chart
    st.subheader("Metric Trends Over Time")
    history_path = Path("./eval_results/metrics_history.jsonl")
    if history_path.exists():
        rows = []
        with open(history_path) as f:
            for line in f:
                if line.strip():
                    run = json.loads(line)
                    rows.append({
                        "timestamp": run["timestamp"],
                        **run.get("scores", {}),
                    })

        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S")
            df = df.set_index("timestamp").sort_index()
            st.line_chart(df[["faithfulness", "answer_relevancy", "context_precision"]])

            st.caption("Lower hallucination rate = better. All other metrics: higher = better.")
            if "hallucination_rate" in df.columns:
                st.area_chart(df[["hallucination_rate"]])
        else:
            st.info("No history data yet.")
    else:
        st.info("Evaluation history will appear here after your first run.")
