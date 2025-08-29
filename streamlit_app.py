from __future__ import annotations



import os
from typing import List
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.config import AppConfig, ensure_dirs
from app.embeddings import EmbeddingProvider
from app.vector_store import VectorStore
from app.faq_loader import load_csv, load_txt, to_documents
from app.llm import LLMFallback


# Load .env from the directory of this file to avoid CWD issues
_DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)


def get_state(reload_cfg: bool = False):
    # Optionally rebuild config from current environment (to catch updated .env)
    if reload_cfg or "cfg" not in st.session_state:
        st.session_state.cfg = AppConfig.from_env()

        # Recreate components that depend on config
        st.session_state.embedder = EmbeddingProvider(st.session_state.cfg)
        st.session_state.store = VectorStore(st.session_state.cfg)
        # Only create LLM if API key is available
        if (st.session_state.cfg.openai_api_key or "").strip():
            st.session_state.llm = LLMFallback(st.session_state.cfg)
        else:
            st.session_state.llm = None
    else:
        # Lazily initialize components if missing
        if "embedder" not in st.session_state:
            st.session_state.embedder = EmbeddingProvider(st.session_state.cfg)
        if "store" not in st.session_state:
            st.session_state.store = VectorStore(st.session_state.cfg)
        if "llm" not in st.session_state:
            # Only create LLM if API key is available
            if (st.session_state.cfg.openai_api_key or "").strip():
                st.session_state.llm = LLMFallback(st.session_state.cfg)
            else:
                st.session_state.llm = None
    return st.session_state


def sidebar_controls(state):
    st.sidebar.header("Settings")
    st.sidebar.write(f"Embeddings: {state.embedder.provider_name}")
    st.sidebar.write(f"Chroma dir: `{state.cfg.chroma_dir}`")
    st.sidebar.write(f"Collection size: {state.store.count()}")
    llm_ok = state.llm.available() if state.llm else False
    masked = "(none)" if not (state.cfg.openai_api_key or "").strip() else ("***" + (state.cfg.openai_api_key or "")[-4:])
    st.sidebar.write("LLM: " + ("configured" if llm_ok else "not configured"))
    st.sidebar.caption(f"Model: {state.cfg.openai_model} | Key: {masked}")

    if st.sidebar.button("Clear index"):
        state.store.clear()
        st.sidebar.success("Cleared.")


def main():
    st.set_page_config(page_title="FAQ Chatbot", page_icon="❓", layout="wide")
    # Ensure .env is loaded even if rerun happens from a different CWD
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
    state = get_state(reload_cfg=True)
    ensure_dirs(state.cfg)

    st.title("AI-Powered FAQ Chatbot")
    st.caption("Upload FAQs, build embeddings, and query with semantic search.")

    sidebar_controls(state)

    tab_index, tab_chat = st.tabs(["Index", "Chat"])

    with tab_index:
        st.subheader("Upload FAQ data")
        uploaded = st.file_uploader("Upload CSV (question,answer) or TXT (Q:/A:)", type=["csv", "txt"], accept_multiple_files=False)
        build_col1, build_col2 = st.columns([1, 2])
        with build_col1:
            top_k = st.number_input("Top K results", min_value=1, max_value=20, value=state.cfg.top_k)
            threshold = st.slider("Distance threshold (cosine)", 0.0, 1.0, value=float(state.cfg.distance_threshold))
        with build_col2:
            st.info("Distance threshold is applied at query time; lower means more similar.")

        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1].lower()
            if st.button("Build / Rebuild Index", use_container_width=True):
                try:
                    if suffix == ".csv":
                        items = load_csv(uploaded)
                    else:
                        items = load_txt(uploaded)
                    ids, docs, metas = to_documents(items)
                    vecs = state.embedder.embed_texts(docs)
                    state.store.clear()
                    state.store.add(ids=ids, embeddings=vecs.tolist(), metadatas=metas, documents=docs)
                    st.success(f"Indexed {len(ids)} FAQs.")
                except Exception as e:
                    st.error(f"Failed to index: {e}")

        st.divider()
        st.markdown("#### Sample format")
        st.code("""Q: What are your hours?\nA: We are open 9am-5pm Mon-Fri.""", language="text")

    with tab_chat:
        st.subheader("Ask a question")
        query = st.text_input("Your question")
        if st.button("Search", type="primary") and query.strip():
            try:
                q_vec = state.embedder.embed_text(query).tolist()
                res = state.store.query(query_embeddings=[q_vec], n_results=top_k)
                distances = res.get("distances", [[ ]])[0]
                documents = res.get("documents", [[ ]])[0]
                metadatas = res.get("metadatas", [[ ]])[0]

                answer_rendered = False
                for doc, meta, dist in zip(documents, metadatas, distances):
                    if dist <= threshold:
                        st.success(meta.get("answer", ""))
                        with st.expander("Matched FAQ"):
                            st.write(f"Question: {meta.get('question')}")
                            st.write(f"Distance: {dist:.3f}")
                        answer_rendered = True
                        break

                if not answer_rendered:
                    if state.llm and state.llm.available():
                        context = "\n\n".join(documents[:3]) if documents else None
                        with st.spinner("Consulting LLM fallback..."):
                            answer = state.llm.answer(query, context=context)
                        st.info(answer)
                    else:
                        st.warning("No close FAQ found and LLM fallback not configured.")

                if documents:
                    with st.expander("Top results"):
                        for doc, dist in zip(documents, distances):
                            st.write(f"Distance: {dist:.3f}")
                            st.code(doc)

            except Exception as e:
                st.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()


