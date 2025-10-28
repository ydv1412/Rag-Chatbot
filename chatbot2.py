from __future__ import annotations

import os
import sys
import streamlit as st

# paths
FAISS_DIR = r"C:\Users\shri\Data_Science\Text Mining\chunk_index_langchain_3"
GGUF_PATH = r"C:\Users\shri\Data_Science\Text Mining\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# vars
TOP_K = 4                # retrieved chunks
SCORE_TH = 0.25          # ignore very weak matches
CTX_LEN = 8192           # model context window
INPUT_BUDGET = 0.80      # fraction of CTX used for prompt (rest for generation)
MAX_NEW_TOKENS = 512     # max tokens to generate
TEMPERATURE = 0.2        # creativity
MEMORY_K_DEFAULT = 3     #  Q↔A pairs to include in context

# llama.cpp 
N_BATCH = 16
N_THREADS = max(1, min(4, os.cpu_count() or 1))

# to keep answers grounded
SYSTEM_PROMPT = (
    "You are a careful assistant in a RAG system. "
    "Answer ONLY using the information in the Context. Do NOT use prior knowledge. "
    "If the Context lacks what is needed, reply exactly: The answer is not in the provided context."
)

# stability env (helps on Windows/CPU)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# UI 
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")
st.markdown(
    """
<style>
#MainMenu, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
footer { visibility: hidden; }
.block-container { max-width: 900px; margin: 0 auto; }
html, body, [data-testid="stAppViewContainer"] { font-size: 18px; }

.input-card {
  border-radius: 16px; background: #0b1220; padding: 16px 16px;
  border: 1px solid rgba(148,163,184,.25); margin-bottom: 18px;
}
.stButton>button {
  width: 120px; border-radius: 12px; font-weight: 600;
  border: 1px solid rgba(248,113,113,.35); background: rgba(248,113,113,.1);
}
.stButton>button:hover { background: rgba(248,113,113,.18); }

.msg {
  border-radius: 16px; border: 1px solid rgba(148,163,184,.25);
  padding: 14px 16px; line-height: 1.55; margin: 12px 0 18px;
  box-shadow: 0 1px 0 rgba(0,0,0,.15);
}
.msg.user   { background: rgba(59,130,246,.10);  border-color: rgba(59,130,246,.35); }
.msg.assist { background: rgba(234,179,8,.10);   border-color: rgba(234,179,8,.35); }
div[data-baseweb="slider"] { margin-top: 8px; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<h1>🧠 RAG Chatbot</h1>", unsafe_allow_html=True)


# LangChain 
def _import_llm():
    try:
        from langchain_community.llms import LlamaCpp
    except Exception:
        from langchain.llms import LlamaCpp
    return LlamaCpp


LlamaCpp = _import_llm()


@st.cache_resource(show_spinner=True)
def load_embeddings():

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


@st.cache_resource(show_spinner=True)
def load_vectorstore(_faiss_dir: str):
    if not os.path.isdir(_faiss_dir):
        raise FileNotFoundError(f"FAISS index folder not found:\n{_faiss_dir}")

    emb = load_embeddings()

    try:
        import faiss
        index_bin = os.path.join(_faiss_dir, "index.faiss")
        if os.path.isfile(index_bin):
            idx = faiss.read_index(index_bin)
            if hasattr(emb, "client") and hasattr(emb.client, "get_sentence_embedding_dimension"):
                emb_dim = emb.client.get_sentence_embedding_dimension()
                if idx.d != emb_dim:
                    raise RuntimeError(
                        f"Embedding dim mismatch: index expects {idx.d}, model outputs {emb_dim}. "
                        "Rebuild the index with the same embedding model."
                    )
    except Exception:
        pass

    try:
        from langchain_community.vectorstores import FAISS as _FAISS
    except Exception:
        from langchain.vectorstores import FAISS as _FAISS

    return _FAISS.load_local(
        _faiss_dir,
        embeddings=emb,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=True)
def load_llm(temp: float):
    if not os.path.isfile(GGUF_PATH):
        raise FileNotFoundError(f"GGUF model file not found:\n{GGUF_PATH}")
    return LlamaCpp(
        model_path=GGUF_PATH,
        temperature=temp,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        n_gpu_layers=0,
        f16_kv=False,
        verbose=False,
        n_ctx=CTX_LEN,
        context_length=CTX_LEN,
    )


# build resources
try:
    _ = load_embeddings()
    vectorstore = load_vectorstore(FAISS_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K, "score_threshold": SCORE_TH})
except Exception as e:
    st.error(f"Retrieval setup failed:\n\n{e}")
    st.stop()

try:
    llm = load_llm(TEMPERATURE)
except Exception as e:
    st.error(f"LLM init failed:\n\n{e}")
    st.stop()


# chat state 
if "history" not in st.session_state:
    st.session_state.history = []           # full transcript
if "memory_k" not in st.session_state:
    st.session_state.memory_k = MEMORY_K_DEFAULT


# helpers 
def format_history() -> str:
    # include the last 3 Q↔A pairs (newest first)
    k = int(st.session_state.get("memory_k", MEMORY_K_DEFAULT))
    msgs = st.session_state.history[-2 * k :]
    lines = []
    for m in msgs[::-1]:
        lines.append(("User: " if m["role"] == "user" else "Assistant: ") + m["content"])
    return "\n".join(lines)


def build_prompt(history: str, context: str, query: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + (f"Conversation History (most recent first):\n{history}\n\n" if history.strip() else "")
        + f"Context:\n{context.strip()}\n\n"
        + f"Question:\n{query.strip()}\n\n"
        + "Answer:"
    )


def clip_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    return text if len(text) <= max_chars else text[-max_chars:]


def rewrite_query(q: str) -> str:
    # tiny rewrite + a short tail of recent turns helps follow-ups
    import re

    q2 = re.sub(r"[^A-Za-z0-9\s]", " ", q).lower()
    stops = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is", "are", "was", "were", "why", "what", "how"}
    toks = [t for t in q2.split() if t not in stops and len(t) > 2]
    tail = " ".join(m["content"] for m in st.session_state.history[-2 * st.session_state.memory_k :])[-600:]
    return " ".join(toks[:15]) + (" " + tail if tail else "")


# input area 
st.markdown('<div class="input-card">', unsafe_allow_html=True)
q = st.text_area("Your question", placeholder="Ask grounded questions…", label_visibility="collapsed", height=110)
c1, c2, c3, _ = st.columns([1, 1, 2, 6])
with c1:
    ask = st.button("Ask")
with c2:
    clear = st.button("Clear")
with c3:
    st.session_state.memory_k = st.slider(
        "History pairs", 1, 6, st.session_state.memory_k, 1, help="How many recent Q↔A pairs to remember."
    )
st.markdown("</div>", unsafe_allow_html=True)

if clear:
    st.session_state.history.clear()
    st.experimental_rerun()

# show transcript
for m in st.session_state.history:
    role = "user" if m["role"] == "user" else "assist"
    st.markdown(f'<div class="msg {role}">{m["content"]}</div>', unsafe_allow_html=True)

# main flow 
if ask and q and q.strip():
    user_q = q.strip()
    st.session_state.history.append({"role": "user", "content": user_q})
    st.markdown(f'<div class="msg user">{user_q}</div>', unsafe_allow_html=True)

    with st.status("Retrieving context…", expanded=False) as status:
        try:
            search_q = rewrite_query(user_q)

            if hasattr(vectorstore, "similarity_search_with_score"):
                docs_scores = vectorstore.similarity_search_with_score(search_q, k=TOP_K)
                docs = [d for d, s in docs_scores]
                scores = [float(s) for _, s in docs_scores]
            else:
                docs = retriever.invoke(search_q)
                scores = [float("nan")] * len(docs)

            if not docs or all((isinstance(s, float) and (s != s or s < SCORE_TH)) for s in scores):
                status.update(label="No relevant context found", state="error")
                answer = "The answer is not in the provided context."
                st.markdown(f'<div class="msg assist">{answer}</div>', unsafe_allow_html=True)
                st.session_state.history.append({"role": "assistant", "content": answer})
            else:
                status.update(label="Building prompt…", state="running")

                context_text = "\n\n---\n\n".join(d.page_content.strip() for d in docs[:TOP_K])
                input_token_budget = int(CTX_LEN * INPUT_BUDGET)
                prompt = build_prompt(format_history(), context_text, user_q)
                prompt = clip_to_tokens(prompt, input_token_budget)

                status.update(label="Generating answer…", state="running")
                try:
                    answer = llm.invoke(prompt).strip()
                except Exception as e:
                    status.update(label="Generation failed", state="error")
                    st.error(f"Generation error: {e}")
                    st.stop()

                if not answer:
                    answer = "The answer is not in the provided context."

                status.update(label="Done ", state="complete")
                st.markdown(f'<div class="msg assist">{answer}</div>', unsafe_allow_html=True)
                st.session_state.history.append({"role": "assistant", "content": answer})

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Retrieval error: {e}")
            st.stop()
