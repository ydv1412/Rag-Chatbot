# RAG Chatbot for European Commission News
---

##  Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers factual questions grounded in real **European Commission news articles** (June–July 2024).  
Instead of searching and reading multiple articles manually, users can ask natural questions and receive concise, context-based answers drawn directly from curated EU news.

---

##  Project Structure

| File | Description |
|------|--------------|
| `Web Scrapping.ipynb` | Web scraping and preprocessing of EU Commission news articles. |
| `Chunking&Vectorisation.ipynb` | Splits news text into overlapping chunks and builds FAISS vector store with MPNet embeddings. |
| `AnswerGeneration.ipynb` | Prototyping and evaluation of answer quality (BLEU, ROUGE, METEOR). |
| `chatbot2.py` | Streamlit chatbot app using Mistral-7B via `llama.cpp` and FAISS retrieval. |


---

##  Components

### 1. **Data & Preprocessing**
- News scraped from the [European Commission website](https://ec.europa.eu/newsroom/homepage)  
- Cleaned, deduplicated, and stored with metadata (title, date, URL).  
- Final dataset: **317 curated articles**

### 2. **Chunking Strategy (Best one)**
- Text split into 500-character chunks with 200 overlap  
- Embeddings: `sentence-transformers/all-mpnet-base-v2` (768-dim)  
- Vector store: **FAISS cosine similarity** (normalized embeddings)

### 3. **Retrieval-Augmented Chat**
- Query → Embedded → Top-K chunks retrieved from FAISS  
- Context + short-term chat history (3 Q↔A pairs) added to prompt  
- Model: **Mistral-7B-Instruct (GGUF)** via `llama.cpp` (CPU)  
- Streamlit UI with memory, clear history, and clean chat layout

---

##  How to Run Locally

```bash
# 1. Create a new conda environment
conda create -n ragbot python=3.10
conda activate ragbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your Mistral model and FAISS index
# Example:
#   C:\Users\<you>\Data_Science\Text Mining\mistral-7b-instruct-v0.1.Q4_K_M.gguf
#   C:\Users\<you>\Data_Science\Text Mining\chunk_index_langchain_3

# 4. Run the chatbot
streamlit run chatbot.py
