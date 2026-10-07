Raag# 🇪🇺 RAG Chatbot for European Commission News

An end-to-end **Retrieval-Augmented Generation (RAG)** system for answering factual questions from European Commission news.

The project explores the complete RAG pipeline — from **web scraping and document preprocessing to chunking, semantic retrieval, prompt construction, LLM-based answer generation, and quantitative evaluation**.

The final system uses:

**MPNet Embeddings → FAISS Cosine Retrieval → Mistral-7B-Instruct → Grounded Answer**

---

##  Project Objective

The European Commission publishes a large number of news articles, making it difficult to quickly locate specific information across multiple documents.

This project investigates whether a RAG pipeline can provide a simpler interface:

> **Ask a natural-language question and receive a concise answer grounded in retrieved European Commission news.**

Rather than relying only on the internal knowledge of an LLM, relevant news passages are retrieved first and supplied as context for answer generation.

---

##  System Architecture

```text
European Commission News
          │
          ▼
    Web Scraping
          │
          ▼
 Cleaning & Preprocessing
          │
          ▼
      Chunking
          │
          ▼
 MPNet Embeddings
          │
          ▼
 FAISS Vector Index
          │
          │
User Query ──► Query Embedding
                  │
                  ▼
          Cosine Similarity
                  │
                  ▼
           Top-4 Chunks
                  │
                  ▼
       Context + Chat History
                  │
                  ▼
        Mistral-7B-Instruct
                  │
                  ▼
          Grounded Answer
```

---

#  Data Collection

European Commission news articles published between **June 6 and July 14, 2024** were collected from the official European Commission website.

The scraping pipeline:

- extracted article content;
- removed headers and footers;
- removed duplicate content;
- preserved article metadata;
- filtered low-quality entries;
- preserved tables separately to avoid splitting them incorrectly during chunking.

Metadata stored for each article included:

- Title
- Publication date
- Source URL

### Final Dataset

**317 curated European Commission news articles**

---

#  Chunking Experiments

Chunking turned out to be an important part of the retrieval pipeline.

Rather than choosing a fixed strategy immediately, I experimented with multiple approaches.

### Semantic Sentence Merging

Articles were first split into sentences using NLTK Punkt.

Adjacent sentences were merged according to semantic similarity.

Initial configuration:

```text
Similarity threshold > 0.75
Maximum length < 300 characters
```

This produced semantically related chunks, but many were too short and lacked sufficient local context.

The strategy was therefore relaxed to:

```text
Similarity threshold > 0.50
Maximum length = 500 characters
```

### Embedding Models for Semantic Merging

Two sentence embedding models were compared:

- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`

MPNet produced stronger semantic grouping.

### Sliding Window Chunking

A simpler sliding-window strategy was also tested.

The final configuration used:

```text
Chunk size: 500 characters
Overlap:    200 characters
```

Interestingly, the simpler sliding-window approach produced better downstream results than the more complex semantic-merging strategy.

---

#  Embedding Models

Two sentence-transformer models were evaluated.

### MiniLM

```text
all-MiniLM-L6-v2
Embedding dimension: 384
```

MiniLM provided a lightweight and fast baseline.

### MPNet

```text
all-mpnet-base-v2
Embedding dimension: 768
```

MPNet produced richer semantic representations and consistently stronger retrieval similarity.

It was therefore selected for the final pipeline.

---

#  Vector Search with FAISS

Document chunks were converted into dense vectors and indexed using **FAISS**.

Two retrieval configurations were investigated:

### Euclidean Search

The initial implementation used a FAISS L2 index.

### Cosine Similarity

The final implementation uses normalized embeddings with an inner-product FAISS index.

```text
Embedding
    ↓
L2 Normalization
    ↓
FAISS Inner Product
    ↓
≈ Cosine Similarity
```

Both indexed documents and incoming queries are normalized, making the inner product equivalent to cosine similarity for retrieval.

The final system retrieves the:

**Top 4 most relevant chunks**

for each user query.

---

#  LLM Answer Generation

Several language models were explored for answer generation.

### Phi-2

A lightweight model that was faster but produced less accurate responses.

### Gemini 1.5 Flash

Produced strong results, but API quota and latency constraints made it less suitable for the final local implementation.

### Mistral-7B-Instruct

The final system uses:

**Mistral-7B-Instruct via `llama.cpp`**

Mistral provided a useful balance between answer quality and the ability to run the complete pipeline locally.

The model was run using a quantized GGUF version on CPU.

---

#  Retrieval-Augmented Generation Pipeline

For each question:

```text
User Question
      │
      ▼
MPNet Query Embedding
      │
      ▼
FAISS Cosine Search
      │
      ▼
Top-4 Relevant Chunks
      │
      ▼
Retrieved Context
      +
Last 3 Question/Answer Pairs
      │
      ▼
Prompt
      │
      ▼
Mistral-7B-Instruct
      │
      ▼
Grounded Response
```

The prompt explicitly instructs the model to:

> Answer only using the provided context and indicate when the requested information cannot be found.

This helps reduce unsupported answers and keeps responses grounded in the retrieved European Commission articles.

---

#  Evaluation

The RAG pipeline was evaluated using a test set of:

**297 question-answer pairs generated using GPT-4o**

Multiple configurations of the pipeline were compared.

Evaluation metrics included:

- BLEU-1
- ROUGE-1
- ROUGE-2
- ROUGE-L
- METEOR
- Semantic Similarity

---

##  Experimental Results

| Configuration | BLEU-1 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | Semantic Similarity |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM + L2 + Mistral-7B | 0.19 | 0.33 | 0.10 | 0.20 | 0.26 | 0.75 |
| MPNet + Cosine + Mistral-7B | 0.22 | 0.38 | 0.13 | 0.24 | 0.35 | **0.88** |
| **MPNet + Sliding Window + Mistral-7B** | **0.27** | **0.43** | **0.17** | **0.29** | **0.39** | 0.84 |
| MPNet + Sliding Window + Gemini | **0.31** | **0.46** | **0.18** | **0.30** | 0.35 | 0.84 |

### Final Local Configuration

The final locally runnable system uses:

```text
Chunking      → 500 characters + 200 overlap
Embeddings    → all-mpnet-base-v2
Vector Search → FAISS cosine similarity
Top-K         → 4
LLM           → Mistral-7B-Instruct
Interface     → Streamlit
```

---

#  Key Findings

The experiments produced several useful observations.

### 1. MPNet improved semantic retrieval

Replacing MiniLM + L2 retrieval with MPNet + cosine similarity increased semantic similarity from:

```text
0.75 → 0.88
```

### 2. Simpler chunking worked better

Semantic sentence merging appeared theoretically attractive, but the simpler **500-character sliding window with 200-character overlap** produced stronger downstream answer-generation metrics.

### 3. Retrieval configuration matters

Changing the embedding and similarity strategy produced measurable improvements without changing the underlying LLM.

This reinforced an important lesson from the project:

> **RAG performance depends heavily on retrieval quality, not only on the language model used for generation.**

---

#  Handling Tables

Some European Commission articles contained structured tables.

Naively splitting these tables during chunking could separate related rows and headers, reducing retrieval quality.

During preprocessing, tables were therefore identified and preserved using:

```html
<table> ... </table>
```

This prevented them from being broken midway during the initial document processing pipeline.

---

#  Conversation Memory

The chatbot maintains short-term conversational context.

The previous:

**3 question-answer pairs**

are included when constructing the next prompt.

This allows follow-up questions while keeping the context size manageable.

Conversation state is maintained using:

```python
st.session_state
```

---

#  Streamlit Application

The complete pipeline is wrapped in a chat-style **Streamlit interface**.

The application supports:

- Natural-language questions
- RAG-based retrieval
- Grounded LLM responses
- Short-term conversation memory
- Conversation history clearing

---

#  Demo

>  **Demo video: Coming soon**


---

# 🛠️ Tech Stack

### Language & Data

- Python
- Pandas
- BeautifulSoup
- NLTK

### Embeddings & Retrieval

- Sentence Transformers
- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`
- FAISS
- Cosine Similarity

### LLMs

- Mistral-7B-Instruct
- llama.cpp
- Gemini 1.5 Flash
- Phi-2

### Application

- Streamlit

### Evaluation

- BLEU
- ROUGE
- METEOR
- Cosine Semantic Similarity

---

#  Project Structure

```text
RAG-EU-Commission-News/
│
├── Web Scrapping.ipynb
│   └── News scraping, cleaning and preprocessing
│
├── Chunking&Vectorisation.ipynb
│   └── Chunking experiments, embeddings and FAISS indexing
│
├── AnswerGeneration.ipynb
│   └── RAG experiments and quantitative evaluation
│
├── chatbot2.py
│   └── Streamlit RAG chatbot
│
├── presentation/
│   └── RAG_Chatbot_EU_Commission_News.pdf
│
├── requirements.txt
└── README.md
```

---

#  Running Locally

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY>
```

## 2. Create the environment

```bash
conda create -n ragbot python=3.10
conda activate ragbot
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the LLM

Download a compatible **Mistral-7B-Instruct GGUF** model for `llama.cpp`.

Update the model path in the application configuration.

## 5. Prepare the FAISS index

Run the preprocessing/vectorization notebook or place the previously generated FAISS index in the expected directory.

## 6. Start the application

```bash
streamlit run chatbot2.py
```

---

#  Limitations

The project has several limitations:

- Mistral-7B inference is relatively slow on CPU.
- The knowledge base covers only a limited period of European Commission news.
- Evaluation is based on automatically generated QA pairs.
- Retrieval performance depends on document preprocessing and chunking.
- Structured tables remain more challenging to retrieve effectively than normal text.

---

#  Future Work

Potential improvements include:

- Hybrid dense + lexical retrieval
- Reranking retrieved documents before generation
- Better retrieval of structured tables
- Query rewriting for conversational questions
- Citation/source display in generated answers
- Evaluation using human-generated questions
- More efficient local LLM inference
- Automatically updating the news knowledge base

---

#  Author

**Shri Prakash Yadav**  
M.Sc. Data Science  
University of Naples Federico II
