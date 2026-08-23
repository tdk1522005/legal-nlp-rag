# Vietnamese Civil Law RAG

A Retrieval-Augmented Generation (RAG) project for question answering over the **Vietnamese Civil Code 2015**. The system retrieves relevant legal passages with dense embeddings and FAISS, then provides the retrieved context to Gemini to generate a grounded answer.

> **Note:** This is an educational AI project and is not a substitute for professional legal advice.

## Overview

The project is designed to reduce unsupported LLM answers by grounding generation in retrieved legal documents.

### Pipeline

```text
Vietnamese Civil Code (.docx)
        ↓
Document loading & text cleaning
        ↓
Recursive text chunking
        ↓
BAAI/bge-m3 dense embeddings
        ↓
FAISS IndexFlatIP
        ↓
Top-k semantic retrieval
        ↓
Context builder
        ↓
Grounded prompt
        ↓
Gemini 2.5 Flash
        ↓
Answer
```

## Tech Stack

- **Language:** Python
- **Embedding:** BAAI/bge-m3 via FlagEmbedding
- **Vector Search:** FAISS (`IndexFlatIP`)
- **LLM:** Gemini 2.5 Flash
- **Document Processing:** LangChain Community, LangChain Text Splitters, docx2txt
- **Utilities:** NumPy, PyTorch, python-dotenv

## Project Structure

```text
legal-nlp-rag/
├── context/          # Build context from retrieved passages
├── data/             # Source legal documents
├── llm/              # LLM abstraction and Gemini implementation
├── models/           # Embedding abstraction and BGE-M3 implementation
├── preprocess/       # Document loading, cleaning and chunking
├── prompt/           # Grounded legal QA prompt construction
├── retrieval/        # Query embedding and top-k retrieval
├── vectorstore/      # FAISS storage, search and persistence
├── build_index.py    # Build the vector index from source documents
├── chat.py           # Command-line RAG chatbot
├── main.py           # Application entry point
├── .env.example      # Environment-variable template
└── requirements.txt  # Python dependencies
```

## How It Works

1. `preprocess/loader.py` loads `.docx` files from `data/`.
2. `preprocess/cleaner.py` normalizes document text.
3. `preprocess/chunker.py` splits documents into overlapping chunks.
4. `models/bge_embedding.py` encodes chunks with `BAAI/bge-m3`.
5. `vectorstore/faiss_store.py` stores embeddings in a FAISS `IndexFlatIP` index.
6. At query time, `retrieval/retriever.py` embeds the user question and retrieves the top-k passages.
7. `context/context_builder.py` combines retrieved passages and source metadata.
8. `prompt/prompt_builder.py` instructs the LLM to answer only from the supplied context.
9. `llm/gemini_llm.py` generates the final response with Gemini 2.5 Flash.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tdk1522005/legal-nlp-rag.git
cd legal-nlp-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Copy `.env.example` to `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Build the FAISS index

```bash
python build_index.py
```

The generated index is stored under `index/`.

### Start the chatbot

```bash
python main.py
```

Type `exit` to stop the command-line chatbot.

## Current Limitations

- Retrieval currently uses dense top-k semantic search without a reranker.
- Evaluation is not yet automated with a dedicated legal QA benchmark.
- Answer quality depends on document coverage, chunking quality and retrieval quality.
- The current interface is command-line based.

## Planned Improvements

- Add retrieval evaluation such as Hit@K / Recall@K.
- Add answer-level evaluation and a curated Vietnamese legal QA test set.
- Explore hybrid retrieval and reranking.
- Improve citation formatting for retrieved legal passages.
- Add automated tests and a web/API interface.

## Author

**Ta Duy Khanh**  
AI Engineering student — Machine Learning, Deep Learning & Natural Language Processing
