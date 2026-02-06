# AI Handbook Generator

This project is an AI-powered chat application that generates long-form
handbooks (~20,000 words) from uploaded PDF documents.

The system allows users to:
- Upload one or more PDF files
- Ask questions about the documents
- Generate a structured handbook based on the content

---

## Features

- PDF upload and parsing
- Contextual AI chat over documents
- Long-form handbook generation
- Export handbook to Markdown
- Fully offline execution

---

## Tech Stack

- Python 3.11
- Streamlit (web UI)
- Ollama (local LLM)
- Custom RAG pipeline:
  - Chunking
  - Embeddings
  - Cosine similarity retrieval

---

## How It Works (High Level)

1. PDFs are split into small text chunks  
2. Each chunk is converted into an embedding  
3. User questions are embedded and matched via cosine similarity  
4. The most relevant chunks are injected into the AI prompt  
5. The AI answers using only the document context  
6. Handbook is generated iteratively (outline → sections)

---

## LLM Choice (Grok vs Local)

This implementation uses Ollama (local LLM) for reliability and offline
execution. The architecture is model-agnostic and can be easily adapted
to cloud LLMs such as Grok by replacing the generation function with an
API call.

Grok was considered, but a local model was chosen to avoid external
dependencies, API limits, and service availability issues during
development.

---

## How to Run

Requirements:
- Python 3.11
- Ollama installed and running

Steps:

(in bash)
pip install streamlit pypdf ollama
python -m streamlit run app.py
