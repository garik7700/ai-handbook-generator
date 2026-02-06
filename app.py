import streamlit as st
from pypdf import PdfReader
import requests
import math

# Constants for memory optimization
MAX_PDF_SIZE_MB = 10
MAX_CONTEXT_CHARS = 8000
CHUNK_SIZE = 1000
TOP_K = 3
MAX_CHUNKS = 500  # Memory limit: max chunks to store

# Initialize session state for local RAG
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_vectors" not in st.session_state:
    st.session_state.chunk_vectors = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache embedding function
@st.cache_data(show_spinner=False, ttl=3600)
def embed(text):
    """Generate embedding for text using Ollama nomic-embed-text model."""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into chunks on sentence boundaries when possible."""
    if not text:
        return []
    
    # Validate parameters
    chunk_size = max(200, min(chunk_size, 2000))
    overlap = min(overlap, chunk_size // 4)
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to split on sentence boundaries near end
        if end < len(text):
            for sep in ['. ', '! ', '? ', '\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.7:
                    end = start + last_sep + 1
                    chunk = text[start:end]
                    break
        
        chunks.append(chunk)
        start = end - overlap
        
        # Safety check
        if len(chunks) > len(text) // 100 + 10:
            break
    
    return chunks

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def retrieve_chunks_local(query, top_k=TOP_K):
    """Retrieve top_k chunks from local in-memory storage using cosine similarity."""
    if not st.session_state.chunks or not st.session_state.chunk_vectors:
        return []
    
    query_embedding = embed(query)
    if not query_embedding:
        return []
    
    # Compute similarities
    similarities = []
    for i, chunk_vec in enumerate(st.session_state.chunk_vectors):
        sim = cosine_similarity(query_embedding, chunk_vec)
        similarities.append((i, sim))
    
    # Sort and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in similarities[:top_k]]
    
    return [st.session_state.chunks[i] for i in top_indices]

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    st.info("Using Ollama at http://localhost:11434")
    st.caption(f"Max context: {MAX_CONTEXT_CHARS} chars")
    st.caption(f"Chunks stored: {len(st.session_state.chunks)}/{MAX_CHUNKS}")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    # Check PDF size
    pdf_size = pdf.size / (1024 * 1024)
    if pdf_size > MAX_PDF_SIZE_MB:
        st.warning(f"PDF is {pdf_size:.1f}MB. Max allowed is {MAX_PDF_SIZE_MB}MB.")
        st.stop()
    
    # Stream PDF text extraction
    reader = PdfReader(pdf)
    text_parts = []
    total_pages = len(reader.pages)
    
    for i, page in enumerate(reader.pages):
        try:
            content = page.extract_text()
            if content:
                text_parts.append(content)
        except Exception as e:
            st.warning(f"Warning: Could not extract text from page {i+1}: {e}")
    
    full_text = "\n".join(text_parts) if text_parts else ""
    del text_parts
    
    # Display PDF text preview
    st.write(full_text[:2000] if full_text else "")
    st.caption(f"Extracted {len(full_text)} characters from {total_pages} pages")
    
    # Export Handbook button
    if st.button("Export Handbook"):
        with open("handbook.md", "w", encoding="utf-8") as f:
            f.write(full_text)
        st.success("Handbook exported to handbook.md")
    
    # Chunk text and generate embeddings
    chunks = chunk_text(full_text)
    del full_text
    
    # Limit chunks to MAX_CHUNKS to control memory
    if len(chunks) > MAX_CHUNKS:
        st.warning(f"PDF has {len(chunks)} chunks. Truncating to {MAX_CHUNKS} for memory efficiency.")
        chunks = chunks[:MAX_CHUNKS]
    
    with st.spinner("Generating embeddings..."):
        # Clear old chunks
        st.session_state.chunks = []
        st.session_state.chunk_vectors = []
        
        # Generate embeddings
        for chunk in chunks:
            vector = embed(chunk)
            if vector:
                st.session_state.chunks.append(chunk)
                st.session_state.chunk_vectors.append(vector)
    
    del chunks

    # Chat interface
    st.divider()
    st.subheader("Chat with PDF")
    
    # Export Context button
    if st.button("Export Context"):
        context = st.session_state.get("last_context", "")
        if context:
            with open("context.txt", "w", encoding="utf-8") as f:
                f.write(context)
            st.success("Context exported to context.txt")
        else:
            st.warning("No context available. Ask a question first.")
    
    # Generate Handbook button
    if st.button("Generate Handbook"):
        context = st.session_state.get("last_context", "")
        if not context:
            st.warning("No context available. Ask a question first.")
        else:
            context = context[:MAX_CONTEXT_CHARS]
            
            progress = st.progress(0)
            status = st.empty()
            
            status.text("Generating Table of Contents...")
            toc_prompt = f"""Based on the following context, create a detailed Table of Contents for a comprehensive handbook.
Use numbered headings (1., 1.1, 1.1.1, etc.) in markdown format.
If information is not in the context, reply exactly: "I don't know based on the uploaded documents."

Context:
{context}"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": toc_prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            toc = response.json()["response"]
            
            sections = [line.strip() for line in toc.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'))]
            
            handbook_text = "# Handbook\n\n## Table of Contents\n\n" + toc + "\n\n## Content\n\n"
            
            for i, section in enumerate(sections):
                progress.progress((i + 1) / len(sections))
                status.text(f"Expanding section {i+1}/{len(sections)}...")
                
                section_prompt = f"""Expand the following section with detailed content in markdown format.
Use the context to provide comprehensive information. Use the same numbered heading.
If the answer is not in the context, reply exactly: "I don't know based on the uploaded documents."

Section: {section}
Context: {context}"""
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3", "prompt": section_prompt, "stream": False},
                    timeout=120
                )
                response.raise_for_status()
                expanded = response.json()["response"]
                
                handbook_text += f"## {section}\n{expanded}\n\n"
            
            st.session_state["handbook_text"] = handbook_text
            progress.progress(1.0)
            status.text("Handbook generation complete!")
            st.success("Handbook generated. Use 'Export Handbook' to save.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Retrieve from local storage
            retrieved_chunks = retrieve_chunks_local(prompt, top_k=TOP_K)
            
            context_string = "\n\n---\n\n".join(retrieved_chunks)[:MAX_CONTEXT_CHARS]
            st.session_state["last_context"] = context_string
            
            prompt_with_context = f"""Context (from PDF):
{context_string}

Question: {prompt}

Instructions: Answer ONLY based on the context above. If the answer is not in the context, say "I don't have enough information to answer this." """
            
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3", "prompt": prompt_with_context, "stream": False},
                    timeout=120
                )
                response.raise_for_status()
                answer = response.json()["response"]
                response_container.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error calling Ollama API: {e}")
