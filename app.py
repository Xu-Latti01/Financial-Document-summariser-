import os
import streamlit as st
from utils import (
    extract_text_from_pdf,
    chunk_text,
    make_embeddings,
    build_faiss_index,
    retrieve_top_k,
    generate_summary
)
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

st.set_page_config(page_title="Financial Document Summariser", layout="wide")
st.title("Financial Document Summariser (RAG + LLM)")

with st.sidebar:
    st.markdown("## Upload Financial Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    max_chunk_chars = st.slider("Chunk size (characters)", 800, 4000, 1200)
    top_k = st.slider("Top chunks to retrieve", 1, 10, 4)
    model = st.text_input("LLM Model", "gpt-4o-mini")
    embed_model = st.text_input("Embedding Model", "text-embedding-3-small")
    st.markdown("---")
    st.markdown("**Privacy:** Do not upload confidential documents.")

st.write("Upload a document in the sidebar or paste text below.")
text_input = st.text_area("Or paste text here", height=150)

if st.button("Build Index & Summarise"):

    if uploaded_file is None and not text_input.strip():
        st.warning("Provide a PDF or paste text to summarise.")
        st.stop()

    with st.spinner("Extracting text..."):
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            else:
                raw_text = uploaded_file.getvalue().decode("utf-8")
        else:
            raw_text = text_input

    if not raw_text.strip():
        st.error("No text found in the uploaded file.")
        st.stop()

    st.success("Text extracted.")
    st.write("---")
    st.header("Indexing document...")

    # Chunk text
    chunks = chunk_text(raw_text, max_chars=max_chunk_chars)
    st.write(f"Document split into **{len(chunks)}** chunks.")

    # Create embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = make_embeddings(chunks, model=embed_model, api_key=OPENAI_API_KEY)

    # Build FAISS index
    index = build_faiss_index(embeddings)
    st.success("Index built.")

    # Retrieval and summarisation
    with st.spinner("Retrieving relevant chunks and generating summary..."):
        query = (
                "Summarise the document focusing on financial highlights, key risks, and opportunities. "
                "Provide a short executive summary (4-6 sentences) and bullet lists for risks and opportunities."
        )
        topk_results = retrieve_top_k(
            index, embeddings, chunks, query, k=top_k, embed_model=embed_model, api_key=OPENAI_API_KEY
        )

        summary = generate_summary(topk_results, model=model, api_key=OPENAI_API_KEY)

    st.header("📌 Summary")
    st.write(summary)

    st.header("🧾 Retrieved Chunks")
    for i, (_, snippet) in enumerate(topk_results, start=1):
        st.markdown(f"**{i}.**")
        st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))