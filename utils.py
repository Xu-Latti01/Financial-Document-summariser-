import PyPDF2
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import io
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"[ERROR reading PDF] {e}"

def chunk_text(text, max_chars=1600):
    """Split text into chunks without splitting in the middle of sentences if possible."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for s in sentences:
        if len(current_chunk) + len(s) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = s + ". "
        else:
            current_chunk += s + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def make_embeddings(texts, model="text-embedding-3-small", api_key=None):
    client = OpenAI(api_key=api_key)

    vectors = []
    for txt in texts:
        resp = client.embeddings.create(
            model=model,
            input=txt
        )
        vectors.append(resp.data[0].embedding)

    return np.array(vectors).astype("float32")

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def retrieve_top_k(index, vectors, chunks, query, k=4, embed_model="text-embedding-3-small", api_key=None):
    client = OpenAI(api_key=api_key)

    q_embed = client.embeddings.create(
        model=embed_model,
        input=query
    ).data[0].embedding
    q_embed = np.array(q_embed).astype("float32").reshape(1, -1)
    
    actual_k = min(k, len(chunks))
    scores, idx = index.search(q_embed, actual_k)

    results = []
    seen_texts = set()
    for score, i in zip(scores[0], idx[0]):
        chunk_text = chunks[i]
        if chunk_text not in seen_texts:
            results.append((float(score), chunk_text))
            seen_texts.add(chunk_text)

    return results

def generate_summary(top_chunks, model="gpt-4o-mini", api_key=None):
    client = OpenAI(api_key=api_key)
    combined_text = "\n\n".join([chunk[1] for chunk in top_chunks])

    prompt = f"""
Summarise the following financial document sections.

Focus on:
- Key financial highlights
- Risks
- Opportunities
- Executive summary (4–6 sentences)

Please provide the summary in a clear and concise manner.
Keep the response structured with bullet points.

Content:
{combined_text}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    message = resp.choices[0].message
    if isinstance(message, dict):
        return message["content"]
    else:
        return message.content