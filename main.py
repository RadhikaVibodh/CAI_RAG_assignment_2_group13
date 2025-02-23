import os
import faiss
import fitz
import re
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from collections import defaultdict

# ------------------- Step 1: Extract & Clean Text -------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return clean_text(text)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

# ------------------- Step 2: Load Models -------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
phi2_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(phi2_model)
model = AutoModelForCausalLM.from_pretrained(phi2_model, torch_dtype=torch.float16, device_map="auto")

# ------------------- Step 3: Indexing (FAISS + BM25) -------------------
def process_files(input_folder):
    text_chunks = []
    chunk_embeddings = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            print(f"Processing file: {filename}")
            file_path = os.path.join(input_folder, filename)
            pdf_text = extract_text_from_pdf(file_path)
            chunks = text_splitter.split_text(pdf_text)
            text_chunks.extend(chunks)
            chunk_embeddings.extend(embed_model.encode(chunks))

    embeddings = np.array(chunk_embeddings, dtype=np.float32)

    # FAISS Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # BM25 Index
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

    return index, text_chunks, bm25

# ------------------- Step 4: Multi-Stage Retrieval -------------------
def retrieve_top_k(index, bm25, text_chunks, query, k=20):
    """Stage 1: Retrieve top candidates from FAISS & BM25."""
    query_embedding = embed_model.encode([query]).astype(np.float32)

    # FAISS Search
    faiss_distances, faiss_indices = index.search(query_embedding, k)

    # BM25 Search
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]

    # Merge results
    combined_results = defaultdict(float)

    for i, d in zip(faiss_indices[0], faiss_distances[0]):
        combined_results[i] += (1 / (1 + d))  # Convert distance to score

    for i in bm25_top_indices:
        combined_results[i] += bm25_scores[i]

    # Sort by combined scores (Hybrid Search)
    top_candidates = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:k]

    # Stage 2: Re-ranking
    ranked_results = reranker.predict([(query, text_chunks[i]) for i, _ in top_candidates])
    ranked_indices = [i for i, _ in sorted(zip([i for i, _ in top_candidates], ranked_results), key=lambda x: x[1], reverse=True)]

    # Stage 3: Merge Adjacent Chunks for Better Context
    final_text = " ".join([text_chunks[i] for i in ranked_indices[:5]])
    return final_text

# ------------------- Step 5: Memory-Augmented Retrieval -------------------
memory_cache = {}

def retrieve_with_memory(index, bm25, text_chunks, query, k=10):
    """Check if query is similar to past queries for faster retrieval."""
    if query in memory_cache:
        print("Using cached retrieval")
        return memory_cache[query]

    retrieved_text = retrieve_top_k(index, bm25, text_chunks, query, k)
    memory_cache[query] = retrieved_text  # Store in memory
    return retrieved_text

# ------------------- Step 6: Generate Answer -------------------
def generate_answer(index, bm25, text_chunks, query):
    relevant_text = retrieve_with_memory(index, bm25, text_chunks, query)
    prompt = f"Based on the following information:\n{relevant_text}\nAnswer the query: {query}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    output_tokens = model.generate(
        **inputs,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_beams=1,
        do_sample=True,
        use_cache=True
    )

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ------------------- Example Usage -------------------
input_folder = './input_folder'
index, text_chunks, bm25 = process_files(input_folder)

query = "What were the total assets in 2023?"
response = generate_answer(index, bm25, text_chunks, query)
print("Generated Response:", response)

query = "What were the total revenue in 2023?"
response = generate_answer(index, bm25, text_chunks, query)
print("Generated Response:", response)
