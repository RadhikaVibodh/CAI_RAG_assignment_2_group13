import os
import faiss
import pdfplumber
import re
import numpy as np
import torch
import multiprocessing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from collections import defaultdict
from joblib import Parallel, delayed

# ------------------- Step 1: Extract & Clean Text -------------------
def extract_text_from_pdf(pdf_path):
    """Extracts plain text from a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return clean_text(text)

def clean_text(text):
    """Removes unwanted characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def extract_tables_from_pdf(pdf_path):
    """Extracts tables from a PDF using pdfplumber."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            tables.extend(extracted_tables)
    return tables  # Returns list of tables (list of lists)

def convert_table_to_text(table):
    """Convert a table (list of lists) into structured text format."""
    if not table or not table[0]:  # Ensure table has rows
        return ""

    headers = table[0]
    formatted_rows = []

    for row in table[1:]:
        if len(row) < len(headers):  # Ensure row has all columns
            row += ["N/A"] * (len(headers) - len(row))  # Fill missing columns
        structured_text = ", ".join(f"{headers[j]}: {row[j]}" for j in range(len(headers)))
        formatted_rows.append(structured_text)

    return "\n".join(formatted_rows)

# ------------------- Step 2: Load Models -------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
phi2_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(phi2_model)
model = AutoModelForCausalLM.from_pretrained(phi2_model, torch_dtype=torch.float16, device_map="auto")

# ------------------- Step 3: Indexing (FAISS + BM25) -------------------
def process_single_pdf(file_path):
    """Processes a single PDF and extracts text and tables."""
    text_chunks = []
    table_chunks = []
    chunk_embeddings = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    # Extract text
    pdf_text = extract_text_from_pdf(file_path)
    text_chunks.extend(text_splitter.split_text(pdf_text))

    # Extract & format tables
    tables = extract_tables_from_pdf(file_path)
    table_texts = [convert_table_to_text(table) for table in tables if table]
    table_chunks.extend(table_texts)

    print(f"Extracted {len(tables)} tables from {file_path}")  # Debugging

    # Encode embeddings
    chunk_embeddings.extend(embed_model.encode(text_chunks))
    table_embeddings = embed_model.encode(table_chunks) if table_chunks else []

    return text_chunks, chunk_embeddings, table_chunks, table_embeddings

def process_files_parallel(input_folder):
    """Parallel processing for multiple PDFs."""
    pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".pdf")]

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process_single_pdf)(file) for file in pdf_files)

    text_chunks, text_embeddings, table_chunks, table_embeddings = [], [], [], []
    for texts, embeds, tables, tbl_embeds in results:
        text_chunks.extend(texts)
        text_embeddings.extend(embeds)
        table_chunks.extend(tables)
        table_embeddings.extend(tbl_embeds)

    # Convert to NumPy for FAISS
    text_embeddings = np.array(text_embeddings, dtype=np.float32)

    if table_embeddings:  # Ensure we have tables before creating FAISS index
        table_embeddings = np.array(table_embeddings, dtype=np.float32)
        table_index = faiss.IndexFlatL2(table_embeddings.shape[1])
        table_index.add(table_embeddings)
    else:
        print("No tables found, skipping FAISS table indexing.")
        table_index = None  # Avoid FAISS error

    index = faiss.IndexFlatL2(text_embeddings.shape[1])
    index.add(text_embeddings)

    # BM25 Indexing
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

    return index, table_index, text_chunks, table_chunks, bm25

# ------------------- Step 4: Multi-Stage Retrieval -------------------
def query_expansion(query):
    """Basic query expansion using synonyms or LLMs (placeholder)."""
    return query + " revenue profit loss assets liabilities"

def retrieve_top_k(index, table_index, bm25, text_chunks, table_chunks, query, k=10):
    query = query_expansion(query)
    query_embedding = embed_model.encode([query]).astype(np.float32)

    # FAISS Search
    faiss_distances, faiss_indices = index.search(query_embedding, k)

    # BM25 Search
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]

    combined_results = defaultdict(float)
    for i, d in zip(faiss_indices[0], faiss_distances[0]):
        combined_results[i] += (1 / (1 + d))

    for i in bm25_top_indices:
        combined_results[i] += bm25_scores[i]

    top_candidates = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:k]

    ranked_results = reranker.predict([(query, text_chunks[i]) for i, _ in top_candidates])
    ranked_indices = [i for i, _ in sorted(zip([i for i, _ in top_candidates], ranked_results), key=lambda x: x[1], reverse=True)]

    final_text = " ".join([text_chunks[i] for i in ranked_indices[:5]])
    return final_text

# ------------------- Step 5: Generate Answer -------------------
def generate_answer(index, table_index, bm25, text_chunks, table_chunks, query):
    relevant_text = retrieve_top_k(index, table_index, bm25, text_chunks, table_chunks, query)

    structured_response = f"Based on:\n{relevant_text}"

    prompt = f"Based on the following information:\n{structured_response}\n Answer the query: {query}. Answer briefly and if you do not know, reply, you cannot infer the answer from the document."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"max token lenght {tokenizer.model_max_length}")  # Usually 2048 for GPT-based models

    output_tokens = model.generate(
        **inputs,
        temperature=0.7,
        top_k=1,
        top_p=0.9,
        num_beams=1,
        do_sample=True,
        use_cache=True,
        max_length=512)


    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ------------------- Example Usage -------------------
input_folder = './input_folder'
index, table_index, text_chunks, table_chunks, bm25 = process_files_parallel(input_folder)

query = "What are the performance highlights of the company?"
response = generate_answer(index, table_index, bm25, text_chunks, table_chunks, query)
print("Generated Response:", response)

print("----------------------------------------------------------------------------------------------------")
query = "What is the revenue of the company?"
response = generate_answer(index, table_index, bm25, text_chunks, table_chunks, query)
print("Generated Response:", response)


print("----------------------------------------------------------------------------------------------------")
query = "Who are the board of directors of the company?"
response = generate_answer(index, table_index, bm25, text_chunks, table_chunks, query)
print("Generated Response:", response)


print("----------------------------------------------------------------------------------------------------")
query = "Who is tom cruise?"
response = generate_answer(index, table_index, bm25, text_chunks, table_chunks, query)
print("Generated Response:", response)
