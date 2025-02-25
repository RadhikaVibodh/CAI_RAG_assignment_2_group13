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
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return clean_text(text)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            tables.extend(extracted_tables)
    return tables

def convert_table_to_text(table):
    if not table or not table[0]:
        return ""
    headers = table[0]
    formatted_rows = []
    for row in table[1:]:
        row += ["N/A"] * (len(headers) - len(row))
        structured_text = ", ".join(f"{headers[j]}: {row[j]}" for j in range(len(headers)))
        formatted_rows.append(structured_text)
    return "\n".join(formatted_rows)

# ------------------- Step 2: Load Models -------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
lang_model = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(lang_model, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(lang_model)

# ------------------- Step 3: Indexing (FAISS + BM25) -------------------
def process_single_pdf(file_path):
    text_chunks = []
    table_chunks = []
    chunk_embeddings = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    pdf_text = extract_text_from_pdf(file_path)
    text_chunks.extend(text_splitter.split_text(pdf_text))

    tables = extract_tables_from_pdf(file_path)
    table_texts = [convert_table_to_text(table) for table in tables if table]
    table_chunks.extend(table_texts)

    chunk_embeddings.extend(embed_model.encode(text_chunks))
    table_embeddings = embed_model.encode(table_chunks) if table_chunks else []

    return text_chunks, chunk_embeddings, table_chunks, table_embeddings

def process_files_parallel(input_folder):
    pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".pdf")]
    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process_single_pdf)(file) for file in pdf_files)

    text_chunks, text_embeddings, table_chunks, table_embeddings = [], [], [], []
    for texts, embeds, tables, tbl_embeds in results:
        text_chunks.extend(texts)
        text_embeddings.extend(embeds)
        table_chunks.extend(tables)
        table_embeddings.extend(tbl_embeds)

    text_embeddings = np.array(text_embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(text_embeddings.shape[1])
    index.add(text_embeddings)

    if table_embeddings:
        table_embeddings = np.array(table_embeddings, dtype=np.float32)
        table_index = faiss.IndexFlatL2(table_embeddings.shape[1])
        table_index.add(table_embeddings)
    else:
        table_index = None

    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])
    return index, table_index, text_chunks, table_chunks, bm25

# ------------------- Step 4: Multi-Stage Retrieval -------------------
def retrieve_top_k(index, table_index, bm25, text_chunks, query, k=10):
    query_embedding = embed_model.encode([query]).astype(np.float32)
    faiss_distances, faiss_indices = index.search(query_embedding, k)
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
    return " ".join([text_chunks[i] for i in ranked_indices[:5]])

# ------------------- Step 5: Generate Answer -------------------
def generate_answer(index, table_index, bm25, text_chunks, query):
    relevant_text = retrieve_top_k(index, table_index, bm25, text_chunks, query)
    prompt = f"Based on the following information:\n{relevant_text}\nAnswer the query: {query}. Answer briefly and if you do not know, reply, you cannot infer the answer from the document."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output_tokens = model.generate(**inputs, temperature=0.7, top_k=1, top_p=0.9, num_beams=1, do_sample=True, max_length=512)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ------------------- Run Once and Query Multiple Times -------------------
input_folder = './input_folder'
index, table_index, text_chunks, table_chunks, bm25 = process_files_parallel(input_folder)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = generate_answer(index, table_index, bm25, text_chunks, query)
    print("Generated Response:", response)
