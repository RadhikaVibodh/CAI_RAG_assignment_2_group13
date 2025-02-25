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
from joblib import Parallel, delayed
import onnxruntime as ort  # ONNX for faster CPU inference


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


# ------------------- Step 2: Load Models (Optimized for CPU) -------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # For ranking results

# Load Phi-2 as ONNX for better CPU inference
onnx_model_path = "phi2.onnx"

if not os.path.exists(onnx_model_path):
    print("Exporting Phi-2 model to ONNX format...")

    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example input text
    example_text = "This is a test input for ONNX export."
    inputs = tokenizer(example_text, return_tensors="pt")

    # Export ONNX model
    torch.onnx.export(
        model,
        (inputs["input_ids"],),
        onnx_model_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "logits": {0: "batch_size", 1: "sequence_length"}},
        opset_version=13
    )
    print(f"ONNX model saved at {onnx_model_path}")

# Load ONNX model
onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


# ------------------- Step 3: Indexing (FAISS + BM25) -------------------
def process_single_pdf(file_path):
    """Processes a single PDF and extracts text and tables."""
    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    # Extract text
    pdf_text = extract_text_from_pdf(file_path)
    text_chunks.extend(text_splitter.split_text(pdf_text))

    # Encode embeddings
    chunk_embeddings = embed_model.encode(text_chunks)

    return text_chunks, chunk_embeddings


def process_files_parallel(input_folder):
    """Parallel processing for multiple PDFs."""
    pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".pdf")]

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process_single_pdf)(file) for file in pdf_files)

    text_chunks, text_embeddings = [], []
    for texts, embeds in results:
        text_chunks.extend(texts)
        text_embeddings.extend(embeds)

    # Convert to NumPy for FAISS
    text_embeddings = np.array(text_embeddings, dtype=np.float32)

    # FAISS Indexing (Optimized with HNSW)
    index = faiss.IndexHNSWFlat(text_embeddings.shape[1], 32)
    index.add(text_embeddings)

    # BM25 Indexing
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])

    return index, text_chunks, bm25


# ------------------- Step 4: Retrieval -------------------
def retrieve_top_k(index, bm25, text_chunks, query, k=5):
    """Retrieve top k relevant text chunks using FAISS + BM25."""
    query_embedding = embed_model.encode([query]).astype(np.float32)

    # FAISS Search
    _, faiss_indices = index.search(query_embedding, k)

    # BM25 Search
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]

    # Combine FAISS and BM25 results
    combined_results = {i: bm25_scores[i] for i in bm25_top_indices}
    for i in faiss_indices[0]:
        combined_results[i] = combined_results.get(i, 0) + 1

    top_indices = sorted(combined_results.keys(), key=lambda x: combined_results[x], reverse=True)[:k]

    ranked_results = reranker.predict([(query, text_chunks[i]) for i in top_indices])
    ranked_indices = [i for i, _ in sorted(zip(top_indices, ranked_results), key=lambda x: x[1], reverse=True)]

    return " ".join([text_chunks[i] for i in ranked_indices[:k]])


# ------------------- Step 5: Generate Answer with ONNX -------------------
def generate_answer(index, bm25, text_chunks, query, top_k=5):
    """Generates a response using Phi-2 ONNX model."""
    relevant_text = retrieve_top_k(index, bm25, text_chunks, query, k=top_k)

    prompt = f"Based on the following information:\n{relevant_text}\n Answer the query: {query}. If you do not know, say 'I cannot infer the answer from the document'."
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run ONNX model with batch processing
    ort_inputs = {onnx_session.get_inputs()[0].name: inputs["input_ids"].cpu().numpy()}
    ort_outputs = onnx_session.run(None, ort_inputs)

    return tokenizer.decode(ort_outputs[0][0], skip_special_tokens=True)


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    input_folder = './input_folder'
    index, text_chunks, bm25 = process_files_parallel(input_folder)

    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        top_k = input("Enter top_k value (default 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5

        response = generate_answer(index, bm25, text_chunks, user_query, top_k)
        print(f"\nQuery: {user_query}\nGenerated Response: {response}\n")
        print("-" * 100)
