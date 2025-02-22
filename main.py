#Data Collection & Preprocessing
import os  # Add this line at the top of your script
import faiss
import fitz  # PyMuPDF for PDF processing
import re  # Regular expressions for text cleaning
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import pandas as pd  # Pandas for CSV processing
import numpy as np  # NumPy for array operations
''
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file and cleans it."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    return clean_text(text)

#def extract_text_from_csv(csv_path):
#   """Extracts text from a CSV file and cleans it."""
#  df = pd.read_csv(csv_path)  # Read CSV
# text = df.to_string(index=False)  # Convert entire DataFrame to a text format
    
#    return clean_text(text)

def clean_text(text):
    """Cleans extracted text by removing unnecessary symbols and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()
    return text

def process_files(input_folder, output_folder):
    """Processes all PDFs and CSVs in a folder and saves cleaned text."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename + ".txt")

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        #elif filename.endswith(".csv"):
         #   text = extract_text_from_csv(file_path)
        else:
            continue  # Skip unsupported files

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed: {filename} -> {output_file}")

# Example Usage
input_folder = r'C:\Users\swara\Downloads\BITS\sem_3\CAI\Assignment\assignment_2\input_folder'
output_folder = r'C:\Users\swara\Downloads\BITS\sem_3\CAI\Assignment\assignment_2\output_folder'
process_files(input_folder, output_folder)

#2. Basic RAG Implementation


# Load preprocessed text files from your output folder

text_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".txt")]

documents = []
for file in text_files:
    with open(file, "r", encoding="utf-8") as f:
        documents.append(f.read())

# Text chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text("\n".join(documents))
text_chunks = chunks  # Each element is a text chunk

# Generate embeddings using Ollama (nomic-embed-text)
def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

embeddings = np.array([get_embedding(chunk) for chunk in text_chunks])

# Store embeddings in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to retrieve top-k relevant chunks
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0]]

# Function to generate response using Ollama LLM
def generate_response(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n".join(retrieved_chunks)
    prompt = f"Based on the following information, answer the question:\n{context}\n\nQuestion: {query}"
   # response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

# Example query
query = "What were the total assets in 2023?"
response = generate_response(query)
print("Generated Response:", response)

