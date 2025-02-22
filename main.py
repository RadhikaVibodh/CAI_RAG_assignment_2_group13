#Data Collection & Preprocessing
import os  # Add this line at the top of your script
import fitz  # PyMuPDF for PDF processing
import re  # Regular expressions for text cleaning
import pandas as pd  # Pandas for CSV processing

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
