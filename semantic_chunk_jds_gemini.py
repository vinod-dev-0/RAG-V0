import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
# Import for Google Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os # Make sure os is imported

# Load environment variables from .env file
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = "\n".join([doc.page_content for doc in documents])
    if not full_text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
    return full_text

def semantic_chunk_text(text, embeddings_model):
    """Chunks the given text into smaller pieces using SemanticChunker."""
    if not text.strip():
        print("Warning: Attempting to chunk empty text.")
        return []
    
    text_splitter = SemanticChunker(
        embeddings_model,
        breakpoint_threshold_type="percentile" 
    )
    chunks = text_splitter.split_text(text)
    print(f"Text semantically chunked into {len(chunks)} pieces using Gemini embeddings.")
    return chunks

def main():
    pdf_folder_path = r"d:\learn\ai-things\v0\Job_Description"
    output_chunks_folder = r"d:\learn\ai-things\v0\Job_Description_Semantic_Chunks_Gemini"

    # Retrieve the API key from the environment variable
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please ensure your GEMINI_API_KEY is set correctly in your .env file.")
        return

    try:
        # Pass the API key explicitly to the constructor
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=gemini_api_key
        )
    except Exception as e:
        print(f"Error initializing Google Generative AI Embeddings: {e}")
        print("Please ensure your GEMINI_API_KEY is valid and the Generative Language API is enabled.")
        return

    if not os.path.exists(output_chunks_folder):
        os.makedirs(output_chunks_folder)
        print(f"Created directory: {output_chunks_folder}")

    processed_files_count = 0

    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Folder not found at {pdf_folder_path}")
        return

    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            try:
                extracted_text = extract_text_from_pdf(pdf_path)
                if extracted_text.strip():
                    processed_files_count += 1
                    
                    chunks = semantic_chunk_text(extracted_text, embeddings)
                    
                    if chunks:
                        output_txt_filename = f"{os.path.splitext(filename)[0]}_semantic_gemini_chunks.txt"
                        output_txt_path = os.path.join(output_chunks_folder, output_txt_filename)
                        
                        with open(output_txt_path, 'w', encoding='utf-8') as f:
                            for i, chunk in enumerate(chunks):
                                f.write(f"--- Semantic Chunk (Gemini) {i+1} ---\n")
                                f.write(chunk)
                                f.write("\n\n--- End of Semantic Chunk (Gemini) ---\n\n")
                        print(f"Semantic chunks (Gemini) for {filename} saved to {output_txt_path}")
                    else:
                        print(f"No semantic chunks generated for {filename} using Gemini.")
                else:
                    print(f"No text to chunk for {filename}.")

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    print(f"\nSuccessfully processed {processed_files_count} PDF files using Gemini semantic chunking.")
    print(f"Gemini semantic chunked text files are saved in: {output_chunks_folder}")

if __name__ == "__main__":
    main()