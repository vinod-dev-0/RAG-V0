import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # loader.load() returns a list of Document objects, each Document might represent a page.
    # We can combine the text from all pages (Documents) into one string for now.
    full_text = "\n".join([doc.page_content for doc in documents])
    if not full_text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
    return full_text, documents # returning documents as well for page-wise context if needed later

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Chunks the given text into smaller pieces."""
    if not text.strip():
        print("Warning: Attempting to chunk empty text.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # The maximum size of each chunk (in characters)
        chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
        length_function=len,
        add_start_index=True, # Adds the start index of the chunk in the original document
    )
    chunks = text_splitter.split_text(text)
    print(f"Text chunked into {len(chunks)} pieces.")
    return chunks

def main():
    pdf_folder_path = r"d:\learn\ai-things\v0\Job_Description"
    output_chunks_folder = r"d:\learn\ai-things\v0\Job_Description_Chunks"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_chunks_folder):
        os.makedirs(output_chunks_folder)
        print(f"Created directory: {output_chunks_folder}")

    # Counter for processed files, to count the number of files processed
    processed_files_count = 0

    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Folder not found at {pdf_folder_path}")
        return

    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            try:
                extracted_text, _ = extract_text_from_pdf(pdf_path)
                if extracted_text.strip():
                    processed_files_count += 1
                    chunks = chunk_text(extracted_text)
                    
                    if chunks:
                        # Define the output file path for the chunks
                        output_txt_filename = f"{os.path.splitext(filename)[0]}_chunks.txt"
                        output_txt_path = os.path.join(output_chunks_folder, output_txt_filename)
                        
                        with open(output_txt_path, 'w', encoding='utf-8') as f:
                            for i, chunk in enumerate(chunks):
                                f.write(f"--- Chunk {i+1} ---\n")
                                f.write(chunk)
                                f.write("\n\n--- End of Chunk ---\n\n")
                        print(f"Chunks for {filename} saved to {output_txt_path}")
                    else:
                        print(f"No chunks generated for {filename}.")
                else:
                    print(f"No text to chunk for {filename}.")

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    print(f"\nSuccessfully processed {processed_files_count} PDF files.")
    print(f"Chunked text files are saved in: {output_chunks_folder}")

if __name__ == "__main__":
    main()