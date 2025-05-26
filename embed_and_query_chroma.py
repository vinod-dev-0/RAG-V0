import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHUNKS_DIR = "Job_Description_Chunks"
CHROMA_PERSIST_DIR = "chroma_db_basic_chunks"
CHROMA_COLLECTION_NAME = "jd_basic_chunks"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    exit()

def load_chunks_from_directory(directory_path):
    """Loads all text chunks from .txt files in the specified directory."""
    all_docs = []
    logging.info(f"Loading chunks from directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return all_docs
    
    for filename in os.listdir(directory_path):
        if filename.endswith("_chunks.txt"):  # Only process files ending with _chunks.txt
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split content by the chunk separators used in chunk_jds.py
                    # Chunks are separated by "--- End of Chunk ---"
                    # and each chunk starts with "--- Chunk X ---"
                    raw_chunks = content.split("--- End of Chunk ---")
                    
                    chunk_counter = 0
                    for raw_chunk in raw_chunks:
                        if raw_chunk.strip():  # Ensure chunk is not empty
                            # Clean up the chunk by removing the header
                            lines = raw_chunk.strip().split('\n')
                            chunk_text = ""
                            
                            # Skip the "--- Chunk X ---" line if it exists
                            start_idx = 0
                            if lines and lines[0].startswith("--- Chunk ") and lines[0].endswith(" ---"):
                                start_idx = 1
                            
                            # Join the remaining lines
                            chunk_text = '\n'.join(lines[start_idx:]).strip()
                            
                            if chunk_text:
                                doc = Document(
                                    page_content=chunk_text,
                                    metadata={
                                        "source": filename.replace("_chunks.txt", ".pdf"),  # Original PDF filename
                                        "chunk_index": chunk_counter,
                                        "chunk_file": filename
                                    }
                                )
                                all_docs.append(doc)
                                chunk_counter += 1
                                
                logging.info(f"Successfully loaded {chunk_counter} chunks from {filename}")
                
            except Exception as e:
                logging.error(f"Error reading or processing file {file_path}: {e}")
    
    logging.info(f"Loaded a total of {len(all_docs)} chunks from directory.")
    return all_docs

def main():
    logging.info("Starting the embedding and ChromaDB storage process with basic chunking...")

    # Initialize Gemini Embeddings
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        logging.info("GoogleGenerativeAIEmbeddings initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
        return

    # Load text chunks
    documents = load_chunks_from_directory(CHUNKS_DIR)
    if not documents:
        logging.warning("No documents found or loaded. Exiting.")
        return

    # Initialize ChromaDB client and collection
    # Using a persistent client to save data to disk
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    logging.info(f"ChromaDB client initialized. Data will be persisted in: {CHROMA_PERSIST_DIR}")

    # Get or create the collection
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
        logging.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' accessed/created.")
    except Exception as e:
        logging.error(f"Failed to get or create ChromaDB collection: {e}")
        return

    # Prepare data for ChromaDB
    chunk_texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"{doc.metadata['source'].replace('.pdf', '')}_chunk_{doc.metadata['chunk_index']}" for doc in documents]

    # Generate embeddings for all documents
    logging.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
    try:
        doc_embeddings = embeddings_model.embed_documents(chunk_texts)
        logging.info("Embeddings generated successfully.")
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        return

    # Add documents, embeddings, and metadatas to the collection
    # This will overwrite if IDs already exist, or add new ones.
    # For large datasets, consider batching.
    try:
        collection.add(
            ids=ids,
            embeddings=doc_embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )
        logging.info(f"Successfully added/updated {len(ids)} documents in ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Failed to add documents to ChromaDB: {e}")
        return

    # --- Example Querying --- 
    logging.info("\n--- Performing a test query ---")
    query_text = "looking for a marketing manager with experience in digital campaigns"
    
    try:
        query_embedding = embeddings_model.embed_query(query_text)
        logging.info(f"Query text: '{query_text}'")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        logging.info("Query executed successfully.")

        logging.info("\nTop relevant chunks found:")
        if results and results.get('documents'):
            for i, doc_content in enumerate(results['documents'][0]):
                doc_metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                doc_distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else 'N/A'
                logging.info(f"  Result {i+1}:")
                logging.info(f"    Source: {doc_metadata.get('source', 'N/A')}, Chunk Index: {doc_metadata.get('chunk_index', 'N/A')}")
                logging.info(f"    Distance: {doc_distance}")
                logging.info(f"    Content: {doc_content[:300]}...")
                logging.info("  ---")
        else:
            logging.info("No results found for the query.")
            
    except Exception as e:
        logging.error(f"Error during querying: {e}")

    logging.info(f"Process finished. ChromaDB data is persisted in '{CHROMA_PERSIST_DIR}'.")

if __name__ == "__main__":
    main()