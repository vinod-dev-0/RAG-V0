# Document Processing Pipeline

## Overview
This project implements an end-to-end document processing pipeline for job descriptions, leveraging AI to extract, chunk, embed, query, and generate standardized content from PDF documents.

## Workflow

### 1. PDF Ingestion & Text Extraction
- **Input**: Job description PDFs stored in `Job_Description/` directory
- **Tool**: PyPDFLoader from langchain_community
- **Process**: Extracts raw text from PDFs while preserving document structure
- **Implementation**: Found in both `chunk_jds.py` and `semantic_chunk_jds_gemini.py`

### 2. Text Chunking
The project offers two different chunking approaches:

#### A. Character-Based Chunking
- **Implementation**: `chunk_jds.py`
- **Method**: RecursiveCharacterTextSplitter
- **Configuration**:
  - Chunk size: 1000 characters
  - Overlap: 200 characters
- **Output**: Chunked text files stored in `Job_Description_Chunks/`

#### B. Semantic Chunking
- **Implementation**: `semantic_chunk_jds_gemini.py`
- **Method**: SemanticChunker with Google's Gemini embeddings
- **Configuration**:
  - Uses breakpoint_threshold_type="percentile"
  - Dynamically determines chunk boundaries based on semantic meaning
- **Output**: Semantically chunked text files stored in `Job_Description_Semantic_Chunks_Gemini/`

### 3. Vector Embedding Generation
- **Implementation**: `embed_and_query_chroma.py`
- **Process**: Converts text chunks into vector embeddings
- **Model**: Google's Generative AI Embeddings (Gemini)
  - Model name: `models/embedding-001`
- **Storage**: ChromaDB vector database
  - Persistent directory: `chroma_db_basic_chunks/`
  - Collection name: `jd_basic_chunks`
- **Metadata**: Each embedding stores:
  - Source PDF filename
  - Chunk index
  - Original text content

### 4. Embedding Visualization
- **Implementation**: `visualize_embeddings.py`
- **Process**: Dimensionality reduction using t-SNE algorithm
- **Configuration**:
  - Reduces to 2 dimensions for visualization
  - Configurable parameters for perplexity, iterations, and learning rate
- **Output**: 2D scatter plot showing relationships between document chunks
  - Color-coded by source document
  - Saved as `jd_embeddings_tsne_visualization.png`

### 5. Query & Generation
- **Implementation**: `generate_standardized_jd.py`
- **Functionality**:
  - Interactive mode: Collects job requirements through user prompts
  - Batch mode: Processes predefined queries
- **Process**:
  1. Converts user query to embedding vector
  2. Performs similarity search against ChromaDB
  3. Retrieves most relevant document chunks
  4. Generates standardized job description using retrieved context
- **Template**: Structured HR-friendly job description format
- **Output**: Standardized job descriptions saved to `outputs/` directory with timestamps

## Technical Stack

### Core Technologies
- **Language**: Python 3.x
- **Environment**: Windows with virtualenv
- **Vector Database**: ChromaDB
- **Embedding Model**: Google Generative AI (Gemini)

### Key Libraries
- **Document Processing**: langchain_community (PyPDFLoader)
- **Text Splitting**: 
  - langchain.text_splitter (RecursiveCharacterTextSplitter)
  - langchain_experimental.text_splitter (SemanticChunker)
- **Embeddings**: langchain_google_genai (GoogleGenerativeAIEmbeddings)
- **Visualization**: sklearn (TSNE), matplotlib, seaborn
- **Data Handling**: pandas, numpy

### Data Flow
1. PDF files → Text extraction
2. Raw text → Chunking (character-based or semantic)
3. Chunks → Vector embeddings
4. Embeddings → ChromaDB storage
5. User query → Relevant chunk retrieval
6. Retrieved chunks → Standardized job description generation

## Usage

1. Run `chunk_jds.py` or `semantic_chunk_jds_gemini.py` to process PDFs
2. Run `embed_and_query_chroma.py` to generate and store embeddings
3. Run `visualize_embeddings.py` to create a visual representation of embeddings
4. Run `generate_standardized_jd.py` to create standardized job descriptions

## Output
- Chunked text files in `Job_Description_Chunks/` or `Job_Description_Semantic_Chunks_Gemini/`
- Vector database in `chroma_db_basic_chunks/`
- Embedding visualization in `jd_embeddings_tsne_visualization.png`
- Generated job descriptions in `outputs/`
        Too many current requests. Your queue position is 1. Please wait for a while or switch to other models for a smoother experience.