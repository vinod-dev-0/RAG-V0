import os
import chromadb
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CHROMA_PERSIST_DIR = "chroma_db_basic_chunks"
CHROMA_COLLECTION_NAME = "jd_basic_chunks" 

# --- For t-SNE Visualization ---
# You might need to adjust these based on your data size and desired plot
TSNE_N_COMPONENTS = 2  # Reduce to 2 dimensions for a 2D plot
TSNE_PERPLEXITY = 30   # Typical values: 5-50. Relates to local neighborhood size.
TSNE_N_ITER = 1000     # Number of iterations for optimization.
TSNE_LEARNING_RATE = 200 # Typical values: 10-1000

def main():
    logging.info("Starting embedding visualization process...")

    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logging.info(f"ChromaDB client initialized from: {CHROMA_PERSIST_DIR}")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client: {e}")
        return

    # Get the collection
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        logging.info(f"Successfully accessed ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Failed to get ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}. Ensure the collection exists and was populated by 'embed_and_query_chroma.py' first.")
        return

    # Fetch all embeddings and metadata from the collection
    # Note: For very large collections, fetching everything might be memory-intensive.
    # You might fetch a sample or implement pagination if needed.
    try:
        results = collection.get(include=['embeddings', 'metadatas', 'documents'])
        embeddings = results.get('embeddings')
        metadatas = results.get('metadatas')
        documents_text = results.get('documents')
        ids = results.get('ids')

        if embeddings is None or len(embeddings) == 0:
            logging.warning("No embeddings found in the collection. Cannot visualize.")
            return
        
        logging.info(f"Fetched {len(embeddings)} embeddings from the collection.")
        embeddings_array = np.array(embeddings)

        # Move perplexity adjustment before logging
        current_perplexity = TSNE_PERPLEXITY
        if len(embeddings_array) < current_perplexity:
            current_perplexity = max(1, len(embeddings_array) - 1)
            logging.warning(f"Number of samples ({len(embeddings_array)}) is less than t-SNE perplexity ({TSNE_PERPLEXITY}). Adjusting perplexity to {current_perplexity}.")
            if current_perplexity == 0:
                logging.error("Cannot run t-SNE with 0 perplexity (not enough samples). Exiting.")
                return

        # Now use current_perplexity in the logging and t-SNE
        logging.info("Performing t-SNE reduction to 2 dimensions...")
        logging.info(f"t-SNE parameters: perplexity={current_perplexity}, n_iter={TSNE_N_ITER}, learning_rate={TSNE_LEARNING_RATE}")

        tsne = TSNE(n_components=TSNE_N_COMPONENTS, 
                    perplexity=current_perplexity,
                    n_iter=TSNE_N_ITER, 
                    learning_rate=TSNE_LEARNING_RATE,
                    random_state=42) # for reproducibility
        reduced_embeddings = tsne.fit_transform(embeddings_array)
        logging.info("t-SNE reduction complete.")
    except Exception as e:
        logging.error(f"Error during t-SNE reduction: {e}")
        return

    # Create a Pandas DataFrame for easier plotting with Seaborn/Matplotlib
    df_viz = pd.DataFrame({
        'tsne-2d-one': reduced_embeddings[:,0],
        'tsne-2d-two': reduced_embeddings[:,1],
        'id': ids,
        'source_file': [meta.get('source', 'N/A') if meta else 'N/A' for meta in metadatas],
        'chunk_index': [meta.get('chunk_index', 'N/A') if meta else 'N/A' for meta in metadatas],
        'text_preview': [doc[:50] + '...' if doc else '' for doc in documents_text] # First 50 chars for preview
    })

    # Add numerical values for plotting
    df_viz['plot_number'] = pd.factorize(df_viz['source_file'])[0]

    # Create a color map for the different source files
    unique_files = df_viz['source_file'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_files)))
    color_dict = dict(zip(range(len(unique_files)), colors))
    
    # Plot the 2D embeddings
    logging.info("Generating plot...")
    plt.figure(figsize=(16, 10))
    
    # Create scatter plot with colored numbers
    for i, txt in enumerate(df_viz['plot_number']):
        plt.scatter(df_viz['tsne-2d-one'].iloc[i], 
                    df_viz['tsne-2d-two'].iloc[i],
                    marker=f"${txt}$",  # Use the number as the marker
                    s=150,  # Increased size of the markers
                    c=[color_dict[txt]],  # Use color from our color map
                    alpha=0.8  # Slight transparency for better visibility
        )

    plt.title('t-SNE visualization of Job Description Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Create legend mapping numbers to source files
    unique_files = df_viz['source_file'].unique()
    legend_elements = [plt.Line2D([0], [0], marker=f"${i}$", color='w',
                                 markerfacecolor='black', markersize=10,
                                 label=f"{i}: {file}")  # Add number prefix to each label
                      for i, file in enumerate(unique_files)]
    
    # Add legend with original file names and ensure all entries are visible
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
              loc=2, borderaxespad=0., title='Source File',
              fontsize='small')  # Adjust font size if needed
    if len(unique_files) > 15:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                  loc=2, borderaxespad=0., title='Source File')
    else:
        plt.legend(handles=legend_elements, title='Source File')

    plt.tight_layout()
    plot_filename = "jd_embeddings_tsne_visualization.png"
    plt.savefig(plot_filename)
    logging.info(f"Plot saved as {plot_filename}")
    plt.show() # Display the plot

    logging.info("Visualization process finished.")

if __name__ == "__main__":
    main()