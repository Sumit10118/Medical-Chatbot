from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss
import pickle
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load configuration from environment variables
INDEX_FILENAME = os.getenv("INDEX_FILENAME", "faiss_index_file.index")
DOCSTORE_FILENAME = os.getenv("DOCSTORE_FILENAME", "faiss_docstore.pkl")
RESULTS_COUNT = int(os.getenv("RESULTS_COUNT", 2))  # Default to 2 results

# Step 1: Load embedding model
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info("HuggingFace embeddings model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading HuggingFace embeddings model: {e}")
    exit()

# Step 2: Load FAISS index and document store
if not os.path.exists(INDEX_FILENAME) or not os.path.exists(DOCSTORE_FILENAME):
    logging.error("FAISS index or document store files are missing!")
    exit()

try:
    logging.info("Loading existing FAISS index from disk...")
    loaded_index = faiss.read_index(INDEX_FILENAME)

    with open(DOCSTORE_FILENAME, 'rb') as f:
        docstore, index_to_docstore_id = pickle.load(f)
        logging.info("Pickled document store loaded successfully.")

    faiss_index = FAISS(
        embeddings.embed_query,
        loaded_index,
        docstore,
        index_to_docstore_id
    )
    logging.info("FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FAISS index or document store: {e}")
    exit()

# Step 3: Define health-check route
@app.route('/health', methods=['GET'])
def health_check():
    try:
        if loaded_index and docstore:
            return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logging.error(f"Health check failed: {e}")
    return jsonify({"status": "unhealthy"}), 500

# Step 4: Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Step 5: Define route for querying the index
@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query')

    # Check if the query is empty
    if not query_text or len(query_text.strip()) == 0:
        return jsonify({"error": "Query text cannot be empty"}), 400

    try:
        logging.info(f"Received query: {query_text}")
        docs = faiss_index.similarity_search(query_text, k=RESULTS_COUNT)
        results = [doc.page_content for doc in docs]

        if not results:
            logging.info("No results found for the query.")
            return jsonify({"error": "No relevant information found"}), 404

        return jsonify({"results": results})

    except Exception as e:
        logging.error(f"Error during query: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
