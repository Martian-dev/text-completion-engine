from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import numpy as np
import os
import json
import google.generativeai as genai

# metrics
import psutil
import sys
import pathlib

app = Flask(__name__)

# loading required environment variables
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# Configuration
DOCS_PATH = os.getenv("DOCS_PATH")
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# configuring gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

# In-memory data
index = None
metadata = []

# Util: Chunk a document


def chunk_text(text, max_len=300):
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


# Util: metrics
def get_deep_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(
            [get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items()]
        )
    elif hasattr(obj, "__dict__"):
        size += get_deep_size(vars(obj), seen)
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_deep_size(i, seen) for i in obj)
    return size


def get_folder_size(path):
    total = 0
    for file in pathlib.Path(path).rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def log_system_metrics():
    process = psutil.Process(os.getpid())
    total_rss = process.memory_info().rss
    metadata_size = get_deep_size(metadata)
    embedding_model_size = get_folder_size(MODEL_NAME)
    faiss_size = index.ntotal * index.d * 4  # float32 = 4 bytes

    with open("metrics.log", "a") as f:
        f.write("=== System Metrics ===\n")
        f.write(f"Total RAM (rss): {total_rss / (1024**2):.2f} MB\n")
        f.write(f"FAISS Index Size: {faiss_size / (1024**2):.2f} MB\n")
        f.write(f"Metadata Size: {metadata_size / (1024**2):.2f} MB\n")
        f.write(f"Embedding Model Size: {embedding_model_size / (1024**2):.2f} MB\n")
        f.write("=======================\n\n")


# Step 1: Prepare documents and embed


def prepare_documents():
    global index, metadata

    if not os.path.exists(DOCS_PATH):
        print(f"Documents path {DOCS_PATH} does not exist.")
        return

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        print("Loading saved index and metadata...")
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

    else:
        print("Reading and embedding documents...")
        all_chunks = []
        metadata = []

        for path in Path(DOCS_PATH).rglob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(text, CHUNK_SIZE)
            all_chunks.extend(chunks)
            metadata.extend([{"file": str(path), "text": chunk} for chunk in chunks])

        embeddings = model.encode(all_chunks)
        print("Embeddings shape:", embeddings.shape)  # Add this line to debug

        np.save(EMBEDDINGS_FILE, embeddings)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("Index built.")

    log_system_metrics()


# Step 2: Search and complete using Ollama


def query_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error querying Gemini:", e)
        return "[Gemini API Error]"


# API: Completion endpoint


@app.route("/complete", methods=["POST"])
def complete():
    data = request.get_json()
    query = data.get("text", "")

    if not query:
        return jsonify({"error": "Missing 'text'"}), 400

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    top_chunks = [metadata[i]["text"] for i in I[0]]

    # prompt = (
    #     f"User typed: {query}...\nRelevant Info:\n- "
    #     + "\n- ".join(top_chunks)
    #     + "\n\nInstructions: Continue the user's sentence (User typed) in 15 words or less based on the Relevant Info provided. if the user's sentence (User typed) has an incomplete word at the end complete that word as well and continue your completion. give only the completed sentence, that too from the exact place the user ended.\n"
    # )
    prompt = (
        f"You are an intelligent writing assistant.\n\n"
        f"User typed:\n{query}\n\n"
        f"Relevant Context:\n" + "\n- ".join(top_chunks) + "\n\n"
        "Instructions:\n"
        "1. Continue the user's input **from exactly where it ends**.\n"
        "2. If the last word is incomplete, complete it naturally before continuing.\n"
        "3. Keep the total completion to **15 words or fewer**.\n"
        "4. Only return the completed sentence—**no extra commentary or explanations**.\n"
    )

    completion = query_gemini(prompt)

    return jsonify({"completion": completion})


# Entry point

if __name__ == "__main__":
    prepare_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)
