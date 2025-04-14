from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import numpy as np
import os
import json

app = Flask(__name__)

# Configuration
DOCS_PATH = "Documents"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# In-memory data
index = None
metadata = []

# Util: Chunk a document


def chunk_text(text, max_len=300):
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


# Step 1: Prepare documents and embed


def prepare_documents():
    global index, metadata

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        print("Loading saved index and metadata...")
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return

    print("Reading and embedding documents...")
    all_chunks = []
    metadata = []

    for path in Path(DOCS_PATH).rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, CHUNK_SIZE)
        all_chunks.extend(chunks)
        metadata.extend([{"file": str(path), "text": chunk} for chunk in chunks])

    embeddings = model.encode(all_chunks)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Index built.")


# Step 2: Search and complete using Ollama

import requests


def query_ollama(prompt):
    try:
        response = requests.post(
            "http://<YOUR_LAPTOP_IP>:11434/api/generate",
            json={"model": "gemma:1b", "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        print("Error querying Ollama:", e)
        return "[LLM Error]"


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

    prompt = (
        f"User typed: {query}\nRelevant Info:\n- "
        + "\n- ".join(top_chunks)
        + "\n\nCompletion:"
    )
    completion = query_ollama(prompt)

    return jsonify({"completion": completion})


# Entry point

if __name__ == "__main__":
    prepare_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)
