# ============================================================
# embeddings.py
# JOB: Load MiniLM model ONCE and convert any text to a
#      384-dimensional vector (list of 384 numbers)
# Used by: ingest.py (to embed documents)
#          retrieval.py (to embed farmer queries)
# ============================================================

# SentenceTransformer is the library that gives us
# pre-trained AI models for converting text to vectors
from sentence_transformers import SentenceTransformer

# numpy comes with sentence-transformers
# we use it to convert model output to a plain Python list
import numpy as np

# ── Step 1: Load the model ONCE at import time ───────────────
# This line runs ONCE when server starts
# NOT every time a query comes in
# paraphrase-MiniLM-L6-v2 is our chosen model because:
#   - completely free
#   - works on CPU (no GPU needed)
#   - fast inference
#   - outputs exactly 384 dimensions
#   - works well with pgvector cosine similarity
# First run will auto-download model (~90MB) from internet
# After first download it is cached locally on your machine
print("⏳ Loading MiniLM embedding model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("✅ MiniLM model loaded successfully")

# ── Step 2: get_embedding function ──────────────────────────
# INPUT:  any string of text (chunk_text or farmer query)
# OUTPUT: list of 384 float numbers (the vector)
# This function is what ingest.py and retrieval.py both call
def get_embedding(text: str) -> list:

    # basic safety check
    # if empty text is passed return a zero vector
    # this prevents crashes from empty strings
    if not text or text.strip() == "":
        print("⚠️  Warning: empty text passed to get_embedding")
        return [0.0] * 384          # return 384 zeros as fallback

    # clean the text before embedding
    # strip removes leading and trailing whitespace
    # replace removes extra newlines that confuse the model
    cleaned_text = text.strip().replace("\n", " ")

    # encode() is the core function that converts text to vector
    # model.encode() returns a numpy array of 384 float numbers
    # convert_to_numpy=True ensures output is always numpy array
    embedding = model.encode(
        cleaned_text,
        convert_to_numpy=True
    )

    # pgvector needs a plain Python list not a numpy array
    # .tolist() converts numpy array [0.23, 0.11, ...]
    # into a regular Python list that can be stored in PostgreSQL
    return embedding.tolist()


# ── Step 3: test function ────────────────────────────────────
# Run this to confirm model is working correctly
# Checks that output is exactly 384 numbers long
def test_embedding():
    sample_text = "wheat yellow rust fungal disease Punjab Pakistan"
    result = get_embedding(sample_text)

    # confirm it returned a list
    print(f"Output type  : {type(result)}")

    # confirm it has exactly 384 numbers
    print(f"Vector length: {len(result)}")

    # print first 5 numbers so you can see what a vector looks like
    print(f"First 5 values: {result[:5]}")

    if len(result) == 384:
        print("✅ Embedding model working correctly")
    else:
        print("❌ Embedding dimension mismatch — check model")


# ── Step 4: run test when file is executed directly ──────────
# Run: python embeddings.py
# to confirm model loads and produces correct output
if __name__ == "__main__":
    test_embedding()
