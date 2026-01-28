# src/embedder.py
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import CORPUS_DIR, EMBED_MODEL, EMBED_CACHE

def build_doc_embeddings():
    """
    Precompute and cache document embeddings for fast semantic reranking.
    Saves compressed .npz with 'ids' and 'embeddings' arrays to EMBED_CACHE.
    """
    corpus_path = CORPUS_DIR / "articles.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}. Run: python3 -m src.main --step crawl")

    df = pd.read_csv(corpus_path).fillna("")
    # Concatenate title + body as the semantic representation
    texts = (df["title"].astype(str) + " " + df["body"].astype(str)).tolist()

    # Load a lightweight, fast sentence transformer (cosine-normalized)
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, ids=np.arange(len(df)), embeddings=embs)
    print(f"[EMBED] Saved {embs.shape[0]} document embeddings -> {EMBED_CACHE}")

def load_doc_embeddings():
    """
    Load cached document embeddings.
    Returns: (ids_np, embeddings_np)
    """
    data = np.load(EMBED_CACHE)
    return data["ids"], data["embeddings"]

def embed_query(texts):
    """
    Encode one or more queries to normalized embeddings (numpy).
    """
    model = SentenceTransformer(EMBED_MODEL)
    q_embs = model.encode(
        texts,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return q_embs
