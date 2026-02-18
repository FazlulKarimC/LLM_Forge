"""
Build RAG Index

One-time script to:
1. Load articles from knowledge base
2. Chunk articles
3. Generate embeddings via HF Inference API
4. Upsert vectors into Qdrant Cloud

Usage:
    cd backend
    python scripts/build_index.py

Requires:
    - HF_TOKEN env var set
    - QDRANT_URL and QDRANT_API_KEY in .env
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.services.rag_service import (
    ChunkingService,
    EmbeddingService,
    QdrantStore,
)


def main():
    print("=" * 60)
    print("RAG Index Builder")
    print("=" * 60)

    # 1. Load articles
    articles_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "articles.json"
    )
    articles_path = os.path.normpath(articles_path)

    print(f"\n[1/4] Loading articles from: {articles_path}")
    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  Loaded {len(articles)} articles")

    # 2. Chunk articles
    print(f"\n[2/4] Chunking articles (chunk_size=256, overlap=50)...")
    chunks = ChunkingService.chunk_articles(articles, chunk_size=256, overlap=50)
    print(f"  Created {len(chunks)} chunks")

    # 3. Generate embeddings
    print(f"\n[3/4] Generating embeddings via HF API...")
    print(f"  Model: sentence-transformers/all-MiniLM-L6-v2")
    embedder = EmbeddingService()

    # Batch embed (API handles batching internally, but do smaller batches to be safe)
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c.text for c in chunks[i : i + batch_size]]
        batch_embs = embedder.embed(batch_texts)
        all_embeddings.append(batch_embs)
        print(f"  Embedded batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
        time.sleep(0.5)  # Rate limit courtesy

    import numpy as np
    embeddings = np.vstack(all_embeddings)
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Upsert into Qdrant
    print(f"\n[4/4] Upserting to Qdrant Cloud...")
    store = QdrantStore()
    store.ensure_collection(vector_size=embeddings.shape[1])
    store.upsert_chunks(chunks, embeddings)

    count = store.count()
    print(f"  Qdrant collection now has {count} vectors")

    print(f"\n{'=' * 60}")
    print(f"Index build complete!")
    print(f"  Articles: {len(articles)}")
    print(f"  Chunks:   {len(chunks)}")
    print(f"  Vectors:  {count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
