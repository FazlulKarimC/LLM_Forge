"""
RAG Service

Retrieval-Augmented Generation pipeline.
All model operations (embeddings, reranking, NLI) use HuggingFace Inference API.
Vector storage uses Qdrant Cloud (free tier).
Keyword search uses BM25 (rank-bm25, ~20KB dependency).

Architecture:
    Query → [Embed via HF API] → [Qdrant Dense Search] → [BM25 Sparse Search]
         → [Merge/Dedup] → [Rerank via HF API] → [Top-K Chunks]
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Chunk:
    """A text chunk from the knowledge base."""
    id: str
    text: str
    title: str
    index: int  # Position within article


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""
    chunks: List[Chunk]
    scores: List[float]
    method: str  # "dense", "bm25", "hybrid", "reranked"
    latency_ms: float = 0.0


# =============================================================================
# Chunking Service
# =============================================================================

class ChunkingService:
    """
    Splits articles into overlapping text chunks.
    
    Uses word-level splitting (not token-level) for simplicity.
    Each chunk retains metadata about its source article.
    """

    @staticmethod
    def chunk_articles(
        articles: List[Dict[str, str]],
        chunk_size: int = 256,
        overlap: int = 50,
    ) -> List[Chunk]:
        """
        Split articles into chunks with overlap.
        
        Args:
            articles: List of {"title": ..., "text": ...}
            chunk_size: Target chunk size in words
            overlap: Overlap between consecutive chunks in words
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_id = 0

        for article in articles:
            title = article["title"]
            words = article["text"].split()

            if len(words) <= chunk_size:
                # Article fits in one chunk
                all_chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    text=article["text"],
                    title=title,
                    index=0,
                ))
                chunk_id += 1
                continue

            # Sliding window chunking
            start = 0
            idx = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                all_chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    title=title,
                    index=idx,
                ))
                chunk_id += 1
                idx += 1

                if end == len(words):
                    break
                start += chunk_size - overlap

        logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} articles")
        return all_chunks


# =============================================================================
# Embedding Service (HF Inference API)
# =============================================================================

class EmbeddingService:
    """
    Generate embeddings via HuggingFace Inference API.
    
    Uses `feature_extraction` endpoint with sentence-transformers models.
    No local model download required.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or self._get_api_key()
        self._model = model or self.DEFAULT_MODEL
        self._client = InferenceClient(token=self._api_key)

    @staticmethod
    def _get_api_key() -> str:
        from app.core.config import settings
        key = settings.HF_TOKEN or os.getenv("HF_TOKEN", "")
        if not key:
            raise ValueError("HF_TOKEN required for embeddings")
        return key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        results = self._client.feature_extraction(
            text=texts,
            model=self._model,
        )

        # result could be nested list or np array
        embeddings = np.array(results)

        # Handle sentence-transformers output: may be (N, seq_len, dim) → mean pool
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=1)

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        result = self.embed([text])
        return result[0]


# =============================================================================
# Qdrant Store
# =============================================================================

class QdrantStore:
    """
    Vector store using Qdrant Cloud.
    
    Free tier: 1GB forever cluster, ~1M vectors.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        from qdrant_client import QdrantClient
        from app.core.config import settings

        self._url = url or settings.QDRANT_URL
        self._api_key = api_key or settings.QDRANT_API_KEY
        self._collection = collection_name or settings.QDRANT_COLLECTION_NAME

        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key if self._api_key else None,
        )

    def ensure_collection(self, vector_size: int = 384):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self._client.get_collections().collections]

        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection}")
        else:
            logger.info(f"Qdrant collection '{self._collection}' already exists")

    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Upsert chunks with their embeddings into Qdrant.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (len(chunks), embedding_dim)
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "title": chunk.title,
                    "index": chunk.index,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        # Batch upsert (100 at a time for free tier)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} vectors to Qdrant")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Returns:
            List of (Chunk, score) tuples, sorted by similarity
        """
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding.tolist(),
            limit=top_k,
        )

        chunks_with_scores = []
        for point in results.points:
            chunk = Chunk(
                id=point.payload["chunk_id"],
                text=point.payload["text"],
                title=point.payload["title"],
                index=point.payload["index"],
            )
            chunks_with_scores.append((chunk, point.score))

        return chunks_with_scores

    def count(self) -> int:
        """Get the number of vectors in the collection."""
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count
        except Exception:
            return 0

    def delete_collection(self):
        """Delete the collection."""
        self._client.delete_collection(self._collection)
        logger.info(f"Deleted Qdrant collection: {self._collection}")


# =============================================================================
# BM25 Index (Keyword Search)
# =============================================================================

class BM25Index:
    """
    BM25 keyword search index.
    
    Uses rank-bm25 library (~20KB dependency).
    Provides sparse retrieval to complement dense embeddings.
    """

    def __init__(self):
        self._index = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info(f"Built BM25 index with {len(chunks)} documents")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks by keyword similarity.
        
        Returns:
            List of (Chunk, score) tuples, sorted by BM25 score
        """
        if self._index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self._chunks[idx], float(scores[idx])))

        return results


# =============================================================================
# Cross-Encoder Reranker (HF Inference API)
# =============================================================================

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    
    Uses HF Inference API text_classification endpoint with
    a cross-encoder model to score (query, passage) pairs.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or self._get_api_key()
        self._model = model or self.DEFAULT_MODEL
        self._client = InferenceClient(token=self._api_key)

    @staticmethod
    def _get_api_key() -> str:
        from app.core.config import settings
        key = settings.HF_TOKEN or os.getenv("HF_TOKEN", "")
        if not key:
            raise ValueError("HF_TOKEN required for reranking")
        return key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def rerank(
        self, query: str, chunks: List[Chunk], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank chunks by relevance to query using cross-encoder.
        
        Args:
            query: The user's question
            chunks: Candidate chunks to rerank
            top_k: Number of top chunks to return
            
        Returns:
            List of (Chunk, score) tuples, sorted by relevance
        """
        if not chunks:
            return []

        scored_chunks = []
        for chunk in chunks:
            try:
                # Cross-encoder expects text pairs
                result = self._client.text_classification(
                    text=f"{query} [SEP] {chunk.text}",
                    model=self._model,
                )
                # Extract score — model returns list of labels with scores
                score = result[0].score if result else 0.0
                scored_chunks.append((chunk, score))
            except Exception as e:
                logger.warning(f"Reranking failed for chunk {chunk.id}: {e}")
                scored_chunks.append((chunk, 0.0))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


# =============================================================================
# Faithfulness Scorer (NLI via HF API)
# =============================================================================

class FaithfulnessScorer:
    """
    Measures faithfulness of generated answers to source context.
    
    Uses zero-shot classification via HF Inference API with
    facebook/bart-large-mnli for Natural Language Inference.
    
    Faithfulness = P(entailment | context → answer)
    Hallucination = faithfulness < 0.5
    """

    DEFAULT_MODEL = "facebook/bart-large-mnli"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or self._get_api_key()
        self._model = model or self.DEFAULT_MODEL
        self._client = InferenceClient(token=self._api_key)

    @staticmethod
    def _get_api_key() -> str:
        from app.core.config import settings
        key = settings.HF_TOKEN or os.getenv("HF_TOKEN", "")
        if not key:
            raise ValueError("HF_TOKEN required for faithfulness scoring")
        return key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def score(self, answer: str, context: str) -> float:
        """
        Compute faithfulness score.
        
        Args:
            answer: The generated answer
            context: The source context used for generation
            
        Returns:
            float 0-1 (higher = more faithful)
        """
        if not answer or not context:
            return 0.0

        # Truncate context to avoid API limits (max ~1024 tokens)
        context_truncated = " ".join(context.split()[:500])

        try:
            result = self._client.zero_shot_classification(
                text=context_truncated,
                candidate_labels=[answer],
                model=self._model,
            )
            # Result has scores for the candidate label
            return float(result.scores[0]) if result.scores else 0.0
        except Exception as e:
            logger.warning(f"Faithfulness scoring failed: {e}")
            return 0.0

    def is_hallucination(self, answer: str, context: str, threshold: float = 0.5) -> bool:
        """Check if an answer is likely hallucinated."""
        return self.score(answer, context) < threshold


# =============================================================================
# RAG Pipeline (Orchestrator)
# =============================================================================

class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.
    
    Supports 3 retrieval modes:
    - naive: Dense retrieval only (Qdrant cosine similarity)
    - hybrid: Dense + BM25, merged and deduplicated
    - reranked: Hybrid + cross-encoder reranking
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_store: Optional[QdrantStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.embedder = embedding_service or EmbeddingService()
        self.qdrant = qdrant_store or QdrantStore()
        self.bm25 = bm25_index or BM25Index()
        self.reranker = reranker or CrossEncoderReranker()

    def load_knowledge_base(self, articles_path: Optional[str] = None, chunk_size: int = 256):
        """Load and index the knowledge base (for BM25 at runtime)."""
        if articles_path is None:
            # rag_service.py → services/ → app/ → backend/ → LlmForge/
            project_root = Path(__file__).parent.parent.parent.parent
            articles_path = str(project_root / "data" / "knowledge_base" / "articles.json")

        with open(articles_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        chunks = ChunkingService.chunk_articles(articles, chunk_size=chunk_size)
        self.bm25.build(chunks)
        self._chunks = chunks
        logger.info(f"Knowledge base loaded: {len(chunks)} chunks")

    def retrieve(
        self,
        question: str,
        method: str = "naive",
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            question: The user's question
            method: One of "naive", "hybrid", "reranked"
            top_k: Number of chunks to return
            
        Returns:
            RetrievalResult with chunks and scores
        """
        start = time.time()

        if method == "naive":
            result = self._naive_retrieve(question, top_k)
        elif method == "hybrid":
            result = self._hybrid_retrieve(question, top_k)
        elif method == "reranked":
            result = self._reranked_retrieve(question, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        result.latency_ms = (time.time() - start) * 1000
        return result

    def _naive_retrieve(self, question: str, top_k: int) -> RetrievalResult:
        """Dense-only retrieval via Qdrant."""
        query_emb = self.embedder.embed_single(question)
        results = self.qdrant.search(query_emb, top_k=top_k)

        return RetrievalResult(
            chunks=[c for c, _ in results],
            scores=[s for _, s in results],
            method="dense",
        )

    def _hybrid_retrieve(self, question: str, top_k: int) -> RetrievalResult:
        """Dense + BM25 hybrid retrieval."""
        # Get more candidates from each source, then merge
        candidate_k = top_k * 2

        # Dense retrieval
        query_emb = self.embedder.embed_single(question)
        dense_results = self.qdrant.search(query_emb, top_k=candidate_k)

        # BM25 retrieval
        bm25_results = self.bm25.search(question, top_k=candidate_k)

        # Merge and deduplicate by chunk text
        seen_texts = set()
        merged = []

        # Add dense results first (higher priority)
        for chunk, score in dense_results:
            if chunk.text not in seen_texts:
                seen_texts.add(chunk.text)
                merged.append((chunk, score))

        # Add BM25 results (normalize scores to [0, 1])
        max_bm25 = max((s for _, s in bm25_results), default=1.0) or 1.0
        for chunk, score in bm25_results:
            if chunk.text not in seen_texts:
                seen_texts.add(chunk.text)
                normalized_score = score / max_bm25
                merged.append((chunk, normalized_score))

        # Sort by score and take top_k
        merged.sort(key=lambda x: x[1], reverse=True)
        merged = merged[:top_k]

        return RetrievalResult(
            chunks=[c for c, _ in merged],
            scores=[s for _, s in merged],
            method="hybrid",
        )

    def _reranked_retrieve(self, question: str, top_k: int) -> RetrievalResult:
        """Hybrid + cross-encoder reranking."""
        # Get hybrid candidates
        hybrid = self._hybrid_retrieve(question, top_k=top_k * 2)

        # Rerank
        reranked = self.reranker.rerank(question, hybrid.chunks, top_k=top_k)

        return RetrievalResult(
            chunks=[c for c, _ in reranked],
            scores=[s for _, s in reranked],
            method="reranked",
        )
