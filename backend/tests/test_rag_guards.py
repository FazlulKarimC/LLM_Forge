"""
Tests for RAG runtime guards.
"""

import json

import pytest

from app.services.rag_service import BM25Index, RAGPipeline


class DummyEmbedder:
    def embed_single(self, text):  # pragma: no cover - not used in this test
        raise AssertionError("embed_single should not be called during load_knowledge_base")


class DummyReranker:
    pass


class MissingCollectionStore:
    _collection = "missing_collection"

    def count(self, strict: bool = False):
        if strict:
            raise RuntimeError("Collection doesn't exist")
        return 0


class EmptyCollectionStore:
    _collection = "empty_collection"

    def count(self, strict: bool = False):
        return 0


def _write_articles(tmp_path):
    articles_path = tmp_path / "articles.json"
    articles_path.write_text(
        json.dumps([{"title": "Doc", "text": "alpha beta gamma delta"}]),
        encoding="utf-8",
    )
    return str(articles_path)


def test_load_knowledge_base_raises_clear_error_for_missing_qdrant_collection(tmp_path):
    pipeline = RAGPipeline(
        embedding_service=DummyEmbedder(),
        qdrant_store=MissingCollectionStore(),
        bm25_index=BM25Index(),
        reranker=DummyReranker(),
    )

    with pytest.raises(RuntimeError, match="Run `python scripts/build_index.py` before using RAG"):
        pipeline.load_knowledge_base(articles_path=_write_articles(tmp_path))


def test_load_knowledge_base_raises_clear_error_for_empty_qdrant_collection(tmp_path):
    pipeline = RAGPipeline(
        embedding_service=DummyEmbedder(),
        qdrant_store=EmptyCollectionStore(),
        bm25_index=BM25Index(),
        reranker=DummyReranker(),
    )

    with pytest.raises(RuntimeError, match="is empty"):
        pipeline.load_knowledge_base(articles_path=_write_articles(tmp_path))
