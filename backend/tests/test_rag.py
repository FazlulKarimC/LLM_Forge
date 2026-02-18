"""
Tests for RAG Service Components

Tests chunking, BM25 index, and RAG prompt template.
API-dependent tests (embedding, Qdrant, reranking, faithfulness) are skipped
unless HF_TOKEN and QDRANT_URL env vars are set.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Chunking Service Tests
# =============================================================================

class TestChunkingService:
    """Tests for the ChunkingService."""

    def test_single_short_article(self):
        """Short article that fits in one chunk."""
        from app.services.rag_service import ChunkingService

        articles = [{"title": "Test", "text": "Hello world from the test."}]
        chunks = ChunkingService.chunk_articles(articles, chunk_size=256, overlap=50)

        assert len(chunks) == 1
        assert chunks[0].title == "Test"
        assert chunks[0].text == "Hello world from the test."
        assert chunks[0].index == 0
        assert chunks[0].id == "chunk_0"

    def test_long_article_creates_multiple_chunks(self):
        """Long article should be split into overlapping chunks."""
        from app.services.rag_service import ChunkingService

        # Create a 600-word article
        words = [f"word{i}" for i in range(600)]
        articles = [{"title": "Long", "text": " ".join(words)}]

        chunks = ChunkingService.chunk_articles(articles, chunk_size=256, overlap=50)

        assert len(chunks) >= 2
        assert all(c.title == "Long" for c in chunks)
        # First chunk should have 256 words
        assert len(chunks[0].text.split()) == 256

    def test_multiple_articles(self):
        """Multiple articles create distinct chunks with correct metadata."""
        from app.services.rag_service import ChunkingService

        articles = [
            {"title": "A", "text": "Short text A."},
            {"title": "B", "text": "Short text B."},
        ]
        chunks = ChunkingService.chunk_articles(articles, chunk_size=256)

        assert len(chunks) == 2
        assert chunks[0].title == "A"
        assert chunks[1].title == "B"
        assert chunks[0].id != chunks[1].id

    def test_overlap_creates_shared_content(self):
        """Overlapping chunks should share some content."""
        from app.services.rag_service import ChunkingService

        words = [f"w{i}" for i in range(400)]
        articles = [{"title": "Overlap", "text": " ".join(words)}]

        chunks = ChunkingService.chunk_articles(articles, chunk_size=256, overlap=50)

        assert len(chunks) == 2
        # Last 50 words of chunk 0 should overlap with first 50 of chunk 1
        words_0 = chunks[0].text.split()
        words_1 = chunks[1].text.split()
        assert words_0[-50:] == words_1[:50]

    def test_empty_articles(self):
        """Empty input returns empty output."""
        from app.services.rag_service import ChunkingService

        chunks = ChunkingService.chunk_articles([], chunk_size=256)
        assert chunks == []

    def test_chunk_ids_are_sequential(self):
        """Chunk IDs should be sequential across all articles."""
        from app.services.rag_service import ChunkingService

        articles = [
            {"title": "A", "text": "Alpha text."},
            {"title": "B", "text": "Beta text."},
            {"title": "C", "text": "Gamma text."},
        ]
        chunks = ChunkingService.chunk_articles(articles, chunk_size=256)

        ids = [c.id for c in chunks]
        assert ids == ["chunk_0", "chunk_1", "chunk_2"]


# =============================================================================
# BM25 Index Tests
# =============================================================================

class TestBM25Index:
    """Tests for the BM25 keyword search index."""

    def test_basic_search(self):
        """BM25 should retrieve relevant chunks."""
        from app.services.rag_service import BM25Index, Chunk

        chunks = [
            Chunk(id="0", text="The capital of France is Paris", title="France", index=0),
            Chunk(id="1", text="Python is a programming language", title="Python", index=0),
            Chunk(id="2", text="The Eiffel Tower is in Paris France", title="Eiffel", index=0),
        ]

        idx = BM25Index()
        idx.build(chunks)

        results = idx.search("capital France Paris", top_k=2)
        assert len(results) >= 1
        # The France chunk should be most relevant
        titles = [c.title for c, _ in results]
        assert "France" in titles or "Eiffel" in titles

    def test_empty_query(self):
        """Empty query should return no results with positive scores."""
        from app.services.rag_service import BM25Index, Chunk

        chunks = [Chunk(id="0", text="Some text here", title="T", index=0)]
        idx = BM25Index()
        idx.build(chunks)

        results = idx.search("", top_k=5)
        assert len(results) == 0  # No positive-score results

    def test_no_index_returns_empty(self):
        """Search on unbuilt index returns empty."""
        from app.services.rag_service import BM25Index

        idx = BM25Index()
        results = idx.search("test query", top_k=5)
        assert results == []

    def test_top_k_limits_results(self):
        """Should return at most top_k results."""
        from app.services.rag_service import BM25Index, Chunk

        chunks = [
            Chunk(id=str(i), text=f"document about topic {i} and search", title=f"T{i}", index=0)
            for i in range(10)
        ]
        idx = BM25Index()
        idx.build(chunks)

        results = idx.search("document topic search", top_k=3)
        assert len(results) <= 3


# =============================================================================
# RAG Prompt Template Tests
# =============================================================================

class TestRAGPromptTemplate:
    """Tests for the RAG prompt template."""

    def test_format_with_context(self):
        """Should format question with context."""
        from app.services.inference.prompting import RAGPromptTemplate

        prompt = RAGPromptTemplate.format(
            "What is the capital of France?",
            ["Paris is the capital of France.", "France is in Europe."]
        )

        assert "Context:" in prompt
        assert "Paris is the capital of France." in prompt
        assert "France is in Europe." in prompt
        assert "Question: What is the capital of France?" in prompt
        assert prompt.endswith("Answer:")

    def test_format_without_context_falls_back_to_naive(self):
        """Empty context falls back to naive prompt."""
        from app.services.inference.prompting import RAGPromptTemplate

        prompt = RAGPromptTemplate.format("What is 2+2?", [])
        assert "Context:" not in prompt
        assert "Question: What is 2+2?" in prompt

    def test_parse_response(self):
        """Parse should clean response same as NaivePromptTemplate."""
        from app.services.inference.prompting import RAGPromptTemplate

        assert RAGPromptTemplate.parse_response("  Paris  \n") == "Paris"
        assert RAGPromptTemplate.parse_response("Answer: Berlin") == "Berlin"
        assert RAGPromptTemplate.parse_response("") == "[No response generated]"


# =============================================================================
# Knowledge Base Validation
# =============================================================================

class TestKnowledgeBase:
    """Tests for the knowledge base data file."""

    def test_articles_file_exists(self):
        """articles.json should exist."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "articles.json"
        )
        assert os.path.exists(os.path.normpath(path))

    def test_articles_format(self):
        """Every article should have title and text."""
        path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "articles.json"
        ))
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        assert len(articles) >= 40  # Should have ~50 articles
        for article in articles:
            assert "title" in article, f"Missing title: {article}"
            assert "text" in article, f"Missing text: {article}"
            assert len(article["text"]) > 50, f"Article too short: {article['title']}"

    def test_articles_chunk_properly(self):
        """All articles should chunk without errors."""
        from app.services.rag_service import ChunkingService

        path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "articles.json"
        ))
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        chunks = ChunkingService.chunk_articles(articles, chunk_size=256, overlap=50)
        assert len(chunks) >= len(articles)  # At least 1 chunk per article
