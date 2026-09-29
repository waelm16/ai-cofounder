"""Tests for the RAG pipeline: document processing, embeddings, vector store, retriever."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.memory.database import Base, User, get_db
from src.rag.document_processor import DocumentChunk, DocumentProcessor
from src.rag.embeddings import embed_query, embed_texts, get_embedding_dimension
from src.rag.retriever import (
    _build_citation_header,
    format_results_for_citation,
    get_knowledge_store,
    retrieve,
    retrieve_multi_source,
    _store_cache,
)
from src.rag.vector_store import VectorStore


# ---------- Fixtures ----------


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    session.add(User(id="founder", name="Founder", role="Founder"))
    session.commit()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_header():
    return {"Authorization": "Bearer founder"}


# ---------- DocumentProcessor Tests ----------


class TestDocumentProcessor:
    def test_chunk_text_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("word " * 500)  # ~2500 chars

        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_file(f, "book")

        assert len(chunks) >= 2
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert chunks[0].metadata["source_type"] == "book"
        assert chunks[0].metadata["source_name"] == "sample"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_html_file(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Hello world</p><p>Startup advice here</p></body></html>")

        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_file(f, "yc_content")

        assert len(chunks) >= 1
        assert "Hello world" in chunks[0].content
        assert chunks[0].metadata["source_type"] == "yc_content"

    def test_skip_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")

        processor = DocumentProcessor()
        chunks = processor.process_file(f, "book")
        assert chunks == []

    def test_process_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("First document content " * 20)
        (tmp_path / "b.txt").write_text("Second document content " * 20)
        (tmp_path / ".hidden").write_text("should be skipped")
        (tmp_path / ".gitkeep").write_text("")

        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_directory(tmp_path, "pg_essay", extensions=[".txt"])

        sources = {c.metadata["source_name"] for c in chunks}
        assert "a" in sources
        assert "b" in sources
        assert ".hidden" not in sources

    def test_chunk_overlap(self, tmp_path):
        # Create multi-sentence content that spans multiple chunks
        sentences = [f"Sentence number {i} about startups. " for i in range(80)]
        content = "".join(sentences)  # ~3000+ chars
        f = tmp_path / "overlap.txt"
        f.write_text(content)

        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_file(f, "book")

        assert len(chunks) >= 2
        # Adjacent chunks should share some content (overlap region)
        shared = set(chunks[0].content[-300:]).intersection(set(chunks[1].content[:300]))
        assert len(shared) > 0  # some characters in common

    def test_sentence_boundary_snapping(self, tmp_path):
        content = (
            "First sentence about MVPs. "
            "Second sentence about customers. "
            "Third sentence about growth. "
            "Fourth sentence about fundraising. "
        ) * 10  # ~1200+ chars
        f = tmp_path / "sentences.txt"
        f.write_text(content)

        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        chunks = processor.process_file(f, "book")

        assert len(chunks) >= 2
        for chunk in chunks[:-1]:  # last chunk can end anywhere
            text = chunk.content.rstrip()
            assert text[-1] in ".?!", f"Chunk does not end at sentence boundary: ...{text[-30:]}"

    def test_extra_metadata_passthrough(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("Some content about startups and venture capital. " * 5)

        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        extra = {"title": "My Book", "author": "Jane Doe", "custom_field": "value"}
        chunks = processor.process_file(f, "book", extra_metadata=extra)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["title"] == "My Book"
            assert chunk.metadata["author"] == "Jane Doe"
            assert chunk.metadata["custom_field"] == "value"
            assert chunk.metadata["source_type"] == "book"

    def test_process_text_method(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "Hello world from a transcript. " * 10
        chunks = processor.process_text(
            text, "customer_call", "call_123",
            extra_metadata={"customer_name": "Alice"},
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata["source_type"] == "customer_call"
        assert chunks[0].metadata["source_name"] == "call_123"
        assert chunks[0].metadata["customer_name"] == "Alice"

    def test_markdown_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Title\n\nSome content about startups.\n\n## Section\n\nMore content.")

        processor = DocumentProcessor()
        chunks = processor.process_file(f, "research_paper")

        assert len(chunks) >= 1
        assert "Title" in chunks[0].content
        assert chunks[0].metadata["source_type"] == "research_paper"

    def test_nonexistent_directory(self):
        processor = DocumentProcessor()
        chunks = processor.process_directory("/nonexistent/path", "book")
        assert chunks == []


# ---------- Embeddings Tests ----------


class TestEmbeddings:
    def test_embed_dimension(self):
        assert get_embedding_dimension() == 384

    def test_single_embed(self):
        vec = embed_query("test query about startups")
        assert len(vec) == 384
        # Should be normalized (unit vector)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_batch_embed(self):
        texts = ["first document", "second document", "third document"]
        vecs = embed_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_semantic_similarity_ordering(self):
        query = embed_query("startup fundraising venture capital")
        vec_related = embed_query("raising a seed round from investors")
        vec_unrelated = embed_query("recipe for chocolate cake baking")

        sim_related = np.dot(query, vec_related)
        sim_unrelated = np.dot(query, vec_unrelated)
        assert sim_related > sim_unrelated


# ---------- VectorStore Tests ----------


class TestVectorStore:
    def test_create_and_add(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))
        assert store.size == 0

        texts = ["doc one", "doc two"]
        embeddings = embed_texts(texts)
        metadata = [
            {"source_type": "book", "source_name": "test_book", "chunk_index": 0},
            {"source_type": "book", "source_name": "test_book", "chunk_index": 1},
        ]
        added = store.add_documents(texts, embeddings, metadata)

        assert added == 2
        assert store.size == 2

    def test_search_with_results(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))

        texts = [
            "Product market fit means making something people want",
            "Revenue models include subscription and one-time purchases",
            "The best way to cook pasta is with salted water",
        ]
        embeddings = embed_texts(texts)
        metadata = [
            {"source_type": "book", "source_name": "startup_guide", "chunk_index": i}
            for i in range(3)
        ]
        store.add_documents(texts, embeddings, metadata)

        query_vec = embed_query("how to achieve product market fit")
        results = store.search(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]
        assert "content" in results[0]
        assert "source_type" in results[0]

    def test_source_type_filter(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))

        texts = ["startup advice from PG", "business book content"]
        embeddings = embed_texts(texts)
        metadata = [
            {"source_type": "pg_essay", "source_name": "pg1", "chunk_index": 0},
            {"source_type": "book", "source_name": "book1", "chunk_index": 0},
        ]
        store.add_documents(texts, embeddings, metadata)

        query_vec = embed_query("startup advice")
        results = store.search(query_vec, top_k=5, source_type="pg_essay")

        assert all(r["source_type"] == "pg_essay" for r in results)

    def test_persistence(self, tmp_path):
        store = VectorStore(name="persist_test", store_dir=str(tmp_path))
        texts = ["persisted document about startups"]
        embeddings = embed_texts(texts)
        metadata = [{"source_type": "book", "source_name": "persist_book", "chunk_index": 0}]
        store.add_documents(texts, embeddings, metadata)

        # Reload from disk
        store2 = VectorStore(name="persist_test", store_dir=str(tmp_path))
        assert store2.size == 1
        assert store2.has_source("persist_book")

        query_vec = embed_query("startups")
        results = store2.search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0]["content"] == "persisted document about startups"

    def test_has_source(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))
        texts = ["some content"]
        embeddings = embed_texts(texts)
        metadata = [{"source_type": "book", "source_name": "the_mom_test", "chunk_index": 0}]
        store.add_documents(texts, embeddings, metadata)

        assert store.has_source("the_mom_test") is True
        assert store.has_source("nonexistent") is False

    def test_empty_search(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))
        query_vec = embed_query("anything")
        results = store.search(query_vec, top_k=5)
        assert results == []

    def test_add_empty_list(self, tmp_path):
        store = VectorStore(name="test", store_dir=str(tmp_path))
        added = store.add_documents([], [], [])
        assert added == 0


# ---------- Retriever Tests ----------


class TestRetriever:
    def setup_method(self):
        _store_cache.clear()

    def test_empty_store_returns_empty(self, tmp_path):
        results = retrieve("test query", store_dir=str(tmp_path))
        assert results == []

    def test_retrieve_finds_results(self, tmp_path):
        store = VectorStore(name="unified_knowledge", store_dir=str(tmp_path))
        texts = [
            "Talk to customers before building anything",
            "Focus on unit economics early",
        ]
        embeddings = embed_texts(texts)
        metadata = [
            {"source_type": "book", "source_name": "mom_test", "chunk_index": 0},
            {"source_type": "pg_essay", "source_name": "growth", "chunk_index": 0},
        ]
        store.add_documents(texts, embeddings, metadata)

        _store_cache[str(tmp_path)] = store
        results = retrieve("customer discovery", top_k=2, store_dir=str(tmp_path))
        assert len(results) >= 1

    def test_format_results_for_citation_empty(self):
        assert format_results_for_citation([]) == "No relevant sources found."

    def test_format_results_for_citation(self):
        results = [
            {
                "content": "Talk to customers first.",
                "source_type": "book",
                "source_name": "the_mom_test",
                "title": "The Mom Test",
                "author": "Rob Fitzpatrick",
                "score": 0.85,
                "chunk_index": 3,
                "page_number": 42,
            },
            {
                "content": "Do things that don't scale.",
                "source_type": "pg_essay",
                "source_name": "do_things",
                "title": "Do Things That Don't Scale",
                "author": "Paul Graham",
                "score": 0.80,
                "chunk_index": 0,
                "page_number": None,
            },
        ]
        formatted = format_results_for_citation(results)

        assert '"The Mom Test"' in formatted
        assert "by Rob Fitzpatrick" in formatted
        assert "p.42" in formatted
        assert "Talk to customers first." in formatted
        assert '"Do Things That Don\'t Scale"' in formatted
        assert "by Paul Graham" in formatted
        assert "Do things that don't scale." in formatted

    def test_citation_header_customer_call(self):
        r = {
            "source_type": "customer_call",
            "source_name": "call_42",
            "customer_name": "Jane Doe",
            "customer_company": "Acme Corp",
            "date": "2025-03-15",
            "content": "test",
        }
        header = _build_citation_header(1, r)
        assert "Call with Jane Doe @ Acme Corp" in header
        assert "2025-03-15" in header

    def test_citation_header_fallback_to_source_name(self):
        r = {"source_type": "book", "source_name": "some_book", "content": "test"}
        header = _build_citation_header(1, r)
        assert '"some_book"' in header

    def test_retrieve_multi_source(self, tmp_path):
        store = VectorStore(name="unified_knowledge", store_dir=str(tmp_path))
        texts = ["book content about startups", "pg essay about growth"]
        embeddings = embed_texts(texts)
        metadata = [
            {"source_type": "book", "source_name": "b1", "chunk_index": 0},
            {"source_type": "pg_essay", "source_name": "pg1", "chunk_index": 0},
        ]
        store.add_documents(texts, embeddings, metadata)

        _store_cache[str(tmp_path)] = store
        results = retrieve_multi_source(
            "startup advice",
            top_k_per_source=2,
            source_types=["book", "pg_essay"],
            store_dir=str(tmp_path),
        )

        assert "book" in results
        assert "pg_essay" in results


# ---------- API Endpoint Test ----------


class TestKnowledgeAPI:
    def setup_method(self):
        _store_cache.clear()

    def teardown_method(self):
        _store_cache.clear()

    def test_search_endpoint(self, client, auth_header, tmp_path, monkeypatch):
        """The endpoint searches the folder named by the VECTOR_STORE_DIR setting."""
        from src.config.settings import get_settings

        monkeypatch.setattr(get_settings(), "vector_store_dir", str(tmp_path))
        text = "Product market fit means customers keep using the product unprompted."
        VectorStore(name="unified_knowledge").add_documents(
            [text],
            embed_texts([text]),
            [{"source_type": "book", "source_name": "fit_notes", "chunk_index": 0}],
        )
        assert (tmp_path / "unified_knowledge.faiss").exists()

        response = client.get(
            "/api/knowledge/search",
            params={"q": "product market fit", "top_k": 3},
            headers=auth_header,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "product market fit"
        assert [r["source_name"] for r in data["results"]] == ["fit_notes"]
        assert data["results"][0]["content"] == text
        assert text in data["formatted"]
        assert list(_store_cache) == [str(tmp_path)]

    def test_search_endpoint_with_empty_store(self, client, auth_header, tmp_path, monkeypatch):
        from src.config.settings import get_settings

        monkeypatch.setattr(get_settings(), "vector_store_dir", str(tmp_path))

        response = client.get(
            "/api/knowledge/search", params={"q": "anything"}, headers=auth_header,
        )
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert response.json()["formatted"] == "No relevant sources found."


class TestConfiguredStoreDir:
    def test_default_store_dir_comes_from_settings(self):
        """With no folder given, stores use the configured folder, which conftest
        points at a temporary directory, not at ``vector_stores`` in the working tree."""
        import os
        from pathlib import Path

        from src.config.settings import get_settings

        configured = Path(get_settings().vector_store_dir)
        assert configured == Path(os.environ["VECTOR_STORE_DIR"])
        assert configured.resolve() != (Path.cwd() / "vector_stores").resolve()
        assert VectorStore(name="probe").store_dir == configured

    def test_settings_default_is_vector_stores(self):
        from src.config.settings import Settings

        assert Settings.model_fields["vector_store_dir"].default == "vector_stores"
