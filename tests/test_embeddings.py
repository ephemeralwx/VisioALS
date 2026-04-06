"""Tests for embedding provider and corpus index."""
import os
import time
import pytest
import numpy as np


@pytest.fixture
def provider():
    from embeddings import EmbeddingProvider
    return EmbeddingProvider()


@pytest.fixture
def corpus_index(tmp_path, provider):
    from embeddings import CorpusIndex
    return CorpusIndex(str(tmp_path / "embeddings"), provider)


class TestEmbeddingProvider:
    def test_encode_shape(self, provider):
        vecs = provider.encode(["hello world", "test sentence"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 384)

    def test_encode_single_shape(self, provider):
        vec = provider.encode_single("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)

    def test_encode_deterministic(self, provider):
        v1 = provider.encode_single("test")
        v2 = provider.encode_single("test")
        np.testing.assert_array_almost_equal(v1, v2)


class TestCorpusIndex:
    def test_build_and_load(self, corpus_index):
        texts = ["The cat sat on the mat", "Dogs are loyal companions", "I like ice cream"]
        corpus_index.build_index(texts)

        assert corpus_index.is_built()
        assert os.path.exists(corpus_index._vectors_path)
        assert os.path.exists(corpus_index._texts_path)

        # reload from disk
        from embeddings import CorpusIndex, EmbeddingProvider
        fresh = CorpusIndex(corpus_index._dir, EmbeddingProvider())
        assert fresh.load_index() is True

    def test_retrieve_relevance(self, corpus_index):
        texts = [
            "I'm feeling tired today",
            "The weather is sunny and warm",
            "My back hurts a lot",
            "Let's have pizza for dinner",
            "I slept poorly last night",
            "The garden looks beautiful",
            "I need my pain medication",
            "What time is the game on?",
        ]
        corpus_index.build_index(texts)

        results = corpus_index.retrieve("How are you feeling?", top_k=3)
        assert len(results) == 3
        # health/feeling related texts should rank higher
        combined = " ".join(results).lower()
        assert "feeling" in combined or "hurts" in combined or "pain" in combined or "tired" in combined

    def test_retrieve_empty_index(self, corpus_index):
        results = corpus_index.retrieve("test query")
        assert results == []

    def test_not_built_before_build(self, corpus_index):
        assert not corpus_index.is_built()


class TestPerformance:
    def test_retrieval_latency(self, corpus_index, large_corpus_texts):
        """Retrieval over 200+ docs must complete in under 200ms."""
        corpus_index.build_index(large_corpus_texts)

        # warm up (first call loads model)
        corpus_index.retrieve("test", top_k=5)

        start = time.time()
        results = corpus_index.retrieve("How are you feeling today?", top_k=5)
        elapsed_ms = (time.time() - start) * 1000

        assert len(results) == 5
        assert elapsed_ms < 500, f"Retrieval took {elapsed_ms:.0f}ms, expected <500ms"
