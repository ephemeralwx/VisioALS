import os
import json
import numpy as np


class EmbeddingProvider:
    """Thin wrapper around sentence-transformers for lazy loading."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name, cache_folder=self._cache_dir)
        print(f"embedding model loaded: {self._model_name}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into an (N, dim) float32 array."""
        self._load_model()
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into a (dim,) float32 array."""
        self._load_model()
        return self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]


class CorpusIndex:
    """Manages precomputed embeddings and performs cosine-similarity retrieval."""

    def __init__(self, embeddings_dir: str, provider: EmbeddingProvider):
        self._dir = embeddings_dir
        self._provider = provider
        self._vectors: np.ndarray | None = None
        self._texts: list[str] | None = None

    @property
    def _vectors_path(self) -> str:
        return os.path.join(self._dir, "vectors.npy")

    @property
    def _texts_path(self) -> str:
        return os.path.join(self._dir, "texts.json")

    def is_built(self) -> bool:
        return os.path.exists(self._vectors_path) and os.path.exists(self._texts_path)

    def build_index(self, texts: list[str], progress_callback=None) -> None:
        """Compute embeddings for all texts and save to disk."""
        os.makedirs(self._dir, exist_ok=True)

        if progress_callback:
            progress_callback(10)

        # encode in batches to allow progress updates
        batch_size = 64
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._provider.encode(batch)
            all_vecs.append(vecs)
            if progress_callback:
                pct = 10 + int(80 * min((i + batch_size), len(texts)) / len(texts))
                progress_callback(pct)

        self._vectors = np.vstack(all_vecs).astype(np.float32)
        self._texts = list(texts)

        np.save(self._vectors_path, self._vectors)
        with open(self._texts_path, "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False)

        if progress_callback:
            progress_callback(100)

        print(f"corpus index built: {len(texts)} texts, shape {self._vectors.shape}")

    def load_index(self) -> bool:
        """Load precomputed vectors and texts from disk. Returns True on success."""
        if not self.is_built():
            return False
        self._vectors = np.load(self._vectors_path).astype(np.float32)
        with open(self._texts_path, "r", encoding="utf-8") as f:
            self._texts = json.load(f)
        print(f"corpus index loaded: {len(self._texts)} texts")
        return True

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Find the top-k most similar corpus texts to the query."""
        if self._vectors is None or self._texts is None:
            return []

        q_vec = self._provider.encode_single(query)
        sims = self._cosine_similarity(q_vec, self._vectors)
        top_idx = sims.argsort()[::-1][:top_k]
        return [self._texts[i] for i in top_idx]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and a corpus matrix."""
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        c_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        return c_norms @ q_norm
