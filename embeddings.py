import os
import json
import urllib.request
import numpy as np

# onnx model files needed for all-MiniLM-L6-v2
_HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/main"
_FILES = {
    "model.onnx": f"{_HF_BASE}/onnx/model.onnx",
    "tokenizer.json": f"{_HF_BASE}/tokenizer.json",
}


class EmbeddingProvider:
    """Lightweight embedding provider using onnxruntime instead of full pytorch."""

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir
        self._session = None
        self._tokenizer = None

    def _model_dir(self) -> str:
        base = self._cache_dir or os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")),
            "VisioALS", "models",
        )
        return os.path.join(base, "all-MiniLM-L6-v2")

    def model_ready(self) -> bool:
        mdir = self._model_dir()
        return all(os.path.exists(os.path.join(mdir, f)) for f in _FILES)

    def download_model(self, progress_callback=None):
        """Download ONNX model files if missing. progress_callback(0-100)."""
        mdir = self._model_dir()
        os.makedirs(mdir, exist_ok=True)

        files_to_download = [
            (name, url) for name, url in _FILES.items()
            if not os.path.exists(os.path.join(mdir, name))
        ]
        if not files_to_download:
            if progress_callback:
                progress_callback(100)
            return

        for i, (name, url) in enumerate(files_to_download):
            dest = os.path.join(mdir, name)
            print(f"downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            print(f"downloaded {name}")
            if progress_callback:
                progress_callback(int(100 * (i + 1) / len(files_to_download)))

    def _load_model(self):
        if self._session is not None:
            return

        if not self.model_ready():
            self.download_model()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        mdir = self._model_dir()
        self._session = ort.InferenceSession(
            os.path.join(mdir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=256)
        print("embedding model loaded: all-MiniLM-L6-v2 (onnx)")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into an (N, dim) float32 array."""
        self._load_model()
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        # check which inputs the model actually expects
        expected = {inp.name for inp in self._session.get_inputs()}
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in expected:
            feeds["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, feeds)
        token_embeddings = outputs[0]  # (batch, seq_len, 384)

        # mean pooling with attention mask
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = np.sum(token_embeddings * mask, axis=1) / np.maximum(
            np.sum(mask, axis=1), 1e-9
        )

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-9)

        return pooled.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into a (dim,) float32 array."""
        return self.encode([text])[0]


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
