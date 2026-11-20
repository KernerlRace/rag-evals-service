"""Embedding model and cross-encoder reranker, both run through ONNX Runtime by fastembed (no torch).

Small fakes with the same interface let the tests run without downloading anything.
"""

import hashlib

import numpy as np

from ragsvc.bm25 import tokenize

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, MIT licence
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # 6-layer cross-encoder trained on MS MARCO, Apache-2.0
# bge v1.5 was trained with this instruction in front of short queries (passages get no prefix).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.dim = 384

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        return _unit(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return _unit(np.array(list(self.model.embed([QUERY_PREFIX + text])), dtype=np.float32))[0]


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, texts)), dtype=np.float32)


class HashEmbedder:
    """Bag of hashed words. Only for tests: deterministic and needs no model download."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _unit(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _unit(self._vec(text)[None, :])[0]


class OverlapReranker:
    """Scores by shared words. Only for tests."""

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) for t in texts], dtype=np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)
