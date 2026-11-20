"""Four retrievers over the same chunks: BM25, dense, hybrid (reciprocal rank fusion) and hybrid + reranker."""

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ragsvc.bm25 import BM25
from ragsvc.chunking import Chunk

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")


class DenseIndex(Protocol):
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]: ...


class NumpyIndex:
    """Exact cosine search: vectors are unit length, so a dot product is the cosine similarity."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.vectors @ query_vec
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank). Ignores raw scores,
    so BM25 scores (unbounded) and cosine similarities (-1..1) never need to be put on one scale."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class SearchResult:
    hits: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, dense: DenseIndex, reranker=None, bm25: BM25 | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.reranker = reranker
        self.bm25 = bm25 or BM25([c.text for c in chunks])

    def search(self, query: str, method: str = "hybrid", k: int = 5, candidates: int = 50, rerank_top: int = 30):
        if method not in METHODS:
            raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
        timings: dict[str, float] = {}
        depth = k if method in ("bm25", "dense") else candidates

        sparse = dense = None
        if method in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            sparse = self.bm25.search(query, depth)
            timings["bm25"] = time.perf_counter() - t
        if method in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed"] = time.perf_counter() - t
            t = time.perf_counter()
            dense = self.dense.search(qvec, depth)
            timings["dense"] = time.perf_counter() - t

        if method == "bm25":
            ranked = sparse
        elif method == "dense":
            ranked = dense
        else:
            ranked = rrf([[i for i, _ in sparse], [i for i, _ in dense]])
            if method == "hybrid_rerank":
                if self.reranker is None:
                    raise ValueError("hybrid_rerank needs a reranker")
                pool = [i for i, _ in ranked[:rerank_top]]
                t = time.perf_counter()
                scores = self.reranker.scores(query, [self.chunks[i].text for i in pool])
                timings["rerank"] = time.perf_counter() - t
                order = np.argsort(-scores)
                ranked = [(pool[j], float(scores[j])) for j in order]

        return SearchResult(hits=[Hit(self.chunks[i], s) for i, s in ranked[:k]], timings=timings)
