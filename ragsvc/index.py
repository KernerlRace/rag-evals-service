"""Build chunks + embeddings once per chunk size and cache them on disk (data/index/size_<n>/).

The evaluation reads this cache directly; the service reads the same rows back from Postgres.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ragsvc.chunking import Chunk, chunk_corpus
from ragsvc.corpus import ARCHIVE, iter_documents

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"


def index_dir(size: int) -> Path:
    return INDEX_DIR / f"size_{size}"


def save(chunks: list[Chunk], vectors: np.ndarray, size: int) -> Path:
    out = index_dir(size)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(out / "vectors.npy", vectors.astype(np.float32))
    return out


def load(size: int) -> tuple[list[Chunk], np.ndarray]:
    out = index_dir(size)
    with open(out / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    vectors = np.load(out / "vectors.npy")
    if len(chunks) != len(vectors):
        raise ValueError(f"{out}: {len(chunks)} chunks but {len(vectors)} vectors")
    return chunks, vectors


def build(size: int, embedder, archive: Path = ARCHIVE) -> tuple[list[Chunk], np.ndarray]:
    docs = list(iter_documents(archive))
    chunks = chunk_corpus(docs, size)
    t = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    print(f"size {size}: {len(docs)} docs, {len(chunks)} chunks, embedded in {time.perf_counter() - t:.0f}s")
    return chunks, vectors


def load_or_build(size: int, embedder) -> tuple[list[Chunk], np.ndarray]:
    if (index_dir(size) / "vectors.npy").exists():
        return load(size)
    chunks, vectors = build(size, embedder)
    save(chunks, vectors, size)
    return chunks, vectors
