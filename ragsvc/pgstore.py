"""Postgres + pgvector storage for chunks and their embeddings.

At ~10k chunks an exact scan (no ANN index) answers in milliseconds, so there is no HNSW index:
it would add recall loss to measure for no latency gain at this size.
"""

import threading

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from ragsvc.chunking import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    idx        integer PRIMARY KEY,
    chunk_id   text NOT NULL,
    doc_id     text NOT NULL,
    title      text NOT NULL,
    section    text NOT NULL,
    start_char integer NOT NULL,
    end_char   integer NOT NULL,
    body       text NOT NULL,
    embedding  vector({dim}) NOT NULL
);
"""


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def write_chunks(conn: psycopg.Connection, chunks: list[Chunk], vectors: np.ndarray) -> None:
    conn.execute(SCHEMA.format(dim=vectors.shape[1]))
    with conn.transaction():
        conn.execute("TRUNCATE chunks")
        with conn.cursor().copy(
            "COPY chunks (idx, chunk_id, doc_id, title, section, start_char, end_char, body, embedding) "
            "FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int4", "text", "text", "text", "text", "int4", "int4", "text", "vector"])
            for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
                copy.write_row([i, c.chunk_id, c.doc_id, c.title, c.section, c.start, c.end, c.body, v])


def load_chunks(conn: psycopg.Connection) -> list[Chunk]:
    rows = conn.execute(
        "SELECT chunk_id, doc_id, title, section, start_char, end_char, body FROM chunks ORDER BY idx"
    ).fetchall()
    return [Chunk(*row) for row in rows]


class PgVectorIndex:
    """Dense search inside Postgres. `<=>` is pgvector's cosine distance, so similarity = 1 - distance."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        self.lock = threading.Lock()  # one connection, FastAPI runs sync handlers in a thread pool

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT idx, 1 - (embedding <=> %s) AS sim FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (query_vec, query_vec, k),
            ).fetchall()
        return [(int(i), float(s)) for i, s in rows]
