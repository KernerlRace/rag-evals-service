"""Runs only when TEST_DATABASE_URL points at a Postgres with pgvector (CI service container, docker compose)."""

import os

import pytest

from ragsvc.retrieval import NumpyIndex

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def test_pgvector_returns_the_same_neighbours_as_numpy(toy_chunks):
    from ragsvc import pgstore
    from ragsvc.models import HashEmbedder

    embedder = HashEmbedder(dim=128)
    vectors = embedder.embed_documents([c.text for c in toy_chunks])
    conn = pgstore.connect(URL)
    conn.execute("DROP TABLE IF EXISTS chunks")
    pgstore.write_chunks(conn, toy_chunks, vectors)

    assert pgstore.load_chunks(conn) == toy_chunks
    q = embedder.embed_query("smallest element heap")
    pg = pgstore.PgVectorIndex(conn).search(q, 3)
    np_ = NumpyIndex(vectors).search(q, 3)
    assert [i for i, _ in pg] == [i for i, _ in np_]
    assert [s for _, s in pg] == pytest.approx([s for _, s in np_], abs=1e-5)
