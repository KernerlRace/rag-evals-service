"""Ingestion: docs archive -> chunks -> embeddings -> on-disk cache and (optionally) Postgres.

    python -m ragsvc.ingest --sizes 128 256 512          # cache only (what the evaluation needs)
    python -m ragsvc.ingest --sizes 256 --postgres URL   # also load one size into pgvector for the service
"""

import argparse
import os
import time

from ragsvc import index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[int(os.environ.get("RAG_CHUNK_SIZE", "256"))])
    parser.add_argument("--postgres", nargs="?", const=os.environ.get("DATABASE_URL"), default=None,
                        help="load the (single) chunk size into this database; defaults to $DATABASE_URL")
    args = parser.parse_args()

    from ragsvc.models import Embedder

    embedder = Embedder()
    for size in args.sizes:
        chunks, vectors = index.load_or_build(size, embedder)
        print(f"size {size}: {len(chunks)} chunks ready in {index.index_dir(size)}")

    if args.postgres:
        if len(args.sizes) != 1:
            parser.error("--postgres takes exactly one --sizes value")
        from ragsvc import pgstore

        t = time.perf_counter()
        conn = pgstore.connect(args.postgres)
        pgstore.write_chunks(conn, chunks, vectors)
        print(f"wrote {len(chunks)} chunks to Postgres in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
