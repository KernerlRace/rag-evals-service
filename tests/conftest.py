import pytest

from ragsvc.chunking import chunk_document
from ragsvc.corpus import Document
from ragsvc.models import HashEmbedder, OverlapReranker
from ragsvc.retrieval import NumpyIndex, Retriever

TOY_DOCS = [
    Document("library/functools", '"functools" --- Higher-order functions', (
        '"functools" --- Higher-order functions\n**************************************\n\n'
        "The lru_cache decorator wraps a function with a memoizing callable. If maxsize is set to None, "
        "the LRU feature is disabled and the cache can grow without bound.\n\n"
        "partial Objects\n===============\n\nPartial objects are callable objects created by partial().\n"
    )),
    Document("library/heapq", '"heapq" --- Heap queue algorithm', (
        '"heapq" --- Heap queue algorithm\n********************************\n\n'
        "The interesting property of a heap is that its smallest element is always the root, heap[0].\n"
    )),
    Document("library/json", '"json" --- JSON encoder and decoder', (
        '"json" --- JSON encoder and decoder\n***********************************\n\n'
        "Keys in key/value pairs of JSON are always of the type str.\n"
    )),
]


@pytest.fixture
def toy_chunks():
    return [c for d in TOY_DOCS for c in chunk_document(d, size=20)]


@pytest.fixture
def toy_retriever(toy_chunks):
    embedder = HashEmbedder(dim=128)
    vectors = embedder.embed_documents([c.text for c in toy_chunks])
    return Retriever(toy_chunks, embedder, NumpyIndex(vectors), OverlapReranker())
