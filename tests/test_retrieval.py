import math

import pytest

from ragsvc.bm25 import BM25, tokenize
from ragsvc.chunking import chunk_document, split_sections
from ragsvc.retrieval import METHODS, rrf
from tests.conftest import TOY_DOCS


def test_tokenize_splits_dotted_names():
    assert tokenize("os.path.join() and lru_cache") == ["os", "path", "join", "and", "lru_cache"]


def test_bm25_matches_formula_by_hand():
    bm = BM25(["cat cat dog", "dog bird", "fish"], k1=1.5, b=0.75)
    avg = (3 + 2 + 1) / 3
    idf_cat = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf_cat * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avg))
    assert bm.scores("cat")[0] == pytest.approx(expected, rel=1e-5)
    assert bm.scores("cat")[1] == 0
    assert [i for i, _ in bm.search("dog bird", 3)][0] == 1


def test_bm25_ignores_unknown_terms():
    assert BM25(["a b"]).search("zzz", 5) == []


def test_sections_and_chunk_spans_point_back_into_the_text():
    doc = TOY_DOCS[0]
    headings = [h for h, _, _ in split_sections(doc.text)]
    assert "partial Objects" in headings
    for chunk in chunk_document(doc, size=8):
        assert doc.text[chunk.start : chunk.end] == chunk.body
        assert len(chunk.body.split()) <= 8


def test_chunks_overlap_and_never_cross_sections():
    doc = TOY_DOCS[0]
    chunks = chunk_document(doc, size=10, overlap=3)
    same_section = [c for c in chunks if c.section == chunks[0].section]
    assert same_section[1].start < same_section[0].end  # overlapping windows
    sections = {h: (s, e) for h, s, e in split_sections(doc.text)}
    for c in chunks:
        if c.section in sections:
            s, e = sections[c.section]
            assert s <= c.start and c.end <= e


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [i for i, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_finds_the_heap_passage(toy_retriever, method):
    result = toy_retriever.search("smallest element of a heap root", method=method, k=2)
    assert result.hits[0].chunk.doc_id == "library/heapq"
    assert len(result.hits) <= 2
    assert result.timings  # every stage reports its time


def test_unknown_method_is_rejected(toy_retriever):
    with pytest.raises(ValueError):
        toy_retriever.search("x", method="magic")
