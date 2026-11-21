import numpy as np
import pytest

from eval import golden
from eval.stats import bootstrap_ci, cohen_kappa, paired_bootstrap_diff, percentiles
from ragsvc.chunking import Chunk
from ragsvc.corpus import ARCHIVE, load_documents


def test_bootstrap_ci_brackets_the_mean():
    ci = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["mean"] == pytest.approx(0.7)
    assert bootstrap_ci([1] * 20)["lo"] == 1.0  # no variance, no interval


def test_paired_diff_detects_a_consistent_gain():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2, 200)
    better = np.maximum(base, rng.integers(0, 2, 200))  # never worse, often better
    assert paired_bootstrap_diff(better, base)["significant"]
    assert not paired_bootstrap_diff(base, base)["significant"]


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohen_kappa([1, 1, 1, 0], [1, 1, 1, 0]) == 1.0


def test_percentiles_in_ms():
    p = percentiles([0.1] * 19 + [1.0])
    assert p["p50_ms"] == pytest.approx(100)
    assert p["p95_ms"] > 100


def test_find_span_ignores_line_wrapping():
    text = "first line of the\n   docs wraps here. Next."
    start, end = golden.find_span(text, "of the docs wraps here.")
    assert text[start:end] == "of the\n   docs wraps here."


def test_relevance_needs_half_the_passage():
    item = golden.Item("q", "?", "a", "d", "p", "draft", "ok", start=100, end=200)
    chunk = lambda s, e, doc="d": Chunk("c", doc, "t", "s", s, e, "")  # noqa: E731
    assert golden.is_relevant(chunk(0, 150), item)  # covers 50 of 100 chars
    assert not golden.is_relevant(chunk(0, 149), item)
    assert not golden.is_relevant(chunk(0, 1000, doc="other"), item)
    assert golden.first_relevant_rank([chunk(0, 10), chunk(90, 300)], item) == 2


@pytest.mark.skipif(not ARCHIVE.exists(), reason="docs archive not present")
def test_every_golden_passage_is_in_the_docs():
    items = golden.load()
    located, problems = golden.locate(items, load_documents())
    assert problems == []
    assert len({it.id for it in items}) == len(items), "duplicate ids"
    assert sum(not it.answerable for it in items) >= 5, "keep some unanswerable questions"


def test_parse_verdict_handles_json_and_junk():
    from eval.judge import parse_verdict

    assert parse_verdict('{"reason": "matches", "verdict": "correct"}') == 1
    assert parse_verdict('{"reason": "wrong default", "verdict": "incorrect"}') == 0
    assert parse_verdict("The answer is INCORRECT.") == 0
    assert parse_verdict("no idea") is None


def test_judge_agreement_counts_the_confusion_matrix():
    from eval.judge import agreement

    judge = {"a": 1, "b": 1, "c": 0, "d": 0}
    hand = {"a": 1, "b": 0, "c": 0, "d": 0, "e": 1}  # "e" has no judge verdict and is ignored
    out = agreement(judge, hand)
    assert out["n"] == 4
    assert out["confusion"] == {"hand1_judge1": 1, "hand1_judge0": 0, "hand0_judge1": 1, "hand0_judge0": 2}
    assert out["agreement"]["mean"] == 0.75


def test_hand_labels_belong_to_one_answers_run(tmp_path):
    from eval.judge import load_hand_labels

    path = tmp_path / "labels.csv"
    path.write_text(
        "id,label,answers_file,note\nq001,1,answers_a.json,\nq001,0,answers_b.json,\nq002,s,answers_a.json,\n",
        encoding="utf-8",
    )
    assert load_hand_labels("answers_a.json", path) == {"q001": 1}
    assert load_hand_labels("answers_b.json", path) == {"q001": 0}


def test_v3_verdict_is_combined_by_code():
    from eval.judge import parse_verdict

    ok = '{"has_key_fact": true, "answers_every_part": true, "contradicts": false, "wrong_name_or_code": false}'
    assert parse_verdict(ok, "v3") == 1
    assert parse_verdict(ok.replace('"contradicts": false', '"contradicts": true'), "v3") == 0
    assert parse_verdict(ok.replace('"answers_every_part": true', '"answers_every_part": false'), "v3") == 0
    assert parse_verdict('{"has_key_fact": "yes"}', "v3") is None
