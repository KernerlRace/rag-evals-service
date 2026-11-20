"""The golden set: questions with a reference answer and the documentation passage that supports it.

eval/golden/golden.csv is the single source of truth (easy to review in a spreadsheet). The passage is
quoted from the docs; whitespace may differ (the docs are hard-wrapped), so it is matched with all runs
of whitespace collapsed, and mapped back to character offsets in the original document.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ragsvc.chunking import Chunk
from ragsvc.corpus import Document

GOLDEN_CSV = Path(__file__).resolve().parent / "golden" / "golden.csv"
FIELDS = ("id", "question", "reference_answer", "doc_id", "passage", "author", "status")


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    reference_answer: str
    doc_id: str
    passage: str
    author: str  # "draft" (written by Claude, reviewed by me) or "mykyta" (written by me)
    status: str  # "draft" until reviewed, then "ok"
    start: int = -1  # passage span in Document.text, filled by locate()
    end: int = -1

    @property
    def answerable(self) -> bool:
        """Unanswerable items (empty doc_id) are outside the docs; the right answer is "I don't know"."""
        return bool(self.doc_id)


def load(path: Path = GOLDEN_CSV) -> list[Item]:
    with open(path, encoding="utf-8", newline="") as f:
        return [Item(**{k: row[k].strip() for k in FIELDS}) for row in csv.DictReader(f)]


def _squash(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space; also return, for each output char, its index in `text`."""
    out, pos = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
        pos.append(i)
    return "".join(out), pos


def find_span(doc_text: str, passage: str, squashed: tuple[str, list[int]] | None = None) -> tuple[int, int] | None:
    squashed, pos = squashed or _squash(doc_text)
    needle = re.sub(r"\s+", " ", passage.strip())
    at = squashed.find(needle)
    if at < 0:
        return None
    return pos[at], pos[at + len(needle) - 1] + 1


def locate(items: list[Item], docs: dict[str, Document]) -> tuple[list[Item], list[str]]:
    """Attach character spans; return the located items and a list of human-readable problems."""
    located, problems = [], []
    squashed: dict[str, tuple[str, list[int]]] = {}  # squash each document once, not once per question
    for it in items:
        if not it.answerable:
            located.append(it)
            continue
        doc = docs.get(it.doc_id)
        if doc is None:
            problems.append(f"{it.id}: unknown doc_id {it.doc_id!r}")
            continue
        if it.doc_id not in squashed:
            squashed[it.doc_id] = _squash(doc.text)
        span = find_span(doc.text, it.passage, squashed[it.doc_id])
        if span is None:
            problems.append(f"{it.id}: passage not found in {it.doc_id}: {it.passage[:60]!r}")
            continue
        located.append(Item(**{**it.__dict__, "start": span[0], "end": span[1]}))
    return located, problems


def is_relevant(chunk: Chunk, item: Item, min_cover: float = 0.5) -> bool:
    """A chunk counts as relevant if it contains at least half of the gold passage's characters."""
    if chunk.doc_id != item.doc_id:
        return False
    overlap = min(chunk.end, item.end) - max(chunk.start, item.start)
    return overlap >= min_cover * (item.end - item.start)


def first_relevant_rank(chunks: list[Chunk], item: Item) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, item):
            return rank
    return None
