"""Split documents into overlapping word windows, section by section.

Each chunk remembers its character span in the source document. The evaluation uses those spans to
decide whether a retrieved chunk contains the gold passage, which works the same for every chunk size.
"""

import re
from dataclasses import dataclass

from ragsvc.corpus import Document

# In the plain-text docs a heading is a line underlined with *, =, - or ~ of the same length.
_UNDERLINE = re.compile(r"^([*=\-~^])\1{2,}$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "library/functools#12"
    doc_id: str
    title: str
    section: str
    start: int  # character offsets into Document.text
    end: int
    body: str

    @property
    def text(self) -> str:
        """What gets embedded and indexed: a short header (where this comes from) plus the body."""
        header = self.title if self.section == self.title else f"{self.title} > {self.section}"
        return f"{header}\n{self.body}"

    @property
    def url(self) -> str:
        return f"https://docs.python.org/3.13/{self.doc_id}.html"


def split_sections(text: str) -> list[tuple[str, int, int]]:
    """Return (heading, body_start, body_end) spans. The heading line and its underline are not part of the
    body (the heading goes into the chunk header instead). Text before the first heading gets heading ""."""
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    # (heading, offset of the heading line, offset where the body starts = line after the underline)
    found: list[tuple[str, int, int]] = []
    for i in range(1, len(lines)):
        prev = lines[i - 1].strip()
        if prev and _UNDERLINE.match(lines[i].strip()) and len(lines[i].strip()) >= len(prev) - 2:
            found.append((prev, offsets[i - 1], offsets[i] + len(lines[i]) + 1))

    if not found or found[0][1] > 0:
        found.insert(0, ("", 0, 0))
    spans = []
    for n, (heading, _, body_start) in enumerate(found):
        end = found[n + 1][1] if n + 1 < len(found) else len(text)
        spans.append((heading, min(body_start, end), end))
    return spans


def chunk_document(doc: Document, size: int, overlap: int | None = None) -> list[Chunk]:
    """Windows of `size` words that step by `size - overlap` words and never cross a section boundary."""
    if overlap is None:
        overlap = size // 5
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for heading, sec_start, sec_end in split_sections(doc.text):
        words = [m.span() for m in re.finditer(r"\S+", doc.text[sec_start:sec_end])]
        if not words:
            continue
        i = 0
        while True:
            window = words[i : i + size]
            start = sec_start + window[0][0]
            end = sec_start + window[-1][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=heading or doc.title,
                    start=start,
                    end=end,
                    body=doc.text[start:end],
                )
            )
            if i + size >= len(words):
                break
            i += step
    return chunks


def chunk_corpus(docs: list[Document], size: int) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, size)]
