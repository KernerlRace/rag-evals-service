"""Read the Python 3.13 documentation (plain-text build) straight from the tar archive.

The archive is committed to the repo (data/raw, PSF licence) so that every character offset in the
golden set stays valid: docs.python.org rebuilds the 3.13 archive on every patch release.
"""

import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "python-3.13-docs-text.tar.bz2"

# What goes into the index. whatsnew/changelog (version noise), the C API and packaging guides are left out.
SECTIONS = ("tutorial", "library", "builtins", "reference", "howto", "faq")
EXTRA_FILES = ("glossary",)


@dataclass(frozen=True)
class Document:
    doc_id: str  # e.g. "library/functools"
    title: str
    text: str

    @property
    def url(self) -> str:
        return f"https://docs.python.org/3.13/{self.doc_id}.html"


def normalize(text: str) -> str:
    """Collapse runs of spaces/tabs and strip trailing spaces, keep newlines (headings depend on them)."""
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def _wanted(doc_id: str) -> bool:
    top = doc_id.split("/", 1)[0]
    return top in SECTIONS or doc_id in EXTRA_FILES


def iter_documents(archive: Path = ARCHIVE) -> Iterator[Document]:
    docs = []
    # Read members in archive order: bz2 cannot seek, so jumping around re-decompresses from the start
    # every time (this was 48 s instead of 2 s). Sort afterwards for a stable chunk order.
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            # "python-3.13-docs-text/library/functools.txt" -> "library/functools"
            doc_id = member.name.split("/", 1)[1].removesuffix(".txt")
            if not _wanted(doc_id):
                continue
            text = normalize(tar.extractfile(member).read().decode("utf-8"))
            title = text.split("\n", 1)[0].strip()
            docs.append(Document(doc_id=doc_id, title=title, text=text))
    yield from sorted(docs, key=lambda d: d.doc_id)


def load_documents(archive: Path = ARCHIVE) -> dict[str, Document]:
    return {doc.doc_id: doc for doc in iter_documents(archive)}
