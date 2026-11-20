"""BM25 written out by hand (Okapi BM25 with the Lucene idf), so every number can be explained.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    # "os.path.join()" -> ["os", "path", "join"]; identifiers with underscores stay whole.
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(texts)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(self.n_docs, dtype=np.float32)
        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            lengths[doc_idx] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_idx, tf))
        self.avg_len = float(lengths.mean()) if self.n_docs else 0.0
        # Length normalisation does not depend on the query, so compute it once per document.
        self._norm = k1 * (1 - b + b * lengths / max(self.avg_len, 1e-9))
        # Per term: document indices, term frequencies and idf, as numpy arrays for fast scoring.
        self._index: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, plist in postings.items():
            docs = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            df = len(plist)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            self._index[term] = (docs, tfs, idf)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._index:
                continue
            docs, tfs, idf = self._index[term]
            scores[docs] += idf * tfs * (self.k1 + 1) / (tfs + self._norm[docs])
        return scores

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
