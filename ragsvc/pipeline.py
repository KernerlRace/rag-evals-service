"""retrieve -> prompt -> generate. The API and the answer evaluation both call this, so they measure the same code."""

import time
from dataclasses import dataclass

from ragsvc.llm import Generation, build_prompt
from ragsvc.retrieval import Hit, Retriever

IDK = "I don't know"


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    retrieval_seconds: float
    generation_seconds: float
    generation: Generation

    @property
    def abstained(self) -> bool:
        return self.text.strip().rstrip(".").lower() == IDK.lower()


def answer(question: str, retriever: Retriever, llm, method: str, k: int) -> Answer:
    t = time.perf_counter()
    result = retriever.search(question, method=method, k=k)
    retrieval_seconds = time.perf_counter() - t

    t = time.perf_counter()
    gen = llm.generate(build_prompt(question, [h.chunk.text for h in result.hits]))
    generation_seconds = time.perf_counter() - t
    return Answer(question, gen.text, result.hits, retrieval_seconds, generation_seconds, gen)
