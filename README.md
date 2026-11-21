# rag-evals-service

Question answering over the Python 3.13 docs, built around its evaluation. The full README (architecture, all
results, how to run) is in progress; this section records the LLM judge, because every answer-accuracy number
depends on it.

## How far the LLM judge can be trusted

The judge (llama3.1:8b, a different model family from the qwen2.5 generator) marks each answer correct or incorrect
against the reference answer. I compared it with my own hand labels.
`python -m eval.judge_compare <answers file>` recomputes the table from the saved verdicts.

| Prompt | Labels | Agreement | Cohen's κ | Wrong answers caught | Right answers rejected | Judged accuracy of the run |
|---|---|---|---|---|---|---|
| v1: "key fact, nothing contradicting" | 40, qwen + hybrid | 0.825 | 0.26 | 2 / 6 | 3 / 34 | 0.855 |
| v2: five rules as one list | same 40 | 0.650 | 0.30 | 6 / 6 | 14 / 34 | 0.455 |
| v3: four yes/no checks, verdict combined in code | same 40 | 0.825 | 0.49 | 5 / 6 | 6 / 34 | 0.745 |
| v1 | 20 fresh, qwen + hybrid + rerank | 0.950 | 0.00 | 0 / 1 | 0 / 19 | 0.918 |
| v2 | same 20 | 0.600 | 0.12 | 1 / 1 | 8 / 19 | 0.464 |
| v3 | same 20 | 0.950 | 0.00 | 0 / 1 | 0 / 19 | 0.818 |

What this does and does not show:

- **v1 is too lenient.** On the first 40 labels it passed 4 of the 6 answers I marked wrong. Examples: an
  explanation of `str.lstrip` that describes it as removing a prefix, and an answer to only half of a two-part
  question. Its 0.855 accuracy is inflated.
- **The 40-label numbers for v2 and v3 are optimistic.** I wrote both prompts after reading v1's 7 disagreements on
  those same 40 answers. The rules are general, not per-question patches, but they were still shaped by those cases.
- **A rule list (v2) made the 8B judge nitpick.** It rejected correct answers for "extra details" even though one
  rule says extra true details are fine. Splitting the decision into narrow yes/no questions and combining them in
  code (v3) fixed most of that.
- **The clean check is too small to rank v1 against v3.** The 20 fresh labels cover questions I never looked at
  while writing the rules, with answers from a different run. Only 1 of the 20 answers was wrong (a reversed
  meaning of `usedforsecurity=False`), and v1, v3 and every other judge passed it.
  - κ is 0 there because the judge never said "incorrect". Agreement of 0.95 only reflects how few wrong answers
    there were.
  - What the fresh set does show: v2 is unusable (8 of 19 right answers rejected), and v3's false rejections
    (6/34 on the tuning set) did not appear here (0/19).
- **Conclusion: report answer accuracy as a range, not one number.** For qwen + hybrid that range is v3's 0.745 to
  v1's 0.855. Next steps:
  - label more wrong answers to make the check decisive;
  - try a stronger judge (8B is the bottleneck: no prompt made it both strict and fair).
- The golden set also had a bug, found while labelling: the reference answer for `bisect.insort` described
  `insort_left`. The reference is fixed now, so it no longer marks a wrong answer as correct.
