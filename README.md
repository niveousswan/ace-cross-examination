# ACE — AI Cross-Examination: reference system and accuracy test

A working implementation of the ACE framework from *Students as AI Critics*, with a
benchmark and an evaluation of how accurately it detects errors in AI answers (H3) and
whether it accepts correct answers while rejecting incorrect ones (H4).

Pure Python (numpy, scipy, scikit-learn). No network or LLM needed.

## Files

| File | What it is |
|---|---|
| `engine.py` | The ACE engine: teacher `Framework` (F1–F6), `ScriptedWitness` (AI), `EvidenceIndex` (lateral reading over admissible sources), `CrossExaminer` (Q1–Q6 probes → accept / revise / reject), `transcript()` |
| `benchmark.py` | 24 development cases (12 science, 12 humanities, 122 labelled claims) |
| `blind.py` | 12 blind cases (60 claims), written **after** the engine was frozen |
| `evaluate.py` | Threshold tuning (dev only), baselines, ablations, d′, AUC, cluster-bootstrap CIs, seeded-mutation stress test |
| `demo.py` | Runs one episode and prints the student-facing cross-examination record |
| `tests.py` | Unit checks for the probes |
| `results/` | `results.json`, `console.txt`, `demo_S1.txt`, `freeze_record.txt` (SHA-256 of the frozen code) |

## Run

```bash
python tests.py            # probe unit checks
python demo.py S1          # the paper's feather/bowling-ball example
python evaluate.py tune    # grid search on development cases only
python evaluate.py         # full report -> results/
```

## How each question type is operationalised

| Q | Probe | Verdict |
|---|---|---|
| Q1 Clarification | formula with undefined symbols; hedged attribution ("many historians believe") | revise |
| Q2 Evidence | citation not in the admissible-source registry; low retrieval support or content coverage | reject / revise |
| Q3 Assumption | absolute (always, never, only, entire, ever…) not licensed by the evidence | revise |
| Q4 Counter-example | minimal-pair contradiction with the best-matching evidence: number/date, direction, polarity, category member, named entity | reject |
| Q5 Consistency | same contradiction checks against the answer to a rephrased question | reject |
| Q6 Implication | same checks against the rest of the admissible record (stricter aboutness gate) | reject |

## Headline results (blind set, 60 claims)

| Condition | Hit rate | False alarms | d′ [95% CI] | AUC |
|---|---|---|---|---|
| Accept all (unstructured use) | 0.00 | 0.00 | −0.16 | 0.50 |
| Evidence check only (Q2) | 0.17 | 0.00 | 1.12 [0.74, 1.40] | 0.71 |
| Re-ask only (Q5) | 0.08 | 0.04 | 0.24 [−0.93, 1.22] | 0.71 |
| **ACE full (Q1–Q6)** | **0.69** | **0.08** | **1.78 [1.08, 2.79]** | **0.86** |

Development set (tuned, optimistic): d′ 3.08. Seeded mutations (279 claims): d′ 3.14.

## Limits

Lexical and rule-based: it misses relational errors (all entities present, wrong link),
substitutions where the wrong entity also appears in the evidence, synonyms outside its
lexicons, and sentence-initial single-word names. The benchmark was written by the same
authors as the engine; the blind set reduces but does not remove that bias.
To use a live model as the witness, pass any `callable(question, rephrased=False) -> str`.
