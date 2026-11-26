"""Run one ACE episode and print the student-facing cross-examination record.
Usage: python demo.py [CASE_ID]   (default S1, the paper's physics example)"""
import sys
import engine as E
from benchmark import CASES
from blind import CASES as BLIND
from evaluate import FROZEN, LIB

cid = sys.argv[1] if len(sys.argv) > 1 else "S1"
case = next(k for k in CASES + BLIND if k["id"] == cid)
fw = E.Framework(goal="Explain falling motion with and without air resistance",
                 case=case["question"], sources=case["sources"], **FROZEN)
ix = E.EvidenceIndex(case["sources"], LIB)
witness = E.ScriptedWitness(" ".join(c["text"] for c in case["answer"]), " ".join(case["reask"]))
reports = E.CrossExaminer(ix, fw).examine(case["question"], witness)
print(E.transcript(case["question"], reports))
