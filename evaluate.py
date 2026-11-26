"""Evaluate ACE error detection (H3) and appropriate reliance (H4) at system level.

Protocol
--------
1. Thresholds (F5) are tuned by grid search on the development cases only.
2. The engine and thresholds are frozen (see FROZEN below).
3. The blind cases (blind.py, written after freezing) are scored once.
4. A programmatic seeded-error set (teacher-style mutations of sound claims)
   is scored as a stress test.

Usage:  python evaluate.py tune      # grid search on dev, prints best thresholds
        python evaluate.py           # full report -> results/
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import engine as E
from benchmark import CASES as DEV

try:
    from blind import CASES as BLIND
except ImportError:
    BLIND = []

ALL_Q = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
LIB = [t for k in DEV + BLIND for s in k["sources"] for t in s["text"]]

# Frozen after tuning on DEV (python evaluate.py tune). Do not edit after blind run.
FROZEN = dict(tau_support=0.20, tau_coverage=0.40, about_q4=0.5, about_q6=0.6, tau_pair=0.15)

EXPECTED_Q = {
    "number": {"Q4", "Q5", "Q6"}, "substitution": {"Q4", "Q5", "Q6"},
    "direction": {"Q4", "Q5", "Q6"}, "negation": {"Q3", "Q4", "Q5", "Q6"},
    "overgeneral": {"Q3", "Q4", "Q6"}, "fabricated_cit": {"Q2"},
    "undefined_form": {"Q1"}, "vague": {"Q1"}, "unsupported": {"Q2"},
    "misconception": set(ALL_Q), "relational": set(ALL_Q),
}


# --------------------------------------------------------------------------
def run_case(case, fw, answer=None, reask=None):
    ix = E.EvidenceIndex(case["sources"], LIB)
    cx = E.CrossExaminer(ix, fw)
    ans = answer if answer is not None else [c["text"] for c in case["answer"]]
    w = E.ScriptedWitness(" ".join(ans), " ".join(reask if reask is not None else case["reask"]))
    reps = cx.examine(case["question"], w)
    if len(reps) != len(ans):  # splitter disagreed with gold claim boundaries
        reps = [cx.examine(case["question"], E.ScriptedWitness(a, " ".join(case["reask"])))[0]
                for a in ans]
    return reps


def score_cases(cases, fw):
    rows = []
    for k in cases:
        for c, r in zip(k["answer"], run_case(k, fw)):
            rows.append(dict(case=k["id"], subject=k["subject"], text=c["text"],
                             flawed=c["label"] == "flawed", flaw=c["flaw"],
                             verdict=r.verdict, score=r.score,
                             qs=sorted({f.q for f in r.flags if f.severity != "weak"}),
                             flags=[(f.q, f.kind, f.finding) for f in r.flags]))
    return rows


def splitter_accuracy(cases):
    ok = sum(len(E.split_claims(" ".join(c["text"] for c in k["answer"]))) == len(k["answer"])
             for k in cases)
    return ok, len(cases)


def metrics(rows):
    fl = [r for r in rows if r["flawed"]]
    so = [r for r in rows if not r["flawed"]]
    hits = sum(r["verdict"] != "accept" for r in fl)
    fas = sum(r["verdict"] != "accept" for r in so)
    H = (hits + 0.5) / (len(fl) + 1)  # log-linear correction
    FA = (fas + 0.5) / (len(so) + 1)
    y = [r["flawed"] for r in rows]
    s = [r["score"] for r in rows]
    return dict(
        n=len(rows), n_flawed=len(fl), n_sound=len(so),
        hit_rate=hits / len(fl) if fl else float("nan"),
        false_alarm=fas / len(so) if so else float("nan"),
        d_prime=float(norm.ppf(H) - norm.ppf(FA)),
        accuracy=(hits + len(so) - fas) / len(rows),
        accept_correct=1 - fas / len(so) if so else float("nan"),
        accept_incorrect=1 - hits / len(fl) if fl else float("nan"),
        auc=float(roc_auc_score(y, s)) if len(set(y)) == 2 and len(set(s)) > 1 else 0.5,
    )


def bootstrap(rows, fn=metrics, keys=("hit_rate", "false_alarm", "d_prime", "accuracy"),
              B=2000, seed=7):
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    ids = list(by)
    rng = random.Random(seed)
    vals = defaultdict(list)
    for _ in range(B):
        samp = [r for i in (rng.choice(ids) for _ in ids) for r in by[i]]
        if not any(r["flawed"] for r in samp) or all(r["flawed"] for r in samp):
            continue
        m = fn(samp)
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def paired_diff(rows_a, rows_b, B=2000, seed=11):
    """Cluster bootstrap CI of d'(a) - d'(b) on the same claims."""
    cases = sorted({r["case"] for r in rows_a})
    ia = defaultdict(list); ib = defaultdict(list)
    for r in rows_a: ia[r["case"]].append(r)
    for r in rows_b: ib[r["case"]].append(r)
    rng = random.Random(seed); d = []
    for _ in range(B):
        pick = [rng.choice(cases) for _ in cases]
        a = [r for i in pick for r in ia[i]]; b = [r for i in pick for r in ib[i]]
        if not any(r["flawed"] for r in a) or all(r["flawed"] for r in a):
            continue
        d.append(metrics(a)["d_prime"] - metrics(b)["d_prime"])
    return float(np.mean(d)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def const_rows(rows, verdict):
    return [dict(r, verdict=verdict, score=0.0, qs=[], flags=[]) for r in rows]


def conditions(cases, base):
    out = {}
    full = score_cases(cases, base)
    out["Accept all (unstructured use)"] = const_rows(full, "accept")
    out["Reject all (blanket doubt)"] = const_rows(full, "reject")
    out["Evidence check only (Q2)"] = score_cases(cases, replace(base, question_types=("Q2",)))
    out["Re-ask only (Q5)"] = score_cases(cases, replace(base, question_types=("Q5",)))
    out["ACE full (Q1-Q6)"] = full
    for q in ALL_Q:
        out[f"ACE without {q}"] = score_cases(cases, replace(base, question_types=tuple(x for x in ALL_Q if x != q)))
    return out


def by_flaw(rows):
    t = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["flawed"]:
            t[r["flaw"]][1] += 1
            t[r["flaw"]][0] += r["verdict"] != "accept"
    return {k: dict(detected=v[0], total=v[1], recall=v[0] / v[1]) for k, v in sorted(t.items())}


def diagnosis(rows):
    det = [r for r in rows if r["flawed"] and r["verdict"] != "accept"]
    ok = sum(bool(set(r["qs"]) & EXPECTED_Q[r["flaw"]]) for r in det)
    return dict(correct=ok, detected=len(det), rate=ok / len(det) if det else float("nan"))


def by_subject(rows):
    return {s: metrics([r for r in rows if r["subject"] == s]) for s in sorted({r["subject"] for r in rows})}


# --------------------------------------------------------------------------
# Programmatic seeded errors (teacher-style mutations of sound claims)
# --------------------------------------------------------------------------
NAME_POOL = ["Hartley", "Moreau", "Okafor", "Lindqvist", "Tanaka", "Ferreira", "Novak", "Castillo"]


def _swap_direction(t):
    for _, a, b in E.DIRECTIONS:
        for w in re.findall(r"[A-Za-z]+", t):
            lw = w.lower()
            if lw in a or lw in b:
                src, dst = (a, b) if lw in a else (b, a)
                # pick the opposite form with the same suffix pattern if possible
                cand = sorted(dst, key=lambda x: (x[-2:] != lw[-2:], len(x)))[0]
                return re.sub(r"\b%s\b" % w, cand, t, count=1)
    return None


def _negate(t):
    m = re.search(r"\b(is|are|was|were|can|does|do|did|has|have)\b", t)
    if m:
        return t[:m.end()] + " not" + t[m.end():]
    m = re.search(r"\b([a-z]+s)\b", t[1:])  # 3rd-person verb, e.g. "boils"
    return None


def _number(t, rng):
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\w|\.\d)", t)
    if not m:
        return None
    v = float(m.group(1))
    if v.is_integer() and 1000 <= v <= 2100:
        nv = str(int(v + rng.choice([-11, -7, 6, 9])))
    elif v.is_integer():
        nv = str(int(v * 2 + 1))
    else:
        nv = f"{v * 1.7:.1f}"
    return t[:m.start()] + nv + t[m.end():]


def _absolute(t):
    m = re.search(r"\b(is|are|was|were|can)\b", t)
    if m:
        return t[:m.end()] + " always" + t[m.end():]
    return None


def _cite(t, rng):
    a, b = rng.sample(NAME_POOL, 2)
    return t.rstrip(".") + f" ({a} & {b}, {rng.randint(1995, 2022)})."


def _entity(t, record, rng):
    pns = [p for p in E.proper_nouns(t) if len(p) > 3]
    for p in pns:
        m = re.search(r"\b%s\b" % re.escape(p), t, re.I)
        if m:
            new = rng.choice([n for n in NAME_POOL if n.lower() not in record])
            return t[:m.start()] + new + t[m.end():]
    return None


def mutation_set(cases, seed=3):
    rng = random.Random(seed)
    rows = []
    for k in cases:
        record = set(E.words(" ".join(t for s in k["sources"] for t in s["text"])))
        for c in k["answer"]:
            if c["label"] != "sound":
                continue
            t = c["text"]
            muts = {"number": _number(t, rng), "direction": _swap_direction(t),
                    "negation": _negate(t), "overgeneral": _absolute(t),
                    "fabricated_cit": _cite(t, rng), "substitution": _entity(t, record, rng)}
            rows.append((k, t, "sound", None))
            for f, m in muts.items():
                if m and m != t:
                    rows.append((k, m, "flawed", f))
    return rows


def score_mutations(cases, fw):
    out = []
    for k, text, lab, flaw in mutation_set(cases):
        ix = E.EvidenceIndex(k["sources"], LIB)
        r = E.CrossExaminer(ix, fw).examine(k["question"], E.ScriptedWitness(text, " ".join(k["reask"])))[0]
        out.append(dict(case=k["id"], subject=k["subject"], text=text, flawed=lab == "flawed",
                        flaw=flaw, verdict=r.verdict, score=r.score,
                        qs=sorted({f.q for f in r.flags if f.severity != "weak"}),
                        flags=[(f.q, f.kind, f.finding) for f in r.flags]))
    return out


# --------------------------------------------------------------------------
def tune():
    grid = dict(tau_support=[0.10, 0.15, 0.20, 0.25], tau_coverage=[0.2, 0.3, 0.4, 0.5],
                about_q4=[0.4, 0.5, 0.6], about_q6=[0.5, 0.6, 0.7])
    best = None
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        m = metrics(score_cases(DEV, E.Framework(**p)))
        key = (round(m["d_prime"], 3), -m["false_alarm"])
        if best is None or key > best[0]:
            best = (key, p, m)
    print("best on dev:", best[1])
    print({k: round(v, 3) for k, v in best[2].items() if isinstance(v, float)})
    return best[1]


def fmt(m, ci=None):
    s = (f"hit {m['hit_rate']:.2f}  FA {m['false_alarm']:.2f}  d' {m['d_prime']:.2f}  "
         f"acc {m['accuracy']:.2f}  AUC {m['auc']:.2f}")
    if ci:
        s += f"   [d' 95% CI {ci['d_prime'][0]:.2f}, {ci['d_prime'][1]:.2f}]"
    return s


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune(); return
    os.makedirs("results", exist_ok=True)
    fw = E.Framework(**FROZEN)
    report = {"frozen_thresholds": FROZEN}
    sets = {"dev": DEV}
    if BLIND:
        sets["blind"] = BLIND
    for name, cases in sets.items():
        conds = conditions(cases, fw)
        res = {}
        for cname, rows in conds.items():
            m = metrics(rows)
            m["ci"] = bootstrap(rows) if not cname.startswith(("Accept", "Reject")) else None
            res[cname] = m
        full = conds["ACE full (Q1-Q6)"]
        report[name] = dict(
            cases=len(cases), splitter=splitter_accuracy(cases), conditions=res,
            by_flaw=by_flaw(full), by_subject=by_subject(full), diagnosis=diagnosis(full),
            ace_vs_evidence_only=paired_diff(full, conds["Evidence check only (Q2)"]),
            ace_vs_reask_only=paired_diff(full, conds["Re-ask only (Q5)"]),
            verdicts=dict(sound=Counter(r["verdict"] for r in full if not r["flawed"]),
                          flawed=Counter(r["verdict"] for r in full if r["flawed"])),
            errors=[dict(case=r["case"], text=r["text"], flaw=r["flaw"], verdict=r["verdict"],
                         flags=r["flags"]) for r in full if (r["verdict"] != "accept") != r["flawed"]],
        )
        print(f"\n=== {name.upper()} ({len(cases)} cases, {len(full)} claims) ===")
        for cname, m in res.items():
            print(f"{cname:32s} {fmt(m, m['ci'])}")
        print("by flaw:", {k: f"{v['detected']}/{v['total']}" for k, v in report[name]['by_flaw'].items()})
        print("diagnosis:", report[name]["diagnosis"])
        print("ACE - evidence-only d':", report[name]["ace_vs_evidence_only"])
        print("ACE - reask-only d':", report[name]["ace_vs_reask_only"])
    mut = score_mutations(DEV + BLIND, fw)
    report["seeded_mutations"] = dict(metrics=metrics(mut), ci=bootstrap(mut), by_flaw=by_flaw(mut),
                                      examples=[dict(text=r["text"], flaw=r["flaw"], verdict=r["verdict"])
                                                for r in mut[:12]])
    print(f"\n=== SEEDED MUTATIONS ({len(mut)} claims) ===\n{fmt(report['seeded_mutations']['metrics'], report['seeded_mutations']['ci'])}")
    print("by flaw:", {k: f"{v['detected']}/{v['total']}" for k, v in report['seeded_mutations']['by_flaw'].items()})
    with open("results/results.json", "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
