"""AI Cross-Examination (ACE) engine.

Roles
-----
Teacher (judge)   -> Framework: F1 goal, F2 case, F3 admissible sources,
                     F4 enabled question types, F5 success criteria
                     (thresholds), F6 boundaries.
AI (witness)      -> any callable question -> answer text. ScriptedWitness
                     replays benchmark answers; plug in an LLM the same way.
Student (counsel) -> the cross-examination cycle implemented here:
                     Elicit -> split into claims -> Q1-Q6 probes with lateral
                     verification against admissible evidence -> verdict.

Every check is deterministic and transparent: each flag records which question
type raised it, the probe a student would ask, and the evidence used, so the
output doubles as a worked example for students.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------
STOP = set("""a an the and or but if of to in on at by for with from into onto over under
about as is are was were be been being am do does did done has have had having it its
this that these those there their they them he she his her him we our you your i me my
which who whom whose what when where why how than then so such very can could would
should will shall may might must also not no never nor only all any each every some
most more less many much few several other another same just even still both either
neither per via while during after before because although though whether up down out
off again further once here s t around roughly approximately typically usually generally
mainly mostly often""".split())

NEGATORS = {"not", "no", "never", "cannot", "can't", "don't", "doesn't", "didn't", "isn't",
            "aren't", "wasn't", "weren't", "won't", "none", "nobody", "neither", "nor",
            "nothing", "n't"}

ABSOLUTES = {  # word -> licence group
    "always": "all", "all": "all", "every": "all", "each": "all", "everyone": "all",
    "entire": "all", "entirely": "all", "completely": "all", "any": "all",
    "never": "none", "none": "none", "nobody": "none", "nothing": "none", "ever": "ever",
    "only": "only", "impossible": "none", "guaranteed": "all",
}

DIRECTIONS = [  # (dimension, pole A forms, pole B forms)
    ("quantity", {"increase", "increases", "increased", "increasing", "raise", "raises", "raised",
                  "rise", "rises", "higher", "greater", "larger", "bigger"},
                 {"decrease", "decreases", "decreased", "decreasing", "lower", "lowers", "lowered",
                  "reduce", "reduces", "reduced", "smaller", "fewer"}),
    ("speed", {"fast", "faster", "fastest", "quickly", "quicker", "rapidly"},
              {"slow", "slower", "slowest", "slowly"}),
    ("volume", {"expand", "expands", "expanded", "expansion"},
               {"contract", "contracts", "contracted", "shrink", "shrinks", "shrank"}),
    ("light", {"absorb", "absorbs", "absorbed"}, {"reflect", "reflects", "reflected"}),
    ("estimate", {"overestimate", "overestimated", "overestimates"},
                 {"underestimate", "underestimated", "underestimates"}),
    ("approach", {"toward", "towards", "together", "closer", "converge"},
                 {"apart", "away", "separate", "diverge"}),
    ("order", {"before", "earlier"}, {"after", "later"}),
    ("heat", {"hotter", "warmer"}, {"colder", "cooler"}),
]

CONTRAST_SETS = {  # mutually exclusive members of a category
    "season": {"summer": "summer", "winter": "winter", "spring": "spring", "autumn": "autumn"},
    "state": {"solid": "solid", "solids": "solid", "liquid": "liquid", "liquids": "liquid",
              "gas": "gas", "gases": "gas"},
    "base": {"adenine": "adenine", "thymine": "thymine", "guanine": "guanine",
             "cytosine": "cytosine", "uracil": "uracil"},
    "organelle": {"chloroplast": "chloroplast", "chloroplasts": "chloroplast",
                  "mitochondria": "mitochondria", "mitochondrion": "mitochondria",
                  "nucleus": "nucleus", "ribosome": "ribosome", "ribosomes": "ribosome"},
    "pathogen": {"bacterium": "bacteria", "bacteria": "bacteria", "bacterial": "bacteria",
                 "virus": "virus", "viruses": "virus", "viral": "virus",
                 "fungus": "fungus", "fungi": "fungus", "parasite": "parasite"},
    "side": {"left": "left", "right": "right"},
    "compass": {"north": "north", "northern": "north", "south": "south", "southern": "south",
                "east": "east", "eastern": "east", "west": "west", "western": "west"},
    "circuit": {"series": "series", "parallel": "parallel"},
    "boundary": {"divergent": "divergent", "convergent": "convergent", "transform": "transform"},
    "shape": {"flat": "flat", "round": "round", "spherical": "round"},
    "colour": {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow"},
    "vessel": {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
               "capillary": "capillary", "capillaries": "capillary"},
}

HEDGES = [
    r"\b(some|many|most|several)\s+(experts|historians|scientists|people|researchers|scholars|sources)\s+(say|believe|think|argue|claim|suggest)",
    r"\bit\s+is\s+(often|widely|commonly|generally)?\s*(said|believed|thought|claimed)\b",
    r"\barguably\b", r"\bsome\s+say\b",
]

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}
CAP_STOP = {"the", "a", "an", "in", "on", "at", "when", "it", "this", "that", "most", "many",
            "some", "there", "during", "near", "from", "by", "as", "if", "pure", "adding",
            "putting", "because", "being", "without", "with", "for", "after", "before", "around",
            "both", "earth", "moon", "sun", "according", "we", "they", "he", "she", "his", "its",
            "these", "those", "between", "construction", "by"} | MONTHS
NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twice": 2}
ORDINAL_WORDS = {"fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
                 "nineteenth": 19, "twentieth": 20}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


def stem(w: str) -> str:
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'")
    for suf, rep in (("ations", "at"), ("ation", "at"), ("ments", ""), ("ment", ""), ("ness", ""),
                     ("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""),
                     ("es", ""), ("s", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            w = w[: -len(suf)] + rep
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def words(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in words(text)
            if w.isalpha() and len(w) >= 3 and w not in STOP and w not in NEGATORS}


def analyzer(text: str) -> list[str]:
    toks = [stem(w) for w in words(text) if w not in STOP]
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def split_claims(answer: str) -> list[str]:
    """Phase 4 step 1: split an answer into individually testable claims."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


# --------------------------------------------------------------------------
# Feature extractors
# --------------------------------------------------------------------------
CIT_ACCORDING = re.compile(r"[Aa]ccording to (?:the )?([A-Z][\w\- ]+?) \((\d{4})\),?\s*")
CIT_PAREN = re.compile(r"\s*\(([A-Z][A-Za-z\-]+)(?: et al\.)?(?: (?:&|and) [A-Z][A-Za-z\-]+)?,\s*(\d{4})\)")
CIT_NARR = re.compile(r"\b([A-Z][a-z]+)(?: et al\.)? \((\d{4})\)\s*")
FORMULA_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[^,;]+?(?=,|;|\s+(?:which|so|gives)\b|\.$|$)")


def extract_citations(text: str):
    cites = []
    for rx in (CIT_ACCORDING, CIT_PAREN, CIT_NARR):
        for m in rx.finditer(text):
            cites.append((m.group(1).strip(), int(m.group(2))))
        text = rx.sub(" ", text)
    return cites, re.sub(r"\s+", " ", text).strip()


def strip_formulas(text: str):
    found = FORMULA_RE.findall(text)
    return found, FORMULA_RE.sub(" ", text)


UNIT_PATTERNS = [
    (r"m/s(?:²|\^?2)", "m/s2"), (r"m/s", "m/s"), (r"°\s?c|degrees? celsius", "degC"),
    (r"degrees?", "deg"), (r"percent|%", "pct"), (r"km\b|kilomet", "km"),
    (r"centimet|cm\b", "cm"), (r"millimet|mm\b", "mm"), (r"met(?:re|er)s?\b|m\b", "m"),
    (r"feet|foot|ft\b", "ft"), (r"inch", "in"), (r"kg\b|kilogram", "kg"),
    (r"times (?:per|a|each) minute|beats per minute|bpm", "per_min"),
    (r"days?\b", "day_dur"), (r"years?\b", "yr_dur"), (r"hours?\b", "hr"),
]


def _unit_nouns(tail: str):
    seg = re.split(r"[,.;:]", tail, maxsplit=1)[0]
    ws = [w for w in words(seg) if w not in STOP and w not in NUM_WORDS and not w[0].isdigit()]
    return [stem(w) for w in ws[:2]]


def _unit_after(tail: str):
    t = tail.lower().lstrip()
    for pat, u in UNIT_PATTERNS:
        if re.match(pat, t):
            return u
    return None


def extract_quantities(text: str):
    """Return list of (unit, lo, hi). Unit 'count:<stem>|<stem>' for bare counts."""
    _, text = strip_formulas(text)
    out, spans = [], []
    low = text.lower()

    def add(u, lo, hi, span):
        out.append((u, lo, hi))
        spans.append(span)

    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)[\s\-]century|\b(" + "|".join(ORDINAL_WORDS) + r")[\s\-]century", low):
        n = int(m.group(1)) if m.group(1) else ORDINAL_WORDS[m.group(2)]
        add("year", (n - 1) * 100, (n - 1) * 100 + 99, m.span())
    for m in re.finditer(r"\b(\d{3,4})s\b", low):
        v = int(m.group(1))
        add("year", v, v + (99 if v % 100 == 0 else 9), m.span())
    num = r"(\d+(?:\.\d+)?)"
    for m in re.finditer(r"(?:between\s+)?(?<![\w.])" + num + r"\s*(?:to|-|–|and)\s*" + num + r"(?!\w|\.\d)", low):
        if any(a <= m.start() < b for a, b in spans):
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi > lo:
            u = _unit_after(low[m.end():]) or ("year" if 1000 <= lo <= 2100 else None)
            if u:
                add(u, lo, hi, m.span())
    for m in re.finditer(r"(?<![\w.])" + num + r"(?!\w|\.\d)", low):
        if any(a <= m.start() < b for a, b in spans):
            continue
        v = float(m.group(1))
        tail = low[m.end():]
        u = _unit_after(tail)
        if u is None:
            nxt = words(tail)[:2]
            if nxt and nxt[0] in MONTHS:
                u = "daynum"
            elif v.is_integer() and 1000 <= v <= 2100:
                u = "year"
            else:
                u = "count:" + "|".join(_unit_nouns(tail))
        add(u, v, v, m.span())
    for m in re.finditer(r"\b(" + "|".join(NUM_WORDS) + r")\b", low):
        add("count:" + "|".join(_unit_nouns(low[m.end():])), NUM_WORDS[m.group(1)], NUM_WORDS[m.group(1)], m.span())
    return out


LENGTH = {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001}


def _norm(q):
    u, lo, hi = q
    if u in LENGTH:
        return ("len_m", lo * LENGTH[u], hi * LENGTH[u])
    return q


def _units_match(u1, u2):
    if u1.startswith("count:") and u2.startswith("count:"):
        a, b = set(u1[6:].split("|")) - {""}, set(u2[6:].split("|")) - {""}
        return bool(a & b)
    return u1 == u2


def _qty_match(q, r, tol=0.03):
    (u, lo1, hi1), (_, lo2, hi2) = q, r
    exact = u in ("year", "daynum") or u.startswith("count")
    slack = 0.5 if exact else tol * max(abs(lo1), abs(hi1), abs(lo2), abs(hi2), 1e-9)
    return not (hi1 + slack < lo2 or hi2 + slack < lo1)  # values / ranges overlap


def number_conflict(claim: str, passage: str):
    cq = [_norm(q) for q in extract_quantities(claim)]
    pq = [_norm(q) for q in extract_quantities(passage)]
    for q in cq:
        same = [r for r in pq if _units_match(q[0], r[0])]
        if same and not any(_qty_match(q, r) for r in same):
            return f"claim gives {_fmt(q)}, evidence gives {', '.join(_fmt(r) for r in same)}"
    return None


def _fmt(q):
    u, lo, hi = q
    u = u.split(":")[0] if u.startswith("count") else u
    v = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
    return f"{v} {u}"


def polarity(text: str) -> int:
    ws = words(text)
    n = sum(w in NEGATORS or w.endswith("n't") for w in ws)
    return n % 2


def claim_coverage(claim: str, passage: str) -> float:
    cs = content_stems(claim)
    return len(cs & content_stems(passage)) / len(cs) if cs else 0.0


def negation_conflict(claim: str, passage: str, min_cov=0.5):
    unshared = content_stems(claim) - content_stems(passage)
    if (polarity(claim) != polarity(passage) and claim_coverage(claim, passage) >= min_cov
            and len(unshared) <= 1):
        return "claim and evidence make the same statement with opposite polarity"
    return None


def _dir_poles(text: str):
    ws = set(words(text))
    poles = {}
    for dim, a, b in DIRECTIONS:
        pa, pb = bool(ws & a), bool(ws & b)
        if pa != pb:
            poles[dim] = "A" if pa else "B"
        elif pa and pb:
            poles[dim] = "both"
    return poles


def direction_conflict(claim: str, passage: str):
    c, p = _dir_poles(claim), _dir_poles(passage)
    for dim, pole in c.items():
        if pole in ("A", "B") and p.get(dim) in ("A", "B") and p[dim] != pole:
            return f"claim and evidence point in opposite directions ({dim})"
    return None


def _members(text: str):
    ws = set(words(text))
    return {cat: {m[w] for w in ws if w in m} for cat, m in CONTRAST_SETS.items()}


def contrast_conflict(claim: str, passage: str):
    c, p = _members(claim), _members(passage)
    for cat in CONTRAST_SETS:
        if c[cat] and p[cat]:
            extra_c, extra_p = c[cat] - p[cat], p[cat] - c[cat]
            if extra_c and extra_p:
                return f"claim says {'/'.join(sorted(extra_c))}, evidence says {'/'.join(sorted(extra_p))} ({cat})"
    return None


def proper_nouns(text: str) -> set[str]:
    _, text = extract_citations(text)
    out = set()
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*|\S", sent)
        seqs, cur = [], []
        for i, t in enumerate(toks):
            if t[0].isalpha() and t[0].isupper() and t.lower().replace("'s", "") not in CAP_STOP:
                cur.append((i, t))
            else:
                if cur:
                    seqs.append(cur)
                cur = []
        if cur:
            seqs.append(cur)
        for s in seqs:
            if s[0][0] == 0 and len(s) == 1:
                continue  # sentence-initial single capital: ambiguous
            for _, t in s:
                t = re.sub(r"'s$", "", t).lower()
                if len(t) >= 2:
                    out.add(t)
    return out


def _present(tok, toks):
    return any(tok == t or (len(tok) >= 4 and len(t) >= 4 and (t.startswith(tok) or tok.startswith(t)))
               for t in toks)


_RECORD: set = set()  # tokens of the whole admissible record for the current case


def entity_conflict(claim: str, passage: str):
    pc, pp = proper_nouns(claim), proper_nouns(passage)
    tc, tp = set(words(extract_citations(claim)[1])), set(words(passage))
    missing = {t for t in pc if not _present(t, tp) and not _present(t, _RECORD)}
    other = {t for t in pp if not _present(t, tc)}
    if missing and other:
        return f"claim names {', '.join(sorted(missing))}; evidence names {', '.join(sorted(other))}"
    return None


CONFLICT_CHECKS = [("number", number_conflict), ("direction", direction_conflict),
                   ("negation", negation_conflict), ("category", contrast_conflict),
                   ("entity", entity_conflict)]


SUBJECT_CHECKS = {"category", "entity"}


def find_conflict(a: str, b: str, min_about=0.5):
    """Minimal-pair contradiction: b must be about the same statement as a
    (aboutness gate); a subject difference (who/what) together with a
    predicate difference (how much / which way / whether) means b describes a
    different statement, so it is not counted as a contradiction."""
    if claim_coverage(a, b) < min_about:
        return None
    fired = [(n, r) for n, fn in CONFLICT_CHECKS if (r := fn(a, b))]
    if not fired:
        return None
    subj = [f for f in fired if f[0] in SUBJECT_CHECKS]
    pred = [f for f in fired if f[0] not in SUBJECT_CHECKS]
    if subj and pred:
        return None
    return fired[0]


# --------------------------------------------------------------------------
# Framework (teacher), witness (AI), evidence index
# --------------------------------------------------------------------------
@dataclass
class Framework:
    goal: str = ""                                    # F1
    case: str = ""                                    # F2
    sources: list = field(default_factory=list)       # F3 admissible sources
    question_types: tuple = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")  # F4
    tau_support: float = 0.20                         # F5 success criteria
    tau_coverage: float = 0.40
    tau_relevant: float = 0.20
    tau_pair: float = 0.15
    about_q4: float = 0.5      # aboutness gate for best-match contradiction
    about_q6: float = 0.6      # stricter gate for the wider record
    boundaries: str = "Use only admissible sources; record every AI exchange."  # F6


@dataclass
class Flag:
    q: str
    kind: str
    severity: str  # reject | revise | weak
    probe: str
    finding: str
    evidence: str = ""


@dataclass
class ClaimReport:
    claim: str
    verdict: str
    score: float
    flags: list
    best_evidence: str
    support: float
    coverage: float


class ScriptedWitness:
    """Replays a benchmark answer; swap for any callable(question)->str."""
    def __init__(self, answer: str, reask: str):
        self._a, self._r = answer, reask

    def __call__(self, question: str, rephrased: bool = False) -> str:
        return self._r if rephrased else self._a


class EvidenceIndex:
    """Lateral reading: retrieve admissible evidence sentences for a claim.

    sources   -- the case's admissible sources (F3); only these are searched
    library   -- optional wider text collection used only to learn term weights
    """
    def __init__(self, sources, library=None):
        self.sents, self.meta = [], []
        for s in sources:
            for t in s["text"]:
                self.sents.append(t)
                self.meta.append(s)
        self.registry = {(s["author"].lower(), s["title"].lower(), s["year"]) for s in sources}
        fit_on = self.sents + list(library or [])
        self.wv = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True).fit(fit_on)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                  lowercase=True).fit(fit_on)
        self.W, self.C = self.wv.transform(self.sents), self.cv.transform(self.sents)

    def sim(self, texts):
        w = (self.wv.transform(texts) @ self.W.T).toarray()
        c = (self.cv.transform(texts) @ self.C.T).toarray()
        return 0.5 * (w + c)

    def pair_sim(self, a: list, b: list):
        if not a or not b:
            return np.zeros((len(a), len(b)))
        w = (self.wv.transform(a) @ self.wv.transform(b).T).toarray()
        c = (self.cv.transform(a) @ self.cv.transform(b).T).toarray()
        return 0.5 * (w + c)

    def search(self, text, k=3):
        s = self.sim([text])[0]
        idx = np.argsort(-s)[:k]
        return [(self.sents[i], float(s[i]), self.meta[i]) for i in idx]

    def cited(self, name: str, year: int) -> bool:
        n = name.lower()
        return any(year == y and (n in a or a in n or n in t) for a, t, y in self.registry)


# --------------------------------------------------------------------------
# Cross-examination
# --------------------------------------------------------------------------
class CrossExaminer:
    def __init__(self, index: EvidenceIndex, fw: Framework | None = None):
        self.ix, self.fw = index, fw or Framework()

    def on(self, q):
        return q in self.fw.question_types

    def examine_claim(self, claim, reask_claims, reask_sims):
        fw, flags = self.fw, []
        cites, core = extract_citations(claim)
        formulas, _ = strip_formulas(core)
        hits = self.ix.search(core, k=3)
        best, s_best, meta = hits[0]
        union = " ".join(h[0] for h in hits)
        cov = claim_coverage(core, union)

        # Q1 Clarification: is it one testable claim?
        if self.on("Q1"):
            for f in formulas:
                if not re.search(r"\bwhere\b|\bin which\b|\bis the\b", core[core.find(f) + len(f):]):
                    flags.append(Flag("Q1", "undefined_formula", "revise",
                                      f"State '{f.strip()}' as one testable claim: what does each symbol mean?",
                                      "formula uses symbols that are never defined"))
            for h in HEDGES:
                if re.search(h, core, re.I):
                    flags.append(Flag("Q1", "vague", "revise",
                                      "Who exactly says this, and what claim could I test?",
                                      "hedged attribution makes the claim untestable"))
                    break
        # Q2 Evidence: what is the source and can I check it?
        if self.on("Q2"):
            for name, yr in cites:
                if not self.ix.cited(name, yr):
                    flags.append(Flag("Q2", "fabricated_citation", "reject",
                                      f"What is the source '{name} ({yr})', and how can I check it?",
                                      "citation not found among admissible sources"))
            if s_best < fw.tau_support or cov < fw.tau_coverage:
                flags.append(Flag("Q2", "unsupported", "revise",
                                  "Where is the evidence for this in the admissible sources?",
                                  f"best support {s_best:.2f}, content coverage {cov:.0%}", best))
        # Q3 Assumption: unlicensed absolutes
        if self.on("Q3"):
            ws = set(words(core))
            lic = {ABSOLUTES[w] for w in set(words(union)) if w in ABSOLUTES}
            lic |= {"none"} if polarity(union) else set()
            bad = sorted(w for w in ws if w in ABSOLUTES and ABSOLUTES[w] not in lic)
            if bad:
                flags.append(Flag("Q3", "overgeneralisation", "revise",
                                  f"What are you assuming when you say '{bad[0]}'? Does it hold in every context?",
                                  f"absolute '{', '.join(bad)}' not licensed by evidence", union))
        # Q4 Counter-example: does the best-matching evidence contradict it?
        if self.on("Q4") and s_best >= fw.tau_relevant:
            cf = find_conflict(core, best, fw.about_q4)
            if cf:
                flags.append(Flag("Q4", cf[0], "reject",
                                  "What about the case in the evidence? It seems to say otherwise.",
                                  cf[1], best))
        # Q5 Consistency: same answer when asked differently?
        if self.on("Q5") and reask_claims:
            for j in np.argsort(-reask_sims):
                cf = find_conflict(core, reask_claims[j], fw.about_q4)
                if cf:
                    flags.append(Flag("Q5", "inconsistent_" + cf[0], "reject",
                                      "Do I get the same answer if I ask differently?",
                                      f"re-asked answer disagrees: {cf[1]}", reask_claims[j]))
                    break
            if max(reask_sims) < fw.tau_pair:
                flags.append(Flag("Q5", "not_reproduced", "weak",
                                  "Do I get the same answer if I ask differently?",
                                  "claim does not reappear when the question is rephrased"))
        # Q6 Implication: what else must be true? check the wider record
        if self.on("Q6"):
            for sent in self.ix.sents:
                if sent != best:
                    cf = find_conflict(core, sent, fw.about_q6)
                    if cf:
                        flags.append(Flag("Q6", cf[0], "reject",
                                          "If this is true, what else must be true? The wider record disagrees.",
                                          cf[1], sent))
                        break

        sev = {f.severity for f in flags}
        verdict = "reject" if "reject" in sev else "revise" if "revise" in sev else "accept"
        score = (sum({"reject": 1.0, "revise": 0.6, "weak": 0.2}[f.severity] for f in flags)
                 + 0.3 * (1 - min(s_best / 0.6, 1)) + 0.3 * (1 - cov))
        return ClaimReport(claim, verdict, score, flags, best, s_best, cov)

    def examine(self, question, witness: Callable):
        global _RECORD
        _RECORD = set(words(" ".join(self.ix.sents)))
        answer = witness(question)
        claims = split_claims(answer)
        reask = split_claims(witness(question, rephrased=True)) if self.on("Q5") else []
        cores = [extract_citations(c)[1] for c in claims]
        S = self.ix.pair_sim(cores, reask) if reask else np.zeros((len(claims), 0))
        return [self.examine_claim(c, reask, S[i]) for i, c in enumerate(claims)]


def transcript(question, reports) -> str:
    """Student-facing record of the cross-examination (phases 3-5)."""
    out = [f"CASE: {question}", ""]
    for i, r in enumerate(reports, 1):
        out.append(f"Claim {i}: {r.claim}")
        out.append(f"  Evidence (support {r.support:.2f}): {r.best_evidence}")
        for f in r.flags:
            out.append(f"  [{f.q}] {f.probe}")
            out.append(f"       -> {f.finding}")
        out.append(f"  VERDICT: {r.verdict.upper()}")
        out.append("")
    return "\n".join(out)
