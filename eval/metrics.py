# -*- coding: utf-8 -*-
"""
Metric primitives for the Mizan baseline.

Deliberately dependency-free and short enough to read in one sitting: the
point of the benchmark is to be trusted, and a metric nobody can check is
worth as little as no metric at all.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence


# ==========================================================================
# Retrieval
# ==========================================================================
def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Share of gold items found in the top k.

    Recall rather than hit-rate because a clause can legitimately be governed
    by more than one article, and finding one of three is not the same result
    as finding all three.
    """
    gold_set = {str(g).strip() for g in gold if str(g).strip()}
    if not gold_set:
        return 0.0
    top = {str(r).strip() for r in retrieved[:k]}
    return len(gold_set & top) / len(gold_set)


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """1/rank of the first gold item, 0 if it never appears.

    Answers a different question from recall: not "did we find it" but "how
    far down the list was it" - which is exactly the question a reranker
    would be trying to fix.
    """
    gold_set = {str(g).strip() for g in gold if str(g).strip()}
    for position, item in enumerate(retrieved, start=1):
        if str(item).strip() in gold_set:
            return 1.0 / position
    return 0.0


# ==========================================================================
# Classification
# ==========================================================================
def prf1(true_labels: Sequence[bool], predicted: Sequence[bool]) -> dict:
    """Precision, recall and F1 for the positive class.

    Positive = "this clause is high risk". Precision is the one that matters
    for trust (how often a red flag was justified); recall is the one that
    matters for safety (how many real problems were missed).
    """
    tp = sum(1 for t, p in zip(true_labels, predicted) if t and p)
    fp = sum(1 for t, p in zip(true_labels, predicted) if not t and p)
    fn = sum(1 for t, p in zip(true_labels, predicted) if t and not p)
    tn = sum(1 for t, p in zip(true_labels, predicted) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ==========================================================================
# OCR
# ==========================================================================
def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over any sequence - characters for CER, words for WER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + (item_a != item_b),  # substitution
            ))
        previous = current
    return previous[-1]


# Arabic OCR output and a hand-typed reference will differ in ways that are
# not recognition errors: presentation forms, tatweel, and hamza seats that
# render identically. Folding them keeps CER measuring the thing it claims to.
_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _DIACRITICS.sub("", text)
    for src in "أإآٱ":
        text = text.replace(src, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. 0.0 is perfect; above 1.0 means more edits than
    reference characters, which happens when the OCR invents text."""
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate over whitespace-separated tokens."""
    ref = normalize_for_cer(reference).split()
    hyp = normalize_for_cer(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# ==========================================================================
# Reporting
# ==========================================================================
def mean(values: Iterable[float]) -> float:
    items: List[float] = list(values)
    return sum(items) / len(items) if items else 0.0
