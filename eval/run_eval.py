# -*- coding: utf-8 -*-
"""
Mizan baseline benchmark.

    python -m eval.run_eval                  # everything
    python -m eval.run_eval --only retrieval # fast, no LLM calls
    python -m eval.run_eval --only risk      # 40 LLM calls, several minutes
    python -m eval.run_eval --only ocr

Reads the pipeline, never writes to it. Results land in eval/results/ so two
runs can be compared.

The one design decision worth knowing about: retrieval is scored twice, over
all clauses and over only those whose gold article exists in the corpus. The
gap between the two separates "the retriever could not find it" from "we
never had it" - which is the question that decides what to fix next, and a
single Recall@K number cannot answer it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from eval.metrics import cer, mean, prf1, recall_at_k, reciprocal_rank, wer  # noqa: E402

GOLD = HERE / "data" / "clauses_gold.jsonl"
OCR_DIR = HERE / "data" / "ocr"
RESULTS = HERE / "results"

RETRIEVE_K = 10


def load_gold() -> List[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(law, number) -> str:
    """Identity of a legal provision: the law it belongs to plus its number."""
    return f"{str(law or '').strip()}|{str(number or '').strip()}"


def corpus_article_numbers() -> set:
    articles = json.loads((REPO / "data" / "legal_articles.json").read_text(encoding="utf-8"))
    return {str(a.get("article_number", "")).strip() for a in articles}


# ==========================================================================
# 1. Retrieval
# ==========================================================================
def eval_retrieval(gold: List[dict]) -> dict:
    from src.orchestrator import load_retriever, retrieve_for_clause, retriever_status

    # Fail loudly. A locked index (another process holding the local Qdrant
    # folder) makes every retrieval return [], which scores a clean 0.0% -
    # a number that looks like a catastrophic regression and is really just a
    # lock. A benchmark that reports a plausible wrong number is worse than
    # one that refuses to run.
    if load_retriever() is None:
        raise RuntimeError(
            "retriever unavailable, refusing to report a score: "
            f"{retriever_status()['error']}"
        )

    in_corpus = corpus_article_numbers()
    rows = []

    print(f"retrieval: {len(gold)} clauses, top-{RETRIEVE_K}")
    started = time.time()

    for i, item in enumerate(gold, start=1):
        articles = retrieve_for_clause(item["clause_text"], item["contract_type"], RETRIEVE_K)
        # Scored on (law, article number), not the number alone. Article
        # numbers repeat across laws - civil code 149 and companies law 149 are
        # different provisions - so a number-only match silently counted the
        # wrong law as a hit. Harmless while the corpus was 82% civil code,
        # badly wrong on a five-law corpus.
        retrieved = [_key(a.get("law_name"), a.get("article_number")
                          or a.get("article_number_suspect")) for a in articles]
        gold_keys = [_key(item.get("gold_law"), g) for g in item["gold_articles"]]
        covered = all(g in in_corpus for g in item["gold_articles"])

        rows.append({
            "id": item["id"],
            "gold": gold_keys,
            "gold_in_corpus": covered,
            "retrieved": retrieved,
            "r@3": recall_at_k(retrieved, gold_keys, 3),
            "r@10": recall_at_k(retrieved, gold_keys, RETRIEVE_K),
            "rr": reciprocal_rank(retrieved, gold_keys),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(gold)}")

    covered_rows = [r for r in rows if r["gold_in_corpus"]]

    return {
        "clauses": len(rows),
        "coverage": len(covered_rows) / len(rows) if rows else 0.0,
        "overall": {
            "recall@3": mean(r["r@3"] for r in rows),
            "recall@10": mean(r["r@10"] for r in rows),
            "mrr": mean(r["rr"] for r in rows),
        },
        "gold_in_corpus_only": {
            "n": len(covered_rows),
            "recall@3": mean(r["r@3"] for r in covered_rows),
            "recall@10": mean(r["r@10"] for r in covered_rows),
            "mrr": mean(r["rr"] for r in covered_rows),
        },
        "seconds": round(time.time() - started, 1),
        "rows": rows,
    }


# ==========================================================================
# 2. Risk detection
# ==========================================================================
def eval_risk(gold: List[dict]) -> dict:
    from src.analysis import analyze_single_clause
    from src.orchestrator import format_legal_context, retrieve_for_clause

    print(f"risk: {len(gold)} clauses (one LLM call each - this takes a while)")
    started = time.time()

    truth, predicted, rows = [], [], []
    for i, item in enumerate(gold, start=1):
        articles = retrieve_for_clause(item["clause_text"], item["contract_type"], 3)
        clause = {
            "clause_id": item["id"],
            "clause_label": f"بند {item['id']}",
            "clause_text": item["clause_text"],
        }
        try:
            result = analyze_single_clause(
                clause, {"contract_type": item["contract_type"]},
                format_legal_context(articles),
            ).model_dump()
            level = getattr(result.get("risk_level"), "value", result.get("risk_level"))
            evidence = getattr(result.get("evidence_status"), "value", result.get("evidence_status"))
            is_red = level == "أحمر"
            score = result.get("risk_score")
            error = None
        except Exception as exc:  # noqa: BLE001 - a failed clause is a miss, not a crash
            level, evidence, is_red, score, error = None, None, False, None, str(exc)[:120]

        truth.append(item["risk"] == "red")
        predicted.append(is_red)
        rows.append({
            "id": item["id"], "gold": item["risk"],
            "predicted": "red" if is_red else "not_red",
            "risk_score": score, "evidence": evidence, "error": error,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(gold)}")

    stats = prf1(truth, predicted)
    evidence_mix: dict = {}
    for r in rows:
        evidence_mix[r["evidence"]] = evidence_mix.get(r["evidence"], 0) + 1

    return {
        "clauses": len(rows),
        **stats,
        "evidence_mix": evidence_mix,
        "failed": sum(1 for r in rows if r["error"]),
        "seconds": round(time.time() - started, 1),
        "rows": rows,
    }


# ==========================================================================
# 3. OCR
# ==========================================================================
def eval_ocr() -> dict:
    from src.ocr import HAS_TESSERACT, refine_text_with_groq, repair_scan_artifacts

    pages = sorted(p.stem.replace("_raw", "") for p in OCR_DIR.glob("*_raw.txt"))
    print(f"ocr: {len(pages)} page(s)")

    rows = []
    for name in pages:
        raw = (OCR_DIR / f"{name}_raw.txt").read_text(encoding="utf-8")
        ref = (OCR_DIR / f"{name}_ref.txt").read_text(encoding="utf-8")

        deterministic = repair_scan_artifacts(raw)
        full = refine_text_with_groq(deterministic)

        rows.append({
            "page": name,
            "raw": {"cer": cer(ref, raw), "wer": wer(ref, raw)},
            "deterministic": {"cer": cer(ref, deterministic), "wer": wer(ref, deterministic)},
            "repaired": {"cer": cer(ref, full), "wer": wer(ref, full)},
        })
        print(f"  {name}: CER {rows[-1]['raw']['cer']:.3f} -> "
              f"{rows[-1]['deterministic']['cer']:.3f} -> {rows[-1]['repaired']['cer']:.3f}")

    return {
        "pages": len(rows),
        # The OCR engine itself needs a Tesseract binary and real page images;
        # what is measured here is the repair stage, which takes damaged text
        # to clean text. That is the stage this project actually changes.
        "engine_measured": bool(HAS_TESSERACT),
        "raw": {"cer": mean(r["raw"]["cer"] for r in rows),
                "wer": mean(r["raw"]["wer"] for r in rows)},
        "deterministic": {"cer": mean(r["deterministic"]["cer"] for r in rows),
                          "wer": mean(r["deterministic"]["wer"] for r in rows)},
        "repaired": {"cer": mean(r["repaired"]["cer"] for r in rows),
                     "wer": mean(r["repaired"]["wer"] for r in rows)},
        "rows": rows,
    }


# ==========================================================================
# Report
# ==========================================================================
def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_report(results: dict) -> None:
    line = "─" * 62
    print("\n" + line)
    print("  MIZAN BASELINE")
    print(line)

    if "retrieval" in results:
        r = results["retrieval"]
        print("\n■ RETRIEVAL")
        print(f"  clauses                  {r['clauses']}")
        print(f"  gold article in corpus   {pct(r['coverage'])}")
        print(f"  Recall@3                 {pct(r['overall']['recall@3'])}")
        print(f"  Recall@10                {pct(r['overall']['recall@10'])}")
        print(f"  MRR                      {r['overall']['mrr']:.3f}")
        c = r["gold_in_corpus_only"]
        print(f"  -- over the {c['n']} clauses whose gold article exists --")
        print(f"  Recall@3                 {pct(c['recall@3'])}")
        print(f"  Recall@10                {pct(c['recall@10'])}")
        print(f"  MRR                      {c['mrr']:.3f}")

    if "risk" in results:
        k = results["risk"]
        print("\n■ RISK DETECTION (positive class = high risk)")
        print(f"  clauses                  {k['clauses']}  (failed: {k['failed']})")
        print(f"  Precision                {pct(k['precision'])}")
        print(f"  Recall                   {pct(k['recall'])}")
        print(f"  F1                       {pct(k['f1'])}")
        print(f"  TP/FP/FN/TN              {k['tp']}/{k['fp']}/{k['fn']}/{k['tn']}")
        print(f"  evidence mix             {k['evidence_mix']}")

    if "ocr" in results:
        o = results["ocr"]
        print("\n■ OCR REPAIR")
        print(f"  pages                    {o['pages']}")
        print(f"  {'stage':<24}{'CER':>8}{'WER':>8}")
        for stage in ("raw", "deterministic", "repaired"):
            print(f"  {stage:<24}{o[stage]['cer']:>8.3f}{o[stage]['wer']:>8.3f}")
        if not o["engine_measured"]:
            print("  note: Tesseract not installed - the OCR engine itself was")
            print("        not measured, only the repair stage on damaged text.")

    print("\n" + line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mizan baseline benchmark")
    parser.add_argument("--only", choices=["retrieval", "risk", "ocr"],
                        help="run a single section")
    parser.add_argument("--out", default=None, help="path for the results JSON")
    args = parser.parse_args()

    gold = load_gold()
    results: dict = {"generated": datetime.now().isoformat(timespec="seconds")}

    if args.only in (None, "retrieval"):
        results["retrieval"] = eval_retrieval(gold)
    if args.only in (None, "risk"):
        results["risk"] = eval_risk(gold)
    if args.only in (None, "ocr"):
        results["ocr"] = eval_ocr()

    print_report(results)

    RESULTS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / f"baseline_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        shown = out.resolve().relative_to(REPO)
    except ValueError:
        shown = out
    print(f"saved -> {shown}")


if __name__ == "__main__":
    main()
