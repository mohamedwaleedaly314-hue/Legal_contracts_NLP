# -*- coding: utf-8 -*-
"""
Corpus A/B: does the new LegalLens corpus actually retrieve better?

Structural quality and retrieval quality are different claims. This scores
both corpora on the same 40-clause gold set, with the same embedding model and
the same dense-only method, so the only variable left is the corpus itself.

Dense-only for both, deliberately: our live system is hybrid and theirs is
not, and comparing 'our hybrid' to 'their dense' would confound the corpus
with the method. What this isolates is whether the articles and their text got
better.

    python -m eval.compare_corpora --new "<path to LegalLens>"
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import List

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from eval.metrics import mean, recall_at_k, reciprocal_rank  # noqa: E402

TOP_K = 10


def load_gold() -> List[dict]:
    path = HERE / "data" / "clauses_gold.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(name: str, vectors, articles: List[dict], gold: List[dict], model) -> dict:
    import numpy as np

    # Cosine over normalised vectors is a dot product; 2443x1024 is small
    # enough that an exact search beats standing up an index for a one-off.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = vectors / norms

    queries = model.encode([g["clause_text"] for g in gold], normalize_embeddings=True)
    sims = queries @ matrix.T

    rows = []
    for i, item in enumerate(gold):
        top = np.argsort(-sims[i])[:TOP_K]
        retrieved = [str(articles[j].get("article_number", "")).strip() for j in top]
        rows.append({
            "id": item["id"],
            "r@3": recall_at_k(retrieved, item["gold_articles"], 3),
            "r@10": recall_at_k(retrieved, item["gold_articles"], TOP_K),
            "rr": reciprocal_rank(retrieved, item["gold_articles"]),
            "retrieved": retrieved[:5],
        })

    return {
        "name": name,
        "articles": len(articles),
        "recall@3": mean(r["r@3"] for r in rows),
        "recall@10": mean(r["r@10"] for r in rows),
        "mrr": mean(r["rr"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True, help="path to the LegalLens folder")
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    gold = load_gold()
    print("loading BAAI/bge-m3 ...")
    model = SentenceTransformer("BAAI/bge-m3")

    # --- ours -------------------------------------------------------------
    ours_articles = json.loads((REPO / "data" / "legal_articles.json").read_text(encoding="utf-8"))
    ours_vectors = np.load(REPO / "data" / "embeddings.npy")
    print(f"ours:   {len(ours_articles)} articles, vectors {ours_vectors.shape}")

    # --- theirs -----------------------------------------------------------
    export = Path(args.new) / "index" / "qdrant_export" / "legallens_legal_only_points.json.gz"
    with gzip.open(export, "rt", encoding="utf-8") as fh:
        points = json.load(fh)
    theirs_articles = [p["payload"] for p in points]
    theirs_vectors = np.asarray([p["vector"] for p in points], dtype="float32")
    print(f"theirs: {len(theirs_articles)} articles, vectors {theirs_vectors.shape}")

    results = [
        score("ours (current corpus)", ours_vectors, ours_articles, gold, model),
        score("theirs (LegalLens)", theirs_vectors, theirs_articles, gold, model),
    ]

    print("\n" + "─" * 64)
    print("  DENSE-ONLY RETRIEVAL, SAME 40 CLAUSES, SAME EMBEDDING MODEL")
    print("─" * 64)
    print(f"{'corpus':<26}{'articles':>10}{'R@3':>10}{'R@10':>10}{'MRR':>8}")
    for r in results:
        print(f"{r['name']:<26}{r['articles']:>10}"
              f"{r['recall@3'] * 100:>9.1f}%{r['recall@10'] * 100:>9.1f}%{r['mrr']:>8.3f}")

    a, b = results
    print("\ndelta (theirs - ours):"
          f"  R@3 {(b['recall@3'] - a['recall@3']) * 100:+.1f}pp"
          f"  R@10 {(b['recall@10'] - a['recall@10']) * 100:+.1f}pp"
          f"  MRR {b['mrr'] - a['mrr']:+.3f}")
    print("─" * 64)

    out = HERE / "results" / "corpus_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
