# -*- coding: utf-8 -*-
"""
Adopt the LegalLens corpus into Mizan's data layout.

    python -m scripts.adopt_legallens_corpus --source "<path to LegalLens>" [--apply]

Without --apply it writes to data_new/ and changes nothing, so the result can
be benchmarked against the live corpus before replacing it.

Why a conversion rather than swapping the retriever: LegalLens ships a
dense-only retriever, and our measured hybrid (dense + BM25 fused by RRF) beats
dense alone by +2.6pp Recall@3 and +6.2pp Recall@10 on the benchmark. Taking
their corpus and keeping our retrieval is strictly better than taking both.

Two things the conversion has to repair:

  law_name  Their payload carries a slug - "civil_code_131_1948". That field is
            what the UI prints under a clause, so adopting it verbatim would
            show a reader "المادة (149) — civil_code_131_1948". Mapped back to
            the Arabic title here.
  law_year  Absent from their payload; recovered from law_number ("131/1948").

Everything else survives, including two fields we did not have and want:
is_invalidity_rule (marks articles that establish nullity) and
primary_contract_type.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import List

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent

# The five laws in the LegalLens corpus, as a reader should see them.
LAW_TITLES = {
    "civil_code_131_1948": "القانون المدني",
    "commercial_law_17_1999": "قانون التجارة",
    "labor_law_14_2025": "قانون العمل",
    "companies_law_159_1981": "قانون الشركات",
    "consumer_protection_law_181_2018": "قانون حماية المستهلك",
}

# Same tokenizer scripts/build_index.py uses, so the BM25 index this writes is
# the one src/retriever.py expects to read.
_DIACRITICS = re.compile(r"[ً-ٰٟ]")


def norm(text: str) -> str:
    text = _DIACRITICS.sub("", text or "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[؀-ۿ]+", norm(text))


def convert(payload: dict) -> dict:
    """One LegalLens payload in Mizan's article schema."""
    law_id = payload.get("law_id") or payload.get("law_name") or ""
    law_number = str(payload.get("law_number") or "")

    number, year = None, None
    if "/" in law_number:
        number, _, year = law_number.partition("/")
    elif law_number:
        number = law_number
    if not year:
        match = re.search(r"(\d{4})", law_id)
        year = match.group(1) if match else None

    return {
        # what our retriever and UI read
        "law_name": LAW_TITLES.get(law_id, law_id),
        "law_number": number,
        "law_year": year,
        "article_number": str(payload.get("article_number") or "").strip(),
        "article_label": payload.get("title"),
        "text": payload.get("text") or "",
        "binding_type": payload.get("binding_type", "general"),
        "contract_types": payload.get("contract_types") or ["general"],
        "is_active": bool(payload.get("is_active", True)),
        "needs_executive_regulation": bool(payload.get("needs_executive_regulation", False)),
        "superseded_by": payload.get("superseded_by"),
        "start_page": payload.get("start_page"),
        "end_page": payload.get("end_page"),
        # kept from LegalLens - we had neither and both are useful
        "law_id": law_id,
        "doc_id": payload.get("doc_id"),
        "is_invalidity_rule": bool(payload.get("is_invalidity_rule", False)),
        "primary_contract_type": payload.get("primary_contract_type"),
        "source_url": payload.get("url"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to the LegalLens folder")
    parser.add_argument("--apply", action="store_true",
                        help="write into data/ instead of data_new/")
    args = parser.parse_args()

    import numpy as np
    from rank_bm25 import BM25Okapi
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    export = Path(args.source) / "index" / "qdrant_export" / "legallens_legal_only_points.json.gz"
    if not export.exists():
        raise SystemExit(f"export not found: {export}")

    out = REPO / ("data" if args.apply else "data_new")
    index_dir = out / "index" / "legal_rag_qdrant"

    print(f"reading {export.name} ...")
    with gzip.open(export, "rt", encoding="utf-8") as fh:
        points = json.load(fh)

    articles = [convert(p["payload"]) for p in points]
    vectors = np.asarray([p["vector"] for p in points], dtype="float32")
    print(f"  {len(articles)} articles, vectors {vectors.shape}")

    unmapped = {a["law_id"] for a in articles if a["law_name"] == a["law_id"]}
    if unmapped:
        print(f"  WARNING: no Arabic title for {sorted(unmapped)} - add it to LAW_TITLES")

    empty = sum(1 for a in articles if not a["text"].strip())
    if empty:
        print(f"  WARNING: {empty} article(s) have empty text")

    out.mkdir(parents=True, exist_ok=True)

    # ---- articles + embeddings ------------------------------------------
    (out / "legal_articles.json").write_text(
        json.dumps(articles, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(out / "embeddings.npy", vectors)
    print(f"  wrote legal_articles.json and embeddings.npy")

    # ---- BM25 ------------------------------------------------------------
    # LegalLens dropped the sparse half; the benchmark says it is worth
    # keeping, so it is rebuilt here from their (cleaner) text.
    corpus = [tokenize(a["text"]) for a in articles]
    with open(out / "bm25.pkl", "wb") as fh:
        pickle.dump({"bm25": BM25Okapi(corpus), "corpus": corpus}, fh)
    print(f"  wrote bm25.pkl")

    # ---- Qdrant ----------------------------------------------------------
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(index_dir))
    collection = "egyptian_legal_articles_contract_types"
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    for start in range(0, len(articles), 200):
        batch = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=articles[i])
            for i in range(start, min(start + 200, len(articles)))
        ]
        client.upsert(collection_name=collection, points=batch, wait=True)
    count = client.count(collection_name=collection).count
    client.close()
    print(f"  wrote qdrant collection '{collection}' with {count} points")

    print(f"\ndone -> {out.relative_to(REPO)}")
    if not args.apply:
        print("benchmark it, then re-run with --apply to replace data/")


if __name__ == "__main__":
    main()
