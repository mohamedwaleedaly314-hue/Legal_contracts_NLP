# -*- coding: utf-8 -*-
"""
Load the corpus into a Qdrant *server*.

    QDRANT_URL=http://localhost:6333 python -m scripts.load_qdrant_server

`src/retriever.py` already talks to a remote Qdrant when QDRANT_URL is set,
but nothing put the collection there - so the setting worked and the app still
found nothing. This is the missing half.

It cannot be a file copy. `qdrant-client` in local mode keeps one SQLite file
with a `points` table; a Qdrant server keeps segments and a write-ahead log.
Mounting one as the other leaves the server with an empty collection and no
error, so the vectors are re-uploaded from data/embeddings.npy instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COLLECTION = "egyptian_legal_articles_contract_types"
BATCH = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "").strip())
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", "").strip())
    parser.add_argument("--data-dir", default=os.environ.get("MIZAN_DATA_DIR", "data"))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--recreate", action="store_true",
                        help="drop the collection first (data loss)")
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("QDRANT_URL is required (or pass --url)")

    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    data = Path(args.data_dir)
    if not data.is_absolute():
        data = REPO / data

    articles = json.loads((data / "legal_articles.json").read_text(encoding="utf-8"))
    vectors = np.load(data / "embeddings.npy")
    if len(articles) != vectors.shape[0]:
        raise SystemExit(
            f"corpus mismatch: {len(articles)} articles, {vectors.shape[0]} vectors"
        )
    print(f"{len(articles)} articles, vectors {vectors.shape}")

    client = QdrantClient(url=args.url, api_key=args.api_key or None)

    exists = client.collection_exists(args.collection)
    if exists and not args.recreate:
        count = client.count(collection_name=args.collection).count
        if count == len(articles):
            print(f"'{args.collection}' already holds {count} points - nothing to do")
            return
        print(f"'{args.collection}' holds {count} points, expected {len(articles)}"
              " - pass --recreate to rebuild it")
        return

    if exists:
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    for start in range(0, len(articles), BATCH):
        end = min(start + BATCH, len(articles))
        client.upsert(
            collection_name=args.collection,
            points=[
                PointStruct(id=i, vector=vectors[i].tolist(), payload=articles[i])
                for i in range(start, end)
            ],
            wait=True,
        )
        print(f"  {end}/{len(articles)}")

    count = client.count(collection_name=args.collection).count
    print(f"done: '{args.collection}' now holds {count} points at {args.url}")


if __name__ == "__main__":
    main()
