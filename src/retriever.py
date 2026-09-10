# ============================================================
# LegalLens - Contract RAG Retriever
# modules/retriever.py
# ============================================================

import json
import pickle
import re
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)


# ============================================================
# PATHS
# ============================================================

# Project root:
# contract-analyzer/
# ├── src/
# │   └── retriever.py
# └── data/
#     ├── legal_articles.json
#     ├── bm25.pkl
#     ├── embeddings.npy
#     └── index/
#         └── legal_rag_qdrant/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MIZAN_DATA_DIR lets a deployment mount the corpus somewhere else, and lets a
# candidate corpus be benchmarked side by side with the live one before it
# replaces anything. Relative values resolve against the repo root.
_DATA_OVERRIDE = os.environ.get("MIZAN_DATA_DIR", "").strip()
if _DATA_OVERRIDE:
    DATA_DIR = Path(_DATA_OVERRIDE)
    if not DATA_DIR.is_absolute():
        DATA_DIR = PROJECT_ROOT / DATA_DIR
else:
    DATA_DIR = PROJECT_ROOT / "data"

INDEX_DIR = DATA_DIR / "index"

ARTICLES_PATH = DATA_DIR / "legal_articles.json"

BM25_PATH = DATA_DIR / "bm25.pkl"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

QDRANT_PATH = INDEX_DIR / "legal_rag_qdrant"

COLLECTION_NAME = "egyptian_legal_articles_contract_types"


# Whether the contract-type tag is used to narrow the search at all. See the
# comment in dense_search. Off is right for a corpus whose type vocabulary
# does not match the classifier's.
# Default off: the shipped corpus tags articles with its own vocabulary
# (lease / residential_lease / agency) which does not match the nine ids the
# classifier emits, and 18% of its articles carry no "general" tag at all.
# Filtering under those conditions cost 16 points of Recall@3 on the benchmark.
# Set MIZAN_TYPE_FILTER=1 for a corpus whose tags match the classifier.
TYPE_FILTER = os.environ.get("MIZAN_TYPE_FILTER", "0").strip().lower() not in {"0", "false", "no"}


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


# ============================================================
# LOAD DATA ONCE
# ============================================================

print("Loading LegalLens RAG...")

with open(
    ARTICLES_PATH,
    "r",
    encoding="utf-8",
) as f:

    ALL_ARTICLES = json.load(f)


with open(
    BM25_PATH,
    "rb",
) as f:

    BM25_DATA = pickle.load(f)


BM25 = BM25_DATA["bm25"]


EMBEDDINGS = np.load(
    EMBEDDINGS_PATH
)


# ============================================================
# LOAD EMBEDDING MODEL ONCE
# ============================================================

EMBEDDING_MODEL = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# CONNECT TO QDRANT ONCE
# ============================================================

# Prefer an HTTP Qdrant if `QDRANT_URL` is provided (useful for docker-compose).
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
if QDRANT_URL:
    QDRANT = QdrantClient(url=QDRANT_URL)
else:
    QDRANT = QdrantClient(path=str(QDRANT_PATH))


# ============================================================
# VALIDATION
# ============================================================

if len(ALL_ARTICLES) != len(EMBEDDINGS):

    raise RuntimeError(
        "Articles and embeddings count mismatch: "
        f"{len(ALL_ARTICLES)} articles vs "
        f"{len(EMBEDDINGS)} embeddings"
    )


if not QDRANT.collection_exists(
    COLLECTION_NAME
):

    raise RuntimeError(
        f"Qdrant collection not found: "
        f"{COLLECTION_NAME}"
    )


print(
    f"RAG loaded successfully: "
    f"{len(ALL_ARTICLES)} articles"
)


# ============================================================
# ARABIC NORMALIZATION
# ============================================================

def normalize_arabic(text: str) -> str:

    if not isinstance(text, str):

        return ""

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # Normalize Arabic letters
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ٱ", "ا")

    # Normalize Alef Maqsura
    text = text.replace("ى", "ي")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_arabic(text: str):

    text = normalize_arabic(
        text
    )

    return re.findall(
        r"[\u0600-\u06FF]+",
        text
    )


# ============================================================
# QUERY REWRITING
# ============================================================

QUERY_SYNONYMS = {

    "ايجار": "إيجار",

    "اجرة": "أجرة",

    "فلوس": "مقابل مالي",

    "مرتب": "أجر",

    "شغل": "عمل",

    "موظف": "عامل",
}


def rewrite_query(
    query: str
) -> str:

    if not isinstance(
        query,
        str
    ):

        raise TypeError(
            "query must be a string"
        )

    query = normalize_arabic(
        query
    )

    for old, new in QUERY_SYNONYMS.items():

        query = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            query,
            flags=re.IGNORECASE
        )

    return query.strip()


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query: str,
    contract_type: str,
    top_k: int = 20,
):

    query_embedding = (
        EMBEDDING_MODEL.encode(
            [rewrite_query(query)],
            normalize_embeddings=True,
        )[0]
    )


    query_filter = Filter(

        must=[

            # Type OR general. Filtering on the type alone starves the
            # search: only 9 of 1352 articles carry "residential", so every
            # clause in a rental contract came back with the same handful of
            # articles no matter what it said. Every article also carries
            # "general", and the civil code genuinely does govern all of
            # these contracts - so let relevance rank the corpus and let the
            # type tag lift its own articles rather than exclude the rest.
            #
            # Whether to filter at all is a property of the corpus, not of the
            # query: a corpus whose type vocabulary differs from the
            # classifier's (lease / residential_lease rather than residential)
            # or whose articles do not all carry "general" gets starved by any
            # filter. MIZAN_TYPE_FILTER=0 turns it off for such a corpus.
            *([FieldCondition(

                key="contract_types",

                match=MatchAny(
                    any=[
                        contract_type,
                        "general",
                    ]
                ),

            )] if TYPE_FILTER else []),

            FieldCondition(

                key="is_active",

                match=MatchValue(
                    value=True
                ),

            ),

        ]

    )


    results = QDRANT.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding.tolist(),

        query_filter=query_filter,

        limit=top_k,

    )


    return results.points


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    contract_type: str,
    top_k: int = 20,
):

    tokens = tokenize_arabic(
        rewrite_query(query)
    )


    scores = BM25.get_scores(
        tokens
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    results = []


    for idx in ranked_indices:

        article = ALL_ARTICLES[
            int(idx)
        ]


        article_types = set(
            article.get(
                "contract_types",
                ["general"]
            )
        )


        # Same rule as the dense filter above. This also revives BM25 at all:
        # legal_articles.json shipped without contract_types, so every article
        # defaulted to ["general"], failed a strict "residential" test, and the
        # sparse half of the hybrid search returned nothing for every query.
        if TYPE_FILTER and not article_types & {contract_type, "general"}:

            continue


        if not article.get(
            "is_active",
            True
        ):

            continue


        results.append({

            "index": int(idx),

            "score": float(
                scores[idx]
            ),

        })


        if len(results) >= top_k:

            break


    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k: int = 60,
):

    fused = {}


    # --------------------------
    # Dense results
    # --------------------------

    for rank, point in enumerate(
        dense_results,
        start=1
    ):

        idx = int(
            point.id
        )


        if idx not in fused:

            fused[idx] = {

                "score": 0.0,

                "dense_score": 0.0,

                "bm25_score": 0.0,

            }


        fused[idx][
            "score"
        ] += 1 / (
            k + rank
        )


        fused[idx][
            "dense_score"
        ] = float(
            point.score
        )


    # --------------------------
    # BM25 results
    # --------------------------

    for rank, item in enumerate(
        bm25_results,
        start=1
    ):

        idx = int(
            item["index"]
        )


        if idx not in fused:

            fused[idx] = {

                "score": 0.0,

                "dense_score": 0.0,

                "bm25_score": 0.0,

            }


        fused[idx][
            "score"
        ] += 1 / (
            k + rank
        )


        fused[idx][
            "bm25_score"
        ] = float(
            item["score"]
        )


    return sorted(

        fused.items(),

        key=lambda x: x[1]["score"],

        reverse=True,

    )


# ============================================================
# BUILD RESULT
# ============================================================

def format_result(
    idx: int,
    scores: dict,
):

    article = ALL_ARTICLES[
        idx
    ]


    return {

        # Retrieval information
        "score": float(
            scores["score"]
        ),

        "dense_score": float(
            scores["dense_score"]
        ),

        "bm25_score": float(
            scores["bm25_score"]
        ),


        # Legal citation
        "law_name": article.get(
            "law_name"
        ),

        "law_number": article.get(
            "law_number"
        ),

        "law_year": article.get(
            "law_year"
        ),

        "article_number": article.get(
            "article_number"
        ),

        "article_label": article.get(
            "article_label"
        ),

        # Legal metadata
        "binding_type": article.get(
            "binding_type",
            "general"
        ),

        "contract_types": article.get(
            "contract_types",
            ["general"]
        ),

        "needs_executive_regulation": (
            article.get(
                "needs_executive_regulation",
                False
            )
        ),

        "superseded_by": article.get(
            "superseded_by"
        ),

        "is_active": article.get(
            "is_active",
            True
        ),

        # Source location
        "start_page": article.get(
            "start_page"
        ),

        "end_page": article.get(
            "end_page"
        ),

        "source_page": article.get(
            "start_page"
        ),

        # Actual legal text
        "text": article.get(
            "text",
            ""
        ),

    }


# ============================================================
# MAIN PUBLIC API
# ============================================================

def retrieve(
    clause_text: str,
    contract_type: str,
    top_k: int = 3,
    exclude_penalty: bool = True,
) -> list[dict]:

    """
    Retrieve relevant legal articles for the Drafter.

    Parameters
    ----------
    clause_text : str
        User requirement or clause description.

    contract_type : str
        Contract type selected by the user.

        Supported values:
            residential
            agricultural
            commercial
            employment
            company
            sale
            supply
            service
            consumer

    top_k : int
        Number of final legal articles to return.

    exclude_penalty : bool
        Drop articles whose binding_type is "penalty". True suits the
        drafting flow (a new contract should not quote penalty provisions).
        Pass False for risk analysis, where penalty and nullity provisions
        are exactly the legal basis a harmful clause must be checked against.

    Returns
    -------
    list[dict]
        Ranked legal articles with citation metadata,
        similarity scores, and legal text.
    """


    if not isinstance(
        clause_text,
        str
    ):

        raise TypeError(
            "clause_text must be a string"
        )


    if not clause_text.strip():

        raise ValueError(
            "clause_text cannot be empty"
        )


    if not isinstance(
        contract_type,
        str
    ):

        raise TypeError(
            "contract_type must be a string"
        )


    if not isinstance(
        top_k,
        int
    ) or top_k <= 0:

        raise ValueError(
            "top_k must be a positive integer"
        )


    supported_types = {

        "residential",
        "agricultural",
        "commercial",
        "employment",
        "company",
        "sale",
        "supply",
        "service",
        "consumer",

    }


    if contract_type not in supported_types:

        raise ValueError(

            f"Unsupported contract_type: "
            f"{contract_type}. "

            f"Supported types: "
            f"{sorted(supported_types)}"

        )


    # ========================================================
    # Retrieve more candidates than final top_k
    # ========================================================

    candidate_k = max(
        20,
        top_k * 5
    )


    # ========================================================
    # Dense Retrieval
    # ========================================================

    dense_results = dense_search(

        query=clause_text,

        contract_type=contract_type,

        top_k=candidate_k,

    )


    # ========================================================
    # BM25 Retrieval
    # ========================================================

    bm25_results = bm25_search(

        query=clause_text,

        contract_type=contract_type,

        top_k=candidate_k,

    )


    # ========================================================
    # Hybrid Fusion
    # ========================================================

    fused_results = reciprocal_rank_fusion(

        dense_results,

        bm25_results,

    )


    # ========================================================
    # Build Final Results
    # ========================================================

    results = []


    for idx, scores in fused_results:

        idx = int(idx)


        article = ALL_ARTICLES[
            idx
        ]


        # Drafter should not retrieve penalty clauses; risk analysis needs them
        if exclude_penalty and article.get(
            "binding_type"
        ) == "penalty":

            continue


        result = format_result(
            idx,
            scores
        )


        results.append(
            result
        )


        if len(results) >= top_k:

            break


    # ========================================================
    # General fallback
    # ========================================================

    if len(results) < top_k:

        dense_general = (
            dense_search_general(
                clause_text,
                top_k=candidate_k
            )
        )


        bm25_general = (
            bm25_search_general(
                clause_text,
                top_k=candidate_k
            )
        )


        fused_general = (
            reciprocal_rank_fusion(
                dense_general,
                bm25_general
            )
        )


        existing_articles = {

            (
                r["law_name"],
                r["article_number"],
                r["text"]
            )

            for r in results

        }


        for idx, scores in fused_general:

            idx = int(idx)


            article = ALL_ARTICLES[
                idx
            ]


            if article.get(
                "contract_types",
                ["general"]
            ) != ["general"]:

                continue


            if exclude_penalty and article.get(
                "binding_type"
            ) == "penalty":

                continue


            key = (

                article.get(
                    "law_name"
                ),

                article.get(
                    "article_number"
                ),

                article.get(
                    "text"
                ),

            )


            if key in existing_articles:

                continue


            results.append(

                format_result(
                    idx,
                    scores
                )

            )


            if len(results) >= top_k:

                break


    return results[:top_k]


# ============================================================
# GENERAL ARTICLE SEARCH
# ============================================================

def dense_search_general(
    query: str,
    top_k: int = 20,
):

    query_embedding = (
        EMBEDDING_MODEL.encode(
            [rewrite_query(query)],
            normalize_embeddings=True,
        )[0]
    )


    query_filter = Filter(

        must=[

            FieldCondition(

                key="contract_types",

                match=MatchAny(
                    any=["general"]
                ),

            ),

            FieldCondition(

                key="is_active",

                match=MatchValue(
                    value=True
                ),

            ),

        ]

    )


    results = QDRANT.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding.tolist(),

        query_filter=query_filter,

        limit=top_k,

    )


    return results.points


def bm25_search_general(
    query: str,
    top_k: int = 20,
):

    tokens = tokenize_arabic(
        rewrite_query(query)
    )


    scores = BM25.get_scores(
        tokens
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    results = []


    for idx in ranked_indices:

        article = ALL_ARTICLES[
            int(idx)
        ]


        if article.get(
            "contract_types",
            ["general"]
        ) != ["general"]:

            continue


        if not article.get(
            "is_active",
            True
        ):

            continue


        results.append({

            "index": int(idx),

            "score": float(
                scores[idx]
            ),

        })


        if len(results) >= top_k:

            break


    return results


# ============================================================
# OPTIONAL: CONTEXT FOR LLM
# ============================================================

def build_legal_context(
    results: list[dict]
) -> str:

    context = []


    for result in results:

        context.append(

            f"""
القانون: {result.get("law_name", "")}
رقم القانون: {result.get("law_number", "")}
سنة القانون: {result.get("law_year", "")}
المادة: {result.get("article_number", "")}
نوع الحكم: {result.get("binding_type", "general")}

النص القانوني:
{result.get("text", "")}
""".strip()

        )


    return "\n\n".join(
        context
    )
