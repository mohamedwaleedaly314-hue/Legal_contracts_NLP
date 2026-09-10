# -*- coding: utf-8 -*-
"""
Smoke checks — the things that must be true before anything is deployed.

    python -m eval.test_smoke

No pytest dependency on purpose: this has to run inside a container where the
only thing installed is requirements.txt.

Scope is deliberately narrow. These are not unit tests for behaviour - the
benchmark in run_eval.py measures quality. What these catch is the class of
failure that has actually bitten this project: a module that stops importing,
a corpus whose vectors no longer line up with its articles, a config that
silently reads nothing, an embedding model that does not match the index it
queries.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run the function, record pass/fail, never raise."""
    def wrap(fn):
        try:
            detail = fn() or ""
            _results.append((name, True, str(detail)))
        except AssertionError as exc:
            _results.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(limit=2)
        return fn
    return wrap


# --------------------------------------------------------------------------
@check("imports")
def _imports():
    import src.analysis, src.classifier, src.config, src.legal_concepts  # noqa: F401
    import src.legal_text, src.llm_client, src.ocr, src.orchestrator, src.preflight  # noqa: F401
    import src.prompts, src.schemas  # noqa: F401
    return "12 modules"


@check("config resolves")
def _config():
    from src.config import settings
    assert settings.provider in {"groq", "ollama"}, f"bad provider {settings.provider!r}"
    assert settings.analysis_model, "no analysis model configured"
    assert settings.tokens_per_minute > 0, "tokens_per_minute must be positive"
    return f"{settings.provider} / {settings.analysis_model}"


@check("corpus integrity")
def _corpus():
    import json

    import numpy as np

    from src.preflight import ARTICLES_PATH, EMBEDDINGS_PATH

    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    assert len(articles) == vectors.shape[0], (
        f"{len(articles)} articles vs {vectors.shape[0]} vectors")
    blank = sum(1 for a in articles if not (a.get("text") or "").strip())
    assert blank == 0, f"{blank} articles have no text"
    unnamed = sum(1 for a in articles if not (a.get("law_name") or "").strip())
    assert unnamed == 0, f"{unnamed} articles have no law_name"
    return f"{len(articles)} articles, dim {vectors.shape[1]}"


@check("embedding dim matches the index")
def _dims():
    import numpy as np
    from qdrant_client import QdrantClient

    from src.preflight import COLLECTION_NAME, EMBEDDINGS_PATH, QDRANT_PATH

    vectors = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        info = client.get_collection(COLLECTION_NAME)
        size = info.config.params.vectors.size
        assert size == vectors.shape[1], f"index dim {size} vs corpus dim {vectors.shape[1]}"
        count = client.count(collection_name=COLLECTION_NAME).count
        assert count == vectors.shape[0], f"index has {count} points, corpus has {vectors.shape[0]}"
        return f"{count} points at dim {size}"
    finally:
        client.close()


@check("clause segmentation")
def _segmentation():
    from src.ocr import extract

    sample = REPO / "samples" / "contract_rent_sample.txt"
    assert sample.exists(), "sample contract missing"
    result = extract(str(sample), refine=False)
    assert result["status"] == "ok", result["errors"]
    labels = [c["clause_label"] for c in result["clauses"]]
    # The heading-corruption bug turned "البند الأول" into "البند الأو ل" and
    # collapsed every contract to paragraph splitting.
    assert "البند الأول" in labels, f"headings not parsed: {labels[:4]}"
    return f"{len(result['clauses'])} clauses"


@check("classifier")
def _classifier():
    from src.classifier import SUPPORTED_TYPE_IDS, classify
    from src.ocr import extract

    result = extract(str(REPO / "samples" / "contract_rent_sample.txt"), refine=False)
    out = classify(result["clean_text"], result["clauses"])
    assert out["contract_type"] in SUPPORTED_TYPE_IDS, out["contract_type"]
    assert out["parties"], "no parties extracted"
    return f"{out['contract_type_ar']} ({out['type_confidence']:.0%})"


@check("legal concept expansion")
def _concepts():
    from src.legal_concepts import matched_rules

    waiver = "يقر المستأجر بتنازله النهائي عن حقه في اللجوء إلى القضاء."
    plain = "يلتزم المستأجر بأداء الأجرة في المواعيد المتفق عليها."
    assert matched_rules(waiver), "no rule fired on a litigation-waiver clause"
    assert not matched_rules(plain), "a rule fired on an ordinary rent clause"
    return f"{len(matched_rules(waiver))} rule(s) on the waiver clause"


@check("retrieval returns governing law")
def _retrieval():
    from src.orchestrator import retrieve_for_clause

    articles = retrieve_for_clause(
        "يقر المستأجر بتنازله النهائي عن حقه في اللجوء إلى القضاء.", "residential", 5)
    assert articles, "retrieval returned nothing"
    assert all(a.get("text") for a in articles), "an article came back with no text"
    assert all(a.get("law_name") for a in articles), "an article came back with no law_name"
    return f"{len(articles)} articles, top = {articles[0].get('law_name')} " \
           f"م.{articles[0].get('article_number')}"


@check("evidence gating")
def _evidence():
    from src.orchestrator import _settle_evidence

    ungrounded = _settle_evidence(
        {"is_void_legal_term": True, "evidence_status": "مسند", "confidence": "عالية"},
        {"verified": [], "unverified": ["المادة 97 من الدستور"]},
        [{"article_number": "568"}],
    )
    assert ungrounded["is_void"] is None, "a nullity claim survived without evidence"
    grounded = _settle_evidence(
        {"is_void_legal_term": True, "evidence_status": "مسند", "confidence": "عالية"},
        {"verified": ["المادة 568"], "unverified": []},
        [{"article_number": "568"}],
    )
    assert grounded["is_void"] is True, "a grounded nullity claim was dropped"
    return "ungrounded verdicts withheld, grounded verdicts kept"


@check("preflight")
def _preflight():
    from src.preflight import run

    report = run(include_qdrant=False)
    failures = [c.name for c in report.failures]
    assert not failures, f"preflight failing: {failures}"
    return f"{len(report.checks)} checks, {len(report.warnings)} warning(s)"


# --------------------------------------------------------------------------
def main() -> None:
    width = max(len(n) for n, _, _ in _results)
    print("\nMizan smoke checks")
    print("─" * (width + 46))
    for name, passed, detail in _results:
        print(f"  {'✓' if passed else '✗'} {name:<{width}}  {detail}")
    print("─" * (width + 46))

    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}\n")
        sys.exit(1)
    print(f"all {len(_results)} checks passed\n")


if __name__ == "__main__":
    main()
