# -*- coding: utf-8 -*-
"""
The glue between the four stages.

    OCR  ->  classify  ->  retrieve (per clause)  ->  analyse (per clause)

Three things live here because they belong to no single stage:

1. Vocabulary translation. The retriever accepts only its nine English
   contract-type ids and raises ValueError on anything else; OCR and the
   analysis prompts speak Arabic. The classifier produces the id, and this
   module is what guarantees nothing else ever reaches retrieve().

2. Per-clause legal context. analysis.run_parallel_analysis takes one
   rag_context for the whole contract, so every clause would be judged
   against the same articles. Retrieving per clause and calling
   analyze_single_clause directly gives each clause the law that actually
   governs it - and makes the citations shown next to a clause real.

3. Degradation. A missing index, an unreachable model or one clause the LLM
   chokes on must not lose the whole run. Every stage failure is recorded in
   `warnings`/`errors` and the rest of the contract still comes back.

Public API:
    analyze_contract_file(path, progress=None) -> dict
"""
from __future__ import annotations

import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from . import ocr as ocr_module
from . import classifier as classifier_module
from .legal_concepts import legal_concepts
from .legal_text import clean_articles

logger = logging.getLogger("contract_ai.orchestrator")

REPO_ROOT = Path(__file__).resolve().parent.parent

# RiskLevel's values are Arabic; the UI keys off colour names.
RISK_COLOURS = {"أحمر": "red", "أصفر": "yellow", "أخضر": "green"}

DEFAULT_TOP_K = 3

ProgressFn = Callable[[str, float, str], None]


def _noop_progress(stage: str, pct: float, message: str) -> None:
    logger.info("[%s %.0f%%] %s", stage, pct * 100, message)


# --------------------------------------------------------------------------
# Retriever loading
# --------------------------------------------------------------------------
_retriever = None
_retriever_error: Optional[str] = None


def load_retriever():
    """Import the retriever once, on first use.

    Imported here rather than at module scope because it builds the embedding
    model and opens the Qdrant index while importing: doing that eagerly would
    stall the UI on startup. The import is cached by Python, so the cost is
    paid once per process and every later call is free.
    """
    global _retriever, _retriever_error
    if _retriever is not None or _retriever_error is not None:
        return _retriever

    try:
        from . import retriever as retriever_module
        _retriever = retriever_module
        logger.info("Retriever loaded")
    except Exception as exc:  # noqa: BLE001 - any failure degrades, not crashes
        _retriever_error = str(exc)
        logger.warning("Retriever unavailable: %s", exc)

    return _retriever


def retriever_status() -> dict:
    """For the UI's diagnostics panel - never raises."""
    return {"loaded": _retriever is not None, "error": _retriever_error}


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def _article_key(article: dict) -> tuple:
    """Identity for fusing lists that came from different queries."""
    return (article.get("law_name"), str(article.get("article_number")))


def _rrf(ranked_lists: List[List[dict]], k: int = 60) -> List[dict]:
    """Reciprocal rank fusion over several ranked article lists.

    Same formula the retriever already uses internally to combine dense with
    BM25; applied one level up to combine the results of different *queries*.
    """
    scores: dict = {}
    seen: dict = {}
    for ranked in ranked_lists:
        for rank, article in enumerate(ranked, start=1):
            key = _article_key(article)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            seen.setdefault(key, article)

    order = sorted(scores, key=lambda key: -scores[key])
    return [seen[key] for key in order]


def retrieve_for_clause(clause_text: str, contract_type: str, top_k: int) -> List[dict]:
    """Legal articles governing one clause. Returns [] when unavailable.

    The clause is searched twice. Once as written, which finds the article
    that reads like it - and once as the abstract legal question it raises,
    which is the only way to reach the general provisions that actually decide
    nullity. Measured: clauses governed by a topic article scored Recall@3 62%,
    clauses governed by a general principle 40%, and article 149 never
    surfaced at all for the four clauses it governs. Those articles share no
    vocabulary with the clause, so reranking a candidate list cannot help -
    they were never in it. See legal_concepts.py.

    exclude_penalty=False on purpose: the retriever drops penalty provisions
    because it was written for the drafting flow, but a void or abusive
    clause is judged precisely against those provisions.
    """
    retriever = load_retriever()
    if retriever is None:
        return []

    def search(query: str) -> List[dict]:
        try:
            return retriever.retrieve(
                clause_text=query,
                contract_type=contract_type,
                # Over-fetch so fusion has something to work with, then trim.
                top_k=max(top_k * 2, 6),
                exclude_penalty=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed for a query: %s", exc)
            return []

    ranked = [search(clause_text)]
    for concept in legal_concepts(clause_text):
        ranked.append(search(concept))

    articles = _rrf(ranked)[:top_k] if len(ranked) > 1 else ranked[0][:top_k]

    # The corpus was itself OCR'd and 58% of it is damaged. Repair it here,
    # once, so both the model's context and the article shown under the clause
    # are readable. See legal_text.py for what is and is not repaired.
    return clean_articles(articles)


# A handful of civil-code articles run to thousands of characters. The clause
# only ever turns on the opening rule, and the tail is what pushes a request
# over the token-per-minute limit, so cap what goes into the prompt. The full
# text is still stored on the clause and shown in the UI.
MAX_ARTICLE_CHARS = 800


def format_legal_context(articles: List[dict]) -> str:
    """Render retrieved articles as the plain text block the prompts expect."""
    if not articles:
        return "لا توجد مواد قانونية مسترجعة لهذا البند."

    lines = []
    for art in articles:
        law = art.get("law_name", "")
        number = art.get("law_number")
        year = art.get("law_year")
        ref = f"{law} رقم {number} لسنة {year}" if number and year else law

        text = art.get("text", "")
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS].rsplit(" ", 1)[0] + " […]"

        lines.append(f"- المادة ({art.get('article_number')}) من {ref}: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Citation verification
# --------------------------------------------------------------------------
# A citation reads "المادة 568 من القانون المدني رقم 131 لسنة 1948" - three
# numbers, only the first of which is the article. Anchor on the word.
_ARTICLE_NUMBER = re.compile(r"(?:المادة|مادة)\s*\(?\s*(\d+)")


def _cited_article_number(citation: str) -> Optional[str]:
    match = _ARTICLE_NUMBER.search(citation)
    if match:
        return match.group(1)
    # No anchor word: fall back to the first number in the string.
    fallback = re.search(r"\d+", citation)
    return fallback.group(0) if fallback else None


def verify_citations(cited: List[str], articles: List[dict]) -> dict:
    """Check every article the model cited against what was actually retrieved.

    The model is asked to cite only from the supplied context, but nothing
    forces it to. Flagging the ones that appear nowhere in the retrieved set
    is the cheapest defence against a confidently invented article number.
    """
    available = {str(a.get("article_number", "")).strip() for a in articles}
    available.discard("")

    verified, unverified = [], []
    for citation in cited or []:
        number = _cited_article_number(citation)
        (verified if number and number in available else unverified).append(citation)

    return {
        "verified": verified,
        "unverified": unverified,
        "all_verified": not unverified and bool(verified),
    }


# --------------------------------------------------------------------------
# Evidence settlement
# --------------------------------------------------------------------------
EVIDENCE_LABELS = {
    "grounded": "مسند إلى المواد المسترجعة",
    "model_knowledge": "استنتاج من معرفة النموذج",
    "insufficient": "لا توجد مواد كافية للحكم",
}


def _settle_evidence(result: dict, check: dict, articles: List[dict]) -> dict:
    """Decide how strongly this verdict is allowed to be stated.

    The model self-reports `evidence_status`, and it has every incentive to be
    generous with itself. verify_citations already knows, mechanically, whether
    the articles it cited were actually in front of it - so the claim is
    checked against that rather than taken at face value, and the two are
    reconciled by taking the weaker of them.

    The consequence that matters: `is_void` survives only when the evidence is
    grounded. A model asserting "باطل" from memory still gets its reasoning
    shown and its risk score kept, but the UI stops printing a legal ruling
    over it.
    """
    declared = result.get("evidence_status")
    declared = getattr(declared, "value", declared)
    confidence = result.get("confidence")
    confidence = getattr(confidence, "value", confidence)

    # Mechanical reading of the same question, independent of what was claimed.
    if not articles:
        observed = "insufficient"
    elif check["unverified"] and not check["verified"]:
        observed = "model_knowledge"
    elif check["verified"]:
        observed = "grounded"
    else:
        observed = "insufficient"  # cited nothing at all

    declared_map = {
        "مسند": "grounded",
        "استنتاج": "model_knowledge",
        "غير كاف": "insufficient",
    }
    claimed = declared_map.get(declared, observed)

    # Take the weaker of the two readings.
    rank = {"grounded": 2, "model_knowledge": 1, "insufficient": 0}
    status = min(claimed, observed, key=lambda s: rank[s])

    is_void = result.get("is_void_legal_term")
    note = None
    if is_void and status != "grounded":
        is_void = None
        note = (
            "رجّح التحليل بطلان هذا البند، لكن لم تُسند المواد المسترجعة هذا "
            "الحكم — فهو مؤشر خطورة يستدعي مراجعة محامٍ، لا حكماً بالبطلان."
        )
    elif status == "insufficient":
        note = "لم تُسترجع مواد قانونية متصلة بهذا البند، والتقييم اجتهادي."

    return {
        "is_void": is_void,
        "status": status,
        "note": note,
        "confidence": confidence or ("عالية" if status == "grounded" else "منخفضة"),
    }


# --------------------------------------------------------------------------
# Per-clause work
# --------------------------------------------------------------------------
def _process_clause(clause: dict, metadata: dict, contract_type: str, top_k: int) -> dict:
    """Retrieve, analyse and verify one clause. Never raises."""
    from .analysis import analyze_single_clause

    clause_id = clause.get("clause_id")
    articles = retrieve_for_clause(clause.get("clause_text", ""), contract_type, top_k)

    base = {
        "id": clause_id,
        "label": clause.get("clause_label", ""),
        "text": clause.get("clause_text", ""),
        "citations": articles,
    }

    try:
        result = analyze_single_clause(
            clause, metadata, format_legal_context(articles)
        ).model_dump()
    except Exception as exc:  # noqa: BLE001 - one bad clause must not sink the run
        from .llm_client import is_quota_error

        logger.warning("Analysis failed for clause %s: %s", clause_id, exc)
        quota = is_quota_error(exc)
        return {
            **base,
            "risk": "unknown",
            "risk_ar": "غير محدد",
            "risk_score": None,
            "is_void": None,
            "is_void_claimed": None,
            "evidence_status": "insufficient",
            "evidence_note": None,
            "confidence": None,
            "reasoning_steps": [],
            "simple_explanation": "",
            "legal_rationale": "",
            "suggestion": None,
            "cited_law_articles": [],
            "citation_check": {"verified": [], "unverified": [], "all_verified": False},
            "quota_exhausted": quota,
            "error": (
                "نفدت حصة الطلبات المتاحة لهذا الموديل."
                if quota else f"تعذر تحليل هذا البند: {exc}"
            ),
        }

    risk_ar = result.get("risk_level")
    risk_ar = getattr(risk_ar, "value", risk_ar)

    cited = result.get("cited_law_articles", [])
    check = verify_citations(cited, articles)
    evidence = _settle_evidence(result, check, articles)

    return {
        **base,
        "risk": RISK_COLOURS.get(risk_ar, "unknown"),
        "risk_ar": risk_ar,
        "risk_score": result.get("risk_score"),
        # The declared verdict, kept for the record...
        "is_void_claimed": result.get("is_void_legal_term"),
        # ...and the one the UI is allowed to state, which the evidence has to
        # earn. See _settle_evidence.
        "is_void": evidence["is_void"],
        "evidence_status": evidence["status"],
        "evidence_note": evidence["note"],
        "confidence": evidence["confidence"],
        "reasoning_steps": result.get("reasoning_steps", []),
        "simple_explanation": result.get("simple_explanation", ""),
        "legal_rationale": result.get("legal_rationale", ""),
        "suggestion": result.get("suggested_balanced_clause"),
        "cited_law_articles": cited,
        "citation_check": check,
        "error": None,
    }


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def build_summary(analysed: List[dict]) -> Optional[dict]:
    """Executive summary over the analysed clauses. None if it fails."""
    from .config import settings
    from .llm_client import call_structured
    from .schemas import OverallSummarySchema

    usable = [c for c in analysed if c.get("risk") != "unknown"]
    if not usable:
        return None

    digest = [
        {
            "clause_id": c["id"],
            "label": c["label"],
            "risk_level": c["risk_ar"],
            "risk_score": c["risk_score"],
            "legal_rationale": c["legal_rationale"],
        }
        for c in usable
    ]

    prompt = (
        "بناءً على نتائج البنود المحللة التالية:\n"
        f"{digest}\n\n"
        "قم بتوليد الملخص التنفيذي وتقييم العقد الإجمالي."
    )

    try:
        summary = call_structured(
            model=settings.analysis_model,
            system_prompt=None,
            user_prompt=prompt,
            response_schema=OverallSummarySchema,
            temperature=settings.summary_temperature,
            retries=settings.clause_retries,
            context="overall_summary",
        ).model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Summary generation failed: %s", exc)
        return None

    # OverallSummarySchema documents a 1-10 score but, unlike ClauseRiskAnalysis,
    # carries no validator - and the model reads a list of per-clause scores and
    # adds them up, which put "54/10" on screen. Rejecting the response would
    # cost the whole summary over one field, so derive the score instead: the
    # highest clause score is what a reader means by how risky the contract is.
    score = summary.get("overall_risk_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        summary["overall_risk_score"] = max(
            (c["risk_score"] for c in usable if c.get("risk_score")), default=None
        )

    return summary


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def analyze_contract_file(
    file_path: str,
    progress: Optional[ProgressFn] = None,
    top_k: int = DEFAULT_TOP_K,
    max_clauses: Optional[int] = None,
) -> dict:
    """Run a contract file through the whole pipeline.

    Parameters
    ----------
    file_path : str
        PDF, image, Word or text file.
    progress : callable(stage, fraction, message), optional
        Called as the run advances so a UI can show where it is.
    top_k : int
        Legal articles retrieved per clause.
    max_clauses : int, optional
        Analyse only the first N clauses. Useful for a quick pass on a long
        contract when time is short.

    Returns a report dict; inspect `status` rather than catching exceptions.
    """
    func_progress = progress or _noop_progress
    started = time.time()

    report = {
        "status": "error",
        "file_name": Path(file_path).name,
        "contract_type": None,
        "contract_type_ar": None,
        "type_confidence": 0.0,
        "is_confident": False,
        "parties": [],
        "date": None,
        "value": None,
        "duration": None,
        "clauses": [],
        "counts": {"red": 0, "yellow": 0, "green": 0, "unknown": 0},
        "summary": None,
        "raw_text": "",
        "clean_text": "",
        "warnings": [],
        "errors": [],
        "timings": {},
    }

    # ---- 1. OCR ----------------------------------------------------------
    func_progress("ocr", 0.05, "جارٍ استخراج النص من الملف...")
    t0 = time.time()
    ocr_result = ocr_module.extract(file_path)
    report["timings"]["ocr"] = round(time.time() - t0, 1)

    if ocr_result["status"] != "ok":
        report["errors"].extend(ocr_result["errors"])
        return report

    report["raw_text"] = ocr_result["raw_text"]
    report["clean_text"] = ocr_result["clean_text"]
    clauses = ocr_result["clauses"]

    if not clauses:
        report["errors"].append("لم يتم التعرف على أي بنود داخل المستند.")
        return report

    if max_clauses:
        clauses = clauses[:max_clauses]

    # ---- 2. Classification ----------------------------------------------
    func_progress("classify", 0.15, "جارٍ تحديد نوع العقد والأطراف...")
    t0 = time.time()
    classification = classifier_module.classify(ocr_result["clean_text"], clauses)
    report["timings"]["classify"] = round(time.time() - t0, 1)

    report.update({
        "contract_type": classification["contract_type"],
        "contract_type_ar": classification["contract_type_ar"],
        "type_confidence": classification["type_confidence"],
        "is_confident": classification["is_confident"],
        "parties": classification["parties"],
        "date": classification["date"],
        "value": classification["value"],
        "duration": classification["duration"],
    })
    report["warnings"].extend(classification["errors"])

    if not classification["is_confident"]:
        report["warnings"].append(
            f"ثقة تحديد نوع العقد منخفضة ({classification['type_confidence']:.0%}) — "
            "راجع النوع قبل الاعتماد على المواد القانونية المسترجعة."
        )

    # ---- 3. Warm the index before fanning out ---------------------------
    func_progress("retrieve", 0.25, "جارٍ تحميل الموسوعة القانونية...")
    t0 = time.time()
    if load_retriever() is None:
        report["warnings"].append(
            "تعذر تحميل قاعدة القوانين — سيتم تحليل البنود دون سند قانوني مسترجع. "
            f"السبب: {_retriever_error}"
        )
    report["timings"]["index_load"] = round(time.time() - t0, 1)

    # ---- 4. Retrieve + analyse, clause by clause ------------------------
    t0 = time.time()
    total = len(clauses)
    func_progress("analyze", 0.3, f"جارٍ تحليل {total} بند...")

    metadata = {"contract_type": classification["contract_type_ar"]}
    results: List[dict] = []
    done = 0

    from .config import settings

    with ThreadPoolExecutor(max_workers=settings.max_concurrent_clauses) as pool:
        futures = {
            pool.submit(
                _process_clause, clause, metadata, classification["contract_type"], top_k
            ): clause
            for clause in clauses
        }
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            func_progress(
                "analyze",
                0.3 + 0.6 * (done / total),
                f"تم تحليل {done} من {total} بند",
            )

    results.sort(key=lambda c: (c["id"] is None, c["id"]))
    report["clauses"] = results
    report["timings"]["analyze"] = round(time.time() - t0, 1)

    for clause in results:
        report["counts"][clause["risk"]] = report["counts"].get(clause["risk"], 0) + 1

    failed = [c for c in results if c.get("error")]
    starved = [c for c in results if c.get("quota_exhausted")]
    if starved:
        from .config import settings

        report["warnings"].append(
            f"نفدت حصة الطلبات اليومية للموديل «{settings.analysis_model}» — "
            f"{len(starved)} بند لم يُحلَّل. غيّر الموديل عبر "
            "CONTRACT_AI_ANALYSIS_MODEL في ملف .env، أو انتظر تجديد الحصة."
        )
    elif failed:
        reasons = {c["error"] for c in failed if c.get("error")}
        detail = f" السبب: {next(iter(reasons))}" if len(reasons) == 1 else ""
        report["warnings"].append(
            f"تعذر تحليل {len(failed)} بند من أصل {total}؛ باقي البنود مكتملة.{detail}"
        )

    unverified = sum(len(c["citation_check"]["unverified"]) for c in results)
    if unverified:
        # Not a malfunction - this is the anti-hallucination check reporting.
        # The wording matters: read as an error it looks like the analysis
        # broke, when what it means is that the model reached past the
        # retrieved articles into its own knowledge, and the reader should
        # confirm those references before relying on them.
        report["warnings"].append(
            f"{unverified} مادة استند إليها التحليل من معرفته العامة وليست ضمن "
            "المواد المسترجعة من الموسوعة. هذا تنبيه وقائي لا خطأ: راجع هذه "
            "المواد بنفسك قبل الاعتماد عليها، وهي معلَّمة عند البند."
        )

    # ---- 5. Summary ------------------------------------------------------
    func_progress("summary", 0.92, "جارٍ إعداد الملخص التنفيذي...")
    t0 = time.time()
    report["summary"] = build_summary(results)
    report["timings"]["summary"] = round(time.time() - t0, 1)
    if report["summary"] is None:
        report["warnings"].append("تعذر إنشاء الملخص التنفيذي.")

    report["timings"]["total"] = round(time.time() - started, 1)
    report["status"] = "ok"
    func_progress("done", 1.0, "اكتمل التحليل")
    return report


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("usage: python -m core.orchestrator <contract-file>")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = analyze_contract_file(sys.argv[1])
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/last_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nstatus={out['status']} counts={out['counts']} timings={out['timings']}")
    print("saved -> outputs/last_report.json")
