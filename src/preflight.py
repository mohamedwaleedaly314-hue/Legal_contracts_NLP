# -*- coding: utf-8 -*-
"""
Startup validation.

A deployment fails in ways that look like bugs. The index was gitignored and
never reached the container, so Qdrant came up empty and every clause returned
"no legal articles found" - which reads as a bad answer, not a bad deploy. The
embedding model in the image differed from the one that built the vectors, so
similarity scores were noise. The API key was missing, so the OCR repair
silently no-opped and the contract came out unreadable.

None of those announce themselves. Each is a config mistake wearing the
costume of a quality problem, and each is cheap to detect before the first
request. That is what this module is for.

    python -m src.preflight          # human-readable, exit 1 on failure
    python -m src.preflight --json   # for a container healthcheck
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent

OK, WARN, FAIL = "ok", "warn", "fail"


# These mirror src/retriever.py deliberately rather than importing it.
# Importing the retriever loads a 2.2 GB embedding model and opens the local
# Qdrant folder - so a preflight that imported it would be slow, and would take
# the very lock it is about to test, reporting a failure it caused itself.
# A container healthcheck has to stay cheap and side-effect free.
_DATA_OVERRIDE = os.environ.get("MIZAN_DATA_DIR", "").strip()
if _DATA_OVERRIDE:
    DATA_DIR = Path(_DATA_OVERRIDE)
    if not DATA_DIR.is_absolute():
        DATA_DIR = REPO / DATA_DIR
else:
    DATA_DIR = REPO / "data"

ARTICLES_PATH = DATA_DIR / "legal_articles.json"
BM25_PATH = DATA_DIR / "bm25.pkl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
QDRANT_PATH = DATA_DIR / "index" / "legal_rag_qdrant"
COLLECTION_NAME = "egyptian_legal_articles_contract_types"
EMBEDDING_MODEL_NAME = os.environ.get("MIZAN_EMBEDDING_MODEL", "BAAI/bge-m3")


@dataclass
class Check:
    name: str
    status: str
    message: str
    hint: Optional[str] = None


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message, "hint": c.hint}
                for c in self.checks
            ],
        }


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------
def _check_data_files(report: Report) -> Optional[Path]:
    """The corpus files the retriever loads at import."""
    missing = [p.name for p in (ARTICLES_PATH, BM25_PATH, EMBEDDINGS_PATH) if not p.exists()]
    if missing:
        report.checks.append(Check(
            "corpus files", FAIL,
            f"ناقص من {DATA_DIR}: {', '.join(missing)}",
            "هذه الملفات مستبعدة من Git. راجع DEPLOYMENT.md — قسم Index.",
        ))
        return None

    report.checks.append(Check("corpus files", OK, f"موجودة في {DATA_DIR}"))
    return DATA_DIR


def _check_corpus_consistency(report: Report) -> None:
    """Article count must equal vector count, or every lookup is off by one."""
    try:
        import numpy as np

        articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
        vectors = np.load(EMBEDDINGS_PATH, mmap_mode="r")

        if len(articles) != vectors.shape[0]:
            report.checks.append(Check(
                "corpus consistency", FAIL,
                f"{len(articles)} مادة مقابل {vectors.shape[0]} متجه",
                "أعد بناء الفهرس: python -m scripts.adopt_legallens_corpus --apply",
            ))
            return

        report.checks.append(Check(
            "corpus consistency", OK,
            f"{len(articles)} مادة، متجهات بأبعاد {vectors.shape[1]}",
        ))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(Check("corpus consistency", FAIL, str(exc)[:160]))


def _check_qdrant(report: Report) -> None:
    """Reachable, has the collection, and the collection is not empty."""
    try:
        from qdrant_client import QdrantClient

        url = os.environ.get("QDRANT_URL", "").strip()
        if url:
            client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
            where = url
        else:
            if not QDRANT_PATH.exists():
                report.checks.append(Check(
                    "qdrant", FAIL, f"مجلد الفهرس غير موجود: {QDRANT_PATH}",
                    "شغّل scripts/adopt_legallens_corpus.py أو اضبط QDRANT_URL.",
                ))
                return
            client = QdrantClient(path=str(QDRANT_PATH))
            where = str(QDRANT_PATH)

        try:
            if not client.collection_exists(COLLECTION_NAME):
                report.checks.append(Check(
                    "qdrant", FAIL,
                    f"المجموعة «{COLLECTION_NAME}» غير موجودة في {where}",
                    "الفهرس لم يُبنَ أو بُني باسم مجموعة مختلف.",
                ))
                return

            count = client.count(collection_name=COLLECTION_NAME).count
            if count == 0:
                # The failure this whole module exists for: an empty collection
                # answers every query with nothing and looks like a bad model.
                report.checks.append(Check(
                    "qdrant", FAIL,
                    f"المجموعة «{COLLECTION_NAME}» فارغة (صفر متجه)",
                    "الفهرس لم يصل إلى النشر. راجع DEPLOYMENT.md — قسم Index.",
                ))
                return

            report.checks.append(Check("qdrant", OK, f"{count} متجه في {where}"))
        finally:
            client.close()

    except Exception as exc:  # noqa: BLE001
        report.checks.append(Check(
            "qdrant", FAIL, str(exc)[:200],
            "إذا كان الخطأ «already accessed»، فهناك عملية أخرى تفتح نفس المجلد؛ "
            "استخدم QDRANT_URL مع خدمة Qdrant.",
        ))


def _check_embedding_model(report: Report) -> None:
    """Cached locally, or the first request pays a 2.2 GB download."""
    cache = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        or (Path.home() / ".cache" / "huggingface")
    )
    slug = "models--" + EMBEDDING_MODEL_NAME.replace("/", "--")
    present = any(cache.rglob(slug)) if cache.exists() else False

    if present:
        report.checks.append(Check("embedding model", OK, f"{EMBEDDING_MODEL_NAME} مخزّن محلياً"))
    else:
        report.checks.append(Check(
            "embedding model", WARN,
            f"{EMBEDDING_MODEL_NAME} غير مخزّن — أول طلب سينزّل ~2.2 جيجا",
            "اخبز الموديل في صورة Docker أو ثبّت HF_HOME على وحدة تخزين دائمة.",
        ))


def _check_llm(report: Report) -> None:
    from .config import settings

    if settings.provider == "groq":
        if not settings.groq_api_key:
            report.checks.append(Check(
                "llm", FAIL, "GROQ_API_KEY غير مضبوط",
                "أضفه إلى أسرار النشر. راجع .env.example.",
            ))
        elif not settings.groq_api_key.startswith("gsk_"):
            report.checks.append(Check("llm", FAIL, "GROQ_API_KEY لا يبدو مفتاحاً صالحاً"))
        else:
            report.checks.append(Check(
                "llm", OK, f"groq / {settings.analysis_model}",
            ))
    else:
        report.checks.append(Check("llm", OK, f"ollama / {settings.analysis_model}"))


def _check_ocr(report: Report) -> None:
    """Optional: only scanned input needs it."""
    from .ocr import HAS_TESSERACT

    binary = False
    if HAS_TESSERACT:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            binary = True
        except Exception:  # noqa: BLE001
            binary = False

    if binary:
        report.checks.append(Check("ocr", OK, "Tesseract متاح"))
    else:
        report.checks.append(Check(
            "ocr", WARN,
            "Tesseract غير مثبت — الملفات الممسوحة ضوئياً لن تُقرأ",
            "ثبّت tesseract-ocr و tesseract-ocr-ara (موجودان في Dockerfile).",
        ))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def run(include_qdrant: bool = True) -> Report:
    """Every check. Never raises - a failure is data, not an exception."""
    report = Report()

    if _check_data_files(report) is not None:
        _check_corpus_consistency(report)
        if include_qdrant:
            _check_qdrant(report)

    _check_embedding_model(report)
    _check_llm(report)
    _check_ocr(report)
    return report


_ICON = {OK: "✓", WARN: "!", FAIL: "✗"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    as_json = "--json" in sys.argv
    # A container healthcheck should not fight the app for the local Qdrant
    # folder lock; --no-qdrant lets it skip that one check.
    report = run(include_qdrant="--no-qdrant" not in sys.argv)

    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        print("\nMizan preflight")
        print("─" * 58)
        for check in report.checks:
            print(f"  {_ICON[check.status]} {check.name:<20} {check.message}")
            if check.hint and check.status != OK:
                print(f"      → {check.hint}")
        print("─" * 58)
        print("READY\n" if report.ok else "NOT READY — fix the ✗ items above\n")

    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
