"""
Central configuration for the contract-AI engine.

Everything here is overridable via environment variables so the same code
runs unchanged in a notebook, a local dev run, and the deployed pipeline
(e.g. different Ollama host per environment, different model per stage).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Load .env before any setting is read - every default below is resolved at
# import time, so a later load would come too late.
#
# The repo root is located from this file, not from the working directory:
# `streamlit run <subdir>/app.py` leaves the cwd above the project, and a
# cwd-relative search then silently finds nothing - which showed up as the app
# reporting provider "ollama" with a perfectly good GROQ_API_KEY sitting in
# .env two directories down.
try:
    from pathlib import Path

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(find_dotenv(usecwd=True))  # then any .env nearer the caller
except ImportError:  # python-dotenv is optional; real env vars still work
    pass

# Also sync Streamlit Cloud Secrets (st.secrets) into os.environ if running inside Streamlit
try:
    import streamlit as _st
    if hasattr(_st, "secrets"):
        for _k, _v in _st.secrets.items():
            if isinstance(_v, str) and _k not in os.environ:
                os.environ[_k] = _v
except Exception:
    pass


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------
# Ollama was the original target, but the analysis models it runs need a GPU
# to be usable: on a CPU-only machine qwen2.5:14b takes minutes per clause,
# which a dozen-clause contract turns into a wait nobody sits through. Groq
# serves the same class of model over HTTP in seconds, so it is the default
# whenever a key is configured. Ollama stays fully supported - set
# CONTRACT_AI_PROVIDER=ollama.
PROVIDER = os.getenv(
    "CONTRACT_AI_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama"
).lower()

# Model ids must exist on the account's Groq catalogue - list it with
# `client.models.list()` if a call comes back 404.
#
# Groq's free tier meters two separate things, and both bite:
#
#   requests per day  compound-mini allows 250, which a day of testing empties
#                     - and an empty bucket fails every clause outright.
#   tokens per minute 8,000 on the 1,000-request models; one clause costs
#                     ~2,455, so roughly three clauses a minute.
#
# qwen3.8-27b is the balance: 1,000 requests a day (~80 twelve-clause
# contracts), 8,000 TPM, ~2s a clause, and the most fluent Arabic of the
# models on this tier. Check the headroom before a demo with
# `client.models.list()` and the x-ratelimit-remaining-* response headers.
_DEFAULT_ANALYSIS_MODEL = {
    "groq": "qwen/qwen3.8-27b",
    "ollama": "qwen2.5:14b",
}.get(PROVIDER, "qwen2.5:14b")

_DEFAULT_DRAFTING_MODEL = {
    "groq": "openai/gpt-oss-20b",
    "ollama": "qwen2.5:7b",
}.get(PROVIDER, "qwen2.5:7b")


@dataclass(frozen=True)
class Settings:
    # "groq" or "ollama" - see PROVIDER above.
    provider: str = PROVIDER
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Ollama connection. OLLAMA_HOST is read natively by the `ollama` client,
    # but we surface it here too so pipeline code can log/validate it.
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Separate models per stage, since drafting used a lighter model (7b)
    # than analysis (14b) in the original notebook. Override independently.
    # Defaults follow the active provider.
    analysis_model: str = os.getenv("CONTRACT_AI_ANALYSIS_MODEL", _DEFAULT_ANALYSIS_MODEL)
    drafting_model: str = os.getenv("CONTRACT_AI_DRAFTING_MODEL", _DEFAULT_DRAFTING_MODEL)

    analysis_temperature: float = _env_float("CONTRACT_AI_ANALYSIS_TEMP", 0.1)
    summary_temperature: float = _env_float("CONTRACT_AI_SUMMARY_TEMP", 0.2)
    drafting_temperature: float = _env_float("CONTRACT_AI_DRAFTING_TEMP", 0.2)

    clause_retries: int = _env_int("CONTRACT_AI_CLAUSE_RETRIES", 2)
    # Cap on concurrent in-flight LLM calls during parallel clause analysis.
    # compound-mini's endpoint advertises 70,000 TPM but routes to
    # llama-3.3-70b, whose own limit is 12,000 - which is the number that
    # actually binds. At ~2,000 tokens a clause that is roughly six clauses a
    # minute, so more than two in flight only produces collisions and backoff.
    max_concurrent_clauses: int = _env_int(
        "CONTRACT_AI_MAX_CONCURRENCY", 1 if PROVIDER == "groq" else 2
    )

    # Tokens-per-minute ceiling the client paces itself against. Must match the
    # active model's real limit: 8,000 for the qwen3/gpt-oss models, 12,000 for
    # what groq/compound routes to. Setting it too high just moves the failure
    # back to the API. Ollama is local and unmetered.
    tokens_per_minute: int = _env_int(
        "CONTRACT_AI_TPM", 8000 if PROVIDER == "groq" else 1_000_000
    )

    # Output-token pacing (Groq enforces 1,000 OTPM ceiling on free tier).
    output_tokens_per_minute: int = _env_int(
        "CONTRACT_AI_OTPM", 900 if PROVIDER == "groq" else 0
    )

    # Declared ceiling on the reply. Groq checks this against the remaining
    # OTPM budget (1,000 limit). Declaring 650 leaves room without tripping the check.
    max_output_tokens: int = _env_int("CONTRACT_AI_MAX_OUTPUT", 650)

    request_timeout_s: float = _env_float("CONTRACT_AI_TIMEOUT_S", 120.0)


settings = Settings()
