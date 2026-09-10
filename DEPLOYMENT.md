# Deploying Mizan

Every command here was run against this repository. Where something could not
be verified on the development machine it says so.

---

## 1. What the app needs

| | requirement | why |
|---|---|---|
| RAM | **≥ 3 GB** | `BAAI/bge-m3` is 2.2 GB of weights and stays resident; measured ~2.1 GB for the process |
| Disk | ~4 GB | image + model + 25 MB corpus |
| Cold start | ~30 s | model load; ~10 min if the model is not baked into the image |
| Network | outbound HTTPS | Groq API |
| System package | `tesseract-ocr`, `tesseract-ocr-ara` | scanned contracts only; in the Dockerfile |

> **Streamlit Community Cloud will not work.** Its 1 GB memory ceiling is below
> what the embedding model needs. This surprises people; it is the first thing
> to check before choosing a host.

Targets that do fit: **Hugging Face Spaces (Docker)**, any VPS with 4 GB, or a
container platform where you can set the memory limit.

---

## 2. Local setup

```bash
git clone https://github.com/Mohamadadel510/Legal_contracts_NLP.git
cd Legal_contracts_NLP
pip install -r requirements.txt
cp .env.example .env          # then put your GROQ_API_KEY in it
python -m src.preflight       # says exactly what is missing, if anything
streamlit run app.py
```

`python -m src.preflight` is the first thing to run when anything looks wrong.
It takes three seconds, loads no model, and checks the corpus, the index, the
API key and Tesseract.

---

## 3. Environment variables

Only the first one is required.

| variable | default | meaning |
|---|---|---|
| `GROQ_API_KEY` | — | **required.** From <https://console.groq.com/keys> |
| `CONTRACT_AI_PROVIDER` | `groq` when a key is set | `groq` or `ollama` |
| `CONTRACT_AI_ANALYSIS_MODEL` | `qwen/qwen3.8-27b` | must exist on your Groq account |
| `CONTRACT_AI_TPM` | `8000` | tokens-per-minute ceiling the client paces to |
| `CONTRACT_AI_MAX_CONCURRENCY` | `2` | clauses analysed in parallel |
| `MIZAN_DATA_DIR` | `data` | where the corpus lives |
| `MIZAN_TYPE_FILTER` | `0` (off) | narrow retrieval by contract type — see §6 |
| `QDRANT_URL` | — | empty = embedded Qdrant; set it to use a server |
| `QDRANT_API_KEY` | — | for a managed Qdrant |
| `HF_HOME` | `~/.cache/huggingface` | model cache; `/opt/models` inside the image |
| `TESSERACT_CMD` | — | path to `tesseract.exe` on Windows |

Never commit `.env`. `.dockerignore` keeps it out of the image and
`.gitignore` keeps it out of the repository — both are verified.

---

## 4. The legal index — read this before deploying

**The index is not in the repository.** `.gitignore` excludes
`data/index/`, `data/embeddings.npy` and `data/bm25.pkl` — 23 MB of the 25 MB
the app needs. A fresh `git clone` gets `legal_articles.json` and nothing else,
and the app will refuse to start with:

```
✗ corpus files    ناقص من .../data: bm25.pkl, embeddings.npy
```

That refusal is deliberate. An empty index answers every clause with "no legal
articles found", which reads as a bad model rather than a bad deploy.

### Rebuilding it

The corpus is reproducible from the LegalLens export:

```bash
python -m scripts.adopt_legallens_corpus --source /path/to/LegalLens --apply
```

That writes all four artefacts — `legal_articles.json`, `embeddings.npy`,
`bm25.pkl`, and the Qdrant collection — from `LegalLens/index/qdrant_export/`.
It takes about a minute and needs no GPU: the vectors ship with the export.

Verify afterwards:

```bash
python -m src.preflight     # corpus + index + collection size
python -m eval.test_smoke   # 10 checks including dimension match
```

### Shipping it instead

If you would rather not rebuild on each host, track the four artefacts with
Git LFS:

```bash
git lfs install
git lfs track "data/embeddings.npy" "data/bm25.pkl" "data/index/**"
# then remove those lines from .gitignore before committing
```

Note GitHub's LFS quota applies. The alternative is a GitHub Release asset
downloaded on first boot.

---

## 5. Docker

```bash
docker compose up --build
# http://localhost:8501
```

One container, embedded Qdrant. Streamlit is single-process, so the folder
lock that mode takes is not a limitation.

The build has two stages: the first downloads the embedding model, the second
copies it to `/opt/models`. That is what keeps the first request from paying a
2.2 GB download. It also makes the image roughly 4 GB — check your platform's
limit before building.

> **Not verified here.** Docker is not installed on the development machine, so
> the Dockerfile and compose file have been reviewed and reasoned about but not
> built. Run `docker compose up --build` once and read the output before
> relying on them.

### Using a Qdrant server

Needed only for more than one worker, or a managed Qdrant.

```bash
docker compose --profile server up -d qdrant
QDRANT_URL=http://localhost:6333 python -m scripts.load_qdrant_server
QDRANT_URL=http://qdrant:6333 docker compose --profile server up --build app
```

The load step is not optional and not a file copy. `qdrant-client` in local
mode writes one SQLite file with a `points` table; a Qdrant server stores
segments and a write-ahead log. Mounting one as the other leaves the server
with an empty collection and no error — the previous compose file did exactly
that.

---

## 6. Retrieval configuration

`MIZAN_TYPE_FILTER` defaults to **off**, and should stay off for the shipped
corpus.

The corpus tags articles with its own vocabulary — `lease`,
`residential_lease`, `agency` — which does not match the nine ids the
classifier emits, and 18% of its articles carry no `general` tag at all.
Filtering on the contract type under those conditions cost **16 points of
Recall@3** on the benchmark (63.7% → 47.5%). Turn it on only for a corpus
whose tags match the classifier's vocabulary.

---

## 7. Rate limits

Groq's free tier is shared across every user of your deployment.

| limit | value | effect |
|---|---|---|
| requests/day | 1,000 | ~80 twelve-clause contracts, total |
| tokens/minute | 8,000 | ~3 clauses/minute; a full contract takes 3–4 minutes |
| output tokens/minute | 1,000 | why `max_tokens` is declared explicitly |

Two users uploading at once queue behind each other. Fine for a demo; a real
deployment wants a paid tier.

Check remaining quota:

```bash
python -c "import os,httpx;from dotenv import load_dotenv;load_dotenv();r=httpx.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']},json={'model':'qwen/qwen3.8-27b','messages':[{'role':'user','content':'hi'}],'max_tokens':1});print({k:v for k,v in r.headers.items() if 'ratelimit' in k})"
```

---

## 8. Hugging Face Spaces

Create a Space with SDK **Docker**, push this repository, and set
`GROQ_API_KEY` under *Settings → Variables and secrets*. The Dockerfile reads
`$PORT`, which Spaces provides.

The index still has to reach the Space — §4 applies.

---

## 9. When something is wrong

Run `python -m src.preflight` first. It names the problem and the fix.

| symptom | cause |
|---|---|
| `corpus files ناقص` | index not deployed — §4 |
| `المجموعة فارغة (صفر متجه)` | Qdrant came up empty; with `QDRANT_URL` set, run the loader |
| `already accessed by another instance` | two processes on the embedded index — use `QDRANT_URL`, or stop the other one |
| `GROQ_API_KEY غير مضبوط` | secret not set on the platform |
| Every clause says "لم تُسترجع مواد" | empty collection, or `MIZAN_TYPE_FILTER=1` on a corpus that does not support it |
| First request hangs for minutes | model not baked in — check `HF_HOME` and that the build stage ran |
| Container restarts repeatedly | healthcheck firing before the model loads — raise `start_period` |

---

## 10. Verifying a deployment

```bash
python -m src.preflight      # config, corpus, index, key, OCR
python -m eval.test_smoke    # 10 checks, end of pipeline included
python -m eval.run_eval --only retrieval   # Recall@3 / Recall@10 / MRR
```

Current measured baseline, 40-clause benchmark:

| | value |
|---|---|
| Recall@3 | 63.7% |
| Recall@10 | 73.8% |
| MRR | 0.554 |
| OCR repair CER | 0.688 → 0.053 |

A deployment that scores materially below these has a configuration problem,
not a model problem.
