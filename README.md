# ⚖️ ميزان — Mizan

**تحليل عقودك على القانون المصري**

Reads an Arabic contract, splits it into numbered clauses, retrieves the
Egyptian legal articles that govern each clause, and reports which clauses are
void or one-sided — with a balanced alternative wording for each.

```
contract file  →  OCR  →  classify  →  retrieve per clause  →  analyse  →  report
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env      # then put your GROQ_API_KEY in it
streamlit run app.py
```

Click **جرّب العقد النموذجي** to run the bundled sample contract, which
contains several genuinely void clauses.

The first analysis loads the embedding model and the legal index — about
30 seconds, once per process. Everything after that is instant.

---

## Layout

```
app.py                  Streamlit UI - reads only the orchestrator's report
requirements.txt
.env.example
.streamlit/config.toml  light theme

src/
  orchestrator.py       wires the four stages together
  ocr.py                stage 1 - text extraction, cleaning, clause segmentation
  classifier.py         stage 2 - contract type + parties/date/value/duration
  retriever.py          stage 3 - hybrid RAG over the Egyptian legal corpus
  analysis.py           stage 4 - per-clause risk analysis
  llm_client.py         Groq / Ollama backends, token pacing, quota detection
  legal_text.py         repairs OCR damage in the legal corpus on read
  config.py             settings, all overridable by environment
  schemas.py            Pydantic contracts between the stages
  prompts.py            system prompts

data/
  legal_articles.json   1352 Egyptian legal articles
  bm25.pkl              sparse index
  embeddings.npy        dense vectors (BAAI/bge-m3, 1024-dim)
  index/                Qdrant collection

scripts/build_index.py  rebuild the index from data/laws.jsonl
samples/                sample contract + sample retriever output
docs/retriever.md       retrieval design notes
notebooks/              the team's original exploration notebooks
```

---

## Configuration

All settings come from the environment; `.env` is loaded automatically, and is
located from the source tree rather than the working directory — so
`streamlit run <subdir>/app.py` from a parent folder still finds it.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | required when the provider is groq |
| `CONTRACT_AI_PROVIDER` | `groq` if a key is set, else `ollama` | which backend runs the analysis |
| `CONTRACT_AI_ANALYSIS_MODEL` | `qwen/qwen3.8-27b` | analysis model |
| `CONTRACT_AI_MAX_CONCURRENCY` | `2` | clauses analysed in parallel |
| `CONTRACT_AI_TPM` | `8000` | tokens-per-minute ceiling the client paces to |
| `TESSERACT_CMD` | — | path to tesseract.exe on Windows |

### Provider

Groq is the default because the local models are unusable without a GPU:
`qwen2.5:14b` on CPU takes minutes per clause, which a twelve-clause contract
turns into half an hour. Ollama remains fully supported — set
`CONTRACT_AI_PROVIDER=ollama`.

### Rate limits — read this before a demo

Groq's free tier meters two separate things, and both bite.

| | limit | what it means here |
|---|---|---|
| requests per day | 250 on `groq/compound*`, 1000 on the rest | a twelve-clause contract costs ~13 requests |
| tokens per minute | 8,000 (12,000 for `groq/compound`) | one clause costs ~2,455 tokens |

`qwen/qwen3.8-27b` is the default because it balances both: 1,000 requests a
day (~80 contracts), 8,000 TPM, about 2 seconds a clause, and the most fluent
Arabic of the models on this tier.

**The tokens-per-minute limit sets the pace, not the code.** At ~2,455 tokens
a clause and 8,000 TPM, that is roughly three clauses a minute — a full
twelve-clause contract takes three to four minutes, and no amount of
concurrency changes it. `llm_client.TokenBudget` paces requests against the
ceiling instead of waiting for the API to refuse them; reacting to 429s alone
was not enough, since one run fired 72 of them and lost two clauses to
exhausted retries.

To demo quickly, switch on **تحليل جزء من البنود فقط** in the sidebar and set
it to five or six clauses.

To check headroom before a demo:

```bash
python -c "import os,httpx;from dotenv import load_dotenv;load_dotenv();r=httpx.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']},json={'model':'qwen/qwen3.8-27b','messages':[{'role':'user','content':'hi'}],'max_tokens':1});print(dict((k,v) for k,v in r.headers.items() if 'ratelimit' in k))"
```

If the daily quota does run out, the app says so explicitly and names the
setting to change — it does not report a dozen mysterious clause failures.

---

## Reaching the article that actually governs a clause

Retrieval finds the article that *reads like* the clause. The articles that
decide whether a clause is void do not read like it, and that gap was the
largest single accuracy problem in the system.

Measured on the 40-clause benchmark: clauses governed by a topic article
(lease, sale) reached Recall@3 62%; clauses governed by a general principle of
contract law reached 40%. Article 149 — abusive terms in an adhesion contract,
the provision this product leans on most — was never retrieved once for any of
the four clauses it governs.

That is not a ranking failure, so a reranker cannot fix it. A clause says
`يتنازل المستأجر عن حقه في اللجوء إلى القضاء`; article 149 says `إذا تم العقد
بطريق الإذعان وكان قد تضمن شروطاً تعسفية`. They share almost no vocabulary. The
article was never a candidate to be reranked.

So `orchestrator.retrieve_for_clause` searches twice: once with the clause as
written, and once with the abstract legal question it raises, fusing the two
with RRF. `legal_concepts.py` derives the second query by rule — the mapping is
small, it is the part a lawyer would want to audit, and it costs no quota.

| | Recall@3 | Recall@10 |
|---|---|---|
| general principle — before | 40.0% | 46.7% |
| general principle — after | **66.7%** | **80.0%** |
| topic-specific — before | 62.0% | 76.0% |
| topic-specific — after | 62.0% | 78.0% |
| **overall** | 53.8% → **63.7%** | 65.0% → **78.8%** |

Zero regressions, and clauses where no rule fires are byte-identical to before:
the second query is only spent when the clause raises an issue that needs it.

---

## How strongly a verdict is stated

A model asked whether a clause is void will answer either way. What changes is
whether a retrieved article backs the answer — and that is the difference the
product lives on, so it is recorded rather than assumed.

Every clause carries an `evidence_status`:

| status | meaning | what the UI does |
|---|---|---|
| `grounded` | every article cited was in the retrieved set | may state **باطل** |
| `model_knowledge` | the model reached past the retrieved articles | states a risk, not a ruling |
| `insufficient` | nothing relevant was retrieved | states a risk, flags the gap |

The model self-reports its status in the response, but `_settle_evidence` does
not take that at face value: `verify_citations` already knows mechanically
whether the cited articles were in front of it, and the weaker of the two
readings wins.

The consequence is deliberate. `is_void_legal_term` is `Optional[bool]` — a
plain bool forces a verdict where the law retrieved does not settle one, and
`false` then reads as "lawful" when it means "could not tell". A claim of
nullity that the evidence does not support is downgraded to `None`, the
reasoning and risk score are still shown, and a note explains why the ruling
was withheld:

> رجّح التحليل بطلان هذا البند، لكن لم تُسند المواد المسترجعة هذا الحكم — فهو
> مؤشر خطورة يستدعي مراجعة محامٍ، لا حكماً بالبطلان.

---

## Scanned contracts

A scan of a justified Arabic page is the hard case. The kashida — the stroke
that stretches the join between letters — reads to Tesseract as a row of dals,
so `الطرف` arrives as `الطدددددرف`, fifty times a page. On top of that come
one-letter fragments and a set of consistent letter confusions
(`ل`→`و`, `ص`→`د`, `ت`→`ه`, `ع`→`ى`).

Tesseract runs in `--psm 6` ("a single uniform block of text") rather than
`psm 3`: a contract page is one justified column, and psm 3's layout analysis
hunts for columns that are not there and mis-slices the line. PDF pages are
rasterised by PyMuPDF in-process, so no Poppler.

`ocr.py` then handles the damage in two passes:

1. `repair_scan_artifacts` — deterministic. Collapses kashida runs, unglues
   digits from words, and rejoins lines the page layout broke mid-sentence.
   It never guesses at a word; whatever it cannot fix it leaves visibly broken.
2. `refine_text_with_groq` — reconstruction. The prompt names each defect and
   lists the letter-confusion table, and the text goes in a paragraph at a
   time: handed a whole page the model repairs the first paragraph and coasts
   through the rest.

On the sample scan this turns

> `رف األول ( البدددددالى ) إلدددددى الطدددددرف ال دددددان( ى ( المشدددددهر )`

into

> `الطرف الأول (البائع) إلى الطرف الثاني (المشتري) … شقة رقم (53) بالدور السابع`

A text-layer PDF skips both passes — unless `scan_damage_ratio` finds the
damage there too, which happens whenever the PDF is a scan someone OCR'd once
and saved.

---

## Dependencies that are not pip packages

Only one, and only for scanned input:

- **Tesseract OCR** + the `ara` language pack — needed for scanned PDFs and
  images. PDFs that carry a text layer are read directly by PyMuPDF and need
  nothing extra.

Poppler is no longer required: PyMuPDF renders PDF pages to images in-process,
so `pip install -r requirements.txt` is the whole setup. `pdf2image` remains
in the requirements only as a fallback for a file PyMuPDF cannot render.

---

## Known limitations

1. **The legal corpus is OCR-damaged.** 787 of 1352 articles carry visible
   damage — `كان` appears as `اان`, `الدائن` as `الداين`, and article
   numbering is broken across lines. `src/legal_text.py` repairs layout and an
   explicit list of non-words on read, but the *indexed* text is still the
   damaged version, so retrieval ranking is affected. A proper fix means
   rebuilding the corpus from a clean source.

2. **Coverage is uneven.** 1110 articles are the civil code, 221 commercial
   law, 16 consumer protection, 3 company law and **2 labour law**. Rental,
   sale and commercial contracts are well supported; employment contracts are
   effectively not.

3. **Contract-type tags are thin.** Only 9 articles are tagged `residential`,
   3 `service`. Retrieval therefore searches `{type, general}` rather than the
   type alone — filtering on the type by itself starved the search and
   returned the same handful of articles for every clause. See the comment in
   `retriever.py`.

4. **The classifier is rule-based**, not trained. It is inspectable and needs
   no model file, but it will miss contract types phrased unusually. Its
   confidence is shown in the UI for exactly that reason.

5. **Drafting is not available.** `drafting.py` was never delivered, so
   generating a new contract from scratch is out of scope for this build.
   `pipeline.py`, which depended on it, was removed.

---

## Not a substitute for a lawyer

The output is advisory, produced with AI assistance, and must not be relied on
as legal advice. Always read the cited legal text before acting on it.
