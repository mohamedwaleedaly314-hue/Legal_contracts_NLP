# -*- coding: utf-8 -*-
"""
ميزان — Mizan. Streamlit front end.

Reads nothing but src.orchestrator's report dict, so the pipeline behind it
can change without touching this file.

The page answers three questions in the order a person reading a contract
actually asks them: what is this contract, how bad is it, and which clause do
I have to fix. Everything else sits one click deeper.
"""
from __future__ import annotations

import html
import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src import orchestrator

APP_NAME = "ميزان"
APP_NAME_LATIN = "Mizan"
TAGLINE = "تحليل عقودك على القانون المصري"

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_CONTRACT = REPO_ROOT / "samples" / "contract_rent_sample.txt"

RISK_META = {
    "red": {"label": "خطر مرتفع", "colour": "#B3261E", "soft": "#FDECEA", "icon": "🔴"},
    "yellow": {"label": "يحتاج مراجعة", "colour": "#B26A00", "soft": "#FFF6E5", "icon": "🟡"},
    "green": {"label": "سليم", "colour": "#1E6B45", "soft": "#EAF6EF", "icon": "🟢"},
    "unknown": {"label": "غير محدد", "colour": "#5F6B76", "soft": "#F1F3F5", "icon": "⚪"},
}
RISK_ORDER = ["red", "yellow", "green", "unknown"]

# Short labels for where a verdict's legal support came from. The risk colour
# says how bad the clause is; this says how much the reader should trust that
# reading. They are different questions and get separate badges.
EVIDENCE_SHORT = {"grounded": "✔︎", "model_knowledge": "◐", "insufficient": "○"}

EVIDENCE_BADGE = {
    "grounded": "✔︎ مسند إلى القانون المسترجع",
    "model_knowledge": "◐ استنتاج غير مسند",
    "insufficient": "○ سند غير كافٍ",
}

STAGE_LABELS = {
    "ocr": "استخراج النص",
    "classify": "تحديد نوع العقد",
    "retrieve": "تحميل الموسوعة القانونية",
    "analyze": "تحليل البنود",
    "summary": "إعداد الملخص",
    "done": "اكتمل",
}

st.set_page_config(
    page_title=f"{APP_NAME} — {TAGLINE}",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Design system
# ==========================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

:root {
  --ink:        #16212B;
  --ink-soft:   #5A6772;
  --line:       #E3E8ED;
  --surface:    #FFFFFF;
  --surface-2:  #F7F9FB;
  --brand:      #0F3D5C;
  --brand-soft: #E8EFF5;
  --radius:     12px;
}

html, body, [class*="css"], .stApp {
  font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif;
}
.stApp .block-container {
  direction: rtl;
  text-align: right;
  max-width: 1180px;
  padding-top: 1.4rem;
}
section[data-testid="stSidebar"] { direction: rtl; text-align: right; }

/* ---------- brand header ---------- */
.brand {
  display: flex; align-items: center; gap: .85rem;
  padding-bottom: .9rem; margin-bottom: 1.3rem;
  border-bottom: 1px solid var(--line);
}
.brand-mark {
  width: 46px; height: 46px; flex: 0 0 46px;
  display: grid; place-items: center;
  background: var(--brand); border-radius: 12px;
}
.brand-text h1 {
  margin: 0; font-size: 1.6rem; font-weight: 800;
  color: var(--ink); line-height: 1.25;
}
.brand-text h1 span { color: var(--ink-soft); font-weight: 600; font-size: .95rem; }
.brand-text p { margin: .1rem 0 0; color: var(--ink-soft); font-size: .9rem; }

/* ---------- cards ---------- */
.card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.1rem 1.25rem;
}

.hero {
  background: linear-gradient(180deg, var(--brand-soft) 0%, var(--surface) 100%);
  border: 1px solid var(--line); border-radius: 16px;
  padding: 1.6rem 1.5rem; margin-bottom: 1.1rem;
}
.hero h2 { margin: 0 0 .4rem; font-size: 1.3rem; color: var(--brand); font-weight: 800; }
.hero p  { margin: 0; color: var(--ink-soft); font-size: .95rem; line-height: 1.95; }

.steps { display: flex; gap: .7rem; margin-top: 1.15rem; flex-wrap: wrap; }
.step {
  /* basis 0 with min-width 0 so the four steps share the row evenly instead
     of sizing to their text and wrapping the last one onto its own line */
  flex: 1 1 0; min-width: 150px;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: .85rem .95rem;
}
.step b { display: block; color: var(--ink); font-size: .92rem; margin-bottom: .2rem; }
.step span { color: var(--ink-soft); font-size: .82rem; line-height: 1.75; }
.step i {
  font-style: normal; display: inline-grid; place-items: center;
  width: 22px; height: 22px; border-radius: 6px; margin-left: .45rem;
  background: var(--brand-soft); color: var(--brand);
  font-size: .75rem; font-weight: 700;
}

/* ---------- chips ---------- */
.meta-strip { display: flex; flex-wrap: wrap; gap: .4rem; padding-top: .55rem; }
.chip {
  display: inline-flex; align-items: center; gap: .35rem;
  background: var(--surface-2); border: 1px solid var(--line);
  border-radius: 999px; padding: .28rem .75rem;
  font-size: .82rem; color: var(--ink);
}
.chip b { color: var(--ink-soft); font-weight: 600; }
.chip-brand {
  background: var(--brand-soft); border-color: #CFDDE8;
  color: var(--brand); font-weight: 700;
}

/* ---------- risk distribution ---------- */
.dist { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: var(--surface-2); }
.dist span { display: block; height: 100%; }
.dist-legend { display: flex; gap: 1.2rem; margin-top: .65rem; flex-wrap: wrap; }
.dist-legend div { font-size: .84rem; color: var(--ink-soft); }
.dist-legend b { color: var(--ink); font-size: 1rem; margin-right: .3rem; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-left: .35rem; }

.score { text-align: center; }
.score-num { font-size: 2.6rem; font-weight: 800; line-height: 1.05; }
.score-lbl { color: var(--ink-soft); font-size: .82rem; margin-top: .3rem; }

/* ---------- clause ---------- */
.clause-head {
  display: flex; align-items: flex-start; gap: .7rem;
  border-right: 5px solid var(--rc); background: var(--bg);
  border-radius: 10px; padding: .75rem .9rem; margin-bottom: .5rem;
}
.clause-head .t { flex: 1; min-width: 0; }
.clause-head .t b { display: block; font-size: 1rem; color: var(--ink); margin-bottom: .15rem; }
.clause-head .t span { color: var(--ink-soft); font-size: .85rem; line-height: 1.8; }
.badge {
  flex: 0 0 auto; background: var(--rc); color: #fff;
  border-radius: 999px; padding: .2rem .7rem;
  font-size: .78rem; font-weight: 700; white-space: nowrap;
}
.void-flag {
  display: inline-block; margin-top: .4rem; font-size: .8rem; font-weight: 700;
  color: #B3261E; background: #fff; border: 1px solid #F3C9C4;
  border-radius: 6px; padding: .14rem .5rem;
}
/* Where the verdict's support came from - shown on every clause, so the
   reader never has to assume a ruling is backed by the retrieved law. */
.ev {
  display: inline-block; margin-top: .4rem; margin-left: .35rem;
  font-size: .76rem; font-weight: 700; border-radius: 6px;
  padding: .14rem .55rem; border: 1px solid transparent;
}
.ev-grounded        { color:#1E6B45; background:#EAF6EF; border-color:#C6E4D3; }
.ev-model_knowledge { color:#B26A00; background:#FFF6E5; border-color:#F0DCB4; }
.ev-insufficient    { color:#5F6B76; background:#F1F3F5; border-color:#DFE4E8; }
.ev-note {
  margin-top: .45rem; font-size: .83rem; line-height: 1.85; color: #5A6772;
  background: #FFF6E5; border-right: 3px solid #B26A00;
  border-radius: 6px; padding: .5rem .7rem;
}

.label { font-weight: 700; color: var(--ink); font-size: .93rem; margin: .95rem 0 .35rem; }
.quote {
  background: var(--surface-2); border: 1px solid var(--line);
  border-radius: 10px; padding: .85rem 1rem; line-height: 2.05; font-size: .93rem;
}
.article {
  background: #F3F7FA; border-right: 4px solid var(--brand);
  border-radius: 8px; padding: .75rem .95rem; margin-bottom: .5rem;
  line-height: 1.95; font-size: .89rem;
}
.article b { color: var(--brand); display: block; margin-bottom: .3rem; font-size: .86rem; }
.fix {
  background: #EAF6EF; border-right: 4px solid #1E6B45;
  border-radius: 10px; padding: .85rem 1rem; line-height: 2.05; font-size: .93rem;
}

/* ---------- widget polish ---------- */
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stExpander"] summary { font-size: .88rem; font-weight: 600; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; font-weight: 700; }
footer, #MainMenu { visibility: hidden; }

.disclaimer {
  color: var(--ink-soft); font-size: .82rem; line-height: 1.9;
  border-top: 1px solid var(--line); padding-top: .9rem; margin-top: 2.2rem;
}
</style>
    """,
    unsafe_allow_html=True,
)

_LOGO = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
    'stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3v18M7 21h10M3 7h18M3 7l-2 5a3 3 0 0 0 6 0L5 7M19 7l-2 5a3 3 0 0 0 6 0l-2-5"/>'
    "</svg>"
)

st.markdown(
    f"""
<div class="brand">
  <div class="brand-mark">{_LOGO}</div>
  <div class="brand-text">
    <h1>{APP_NAME} <span>· {APP_NAME_LATIN}</span></h1>
    <p>{TAGLINE}</p>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# State
# ==========================================================================
for key, default in [("report", None), ("source_name", None)]:
    st.session_state.setdefault(key, default)


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown("#### ⚙️ إعدادات التحليل")

    top_k = st.slider(
        "المواد القانونية لكل بند", 1, 6, 3,
        help="عدد المواد التي تُسترجع من الموسوعة لمقارنة كل بند بها.",
    )

    limit_clauses = st.toggle(
        "تحليل جزء من البنود فقط", value=False,
        help="أسرع كثيراً — مناسب للتجربة والعرض التقديمي.",
    )
    max_clauses = st.number_input(
        "عدد البنود", 1, 60, 6,
        disabled=not limit_clauses, label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### حالة النظام")

    try:
        from src.config import settings

        st.markdown(
            f"<div class='chip'><b>المزوّد</b> {html.escape(settings.provider)}</div>"
            f"<div style='height:.35rem'></div>"
            f"<div class='chip'><b>الموديل</b> {html.escape(settings.analysis_model)}</div>",
            unsafe_allow_html=True,
        )
        if settings.provider == "groq" and not settings.groq_api_key:
            st.error("GROQ_API_KEY غير موجود في ملف .env")
    except Exception as exc:  # noqa: BLE001
        st.error(f"تعذر قراءة الإعدادات: {exc}")

    st.write("")
    status = orchestrator.retriever_status()
    if status["loaded"]:
        st.success("الموسوعة القانونية جاهزة", icon="✅")
    elif status["error"]:
        st.error(f"الموسوعة غير متاحة — {status['error'][:110]}", icon="⚠️")
    else:
        st.info("تُحمَّل الموسوعة عند أول تحليل (~٣٠ ثانية).", icon="ℹ️")
        if st.button("تحميل الموسوعة الآن", use_container_width=True):
            with st.spinner("جارٍ تحميل الفهرس وموديل التضمين..."):
                orchestrator.load_retriever()
            st.rerun()

    if st.session_state.report is not None:
        st.divider()
        if st.button("↻ تحليل عقد آخر", use_container_width=True):
            st.session_state.report = None
            st.session_state.source_name = None
            st.rerun()


# ==========================================================================
# Pipeline runner
# ==========================================================================
def run_pipeline(path: str, display_name: str) -> None:
    """Run the pipeline, streaming progress into the page."""
    slot = st.container()
    with slot:
        bar = st.progress(0.0)
        line = st.empty()

    def on_progress(stage: str, pct: float, message: str) -> None:
        bar.progress(min(max(pct, 0.0), 1.0))
        line.markdown(
            f"<div class='meta-strip'>"
            f"<div class='chip chip-brand'>{html.escape(STAGE_LABELS.get(stage, stage))}</div>"
            f"<div class='chip'>{html.escape(message)}</div></div>",
            unsafe_allow_html=True,
        )

    try:
        report = orchestrator.analyze_contract_file(
            path,
            progress=on_progress,
            top_k=top_k,
            max_clauses=int(max_clauses) if limit_clauses else None,
        )
    except Exception as exc:  # noqa: BLE001 - surface, never a stack trace
        slot.empty()
        st.error(f"توقف التحليل بسبب خطأ غير متوقع: {exc}", icon="⚠️")
        return

    slot.empty()
    st.session_state.report = report
    st.session_state.source_name = display_name


# ==========================================================================
# Rendering
# ==========================================================================
def render_landing() -> None:
    st.markdown(
        """
<div class="hero">
  <h2>اعرف ما الذي وقّعت عليه</h2>
  <p>ارفع ملف العقد أو صورته، فيقرأ ميزان بنوده بنداً بنداً، ويقارن كل بند
     بنصوص القانون المصري، ثم يخبرك أي البنود باطل أو مجحف — ويقترح صياغة
     بديلة متوازنة تصلح لأن توضع مكانه.</p>
  <div class="steps">
    <div class="step"><b><i>١</i>قراءة المستند</b>
      <span>استخراج النص العربي وتقسيمه إلى بنود مرقّمة.</span></div>
    <div class="step"><b><i>٢</i>تحديد العقد</b>
      <span>نوعه وأطرافه وقيمته ومدته وتاريخه.</span></div>
    <div class="step"><b><i>٣</i>استدعاء القانون</b>
      <span>المواد التي تحكم كل بند من ١٣٥٢ مادة.</span></div>
    <div class="step"><b><i>٤</i>تقييم وبديل</b>
      <span>درجة خطورة وسند قانوني وصياغة مقترحة.</span></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.1, 1])
    with left:
        uploaded = st.file_uploader(
            "ارفع ملف العقد",
            type=["pdf", "png", "jpg", "jpeg", "docx", "txt"],
            help="ملفات PDF التي تحتوي على نص تُقرأ مباشرة؛ الممسوحة ضوئياً تحتاج Tesseract.",
        )
        if st.button("🔍 حلّل العقد", type="primary", disabled=uploaded is None,
                     use_container_width=True):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            run_pipeline(tmp_path, uploaded.name)
            st.rerun()

    with right:
        st.markdown(
            "<div class='card' style='margin-top:1.75rem'>"
            "<b style='font-size:.92rem'>ليس لديك عقد جاهز؟</b><br>"
            "<span style='color:#5A6772;font-size:.85rem;line-height:1.8'>"
            "جرّب عقد إيجار نموذجياً يحتوي على بنود باطلة معروفة.</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("📄 جرّب العقد النموذجي", use_container_width=True,
                     disabled=not SAMPLE_CONTRACT.exists()):
            run_pipeline(str(SAMPLE_CONTRACT), SAMPLE_CONTRACT.name)
            st.rerun()


def render_meta(report: dict) -> None:
    chips = [
        "<div class='chip chip-brand'>"
        f"{html.escape(report['contract_type_ar'] or 'نوع غير محدد')}</div>"
    ]
    confidence = report.get("type_confidence") or 0
    if confidence:
        chips.append(f"<div class='chip'><b>الثقة</b> {confidence:.0%}</div>")
    for key, title in (("date", "التاريخ"), ("value", "القيمة")):
        if report.get(key):
            chips.append(f"<div class='chip'><b>{title}</b> {html.escape(report[key])}</div>")
    for party in report.get("parties", []):
        chips.append(
            f"<div class='chip'><b>{html.escape(party['role'])}</b> "
            f"{html.escape(party['name'])}</div>"
        )
    chips.append(f"<div class='chip'><b>البنود</b> {len(report['clauses'])}</div>")

    st.markdown(
        "<div class='card'>"
        "<div style='font-size:.8rem;color:#5A6772'>الملف</div>"
        f"<div style='font-weight:700;font-size:1.05rem'>{html.escape(report['file_name'])}</div>"
        f"<div class='meta-strip'>{''.join(chips)}</div></div>",
        unsafe_allow_html=True,
    )


def render_overview(report: dict) -> None:
    counts = report["counts"]
    total = max(sum(counts.get(k, 0) for k in RISK_ORDER), 1)

    segments = "".join(
        f"<span style='width:{counts.get(k, 0) / total * 100:.4f}%;"
        f"background:{RISK_META[k]['colour']}'></span>"
        for k in RISK_ORDER if counts.get(k, 0)
    )
    legend = "".join(
        f"<div><span class='dot' style='background:{RISK_META[k]['colour']}'></span>"
        f"{RISK_META[k]['label']}<b>{counts.get(k, 0)}</b></div>"
        for k in RISK_ORDER if counts.get(k, 0)
    )

    summary = report.get("summary") or {}
    score = summary.get("overall_risk_score")

    bar_col, score_col = st.columns([3, 1])
    bar_col.markdown(
        "<div class='card'><div class='label' style='margin-top:0'>توزيع المخاطر</div>"
        f"<div class='dist'>{segments}</div>"
        f"<div class='dist-legend'>{legend}</div></div>",
        unsafe_allow_html=True,
    )

    if score is None:
        dial = "<div class='score-num' style='color:#5F6B76'>—</div>"
    else:
        tone = "#B3261E" if score >= 7 else "#B26A00" if score >= 4 else "#1E6B45"
        dial = (f"<div class='score-num' style='color:{tone}'>{score}"
                "<span style='font-size:1.05rem;font-weight:600'>/10</span></div>")
    score_col.markdown(
        f"<div class='card score'>{dial}"
        "<div class='score-lbl'>التقييم الإجمالي</div></div>",
        unsafe_allow_html=True,
    )

    if summary.get("overall_risk_summary"):
        st.markdown(
            "<div class='card' style='margin-top:.7rem'>"
            "<div class='label' style='margin-top:0'>🧾 الملخص التنفيذي</div>"
            "<div style='line-height:2.05;font-size:.93rem'>"
            f"{html.escape(summary['overall_risk_summary'])}</div></div>",
            unsafe_allow_html=True,
        )


def render_clause(clause: dict) -> None:
    meta = RISK_META.get(clause["risk"], RISK_META["unknown"])
    score = clause.get("risk_score")
    title = clause["label"] or f"بند {clause['id']}"
    badge = meta["label"] + (f" · {score}/10" if score is not None else "")
    excerpt = clause.get("simple_explanation") or clause["text"]

    # "باطل" is a legal ruling, so it appears only when the orchestrator was
    # able to tie it to a retrieved article. An ungrounded claim of nullity
    # arrives here as is_void=None with a note explaining why.
    void = ("<div class='void-flag'>⚠️ باطل وفقاً للقانون المصري</div>"
            if clause.get("is_void") else "")

    status = clause.get("evidence_status") or "insufficient"
    evidence = (
        f"<div class='ev ev-{status}'>"
        f"{html.escape(EVIDENCE_BADGE.get(status, status))}</div>"
        if clause.get("risk") != "unknown" else ""
    )
    note = (
        f"<div class='ev-note'>{html.escape(clause['evidence_note'])}</div>"
        if clause.get("evidence_note") else ""
    )

    st.markdown(
        f"<div class='clause-head' style='--rc:{meta['colour']};--bg:{meta['soft']}'>"
        f"<div class='badge'>{html.escape(badge)}</div>"
        f"<div class='t'><b>{html.escape(title)}</b>"
        f"<span>{html.escape(excerpt[:200])}{'…' if len(excerpt) > 200 else ''}</span>"
        f"{void}{evidence}{note}</div></div>",
        unsafe_allow_html=True,
    )

    with st.expander("عرض النص الكامل والسند القانوني"):
        if clause.get("error"):
            st.warning(clause["error"], icon="⚠️")

        st.markdown("<div class='label'>نص البند كما ورد في العقد</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='quote'>{html.escape(clause['text'])}</div>",
                    unsafe_allow_html=True)

        if clause.get("legal_rationale"):
            st.markdown("<div class='label'>التعليل القانوني</div>", unsafe_allow_html=True)
            st.write(clause["legal_rationale"])

        if clause.get("citations"):
            st.markdown("<div class='label'>السند القانوني المسترجع</div>",
                        unsafe_allow_html=True)
            for art in clause["citations"]:
                number = art.get("article_number")
                ref = (f"المادة ({number})" if number else "مادة (رقمها غير موثوق)")
                ref += f" — {art.get('law_name', '')}"
                if art.get("law_number"):
                    ref += f" رقم {art['law_number']} لسنة {art.get('law_year', '')}"
                st.markdown(
                    f"<div class='article'><b>{html.escape(ref)}</b>"
                    f"{html.escape(art.get('text', ''))}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("لم تُسترجع مواد قانونية لهذا البند.")

        check = clause.get("citation_check", {})
        if check.get("unverified"):
            st.warning(
                "استشهادات ذكرها النموذج ولم يُعثر عليها في المواد المسترجعة، "
                "تعامل معها بحذر: " + "، ".join(check["unverified"]),
                icon="⚠️",
            )

        if clause.get("suggestion"):
            st.markdown("<div class='label'>الصياغة البديلة المقترحة</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='fix'>{html.escape(clause['suggestion'])}</div>",
                        unsafe_allow_html=True)
            st.caption("انسخ الصياغة من هنا:")
            st.code(clause["suggestion"], language=None)

        if clause.get("reasoning_steps"):
            with st.popover("خطوات التحليل"):
                for i, step in enumerate(clause["reasoning_steps"], 1):
                    st.write(f"{i}. {step}")


def build_html_report(report: dict) -> str:
    """A self-contained RTL report, written to be printed and handed over.

    Structured the way a reader uses it rather than the way the data happens
    to be shaped: the verdict first, then an index that fits on one screen, and
    only then the clause-by-clause detail. Page breaks and a print stylesheet
    are included because this ends up as a PDF more often than not.
    """
    counts = report["counts"]
    summary = report.get("summary") or {}
    score = summary.get("overall_risk_score")
    clauses = report["clauses"]
    total = max(sum(counts.get(k, 0) for k in RISK_ORDER), 1)

    # ---- headline ---------------------------------------------------------
    if score is None:
        verdict_tone, verdict_text = "#5F6B76", "تعذّر حساب تقييم إجمالي"
    elif score >= 7:
        verdict_tone, verdict_text = "#B3261E", "عقد عالي المخاطر — يحتاج تعديلاً قبل التوقيع"
    elif score >= 4:
        verdict_tone, verdict_text = "#B26A00", "عقد متوسط المخاطر — يحتاج مراجعة"
    else:
        verdict_tone, verdict_text = "#1E6B45", "عقد منخفض المخاطر"

    bar = "".join(
        f"<span style='width:{counts.get(k, 0) / total * 100:.4f}%;"
        f"background:{RISK_META[k]['colour']}'></span>"
        for k in RISK_ORDER if counts.get(k, 0)
    )
    legend = "".join(
        f"<li><i style='background:{RISK_META[k]['colour']}'></i>"
        f"{RISK_META[k]['label']} <b>{counts.get(k, 0)}</b></li>"
        for k in RISK_ORDER if counts.get(k, 0)
    )

    # ---- clause index -----------------------------------------------------
    index_rows = "".join(
        "<tr>"
        f"<td class='n'>{i}</td>"
        f"<td><a href='#c{i}'>{html.escape(c['label'] or ('بند ' + str(c['id'])))}</a></td>"
        f"<td><span class='pill' style='background:"
        f"{RISK_META.get(c['risk'], RISK_META['unknown'])['colour']}'>"
        f"{RISK_META.get(c['risk'], RISK_META['unknown'])['label']}</span></td>"
        f"<td class='n'>{c['risk_score'] if c.get('risk_score') is not None else '—'}</td>"
        f"<td class='n'>{'نعم' if c.get('is_void') else '—'}</td>"
        f"<td class='n'>{EVIDENCE_SHORT.get(c.get('evidence_status'), '—')}</td>"
        "</tr>"
        for i, c in enumerate(clauses, 1)
    )

    # ---- clause detail ----------------------------------------------------
    blocks = []
    for i, clause in enumerate(clauses, 1):
        meta = RISK_META.get(clause["risk"], RISK_META["unknown"])
        clause_score = clause.get("risk_score")
        score_text = f" · {clause_score}/10" if clause_score is not None else ""

        articles = "".join(
            "<div class='art'><b>"
            + (f"المادة ({html.escape(str(a['article_number']))})"
               if a.get("article_number") else "مادة (رقمها غير موثوق)")
            + f" — {html.escape(a.get('law_name', ''))}"
            + (f" رقم {html.escape(str(a.get('law_number')))} لسنة "
               f"{html.escape(str(a.get('law_year', '')))}" if a.get("law_number") else "")
            + f"</b>{html.escape(a.get('text', ''))}</div>"
            for a in clause.get("citations", [])
        ) or "<p class='muted'>لم تُسترجع مواد قانونية لهذا البند.</p>"

        unverified = clause.get("citation_check", {}).get("unverified") or []
        caveat = (
            "<p class='caveat'>⚠️ استشهادات ذكرها التحليل ولم يُعثر عليها ضمن المواد "
            "المسترجعة، تُقرأ بحذر: " + html.escape("، ".join(unverified)) + "</p>"
            if unverified else ""
        )

        suggestion = (
            "<div class='fix'><b>الصياغة البديلة المقترحة</b>"
            f"{html.escape(clause['suggestion'])}</div>"
            if clause.get("suggestion") else ""
        )
        void = ("<p class='void'>هذا البند يُعد باطلاً وفقاً للقانون المصري.</p>"
                if clause.get("is_void") else "")
        status = clause.get("evidence_status") or "insufficient"
        evidence = (
            f"<p class='ev ev-{status}'>السند: "
            f"{html.escape(EVIDENCE_BADGE.get(status, status))}"
            + (f" · الثقة: {html.escape(clause['confidence'])}"
               if clause.get("confidence") else "")
            + "</p>"
        )
        if clause.get("evidence_note"):
            evidence += f"<p class='ev-note'>{html.escape(clause['evidence_note'])}</p>"

        explanation = (
            f"<p class='plain'>{html.escape(clause['simple_explanation'])}</p>"
            if clause.get("simple_explanation") else ""
        )
        rationale = (
            "<h4>التعليل القانوني</h4>"
            f"<p>{html.escape(clause['legal_rationale'])}</p>"
            if clause.get("legal_rationale") else ""
        )

        blocks.append(
            f"<section class='clause' id='c{i}' style='border-right-color:{meta['colour']}'>"
            f"<h3><span class='pill' style='background:{meta['colour']}'>"
            f"{meta['label']}{score_text}</span>"
            f"{html.escape(clause['label'] or ('بند ' + str(clause['id'])))}</h3>"
            f"{void}{evidence}{explanation}"
            "<h4>نص البند كما ورد في العقد</h4>"
            f"<blockquote>{html.escape(clause['text'])}</blockquote>"
            f"{rationale}"
            "<h4>السند القانوني</h4>"
            f"{articles}{caveat}{suggestion}"
            "</section>"
        )

    parties = "".join(
        f"<tr><td>{html.escape(p['role'])}</td><td>{html.escape(p['name'])}</td></tr>"
        for p in report.get("parties", [])
    ) or "<tr><td colspan='2' class='muted'>لم تُستخرج بيانات الأطراف</td></tr>"

    return f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME} — تقرير تحليل {html.escape(report['file_name'])}</title>
<style>
  :root {{
    --ink:#16212B; --soft:#5A6772; --line:#E3E8ED;
    --brand:#0F3D5C; --brand-soft:#E8EFF5; --paper:#fff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Tahoma, 'Noto Naskh Arabic', sans-serif;
    margin: 0; background: #EEF1F5; color: var(--ink); line-height: 1.95;
  }}
  .sheet {{
    max-width: 860px; margin: 2rem auto; background: var(--paper);
    padding: 2.4rem 2.6rem; border-radius: 14px;
    box-shadow: 0 2px 18px rgba(15,61,92,.09);
  }}

  header.cover {{ border-bottom: 3px solid var(--brand); padding-bottom: 1rem; }}
  .brandline {{ display: flex; align-items: center; gap: .7rem; }}
  .mark {{
    width: 40px; height: 40px; border-radius: 10px; background: var(--brand);
    color: #fff; display: grid; place-items: center; font-size: 1.3rem;
  }}
  .brandline h1 {{ margin: 0; font-size: 1.45rem; color: var(--brand); }}
  .brandline p {{ margin: 0; font-size: .85rem; color: var(--soft); }}

  .verdict {{
    margin: 1.5rem 0; padding: 1.1rem 1.3rem; border-radius: 12px;
    background: var(--brand-soft); display: flex; align-items: center; gap: 1.3rem;
  }}
  .verdict .dial {{
    flex: 0 0 92px; text-align: center; background: #fff;
    border-radius: 12px; padding: .7rem .3rem;
  }}
  .verdict .dial b {{ font-size: 2rem; display: block; line-height: 1; }}
  .verdict .dial span {{ font-size: .72rem; color: var(--soft); }}
  .verdict .say {{ font-size: 1.08rem; font-weight: 700; }}
  .verdict .say small {{ display: block; font-weight: 400; font-size: .86rem; color: var(--soft); }}

  .dist {{ display: flex; height: 11px; border-radius: 999px; overflow: hidden;
           background: #EDF0F3; margin-top: .8rem; }}
  .dist span {{ display: block; }}
  ul.legend {{ list-style: none; padding: 0; margin: .6rem 0 0;
               display: flex; gap: 1.3rem; flex-wrap: wrap; font-size: .87rem; color: var(--soft); }}
  ul.legend i {{ display: inline-block; width: 9px; height: 9px;
                 border-radius: 50%; margin-left: .35rem; }}
  ul.legend b {{ color: var(--ink); }}

  h2 {{ font-size: 1.12rem; color: var(--brand); margin: 2rem 0 .6rem;
        padding-bottom: .35rem; border-bottom: 1px solid var(--line); }}
  h4 {{ font-size: .92rem; margin: 1rem 0 .3rem; color: var(--ink); }}

  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  table.meta td {{ padding: .3rem 0; vertical-align: top; }}
  table.meta td:first-child {{ color: var(--soft); width: 130px; }}
  table.index th {{ text-align: right; font-size: .82rem; color: var(--soft);
                    border-bottom: 1px solid var(--line); padding: .4rem .5rem; }}
  table.index td {{ padding: .42rem .5rem; border-bottom: 1px solid #F1F4F7; }}
  table.index td.n {{ text-align: center; color: var(--soft); width: 58px; }}
  table.index a {{ color: var(--brand); text-decoration: none; }}

  .pill {{ color: #fff; border-radius: 999px; padding: .12rem .6rem;
           font-size: .76rem; margin-left: .5rem; white-space: nowrap; }}

  section.clause {{ border-right: 6px solid #999; background: #FAFBFC;
                    padding: 1.1rem 1.3rem; margin: 1.1rem 0; border-radius: 10px;
                    page-break-inside: avoid; }}
  section.clause h3 {{ margin: 0 0 .5rem; font-size: 1.02rem; }}
  .plain {{ margin: .3rem 0 .6rem; }}
  .void {{ color: #B3261E; font-weight: 700; margin: .2rem 0 .5rem; }}
  .ev {{ display: inline-block; font-size: .82rem; font-weight: 700; margin: .3rem 0;
         border-radius: 6px; padding: .15rem .6rem; border: 1px solid transparent; }}
  .ev-grounded {{ color:#1E6B45; background:#EAF6EF; border-color:#C6E4D3; }}
  .ev-model_knowledge {{ color:#B26A00; background:#FFF6E5; border-color:#F0DCB4; }}
  .ev-insufficient {{ color:#5F6B76; background:#F1F3F5; border-color:#DFE4E8; }}
  .ev-note {{ background:#FFF6E5; border-right:3px solid #B26A00; padding:.55rem .8rem;
              border-radius:6px; font-size:.88rem; margin:.4rem 0; }}
  blockquote {{ margin: 0; background: #fff; border: 1px solid var(--line);
                padding: .8rem 1rem; border-radius: 8px; }}
  .art {{ background: #F3F7FA; border-right: 4px solid var(--brand);
          padding: .7rem .9rem; margin: .5rem 0; border-radius: 8px; font-size: .9rem; }}
  .art b {{ display: block; color: var(--brand); margin-bottom: .25rem; font-size: .86rem; }}
  .fix {{ background: #EAF6EF; border-right: 4px solid #1E6B45;
          padding: .8rem .95rem; border-radius: 8px; margin-top: .8rem; }}
  .fix b {{ display: block; color: #1E6B45; margin-bottom: .3rem; }}
  .caveat {{ background: #FFF6E5; border-right: 4px solid #B26A00;
             padding: .6rem .9rem; border-radius: 8px; font-size: .88rem; }}
  .muted {{ color: var(--soft); }}

  footer {{ margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid var(--line);
            color: var(--soft); font-size: .84rem; }}

  @media print {{
    body {{ background: #fff; }}
    .sheet {{ box-shadow: none; margin: 0; max-width: none; padding: 0; }}
    h2 {{ page-break-after: avoid; }}
  }}
</style></head><body>
<div class="sheet">

  <header class="cover">
    <div class="brandline">
      <div class="mark">⚖️</div>
      <div>
        <h1>{APP_NAME} — تقرير تحليل عقد</h1>
        <p>{TAGLINE}</p>
      </div>
    </div>
  </header>

  <div class="verdict">
    <div class="dial">
      <b style="color:{verdict_tone}">{score if score is not None else '—'}</b>
      <span>من 10</span>
    </div>
    <div class="say" style="color:{verdict_tone}">{verdict_text}
      <small>{counts.get('red', 0)} بند مرتفع الخطورة من أصل {len(clauses)} بنداً</small>
      <div class="dist">{bar}</div>
      <ul class="legend">{legend}</ul>
    </div>
  </div>

  <h2>بيانات العقد</h2>
  <table class="meta">
    <tr><td>الملف</td><td>{html.escape(report['file_name'])}</td></tr>
    <tr><td>نوع العقد</td><td>{html.escape(report['contract_type_ar'] or '—')}</td></tr>
    <tr><td>التاريخ</td><td>{html.escape(report['date'] or '—')}</td></tr>
    <tr><td>القيمة</td><td>{html.escape(report['value'] or '—')}</td></tr>
    <tr><td>المدة</td><td>{html.escape(report['duration'] or '—')}</td></tr>
  </table>
  <table class="meta" style="margin-top:.5rem">{parties}</table>

  <h2>الملخص التنفيذي</h2>
  <p>{html.escape(summary.get('overall_risk_summary', 'غير متاح.'))}</p>

  <h2>فهرس البنود</h2>
  <table class="index">
    <tr><th>#</th><th>البند</th><th>التقييم</th><th>الدرجة</th><th>باطل</th><th>السند</th></tr>
    {index_rows}
  </table>

  <h2>التحليل التفصيلي</h2>
  {''.join(blocks)}

  <footer>
    صدر هذا التقرير من <b>{APP_NAME}</b> بتاريخ {datetime.now():%Y-%m-%d} الساعة
    {datetime.now():%H:%M}. التحليل استرشادي أُعدّ بمساعدة الذكاء الاصطناعي ولا
    يغني عن استشارة محامٍ مختص؛ راجع النصوص القانونية المستشهد بها قبل الاعتماد عليها.
  </footer>

</div>
</body></html>"""


def render_report(report: dict) -> None:
    if report["status"] != "ok":
        st.error("تعذر تحليل هذا الملف.", icon="⚠️")
        for err in report["errors"]:
            st.write(f"• {err}")
        return

    for warning in report["warnings"]:
        st.warning(warning, icon="⚠️")

    render_meta(report)
    st.write("")
    render_overview(report)

    st.markdown("<div class='label' style='font-size:1.05rem'>📑 البنود</div>",
                unsafe_allow_html=True)

    present = [k for k in RISK_ORDER if report["counts"].get(k, 0)]
    filter_col, search_col, order_col = st.columns([2, 2, 1.2])
    chosen = filter_col.pills(
        "التصفية", options=present, selection_mode="multi", default=present,
        format_func=lambda k: f"{RISK_META[k]['icon']} {RISK_META[k]['label']}",
        label_visibility="collapsed",
    )
    query = search_col.text_input(
        "بحث", "", placeholder="ابحث في نص البنود…", label_visibility="collapsed",
    )
    order = order_col.selectbox(
        "الترتيب", ["ترتيب العقد", "الأخطر أولاً"], label_visibility="collapsed",
    )

    visible = [
        c for c in report["clauses"]
        if c["risk"] in (chosen or present)
        and (not query or query.strip() in c["text"])
    ]
    if order == "الأخطر أولاً":
        visible.sort(key=lambda c: -(c.get("risk_score") or 0))

    if not visible:
        st.info("لا توجد بنود مطابقة للتصفية.")
    for clause in visible:
        render_clause(clause)

    st.divider()
    stem = Path(report["file_name"]).stem
    html_col, json_col, _ = st.columns([1, 1, 2])
    html_col.download_button(
        "⬇️ تحميل التقرير (HTML)",
        data=build_html_report(report).encode("utf-8"),
        file_name=f"تقرير_{stem}.html",
        mime="text/html", use_container_width=True,
    )
    json_col.download_button(
        "⬇️ تحميل النتائج (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{stem}_analysis.json",
        mime="application/json", use_container_width=True,
    )

    with st.expander("⏱️ تفاصيل التنفيذ والنص المستخرج"):
        st.json(report["timings"])
        st.text_area("النص بعد التنظيف", report["clean_text"], height=240)


# ==========================================================================
# Preflight - a misconfigured deployment says so, up front
# ==========================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _preflight() -> dict:
    """Cached: the checks are cheap but not free, and Streamlit reruns often."""
    from src.preflight import run
    # Skip the Qdrant probe: the app process already holds the embedded
    # folder's lock, so opening it a second time would report a failure the
    # check itself caused.
    return run(include_qdrant=False).to_dict()


_status = _preflight()
if not _status["ok"]:
    st.error("⚠️ إعداد النشر غير مكتمل — التطبيق لن يعمل بشكل صحيح:", icon="⚠️")
    for check in _status["checks"]:
        if check["status"] == "fail":
            st.markdown(f"**{check['name']}** — {check['message']}")
            if check["hint"]:
                st.caption(f"↳ {check['hint']}")
    st.stop()

for _check in _status["checks"]:
    if _check["status"] == "warn":
        st.warning(f"{_check['message']}"
                   + (f" — {_check['hint']}" if _check["hint"] else ""), icon="ℹ️")


# ==========================================================================
# Route
# ==========================================================================
if st.session_state.report is None:
    render_landing()
else:
    render_report(st.session_state.report)

st.markdown(
    f"<div class='disclaimer'><b>{APP_NAME}</b> يقدّم تحليلاً استرشادياً بمساعدة "
    "الذكاء الاصطناعي، ولا يغني عن استشارة محامٍ مختص. راجع دائماً النصوص "
    "القانونية المستشهد بها قبل الاعتماد عليها.</div>",
    unsafe_allow_html=True,
)
