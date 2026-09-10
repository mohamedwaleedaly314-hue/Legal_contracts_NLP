# -*- coding: utf-8 -*-
"""
Stage 1 - OCR and Arabic text preparation.

Module form of notebooks/contracts_ocr_pipeline.ipynb. The extraction,
cleaning and clause-segmentation logic is the notebook's; what changed is
everything that tied it to Colab:

  * no !apt-get / google.colab / files.upload() - extract() takes a path
  * PDFs with a real text layer go through PyMuPDF first. The notebook
    imported fitz but always rasterised, so every PDF needed Poppler +
    Tesseract; a born-digital contract now needs neither and is far more
    accurate besides.
  * Tesseract and Groq are optional. Missing either degrades the result,
    it does not raise.

Public API:
    extract(file_path) -> dict
"""
from __future__ import annotations

import io
import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("contract_ai.ocr")

# --------------------------------------------------------------------------
# Optional dependencies - each stage degrades on its own
# --------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import pytesseract
    # Windows installs Tesseract outside PATH; TESSERACT_CMD overrides.
    _tess_cmd = os.environ.get("TESSERACT_CMD")
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
    ".docx", ".doc",
    ".txt", ".csv", ".json",
}

# Below this many characters a PDF text layer is treated as absent/broken
# and we fall back to OCR.
MIN_TEXT_LAYER_CHARS = 120

PAGE_SEPARATOR = "\n\n--- صفحة جديدة ---\n\n"


# ==========================================================================
# File type
# ==========================================================================
def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        return "image"
    if ext in {".docx", ".doc"}:
        return "word"
    if ext in {".txt", ".csv", ".json"}:
        return "text"
    return "unknown"


# ==========================================================================
# Groq refinement (optional)
# ==========================================================================
# The notebook listed llama-3.3-70b-versatile and friends, none of which exist
# on this account. Every refinement call 404'd, the except branch swallowed it,
# and the caller got the raw OCR back believing it had been cleaned - which is
# why scanned contracts came out unreadable. These ids are on the account:
# allam is Arabic-native and has a 7,000/day quota, so it carries the bulk and
# leaves the analysis model's 1,000 for the analysis.
GROQ_TEXT_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

# Small on purpose. Handed a whole page, the model reconstructs the first
# paragraph beautifully and then coasts, leaving the rest of the page as
# broken as it found it. Roughly a paragraph at a time keeps its attention on
# all of the text, at the cost of a few more requests.
MAX_REFINE_CHARS = 1200

_REFINE_PROMPT = """أنت مدقق نصوص قانونية عربية، ومهمتك إصلاح مخرجات OCR لعقد ممسوح ضوئياً.

العيوب المتوقعة في هذا النص، أصلحها جميعاً:
1. حرف «د» مكرر داخل الكلمات ناتج عن مدّ الأحرف (الكشيدة) في النص المضبوط:
   «الطدددرف» = «الطرف»، «بالددددور» = «بالدور». احذف التكرار الزائد فقط،
   وأبقِ الدال الأصلية إن كانت جزءاً من الكلمة.
2. كلمات مقطّعة بمسافات داخلية: «ط رف أول بائ ع» = «طرف أول بائع».
3. أرقام وأقواس في غير موضعها بسبب اتجاه النص: «رقم53 )» = «رقم (53)».
4. استبدالات حرفية ثابتة يخطئ فيها المحرك، صحّحها حيثما تُنتج كلمة صحيحة:
   ل ← و   «موك» = «ملك»، «عوى» = «على»، «لوعاموين» = «للعاملين»
   ص ← د   «دندوق» = «صندوق»
   ت ← ه   «الهامين» = «التأمين»، «مسااحتها» = «مساحتها»
   ت ← ه   «مهر» = «متر»، «بهقسيم» = «بتقسيم»
   ع ← ى   «السابى» = «السابع»، «مربى» = «مربع»
   ئ ← ل   «الكالنة» = «الكائنة»
   ث ← فراغ «ال ان( ى» = «الثاني»
   ق ← م   «مانون» = «قانون»
   ي ← ل   «القبوى» = «القبلي»

قواعد ملزمة:
- لا تخترع أي معلومة. إن تعذّر تخمين كلمة، اتركها كما هي.
- لا تغيّر أي اسم أو رقم أو تاريخ أو مبلغ.
- لا تلخّص ولا تحذف أي بند.
- أعد النص المصحّح فقط، دون مقدمة أو تعليق.

النص:
{raw_text}"""


def _groq_key() -> str:
    """The key, from settings rather than the raw environment.

    os.environ alone is not enough: .env is loaded by config, and this module
    is imported before it on some paths (the benchmark hit exactly that).
    Reading the environment directly then found nothing, refinement silently
    no-opped, and the caller got raw OCR back believing it had been cleaned -
    the same silent-skip failure the wrong model ids caused earlier.
    """
    try:
        from .config import settings
        return settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    except Exception:  # noqa: BLE001 - config is optional for this module
        return os.environ.get("GROQ_API_KEY", "")


def groq_available() -> bool:
    return HAS_GROQ and _groq_key().startswith("gsk_")


def _split_for_refinement(text: str, limit: int = MAX_REFINE_CHARS) -> List[str]:
    """Break text into chunks on blank lines, never mid-paragraph."""
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) + 2 > limit:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


# Models sometimes answer with "بعد المراجعة، يصبح النص كالتالي:" before the
# text itself. Drop that lead-in rather than letting it into the contract.
_PREAMBLE = re.compile(
    r"\A[^\n]{0,140}?(?:النص\s+(?:المصحح|المنقح|بعد)|كالتالي|كما\s+يلي)[^\n]{0,40}?:\s*",
)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE.sub("", text.strip(), count=1).strip()


def _refine_chunk(client, text: str) -> str:
    for model_name in GROQ_TEXT_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user",
                           "content": _REFINE_PROMPT.format(raw_text=text)}],
                temperature=0.0,
            )
            result = _strip_preamble(response.choices[0].message.content or "")
            # A refusal or a summary is worse than the raw text. Anything much
            # shorter than the input is one of those, so keep the original.
            if result and len(result) >= len(text) * 0.5:
                return result
            logger.warning("Refinement from %s looked truncated; keeping raw", model_name)
        except Exception as exc:  # try the next model in the list
            logger.warning("Groq model %s failed: %s", model_name, exc)
    return text


def refine_text_with_groq(raw_text: str) -> str:
    """Repair OCR artefacts via Groq. Returns raw_text unchanged on any failure."""
    if not raw_text or not raw_text.strip():
        return ""
    if not groq_available():
        logger.info("No GROQ_API_KEY - skipping LLM refinement of the OCR text")
        return raw_text

    try:
        client = Groq(api_key=_groq_key())
        return "\n\n".join(
            _refine_chunk(client, chunk) for chunk in _split_for_refinement(raw_text)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq refinement unavailable (%s); keeping raw text", exc)
        return raw_text


# ==========================================================================
# Image pre-processing
# ==========================================================================
def preprocess_image_for_ocr(pil_img):
    """CLAHE contrast boost before Tesseract. No-op without OpenCV."""
    if not HAS_CV2:
        return pil_img.convert("L")
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray))


# psm 6 - "assume a single uniform block of text" - rather than psm 3's full
# automatic page segmentation. A contract page is one justified column, and
# psm 3's layout analysis on justified Arabic is fragile: it hunts for columns
# that are not there and mis-slices the line, which is part of how the kashida
# stretch ends up read as a row of dals.
TESS_CONFIG = r"--oem 3 --psm 6 -l ara+eng"


def _ocr_image(pil_img) -> str:
    if not HAS_TESSERACT:
        raise RuntimeError(
            "Tesseract غير متاح. ثبّت tesseract-ocr مع حزمة اللغة العربية، "
            "أو اضبط متغير البيئة TESSERACT_CMD على مسار الملف التنفيذي."
        )
    return pytesseract.image_to_string(preprocess_image_for_ocr(pil_img),
                                       config=TESS_CONFIG)


# ==========================================================================
# Extraction per file type
# ==========================================================================
def extract_pdf_text_layer(pdf_path: str) -> List[str]:
    """Read an existing PDF text layer. Returns [] when there isn't a usable one."""
    if not HAS_FITZ:
        return []
    try:
        with fitz.open(pdf_path) as doc:
            pages = [page.get_text("text") or "" for page in doc]
    except Exception as exc:
        logger.warning("PyMuPDF could not read %s: %s", pdf_path, exc)
        return []

    if sum(len(p.strip()) for p in pages) < MIN_TEXT_LAYER_CHARS:
        return []
    return pages


def rasterize_pdf(pdf_path: str, dpi: int = 300) -> List["Image.Image"]:
    """Render each PDF page to an image.

    PyMuPDF first: it is already a dependency (the text-layer path uses it) and
    it renders in-process, so a scanned PDF no longer needs Poppler installed
    on the machine - which was the one system package standing between this
    pipeline and a plain `pip install`. pdf2image stays as a fallback for the
    rare file PyMuPDF cannot render.
    """
    if HAS_FITZ:
        try:
            images = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=dpi)
                    images.append(
                        Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    )
            if images:
                return images
            logger.warning("PyMuPDF rendered no pages from %s", pdf_path)
        except Exception as exc:  # noqa: BLE001 - fall through to pdf2image
            logger.warning("PyMuPDF could not rasterise %s: %s", pdf_path, exc)

    if HAS_PDF2IMAGE:
        return convert_from_path(pdf_path, dpi=dpi)

    raise RuntimeError(
        "تعذر تحويل ملف PDF إلى صور: PyMuPDF غير متاح ولا pdf2image. "
        "ثبّت pymupdf عبر pip."
    )


def extract_pdf_ocr(pdf_path: str, dpi: int = 300, refine: bool = True) -> List[str]:
    """Rasterise then OCR - for scanned PDFs. Needs Tesseract."""
    images = rasterize_pdf(pdf_path, dpi=dpi)
    logger.info("OCR over %d page(s)", len(images))

    page_texts = []
    for i, img in enumerate(images):
        text = repair_scan_artifacts(_ocr_image(img))
        page_texts.append(refine_text_with_groq(text) if refine else text)
        logger.info("  page %d/%d done", i + 1, len(images))
    return page_texts


def extract_image(image_path: str, refine: bool = True) -> List[str]:
    img = Image.open(image_path)
    text = repair_scan_artifacts(_ocr_image(img))
    return [refine_text_with_groq(text) if refine else text]


def extract_word(docx_path: str) -> List[str]:
    if not HAS_DOCX:
        raise RuntimeError("python-docx غير مثبت — لا يمكن قراءة ملفات Word.")
    doc = docx.Document(docx_path)
    return ["\n".join(p.text for p in doc.paragraphs if p.text.strip())]


def extract_plain_text(file_path: str) -> List[str]:
    return [Path(file_path).read_text(encoding="utf-8", errors="ignore")]


def extract_document_text(file_path: str, refine: bool = True) -> Tuple[List[str], str, int]:
    """Dispatch on file type. Returns (page_texts, method, page_count)."""
    file_type = detect_file_type(file_path)

    if file_type == "pdf":
        pages = extract_pdf_text_layer(file_path)
        if pages:
            logger.info("PDF text layer found - skipping OCR")
            return pages, "pymupdf_text_layer", len(pages)
        logger.info("No usable text layer - falling back to OCR")
        pages = extract_pdf_ocr(file_path, refine=refine)
        return pages, "tesseract_ocr", len(pages)

    if file_type == "image":
        pages = extract_image(file_path, refine=refine)
        return pages, "tesseract_image", 1

    if file_type == "word":
        pages = extract_word(file_path)
        return pages, "docx_direct", 1

    if file_type == "text":
        pages = extract_plain_text(file_path)
        return pages, "plain_text", 1

    raise ValueError(f"نوع ملف غير مدعوم: {Path(file_path).suffix or file_path}")


# ==========================================================================
# Arabic cleaning
# ==========================================================================
# --------------------------------------------------------------------------
# Scanned-document repair
# --------------------------------------------------------------------------
# Justified Arabic stretches the join between letters with a kashida (ـ).
# Tesseract reads that stroke as a row of dals, so "الطرف" arrives as
# "الطدددددرف" and "بالدور" as "بالددددددور" - fifty such runs in a single
# page of the sample contract. Collapsing each run to one dal is the right
# call rather than deleting it: where the word really did contain a dal it is
# now correct outright, and where it did not, what is left ("الطدرف") is an
# obvious non-word that the LLM pass repairs. Deleting the run instead would
# turn "بالدور" into "بالور" and hide the damage.
_KASHIDA_AS_DAL = re.compile(r"د{3,}")

# No Arabic word carries the same letter three times in a row; a run like that
# is always the scanner smearing one glyph.
_LETTER_SMEAR = re.compile(r"([ء-ي])\1{2,}")

# "رقم53" - the digit run is glued to the word it belongs to.
_DIGIT_GLUED = re.compile(r"([ء-ي])(\d)")
_WORD_GLUED = re.compile(r"(\d)([ء-ي])")


# A scan of a justified page comes back as a column of short fragments -
# "حلوان–", "المكونة ( من عد 5". Handed those, the model treats each as its
# own item and copies them through untouched; rejoined into flowing prose it
# reconstructs them. A line is a continuation unless the previous one closed a
# sentence or this one opens a new clause.
_BLOCK_START = re.compile(r"^\s*(?:البند|المادة|أولا|أولاً|ثانيا|ثانياً|ثالثا|ثالثاً|"
                          r"رابعا|رابعاً|خامسا|خامساً|\d{1,2}\s*[-.)]|\(\d{1,2}\))")
_SENTENCE_END = re.compile(r"[.:؟!]\s*$")
WRAPPED_LINE_CHARS = 60


def merge_wrapped_lines(text: str) -> str:
    """Rejoin lines the page layout broke mid-sentence."""
    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []

    for line in lines:
        if not line:
            continue
        if (
            out
            and len(out[-1]) < WRAPPED_LINE_CHARS * 4
            and not _SENTENCE_END.search(out[-1])
            and not _BLOCK_START.match(line)
        ):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)

    return "\n".join(out)


def repair_scan_artifacts(text: str) -> str:
    """Undo the mechanical damage a scan leaves behind.

    Deterministic only - it never guesses at a word. Whatever it cannot fix
    is left visibly broken for `refine_text_with_groq` to reconstruct.
    """
    if not text:
        return ""

    out = _KASHIDA_AS_DAL.sub("د", text)
    out = _LETTER_SMEAR.sub(r"\1", out)
    out = _DIGIT_GLUED.sub(r"\1 \2", out)
    out = _WORD_GLUED.sub(r"\1 \2", out)
    return merge_wrapped_lines(out)


# A born-digital PDF is usually clean, but not always: some are a scan someone
# ran through an OCR tool before saving, and the text layer carries the same
# kashida runs and one-letter fragments. Measuring the damage rather than
# assuming it from the file type decides whether the repair pass is worth a
# request.
_DAMAGE_TOKENS = re.compile(r"د{3,}|ـ{2,}")
DAMAGE_THRESHOLD = 0.06


def scan_damage_ratio(text: str) -> float:
    """Share of words that look mangled. 0.0 for clean text."""
    words = re.findall(r"[ء-ي]+", text)
    if not words:
        return 0.0

    suspect = sum(
        1 for w in words
        if len(w) == 1 or _DAMAGE_TOKENS.search(w) or re.search(r"([ء-ي])\1{2,}", w)
    )
    return suspect / len(words)


def clean_arabic_text(text: str) -> str:
    """Normalisation only - never rewrites or drops legal wording.

    Implements the cleaning the notebook documented but left unwritten
    (its `clean_text` variable was never assigned).
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("ـ", "")                          # tatweel
    text = re.sub(r"[​-‏‪-‮]", "", text)  # bidi / zero-width
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.،؛:!؟])\1{1,}", r"\1", text)
    return text.strip()


# Shared by the heading repair and the clause pattern. One copy matters: the
# repair step has to recognise exactly the headings the segmenter looks for.
ORDINAL_WORDS = (
    r"(?:الأول|الأولى|الثاني|الثانية|الثالث|الثالثة|الرابع|الرابعة|الخامس|الخامسة|"
    r"السادس|السادسة|السابع|السابعة|الثامن|الثامنة|التاسع|التاسعة|العاشر|العاشرة|"
    r"الحادي\s+عشر|الثاني\s+عشر|الثالث\s+عشر|الرابع\s+عشر|الخامس\s+عشر|السادس\s+عشر|"
    r"السابع\s+عشر|الثامن\s+عشر|التاسع\s+عشر|العشرون|الثلاثون|رقم\s*\d+|\d+)"
)

STANDALONE_ORDINALS = (
    r"(?:أولاً|أولا|ثانياً|ثانيا|ثالثاً|ثالثا|رابعاً|رابعا|خامساً|خامسا|"
    r"سادساً|سادسا|سابعاً|سابعا|ثامناً|ثامنا|تاسعاً|تاسعا|عاشراً|عاشرا)"
)


def fix_ocr_clause_headers(text: str) -> str:
    """Repair OCR damage in clause headings. From the notebook, one rule fixed."""
    if not text:
        return ""

    text = re.sub(r'البند\s*العاشر', 'البند العاشر', text)
    text = re.sub(r'البند\s*الرا[بع]*', 'البند الرابع', text)
    text = re.sub(r'البند\s*السا[يئ]ع', 'البند السابع', text)
    text = re.sub(r'اليند', 'البند', text)

    text = re.sub(r'(البند\s+[أ-ي]+)\s*[\n\r_]+(عشر)', r'\1 \2', text)
    text = re.sub(r'(المادة\s+\w+)\s*[\n\r_]+(عشر)', r'\1 \2', text)
    # Detach a heading that OCR glued onto the clause body. The notebook's rule
    # was r'(البند\s*[أ-ي]+)(?=[أ-ي])' — [أ-ي]+ is greedy, so the lookahead
    # forced it to give a letter back and the space landed *inside* the ordinal:
    # "البند الأول" became "البند الأو ل". That corrupted every heading in the
    # document, the segmenter then matched nothing, and each contract silently
    # fell back to splitting on blank lines. Anchoring on the known ordinals
    # matches the heading as one unit.
    text = re.sub(rf'(البند\s+{ORDINAL_WORDS})(?=[أ-ي])', r'\1 ', text)
    text = re.sub(rf'(المادة\s+{ORDINAL_WORDS})(?=[أ-ي])', r'\1 ', text)

    def clean_numeric_brackets(match):
        content = match.group(1)
        content = content.replace('"', '2').replace('؟', '2').replace("'", '')
        return f"({content.strip()})"

    text = re.sub(r'\(([^)\n]*[\d٠-٩"؟][^)\n]*)\)', clean_numeric_brackets, text)
    return text


# ==========================================================================
# Clause segmentation
# ==========================================================================
def build_advanced_clause_pattern():
    """Arabic legal numbering patterns. Verbatim from the notebook."""
    words_pattern = ORDINAL_WORDS
    ordinals_pattern = STANDALONE_ORDINALS
    patterns = [
        rf"(?:^|\n)\s*(البند\s+{words_pattern})\s*[:\-–—]?",
        rf"(?:^|\n)\s*(المادة\s+{words_pattern})\s*[:\-–—]?",
        rf"(?:^|\n)\s*({ordinals_pattern})\s*[:\-–—]?",
        r"(?:^|\n)\s*(\d{1,2}\s*[-.\)])\s*",
        r"(?:^|\n)\s*(\(\d{1,2}\))\s*",
    ]
    return re.compile("|".join(patterns), flags=re.MULTILINE)


def segment_legal_clauses(text: str) -> List[dict]:
    """Split into numbered clauses. Never invents a clause that isn't there."""
    if not text or not text.strip():
        return []

    prepared_text = fix_ocr_clause_headers(text)
    matches = list(build_advanced_clause_pattern().finditer(prepared_text))
    clauses: List[dict] = []

    if matches:
        preamble = prepared_text[:matches[0].start()].strip()
        preamble = re.sub(r'---\s*صفحة جديدة\s*---', '', preamble).strip()
        if preamble:
            clauses.append({
                "clause_id": 1,
                "clause_label": "تمهيد / مقدمة العقد",
                "clause_text": preamble,
            })

        for idx, match in enumerate(matches):
            label = next((g for g in match.groups() if g), "").strip()
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(prepared_text)

            clause_body = prepared_text[start_pos:end_pos].strip()
            clause_body = re.sub(r'---\s*صفحة جديدة\s*---', '', clause_body).strip()

            if clause_body:
                clauses.append({
                    "clause_id": len(clauses) + 1,
                    "clause_label": label,
                    "clause_text": clause_body,
                })
        return clauses

    # Fallback: paragraph split when no explicit numbering is present
    paragraphs = [p.strip() for p in prepared_text.split("\n\n") if p.strip()]
    return [
        {"clause_id": i + 1, "clause_label": f"فقرة {i + 1}", "clause_text": p}
        for i, p in enumerate(paragraphs)
    ]


# ==========================================================================
# Public API
# ==========================================================================
def extract(file_path: str, refine: bool = True) -> dict:
    """Turn a contract file into clean text plus numbered clauses.

    Returns
    -------
    dict with keys:
        status        "ok" | "error"
        metadata      file_name, file_type, num_pages, method, num_clauses
        raw_text      concatenated pages, exactly as extracted
        clean_text    normalised text
        clauses       [{clause_id, clause_label, clause_text}, ...]
        errors        list of human-readable messages (Arabic)

    Never raises for an unreadable document - inspect `status` instead.
    """
    path = Path(file_path)
    result = {
        "status": "error",
        "metadata": {
            "file_name": path.name,
            "file_type": detect_file_type(file_path),
            "num_pages": 0,
            "method": None,
            "num_clauses": 0,
        },
        "raw_text": "",
        "clean_text": "",
        "clauses": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"الملف غير موجود: {file_path}")
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["errors"].append(f"نوع ملف غير مدعوم: {path.suffix}")
        return result

    try:
        page_texts, method, page_count = extract_document_text(file_path, refine=refine)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    raw_text = PAGE_SEPARATOR.join(p for p in page_texts if p and p.strip())
    if not raw_text.strip():
        result["metadata"]["method"] = method
        result["errors"].append(
            "تمت المعالجة لكن لم يُعثر على نص قابل للقراءة داخل الملف."
        )
        return result

    # The scan paths already repaired and refined their own pages. A text layer
    # skipped both, on the assumption it was born digital - but plenty of PDFs
    # are a scan that was OCR'd once and saved, and those carry exactly the same
    # damage. Measure it and repair when it is actually there.
    repaired = repair_scan_artifacts(raw_text)
    damage = scan_damage_ratio(repaired)
    result["metadata"]["damage_ratio"] = round(damage, 3)

    if method == "pymupdf_text_layer" and damage >= DAMAGE_THRESHOLD:
        logger.info("Text layer looks OCR-damaged (%.0f%%) - repairing", damage * 100)
        result["metadata"]["method"] = method = "pymupdf_text_layer+repair"
        if refine:
            repaired = refine_text_with_groq(repaired)

    clean_text = clean_arabic_text(repaired)
    clauses = segment_legal_clauses(clean_text)

    result.update({
        "status": "ok",
        "raw_text": raw_text,
        "clean_text": clean_text,
        "clauses": clauses,
    })
    result["metadata"].update({
        "num_pages": page_count,
        "method": method,
        "num_clauses": len(clauses),
    })
    return result


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("usage: python -m modules.ocr <file>")
        raise SystemExit(1)
    print(json.dumps(extract(sys.argv[1]), ensure_ascii=False, indent=2)[:4000])
