# -*- coding: utf-8 -*-
"""
Build the synthetic half of the OCR benchmark.

Page 1 is a real scan: raw Tesseract output a user reported, paired with a
hand transcription. It is the honest case but there is only one of it.

Pages 2 and 3 are generated here, from contract text whose reference is
exact by construction. The damage applied is not invented - it is the
distribution measured on page 1: kashida runs read as dals, a fixed table of
letter confusions, digits glued to words, and lines broken mid-sentence.

Why generate rather than hand-collect more scans: the repair stage has to
undo damage without being told which dal was inserted, so a generator that
knows the answer still cannot help it. What this does buy is exact ground
truth and a benchmark anyone can rebuild with one command.

    python -m eval.make_ocr_pages
"""
from __future__ import annotations

import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "ocr"

# Measured on page 1: 50 kashida runs, mean length ~6, over ~1100 characters.
KASHIDA_RATE = 0.30      # share of eligible letter joins that get stretched
KASHIDA_LEN = (3, 7)

# The substitutions observed on page 1, with the direction Tesseract makes
# the mistake in (correct -> what the engine produced).
LETTER_CONFUSIONS = [
    ("ل", "و"), ("ص", "د"), ("ت", "ه"), ("ع", "ى"), ("ئ", "ل"), ("ق", "م"),
]
CONFUSION_RATE = 0.06

# Letters that connect to the following letter, so a kashida can follow them.
CONNECTING = set("بتثجحخسشصضطظعغفقكلمنهي")

SOURCES = {
    "page2": (
        "البند الرابع: الإخلاء. إذا تأخر المستأجر عن سداد الأجرة يومين فقط، "
        "يحق للمؤجر طرده فوراً من العين وتغيير الأقفال والاستيلاء على منقولاته "
        "دون حاجة إلى إنذار أو حكم قضائي. "
        "البند الخامس: التنازل عن حق التقاضي. يقر المستأجر بتنازله النهائي عن "
        "حقه في اللجوء إلى القضاء أو الطعن على أي إجراء يتخذه المؤجر، ويعتبر "
        "قرار المؤجر نهائياً وغير قابل للاعتراض."
    ),
    "page3": (
        "البند السادس: الصيانة وضمان العيوب. يتنازل المستأجر عن حقه في مطالبة "
        "المؤجر بأي صيانة للعين المؤجرة أو ضمان العيوب الخفية مهما كانت جسامتها. "
        "البند السابع: التأمين. دفع المستأجر مبلغ عشرة آلاف جنيه تأميناً، ويقر "
        "بأن هذا المبلغ غير مسترد بأي حال من الأحوال ولو انتهى العقد بصورة طبيعية. "
        "البند الثامن: الشرط الجزائي. في حالة إخلال المستأجر بأي بند من بنود هذا "
        "العقد يلتزم بدفع مبلغ مائة ألف جنيه كشرط جزائي غير قابل للتخفيض."
    ),
}


def damage(text: str, rng: random.Random) -> str:
    """Apply scanner-shaped damage to clean text."""
    out = []
    for i, ch in enumerate(text):
        # Letter confusion first, so a confused letter can still be stretched.
        for correct, wrong in LETTER_CONFUSIONS:
            if ch == correct and rng.random() < CONFUSION_RATE:
                ch = wrong
                break
        out.append(ch)

        # A kashida only stretches a join, so it needs a connecting letter
        # before it and a letter after it.
        following = text[i + 1] if i + 1 < len(text) else ""
        if ch in CONNECTING and following and following not in " \n" and rng.random() < KASHIDA_RATE:
            out.append("د" * rng.randint(*KASHIDA_LEN))

    damaged = "".join(out)
    damaged = re.sub(r"(\S) (\d)", r"\1\2", damaged)          # glue digits on
    words = damaged.split(" ")
    lines, current = [], []
    for w in words:                                            # break lines short
        current.append(w)
        if len(" ".join(current)) > rng.randint(28, 46):
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    return "\n\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260910)  # fixed seed: the benchmark must not drift

    for name, source in SOURCES.items():
        (OUT / f"{name}_ref.txt").write_text(source + "\n", encoding="utf-8")
        (OUT / f"{name}_raw.txt").write_text(damage(source, rng) + "\n", encoding="utf-8")
        print(f"wrote {name}_ref.txt / {name}_raw.txt")


if __name__ == "__main__":
    main()
