# -*- coding: utf-8 -*-
"""
Legal-issue expansion for retrieval.

The measured problem this exists to fix: retrieval finds the article that
*reads like* the clause, and the articles that decide whether a clause is void
never read like it. On the 40-clause benchmark, clauses governed by a topic
article (lease, sale) reached Recall@3 62%; clauses governed by a general
principle of contract law reached 40%. Article 149 - abusive terms in an
adhesion contract, the single most load-bearing provision for this product -
was never retrieved once across four different clauses that it governs.

The reason is not a ranking failure. A clause says "يتنازل المستأجر عن حقه في
اللجوء إلى القضاء"; article 149 says "إذا تم العقد بطريق الإذعان وكان قد تضمن
شروطاً تعسفية". They share almost no vocabulary and little surface semantics,
so no amount of reordering the candidate list will surface it - it was never a
candidate.

So the clause is searched twice: once as written, and once as the abstract
legal question it raises. The second query is built by rule rather than by a
model call: the mapping is small, it is the part a lawyer would want to audit,
and it costs no quota.

    legal_concepts("... يتنازل عن حقه في اللجوء إلى القضاء ...")
    -> ["شرط تعسفي في عقد إذعان يجوز للقاضي تعديله أو الإعفاء منه ..."]
"""
from __future__ import annotations

import re
from typing import List, Tuple

# --------------------------------------------------------------------------
# Normalisation - patterns are matched against folded text so that spelling
# variants in a scanned contract still hit.
# --------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def _fold(text: str) -> str:
    text = _DIACRITICS.sub("", text or "")
    for src in "أإآٱ":
        text = text.replace(src, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", text)


def _any(*alternatives: str) -> str:
    return "(?:" + "|".join(alternatives) + ")"


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------
# Each rule is (name, trigger pattern, query text). The query is phrased in the
# vocabulary the *statute* uses, not the vocabulary the contract uses - that is
# the whole point. Article numbers in the comments are what the rule is aiming
# at; nothing depends on them, they document intent and make the rules
# reviewable.
CONCEPT_RULES: List[Tuple[str, re.Pattern, str]] = [
    (
        "adhesion_abusive",  # م.149
        re.compile(_fold(_any(
            r"تنازل[^.]{0,40}عن حقه",
            r"تنازل[^.]{0,30}عن (?:اي|كل|جميع)",
            r"نهاييا? وغير قابل",
            r"غير قابل للاعتراض",
            r"غير قابله للتعديل",
            r"دون حق[^.]{0,20}في الاعتراض",
            r"لا يجوز[^.]{0,25}الطعن",
            r"قراره? نهايي",
            r"بصفه تلقاييه دون",
            r"مهما كانت جسامتها",
        ))),
        "شرط تعسفي في عقد إذعان يجوز للقاضي أن يعدله أو أن يعفي الطرف المذعن "
        "منه، ويقع باطلاً كل اتفاق على خلاف ذلك",
    ),
    (
        "liability_waiver",  # م.217
        re.compile(_fold(_any(
            r"يعفي[^.]{0,35}مس[وي]ولي",
            r"اخلاء[^.]{0,15}المس[وي]ولي",
            r"لا يتحمل[^.]{0,25}مس[وي]ولي",
            r"اسقاط[^.]{0,20}الضمان",
            r"تنازل[^.]{0,30}الضمان",
            r"دون ادني مس[وي]ولي",
        ))),
        "الاتفاق على الإعفاء من المسؤولية عن الغش أو الخطأ الجسيم باطل، "
        "وشرط عدم الضمان لا يعفي من الغش",
    ),
    (
        "penalty_clause",  # م.224
        re.compile(_fold(_any(
            r"شرط جزايي",
            r"تعويض اتفاقي",
            r"غرامه[^.]{0,20}تاخير",
            r"غير قابل للتخفيض",
            r"يلتزم بدفع مبلغ[^.]{0,40}كشرط",
        ))),
        "التعويض الاتفاقي والشرط الجزائي، وجواز تخفيضه إذا كان مبالغاً فيه "
        "إلى درجة كبيرة أو إذا نُفذ الالتزام في جزء منه",
    ),
    (
        "unilateral_termination",  # م.147
        re.compile(_fold(_any(
            r"بارادته المنفرده",
            r"في اي وقت دون ابداء اسباب",
            r"يحق[^.]{0,30}انهاء[^.]{0,25}دون",
            r"فسخ[^.]{0,25}دون[^.]{0,20}حكم",
            r"تعديل[^.]{0,25}من طرف واحد",
        ))),
        "العقد شريعة المتعاقدين فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين "
        "أو للأسباب التي يقررها القانون",
    ),
    (
        "unjust_enrichment",  # م.143 وما بعدها
        re.compile(_fold(_any(
            r"غير مسترد",
            r"لا يرد[^.]{0,20}باي حال",
            r"يسقط حقه في استرداد",
            r"يستولي[^.]{0,25}المبلغ",
        ))),
        "استرداد ما دفع بغير حق وأثر بطلان العقد في رد المتعاقدين إلى الحالة "
        "التي كانا عليها، والإثراء بلا سبب",
    ),
    (
        "self_help_eviction",  # الإخلاء بغير حكم
        re.compile(_fold(_any(
            r"دون حاجه الي[^.]{0,20}حكم قضايي",
            r"بدون حكم",
            r"تغيير[^.]{0,15}الاقفال",
            r"الاستيلاء علي منقولاته",
            r"طرده فورا",
        ))),
        "لا يجوز اقتضاء الحق بالذات، والإخلاء يكون بحكم قضائي، والتزام "
        "المستأجر برد العين عند انتهاء الإيجار",
    ),
    (
        "access_and_quiet_enjoyment",  # م.571/572
        re.compile(_fold(_any(
            r"دخول العين[^.]{0,30}اي وقت",
            r"بدون اخطار",
            r"في اي وقت يشاء",
        ))),
        "التزام المؤجر بالامتناع عن كل ما يحول دون انتفاع المستأجر بالعين "
        "وضمان عدم التعرض له",
    ),
]

MAX_CONCEPTS = 2


def legal_concepts(clause_text: str) -> List[str]:
    """Abstract legal questions this clause raises, most specific first.

    Returns [] when no rule matches, which is the common case for an ordinary
    clause - and the right answer, since a second query costs a search and
    should only be spent when there is a reason.
    """
    folded = _fold(clause_text)
    hits = [query for _, pattern, query in CONCEPT_RULES if pattern.search(folded)]
    return hits[:MAX_CONCEPTS]


def matched_rules(clause_text: str) -> List[str]:
    """Rule names that fired - for the benchmark and for debugging."""
    folded = _fold(clause_text)
    return [name for name, pattern, _ in CONCEPT_RULES if pattern.search(folded)]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        "يقر المستأجر بتنازله النهائي عن حقه في اللجوء إلى القضاء أو الطعن على أي إجراء يتخذه المؤجر.",
        "يلتزم بدفع مائة ألف جنيه كشرط جزائي غير قابل للتخفيض أو المنازعة أمام القضاء.",
        "يلتزم المستأجر بأداء الأجرة في المواعيد المتفق عليها في العقد.",
    ]
    for s in samples:
        print(f"{s[:60]}...\n  -> {matched_rules(s) or '(no rule)'}\n")
