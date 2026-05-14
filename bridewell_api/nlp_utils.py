"""
nlp_utils.py — NLP extraction logic for the Bridewell AI dashboard metrics.

Dependencies: spaCy (en_core_web_sm), scikit-learn

Install the model once before first use:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Lazy spaCy loader — avoids crashing at import time if model not yet
# installed (e.g. fresh checkout before `spacy download` is run).
# ---------------------------------------------------------------------------

_nlp = None  # loaded on first use


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIVE_WINDOW_MINUTES = 60

_CONFUSION_PATTERNS = re.compile(
    r"\b(don'?t understand|confused|confusing|not sure|unsure|help|"
    r"struggling|stuck|why|how do|what does|what is|don'?t get|"
    r"doesn'?t make sense|lost|explain|clarify)\b",
    re.IGNORECASE,
)

_EXPLANATION_REQUEST_PATTERNS = re.compile(
    r"\b(explain|clarify|show me|walk me through|help me understand|help me|"
    r"can you help|what does|how do i|don't understand|don't get|just tell me|tell me the answer)\b",
    re.IGNORECASE,
)

_SUPPORT_NEEDS_PATTERNS = re.compile(
    r"\b(explicit|clear structure|scaffold|step by step|check[- ]in|say it again|repeat|simplify|break it down)\b",
    re.IGNORECASE,
)

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "deep_questions": [
        "why", "how does", "what if", "what would happen", "explain",
        "difference between", "reason", "cause", "effect", "compare",
        "relationship", "meaning", "purpose", "significance",
    ],
    "scaffolded": [
        "hint", "clue", "help me", "guide", "show me", "step by step",
        "walk me through", "start me off", "where do i begin",
    ],
    "off_topic": [
        "lunch", "game", "football", "movie", "weekend", "tired",
        "bored", "hungry", "funny", "meme", "random",
    ],
    "answer_seeking": [
        "what is the answer", "just tell me", "give me the answer",
        "what's the answer", "is it", "is the answer", "correct answer",
        "tell me the answer", "right answer",
    ],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _avg_sentence_length(text: str) -> float:
    nlp = _get_nlp()
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return 0.0
    return sum(len(s.split()) for s in sents) / len(sents)


def _subordinate_clause_ratio(text: str) -> float:
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return 0.0
    complex_deps = {"mark", "relcl", "advcl", "ccomp", "xcomp"}
    return len([t for t in tokens if t.dep_ in complex_deps]) / len(tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _normalize_passport_data(passport: Any) -> dict[str, Any]:
    if passport is None:
        return {
            "access_arrangements": [],
            "declared_needs": [],
            "preferred_mode": "",
            "support_needs": [],
        }

    if isinstance(passport, dict):
        return {
            "access_arrangements": passport.get("access_arrangements", []) or [],
            "declared_needs": passport.get("declared_needs", []) or [],
            "preferred_mode": passport.get("preferred_mode", "") or "",
            "support_needs": passport.get("support_needs", []) or [],
        }

    return {
        "access_arrangements": getattr(passport, "access_arrangements", []) or [],
        "declared_needs": getattr(passport, "declared_needs", []) or [],
        "preferred_mode": getattr(passport, "preferred_mode", "") or "",
        "support_needs": getattr(passport, "support_needs", []) or [],
    }


def _passport_adjustment(passport: Any) -> dict[str, float]:
    normalized = _normalize_passport_data(passport)
    access_arrangements = [a.lower() for a in normalized["access_arrangements"]]
    declared_needs = [n.lower() for n in normalized["declared_needs"]]
    support_needs = [n.lower() for n in normalized["support_needs"]]

    adjustment = {"complexity": 1.0, "delay": 1.0, "help": 1.0}

    if any("dyslexia" in need for need in declared_needs):
        adjustment["complexity"] = 0.6
    if any("adhd" in need for need in declared_needs):
        adjustment["help"] = 0.8
    if any("extra time" in arrangement or "rest break" in arrangement for arrangement in access_arrangements):
        adjustment["delay"] = 0.5

    if any(_SUPPORT_NEEDS_PATTERNS.search(need) for need in support_needs):
        adjustment["help"] = min(adjustment["help"], 0.7)

    return adjustment


def _student_confusion_score(student_msgs: list[dict[str, Any]]) -> tuple[float, float]:
    confusion_hits = 0
    explanation_hits = 0

    for msg in student_msgs:
        text = msg["text"]
        if _CONFUSION_PATTERNS.search(text):
            confusion_hits += 1
        if _EXPLANATION_REQUEST_PATTERNS.search(text):
            explanation_hits += 1

    total = max(len(student_msgs), 1)
    confusion_fraction = min(confusion_hits / total, 1.0)
    explanation_fraction = min(explanation_hits / total, 1.0)
    return confusion_fraction, explanation_fraction


def compute_cognitive_load(
    messages_by_student: dict[str, list[dict[str, Any]]],
    passports_by_student: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}

    passports_by_student = passports_by_student or {}

    for student_id, msgs in messages_by_student.items():
        student_msgs = [m for m in msgs if m["sender_type"] == "student"]
        if not student_msgs:
            scores[student_id] = 0.0
            continue

        combined = " ".join(m["text"] for m in student_msgs)
        avg_len = _avg_sentence_length(combined)
        sub_ratio = _subordinate_clause_ratio(combined)
        len_score = min(max((avg_len - 5) / 15, 0.0), 1.0)
        complexity_score = 0.5 * len_score + 0.5 * min(sub_ratio * 5, 1.0)

        confusion_fraction, explanation_fraction = _student_confusion_score(student_msgs)
        help_score = min(confusion_fraction + 0.5 * explanation_fraction, 1.0)
        if explanation_fraction > 0.2 and confusion_fraction < 0.1:
            help_score *= 0.6

        all_msgs = sorted(msgs, key=lambda m: m["timestamp"])
        delays: list[float] = []
        for i, msg in enumerate(all_msgs[:-1]):
            nxt = all_msgs[i + 1]
            if msg["sender_type"] == "assistant" and nxt["sender_type"] == "student":
                delays.append(max((nxt["timestamp"] - msg["timestamp"]).total_seconds(), 0.0))

        delay_score = 0.0
        if delays:
            delay_score = min(max((sum(delays) / len(delays) - 10) / 110, 0.0), 1.0)

        passport = passports_by_student.get(student_id)
        adjustment = _passport_adjustment(passport)

        final_score = (
            0.35 * complexity_score * adjustment["complexity"]
            + 0.25 * delay_score * adjustment["delay"]
            + 0.35 * help_score * adjustment["help"]
            + 0.05 * explanation_fraction * adjustment["help"]
        )

        scores[student_id] = min(max(final_score, 0.0), 1.0)

    return [
        {
            "student_id": sid,
            "score": round(scores.get(sid, 0.0), 2),
        }
        for sid in sorted(messages_by_student.keys())
    ]


def _passport_prefers_explicit_support(passport: Any) -> bool:
    normalized = _normalize_passport_data(passport)
    preferred_mode = normalized["preferred_mode"].lower()
    support_needs = [need.lower() for need in normalized["support_needs"]]

    if any(_SUPPORT_NEEDS_PATTERNS.search(need) for need in support_needs):
        return True

    return any(
        keyword in preferred_mode
        for keyword in ("explicit", "structured", "scaffold", "step by step", "clear structure")
    )


def compute_topic_wrestling(
    student_messages: list[dict[str, Any]],
    passports_by_student: dict[str, Any] | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    passports_by_student = passports_by_student or {}
    struggle_texts: list[str] = []

    for msg in student_messages:
        if msg["sender_type"] != "student":
            continue

        text = msg["text"]
        passport = passports_by_student.get(msg["student_id"])
        prefers_explicit = _passport_prefers_explicit_support(passport)

        if _CONFUSION_PATTERNS.search(text):
            struggle_texts.append(text)
            continue

        if _EXPLANATION_REQUEST_PATTERNS.search(text):
            if not prefers_explicit:
                struggle_texts.append(text)

    if not struggle_texts:
        return []

    nlp = _get_nlp()
    phrase_counter: Counter = Counter()

    for text in struggle_texts:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            lemma = chunk.root.lemma_.lower()
            if chunk.root.pos_ not in ("PRON",) and not chunk.root.is_stop and len(lemma) > 2:
                phrase_counter[chunk.text.strip().lower()] += 1

    if not phrase_counter:
        # TF-IDF fallback, but do not fail if scikit-learn is unavailable.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return []

        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n, ngram_range=(1, 2))
        try:
            vectorizer.fit(struggle_texts)
            for term in vectorizer.get_feature_names_out():
                phrase_counter[term] = 1
        except ValueError:
            pass

    return [{"topic": t, "count": c} for t, c in phrase_counter.most_common(top_n)]


def compute_engagement_timeline(
    messages: list[dict[str, Any]],
    window_minutes: int = 30,
    bucket_minutes: int = 2,
) -> list[dict[str, Any]]:
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(minutes=window_minutes)
    num_buckets = window_minutes // bucket_minutes

    buckets = []
    for i in range(num_buckets):
        b_start = start + timedelta(minutes=i * bucket_minutes)
        b_end = b_start + timedelta(minutes=bucket_minutes)
        b_msgs = [m for m in messages if b_start <= m["timestamp"] < b_end]
        s_words = sum(len(m["text"].split()) for m in b_msgs if m["sender_type"] == "student")
        t_words = sum(len(m["text"].split()) for m in b_msgs)
        buckets.append({"time": b_start.strftime("%H:%M"), "count": len(b_msgs),
                        "s_words": s_words, "t_words": t_words})

    max_msgs = max((b["count"] for b in buckets), default=1) or 1
    return [
        {
            "time": b["time"],
            "score": round(
                0.5 * (b["count"] / max_msgs) +
                0.5 * (min(b["s_words"] / b["t_words"], 1.0) if b["t_words"] else 0.0),
                2,
            ),
        }
        for b in buckets
    ]


def _classify_intent(text: str) -> str:
    text_lower = text.lower()
    hits = {intent: sum(1 for kw in kws if kw in text_lower)
            for intent, kws in _INTENT_KEYWORDS.items()}
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] > 0 else "scaffolded"


def compute_behavior_mix(student_messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for msg in student_messages:
        counts[_classify_intent(msg["text"])] += 1
    return {
        "deep_questions": counts.get("deep_questions", 0),
        "scaffolded":     counts.get("scaffolded", 0),
        "off_topic":      counts.get("off_topic", 0),
        "answer_seeking": counts.get("answer_seeking", 0),
    }


# ---------------------------------------------------------------------------
# New NLP Functions (spaCy-based chat analysis)
# ---------------------------------------------------------------------------


def analyse_chat_messages(messages):
    """
    Expected GPT-5 mini chat log format:
    [
        {
            "role": "user",
            "content": "what if the dragon is the narrator?",
            "created_at": "2026-04-24T12:20:00Z"
        }
    ]
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    text_blocks = [m.get("content", "") for m in user_messages if m.get("content")]
    combined = " ".join(text_blocks)

    if not combined:
        return {
            "summary": "No recent student interaction available.",
            "highlights": ["No recent messages"],
            "engagement": 0.0,
            "confidence": 0.5,
            "support_need": "unknown",
            "top_quote": "No recent message",
            "recommended_next_step": "Prompt the student with a low-stakes starter question."
        }

    nlp = _get_nlp()
    doc = nlp(combined)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "VERB", "ADJ"}
        and not token.is_stop
        and token.is_alpha
        and len(token.text) > 2
    ]
    keyword_counts = Counter(keywords)
    top_keywords = [word for word, _ in keyword_counts.most_common(5)]

    question_count = combined.count("?")
    engagement = min(1.0, round(len(user_messages) / 10, 2))
    confidence = min(1.0, round(0.55 + (question_count * 0.05), 2))

    if any(word in top_keywords for word in ["story", "narrator", "character", "metaphor"]):
        summary = "Student shows strong engagement with narrative thinking and reflective questioning."
        support_need = "low"
        next_step = "Offer an extended writing prompt with comparison or justification."
        highlights = ["Strong narrative reasoning", "Creative thinking"]
    elif any(word in top_keywords for word in ["fraction", "number", "divide", "equivalent"]):
        summary = "Student is actively reasoning through mathematical ideas and benefits from structured explanation."
        support_need = "medium"
        next_step = "Use worked examples followed by explanation in their own words."
        highlights = ["Conceptual mathematical reasoning", "Benefits from scaffolding"]
    else:
        summary = "Student is participating consistently and benefits from guided questioning and regular check-ins."
        support_need = "medium"
        next_step = "Use probing follow-up questions to deepen understanding."
        highlights = ["Steady classroom engagement"]

    return {
        "summary": summary,
        "highlights": highlights,
        "engagement": engagement,
        "confidence": confidence,
        "support_need": support_need,
        "top_quote": text_blocks[-1] if text_blocks else "",
        "recommended_next_step": next_step
    }