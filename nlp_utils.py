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


def compute_cognitive_load(
    messages_by_student: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    delay_scores: dict[str, float] = {}

    for student_id, msgs in messages_by_student.items():
        student_msgs = [m for m in msgs if m["sender_type"] == "student"]
        if not student_msgs:
            scores[student_id] = 0.0
            delay_scores[student_id] = 0.0
            continue

        combined = " ".join(m["text"] for m in student_msgs)
        avg_len = _avg_sentence_length(combined)
        sub_ratio = _subordinate_clause_ratio(combined)
        len_score = min(max((avg_len - 5) / 15, 0.0), 1.0)
        complexity_score = 0.6 * len_score + 0.4 * min(sub_ratio * 5, 1.0)

        all_msgs = sorted(msgs, key=lambda m: m["timestamp"])
        delays: list[float] = []
        for i, msg in enumerate(all_msgs[:-1]):
            nxt = all_msgs[i + 1]
            if msg["sender_type"] == "assistant" and nxt["sender_type"] == "student":
                delays.append(max((nxt["timestamp"] - msg["timestamp"]).total_seconds(), 0.0))

        delay_score = 0.0
        if delays:
            delay_score = min(max((sum(delays) / len(delays) - 10) / 110, 0.0), 1.0)

        scores[student_id] = complexity_score
        delay_scores[student_id] = delay_score

    return [
        {
            "student_id": sid,
            "score": round(0.5 * scores.get(sid, 0.0) + 0.5 * delay_scores.get(sid, 0.0), 2),
        }
        for sid in sorted(messages_by_student.keys())
    ]


def compute_topic_wrestling(
    student_messages: list[dict[str, Any]],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    confusion_texts = [
        m["text"] for m in student_messages if _CONFUSION_PATTERNS.search(m["text"])
    ]
    if not confusion_texts:
        return []

    nlp = _get_nlp()
    phrase_counter: Counter = Counter()

    for text in confusion_texts:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            lemma = chunk.root.lemma_.lower()
            if chunk.root.pos_ not in ("PRON",) and not chunk.root.is_stop and len(lemma) > 2:
                phrase_counter[chunk.text.strip().lower()] += 1

    if not phrase_counter:
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n, ngram_range=(1, 2))
        try:
            vectorizer.fit(confusion_texts)
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
