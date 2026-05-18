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
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Lazy spaCy loader — avoids crashing at import time if model not yet
# installed (e.g. fresh checkout before `spacy download` is run).
# ---------------------------------------------------------------------------

_nlp = None  # loaded on first use
_bert_embedder = None  # loaded on first use


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                _nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        if _nlp is None:
            raise RuntimeError(
                "spaCy model 'en_core_web_md' or 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_md"
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

_SEMANTIC_SEED_PHRASES: dict[str, list[str]] = {
    "confusion": [
        "I don't understand", "I'm confused", "I am stuck",
        "this doesn't make sense", "I don't get it", "lost",
        "not sure", "confusing", "help",
    ],
    "explanation": [
        "explain this", "clarify", "walk me through",
        "help me understand", "tell me why", "show me how",
        "explain why", "explain how",
    ],
}

_INTENT_SEED_PHRASES: dict[str, list[str]] = {
    "deep_questions": [
        "why", "how does", "what if", "what would happen", "explain",
        "difference between", "reason", "cause", "effect", "compare",
        "relationship", "meaning", "purpose", "significance",
    ],
    "scaffolded": [
        "help me", "guide me", "show me", "step by step",
        "walk me through", "give me a hint", "i need help",
        "where do i begin",
    ],
    "off_topic": [
        "lunch", "movie", "game", "weekend", "tired", "bored",
        "funny", "meme", "random", "music", "sport",
    ],
    "answer_seeking": [
        "what is the answer", "just tell me", "give me the answer",
        "what's the answer", "is it", "correct answer",
        "tell me the answer",
    ],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_bert_embedder():
    global _bert_embedder
    if _bert_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use BERT mode: "
                "pip install sentence-transformers"
            ) from exc
        _bert_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _bert_embedder


def _bert_vector(text: str):
    embedder = _get_bert_embedder()
    return embedder.encode(text, convert_to_numpy=True)


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    try:
        if len(vec1) == 0 or len(vec2) == 0 or len(vec1) != len(vec2):
            return 0.0
    except TypeError:
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def _seed_similarity_score(text: str, seed_phrases: list[str], engine: str = "spacy") -> float:
    if engine == "bert":
        source_vec = _bert_vector(text)
        return max(
            (_cosine_similarity(source_vec, _bert_vector(seed)) for seed in seed_phrases),
            default=0.0,
        )

    nlp = _get_nlp()
    doc = nlp(text)
    if not doc.vector_norm:
        return 0.0
    scores = []
    for seed in seed_phrases:
        seed_doc = nlp(seed)
        if seed_doc.vector_norm:
            try:
                scores.append(doc.similarity(seed_doc))
            except Exception:
                continue
    return max(scores, default=0.0)


def _semantic_similarity_matches(text: str, seed_phrases: list[str], threshold: float = 0.6, engine: str = "spacy") -> bool:
    return _seed_similarity_score(text, seed_phrases, engine=engine) >= threshold


def _message_identity(msg: dict[str, Any]) -> str:
    return str(
        msg.get("source_id")
        or msg.get("message_id")
        or f"{msg.get('student_id', 'unknown')}:{msg.get('timestamp', '')}"
    )


def _message_evidence(msg: dict[str, Any], signal: str = "") -> dict[str, Any]:
    evidence = {
        "source": msg.get("source") or "chat_history",
        "message_id": _message_identity(msg),
        "student_id": msg.get("student_id", ""),
        "quote": msg.get("text", ""),
        "signal": signal,
    }
    if msg.get("timestamp"):
        evidence["timestamp"] = msg["timestamp"]
    return evidence


def _append_unique_evidence(
    evidence: list[dict[str, Any]],
    msg: dict[str, Any],
    signal: str,
    limit: int = 3,
) -> None:
    if len(evidence) >= limit:
        return
    item = _message_evidence(msg, signal=signal)
    if all(existing["message_id"] != item["message_id"] for existing in evidence):
        evidence.append(item)


def _keyword_evidence(messages: list[dict[str, Any]], keywords: list[str], limit: int = 3) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    lowered_keywords = [kw.lower() for kw in keywords if kw]
    for msg in messages:
        text = msg.get("text", "")
        text_lower = text.lower()
        if lowered_keywords and not any(kw in text_lower for kw in lowered_keywords):
            continue
        _append_unique_evidence(evidence, msg, signal="student_chat", limit=limit)
    if not evidence:
        for msg in messages:
            _append_unique_evidence(evidence, msg, signal="student_chat", limit=limit)
    return evidence


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


def _student_confusion_score(
    student_msgs: list[dict[str, Any]],
    engine: str = "spacy",
) -> tuple[float, float, int, int, list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    confusion_hits = 0
    explanation_hits = 0
    confusion_examples: list[str] = []
    explanation_examples: list[str] = []
    confusion_evidence: list[dict[str, Any]] = []
    explanation_evidence: list[dict[str, Any]] = []

    for msg in student_msgs:
        text = msg["text"]
        is_confused = (
            _CONFUSION_PATTERNS.search(text)
            or _semantic_similarity_matches(text, _SEMANTIC_SEED_PHRASES["confusion"], engine=engine)
        )
        is_explaining = (
            _EXPLANATION_REQUEST_PATTERNS.search(text)
            or _semantic_similarity_matches(text, _SEMANTIC_SEED_PHRASES["explanation"], engine=engine)
        )

        if is_confused:
            confusion_hits += 1
            if len(confusion_examples) < 3:
                confusion_examples.append(text)
            _append_unique_evidence(confusion_evidence, msg, signal="confusion", limit=3)
        if is_explaining:
            explanation_hits += 1
            if len(explanation_examples) < 3:
                explanation_examples.append(text)
            _append_unique_evidence(explanation_evidence, msg, signal="explanation_request", limit=3)

    total = max(len(student_msgs), 1)
    confusion_fraction = min(confusion_hits / total, 1.0)
    explanation_fraction = min(explanation_hits / total, 1.0)
    return (
        confusion_fraction,
        explanation_fraction,
        confusion_hits,
        explanation_hits,
        confusion_examples,
        explanation_examples,
        confusion_evidence,
        explanation_evidence,
    )


def compute_cognitive_load(
    messages_by_student: dict[str, list[dict[str, Any]]],
    passports_by_student: dict[str, Any] | None = None,
    engine: str = "spacy",
    include_details: bool = False,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    detail_store: dict[str, dict[str, Any]] = {}

    passports_by_student = passports_by_student or {}

    for student_id, msgs in messages_by_student.items():
        student_msgs = [m for m in msgs if m["sender_type"] == "student"]
        if not student_msgs:
            scores[student_id] = 0.0
            detail_store[student_id] = {
                "message_count": 0,
                "confusion_hits": 0,
                "explanation_hits": 0,
                "confusion_fraction": 0.0,
                "explanation_fraction": 0.0,
                "confusion_examples": [],
                "explanation_examples": [],
                "average_sentence_length": 0.0,
                "subordinate_clause_ratio": 0.0,
                "delay_score": 0.0,
                "complexity_score": 0.0,
                "help_score": 0.0,
                "evidence": [],
                "reasons": ["No student chat messages found for this window."],
            }
            continue

        combined = " ".join(m["text"] for m in student_msgs)
        avg_len = _avg_sentence_length(combined)
        sub_ratio = _subordinate_clause_ratio(combined)
        len_score = min(max((avg_len - 5) / 15, 0.0), 1.0)
        complexity_score = 0.5 * len_score + 0.5 * min(sub_ratio * 5, 1.0)

        (
            confusion_fraction,
            explanation_fraction,
            confusion_hits,
            explanation_hits,
            confusion_examples,
            explanation_examples,
            confusion_evidence,
            explanation_evidence,
        ) = _student_confusion_score(student_msgs, engine=engine)
        help_score = min(confusion_fraction + 0.5 * explanation_fraction, 1.0)
        if explanation_fraction > 0.2 and confusion_fraction < 0.1:
            help_score *= 0.6

        all_msgs = sorted(msgs, key=lambda m: m["timestamp"])
        delays: list[float] = []
        for i, msg in enumerate(all_msgs[:-1]):
            nxt = all_msgs[i + 1]
            if msg["sender_type"] == "assistant" and nxt["sender_type"] == "student":
                delays.append(max((nxt["timestamp"] - msg["timestamp"]).total_seconds(), 0.0))

        delay_score = min(max((sum(delays) / len(delays) - 10) / 110, 0.0), 1.0) if delays else 0.0

        passport = passports_by_student.get(student_id)
        adjustment = _passport_adjustment(passport)

        final_score = (
            0.35 * complexity_score * adjustment["complexity"]
            + 0.25 * delay_score * adjustment["delay"]
            + 0.35 * help_score * adjustment["help"]
            + 0.05 * explanation_fraction * adjustment["help"]
        )

        scores[student_id] = min(max(final_score, 0.0), 1.0)
        evidence = (confusion_evidence + explanation_evidence)[:3]
        if not evidence:
            evidence = _keyword_evidence(student_msgs, [], limit=2)

        reasons = []
        if confusion_hits:
            reasons.append(f"{confusion_hits} student confusion cue(s)")
        if explanation_hits:
            reasons.append(f"{explanation_hits} explanation/help request(s)")
        if delay_score >= 0.5:
            reasons.append("Longer assistant-to-student response gaps")
        if complexity_score >= 0.5:
            reasons.append("Higher language complexity in student messages")
        if not reasons:
            reasons.append("Score is based on low observed struggle signals in the selected chat window")

        detail_store[student_id] = {
            "message_count": len(student_msgs),
            "confusion_hits": confusion_hits,
            "explanation_hits": explanation_hits,
            "confusion_fraction": round(confusion_fraction, 2),
            "explanation_fraction": round(explanation_fraction, 2),
            "confusion_examples": confusion_examples,
            "explanation_examples": explanation_examples,
            "average_sentence_length": round(avg_len, 2),
            "subordinate_clause_ratio": round(sub_ratio, 2),
            "delay_score": round(delay_score, 2),
            "complexity_score": round(complexity_score, 2),
            "help_score": round(help_score, 2),
            "evidence": evidence,
            "reasons": reasons,
        }

    results: list[dict[str, Any]] = []
    for student_id in sorted(messages_by_student.keys()):
        score = round(scores.get(student_id, 0.0), 2)
        if not include_details:
            student_details = detail_store.get(student_id, {})
            results.append(
                {
                    "student_id": student_id,
                    "score": score,
                    "engine": engine,
                    "message_count": student_details.get("message_count", 0),
                    "evidence": student_details.get("evidence", []),
                    "reasons": student_details.get("reasons", []),
                }
            )
            continue

        student_details = detail_store.get(student_id, {})
        results.append(
            {
                "student_id": student_id,
                "score": score,
                "engine": engine,
                "message_count": student_details.get("message_count", 0),
                "confusion_hits": student_details.get("confusion_hits", 0),
                "explanation_hits": student_details.get("explanation_hits", 0),
                "confusion_fraction": student_details.get("confusion_fraction", 0.0),
                "explanation_fraction": student_details.get("explanation_fraction", 0.0),
                "average_sentence_length": student_details.get("average_sentence_length", 0.0),
                "subordinate_clause_ratio": student_details.get("subordinate_clause_ratio", 0.0),
                "delay_score": student_details.get("delay_score", 0.0),
                "complexity_score": student_details.get("complexity_score", 0.0),
                "help_score": student_details.get("help_score", 0.0),
                "confusion_examples": student_details.get("confusion_examples", []),
                "explanation_examples": student_details.get("explanation_examples", []),
                "evidence": student_details.get("evidence", []),
                "reasons": student_details.get("reasons", []),
            }
        )

    return results


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
    engine: str = "spacy",
) -> list[dict[str, Any]]:
    passports_by_student = passports_by_student or {}
    struggle_messages: list[dict[str, Any]] = []

    for msg in student_messages:
        if msg["sender_type"] != "student":
            continue

        text = msg["text"]
        passport = passports_by_student.get(msg["student_id"])
        prefers_explicit = _passport_prefers_explicit_support(passport)

        if _EXPLANATION_REQUEST_PATTERNS.search(text) and prefers_explicit:
            continue

        if _CONFUSION_PATTERNS.search(text):
            struggle_messages.append(msg)
            continue

        if _EXPLANATION_REQUEST_PATTERNS.search(text):
            struggle_messages.append(msg)

    if not struggle_messages:
        return []

    struggle_texts = [msg["text"] for msg in struggle_messages]
    nlp = _get_nlp()
    phrase_counter: Counter = Counter()
    phrase_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for msg in struggle_messages:
        text = msg["text"]
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ("PRON",) or chunk.root.is_stop:
                continue

            phrase = chunk.text.strip().lower()
            if len(phrase) <= 2:
                continue

            similarity = 0.0
            try:
                if engine == "bert":
                    similarity = _cosine_similarity(_bert_vector(chunk.text), _bert_vector(text))
                else:
                    similarity = chunk.similarity(doc)
            except Exception:
                pass

            if similarity >= 0.45:
                phrase_counter[phrase] += 1
                _append_unique_evidence(phrase_evidence[phrase], msg, signal="topic_struggle", limit=3)

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
                phrase_evidence[term] = _keyword_evidence(struggle_messages, term.split(), limit=3)
        except ValueError:
            pass

    return [
        {
            "topic": t,
            "count": c,
            "engine": engine,
            "evidence": phrase_evidence.get(t) or _keyword_evidence(struggle_messages, t.split(), limit=3),
        }
        for t, c in phrase_counter.most_common(top_n)
    ]


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
                        "s_words": s_words, "t_words": t_words, "messages": b_msgs})

    max_msgs = max((b["count"] for b in buckets), default=1) or 1
    return [
        {
            "time": b["time"],
            "score": round(
                0.5 * (b["count"] / max_msgs) +
                0.5 * (min(b["s_words"] / b["t_words"], 1.0) if b["t_words"] else 0.0),
                2,
            ),
            "message_count": b["count"],
            "student_count": len({m["student_id"] for m in b["messages"] if m["sender_type"] == "student"}),
            "evidence": [
                _message_evidence(m, signal="engagement")
                for m in b["messages"]
                if m["sender_type"] == "student"
            ][:3],
        }
        for b in buckets
    ]


def _classify_intent(text: str, engine: str = "spacy") -> str:
    if engine == "bert":
        semantic_scores = {
            intent: _seed_similarity_score(text, seeds, engine=engine)
            for intent, seeds in _INTENT_SEED_PHRASES.items()
        }
        best_semantic = max(semantic_scores, key=semantic_scores.get)
        if semantic_scores[best_semantic] >= 0.65:
            return best_semantic

    text_lower = text.lower()
    hits = {intent: sum(1 for kw in kws if kw in text_lower)
            for intent, kws in _INTENT_KEYWORDS.items()}
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] > 0 else "scaffolded"


def compute_behavior_mix(
    student_messages: list[dict[str, Any]],
    engine: str = "spacy",
    include_evidence: bool = False,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in student_messages:
        intent = _classify_intent(msg["text"], engine=engine)
        counts[intent] += 1
        _append_unique_evidence(evidence[intent], msg, signal=intent, limit=3)
    result = {
        "deep_questions": counts.get("deep_questions", 0),
        "scaffolded":     counts.get("scaffolded", 0),
        "off_topic":      counts.get("off_topic", 0),
        "answer_seeking": counts.get("answer_seeking", 0),
    }
    if include_evidence:
        result["engine"] = engine
        result["evidence"] = {
            "deep_questions": evidence.get("deep_questions", []),
            "scaffolded": evidence.get("scaffolded", []),
            "off_topic": evidence.get("off_topic", []),
            "answer_seeking": evidence.get("answer_seeking", []),
        }
    return result


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
