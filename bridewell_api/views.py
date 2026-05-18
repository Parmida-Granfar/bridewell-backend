"""
views.py — DRF APIViews for Bridewell AI dashboard metrics.

All endpoints are read-only and operate on a rolling 60-minute window
("live" data) sourced from the ChatMessage table.

Routes
------
GET /api/v1/metrics/cognitive-load/
GET /api/v1/metrics/topic-wrestling/
GET /api/v1/metrics/engagement-timeline/
GET /api/v1/metrics/behavior-mix/
"""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
from datetime import datetime, timedelta, timezone

from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, StudentPassport
from .nlp_utils import (
    LIVE_WINDOW_MINUTES,
    compute_behavior_mix,
    compute_cognitive_load,
    compute_engagement_timeline,
    compute_topic_wrestling,
)
from .serializers import (
    BehaviorMixSerializer,
    ClassInteractionSummarySerializer,
    ClassSummarySerializer,
    CognitiveLoadItemSerializer,
    EngagementTimelineItemSerializer,
    TopicWrestlingItemSerializer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _live_queryset():
    """Return ChatMessages from the last LIVE_WINDOW_MINUTES."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=LIVE_WINDOW_MINUTES)
    return ChatMessage.objects.filter(timestamp__gte=cutoff).order_by("timestamp")


def _all_queryset():
    """Return all ChatMessages ordered by timestamp."""
    return ChatMessage.objects.order_by("timestamp")


def _use_live_messages(request: Request) -> bool:
    raw = request.query_params.get("live", "true").strip().lower()
    return raw not in {"0", "false", "no"}


def _to_dict(msg: ChatMessage) -> dict:
    """Convert a ChatMessage ORM object to a plain dict for nlp_utils."""
    return {
        "text": msg.text,
        "sender_type": msg.sender_type,
        "student_id": msg.student_id,
        "timestamp": msg.timestamp,
    }


def _load_passports(student_ids: set[str]) -> dict[str, dict]:
    passports = StudentPassport.objects.filter(student_id__in=student_ids)
    return {
        p.student_id: {
            "access_arrangements": p.access_arrangements or [],
            "declared_needs": p.declared_needs or [],
            "preferred_mode": p.preferred_mode or "",
            "support_needs": p.support_needs or [],
        }
        for p in passports
    }


def _passport_summary(passport_map: dict[str, dict], total_students: int) -> dict[str, object]:
    access_counter = Counter()
    declared_needs_counter = Counter()

    for passport in passport_map.values():
        access_counter.update(passport.get("access_arrangements", []))
        declared_needs_counter.update(passport.get("declared_needs", []))

    return {
        "students_with_passports": len(passport_map),
        "total_students": total_students,
        "common_access_arrangements": [item for item, _ in access_counter.most_common(5)],
        "common_declared_needs": [item for item, _ in declared_needs_counter.most_common(5)],
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class CognitiveLoadView(APIView):
    """
    GET /api/v1/metrics/cognitive-load/

    Returns a list of per-student cognitive-load scores computed from
    sentence complexity and response-delay proxies.

    Query params
    -----------
    - live: bool (default true) — set false to compute on all stored messages

    Response schema
    ---------------
    [
        {"student_id": "TOM",  "score": 0.82},
        {"student_id": "PRIYA","score": 0.64},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset() if _use_live_messages(request) else _all_queryset()

        # Group messages by student_id
        messages_by_student: dict[str, list[dict]] = defaultdict(list)
        for msg in qs:
            messages_by_student[msg.student_id].append(_to_dict(msg))

        passport_map = _load_passports(set(messages_by_student.keys()))
        data = compute_cognitive_load(
            dict(messages_by_student),
            passports_by_student=passport_map,
        )

        serializer = CognitiveLoadItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class TopicWrestlingView(APIView):
    """
    GET /api/v1/metrics/topic-wrestling/

    Returns topics that students are currently struggling with, based on
    confusion signals in their messages over the last 60 minutes.

    Query params
    -----------
    - top_n: int (default 10) — maximum number of topics to return
    - live: bool (default true) — set false to compute on all stored messages

    Response schema
    ---------------
    [
        {"topic": "fractions", "count": 12},
        {"topic": "division",  "count":  8},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset() if _use_live_messages(request) else _all_queryset()
        top_n = int(request.query_params.get("top_n", 10))

        student_msgs = [_to_dict(m) for m in qs if m.sender_type == "student"]
        passport_map = _load_passports({m["student_id"] for m in student_msgs})
        data = compute_topic_wrestling(
            student_msgs,
            passports_by_student=passport_map,
            top_n=top_n,
        )

        serializer = TopicWrestlingItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class ClassSummaryView(APIView):
    """GET /api/v1/metrics/class-summary/ - teacher view of overall class metrics.

    Query params
    -----------
    - live: bool (default true) — set false to compute on all stored messages
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset() if _use_live_messages(request) else _all_queryset()
        student_msgs = [_to_dict(m) for m in qs if m.sender_type == "student"]
        messages_by_student: dict[str, list[dict]] = defaultdict(list)
        for msg in student_msgs:
            messages_by_student[msg["student_id"]].append(msg)

        passport_map = _load_passports(set(messages_by_student.keys()))
        cognitive_data = compute_cognitive_load(
            dict(messages_by_student),
            passports_by_student=passport_map,
        )

        scores = [item["score"] for item in cognitive_data]
        student_count = len(scores)
        average_score = round(statistics.mean(scores), 2) if scores else 0.0
        median_score = round(statistics.median(scores), 2) if scores else 0.0
        high_risk_count = sum(1 for score in scores if score >= 0.75)
        warning_risk_count = sum(1 for score in scores if 0.5 <= score < 0.75)
        top_students = sorted(cognitive_data, key=lambda item: item["score"], reverse=True)[:5]

        topic_data = compute_topic_wrestling(
            student_msgs,
            passports_by_student=passport_map,
            top_n=10,
        )

        behavior_data = compute_behavior_mix(student_msgs)

        engagement_data = compute_engagement_timeline(student_msgs)
        engagement_scores = [item["score"] for item in engagement_data]
        average_engagement = round(statistics.mean(engagement_scores), 2) if engagement_scores else 0.0
        highest_bucket_score = round(max(engagement_scores), 2) if engagement_scores else 0.0
        lowest_bucket_score = round(min(engagement_scores), 2) if engagement_scores else 0.0

        class_summary = {
            "cognitive_load": {
                "average_score": average_score,
                "median_score": median_score,
                "high_risk_count": high_risk_count,
                "warning_risk_count": warning_risk_count,
                "total_students": student_count,
                "top_students_needing_support": top_students,
            },
            "topic_wrestling": topic_data,
            "behavior_mix": behavior_data,
            "engagement": {
                "average_score": average_engagement,
                "active_students": len(set(m["student_id"] for m in student_msgs)),
                "bucket_count": len(engagement_data),
                "highest_bucket_score": highest_bucket_score,
                "lowest_bucket_score": lowest_bucket_score,
            },
            "passport_summary": _passport_summary(passport_map, student_count),
        }

        serializer = ClassSummarySerializer(class_summary)
        return Response(serializer.data)


class EngagementTimelineView(APIView):
    """
    GET /api/v1/metrics/engagement-timeline/

    Returns a time-series of engagement scores over the last 30 minutes,
    bucketed into 2-minute intervals.

    Query params
    -----------
    - window_minutes: int (default 30)
    - bucket_minutes: int (default 2)

    Response schema
    ---------------
    [
        {"time": "14:00", "score": 0.85},
        {"time": "14:02", "score": 0.72},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset()
        window = int(request.query_params.get("window_minutes", request.query_params.get("window", 30)))
        bucket = int(request.query_params.get("bucket_minutes", request.query_params.get("bucket", 2)))

        messages = [_to_dict(m) for m in qs]
        data = compute_engagement_timeline(messages, window_minutes=window, bucket_minutes=bucket)

        serializer = EngagementTimelineItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class InteractionSummaryView(APIView):
    """GET /api/v1/metrics/interaction-summary/ - summary of student help-seeking patterns.

    Breaks down student interactions by action type (chat, hint, rephrase, simplify).
    Shows frequency of each action per student and overall class patterns.

    Query params
    -----------
    - live: bool (default true) — set false to compute on all stored messages

    Response schema
    ---------------
    {
      "total_interactions": 145,
      "students_count": 4,
      "action_breakdown": {"chat": 80, "hint": 30, "rephrase": 20, "simplify": 15},
      "top_actions_per_student": {
        "TOM": {"chat": 25, "hint": 10},
        "PRIYA": {"chat": 20, "hint": 15}
      },
      "by_student": [
        {
          "student_id": "TOM",
          "total_interactions": 35,
          "by_action": {"chat": 25, "hint": 10},
          "sessions_count": 3
        }
      ]
    }
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset() if _use_live_messages(request) else _all_queryset()

        # Group by student
        summary_by_student = defaultdict(lambda: {"by_action": Counter(), "sessions": set()})
        action_counter = Counter()

        for msg in qs:
            action = msg.action_type or "chat"
            summary_by_student[msg.student_id]["by_action"][action] += 1
            summary_by_student[msg.student_id]["sessions"].add(msg.session_id)
            action_counter[action] += 1

        student_summaries = [
            {
                "student_id": sid,
                "total_interactions": sum(data["by_action"].values()),
                "by_action": dict(data["by_action"]),
                "sessions_count": len(data["sessions"]),
            }
            for sid, data in sorted(summary_by_student.items())
        ]

        class_summary = {
            "total_interactions": sum(s["total_interactions"] for s in student_summaries),
            "students_count": len(student_summaries),
            "action_breakdown": dict(action_counter),
            "top_actions_per_student": {
                sid: dict(data["by_action"])
                for sid, data in summary_by_student.items()
            },
            "by_student": student_summaries,
        }

        serializer = ClassInteractionSummarySerializer(class_summary)
        return Response(serializer.data)


class BehaviorMixView(APIView):
    """
    GET /api/v1/metrics/behavior-mix/

    Returns a distribution of student behaviour types across all messages
    in the last 60 minutes.

    Response schema
    ---------------
    {
        "deep_questions": 15,
        "scaffolded":      8,
        "off_topic":       3,
        "answer_seeking":  2
    }
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset()
        student_msgs = [_to_dict(m) for m in qs if m.sender_type == "student"]

        data = compute_behavior_mix(student_msgs)

        serializer = BehaviorMixSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# New Service Functions
# ---------------------------------------------------------------------------


def get_chat_summary(student_id: str):
    """
    Uses NLP analysis of GPT-5 mini logs rather than static mock values.
    In production this should fetch recent chat history from DB.
    """
    # Replace with DB fetch for real student logs
    sample_messages = [
        {
            "role": "user",
            "content": "what if the dragon is the narrator?",
            "created_at": "2026-04-24T12:20:00Z"
        },
        {
            "role": "user",
            "content": "maybe that makes the story more interesting?",
            "created_at": "2026-04-24T12:22:00Z"
        }
    ]

    from .nlp_utils import analyse_chat_messages
    analysis = analyse_chat_messages(sample_messages)

    return {
        "student_id": student_id,
        "generated_at": datetime.now(tz=timezone.utc),
        "summary": analysis["summary"],
        "highlights": [
            "Asks advanced questions",
            "Shows meta-narrative thinking",
            "Responds well to stretch prompts",
        ],
        "signals": {
            "engagement": analysis["engagement"],
            "confidence": analysis["confidence"],
            "support_need": analysis["support_need"],
        },
        "evidence": [
            {
                "source": "chat_history",
                "message_id": "msg_1842",
                "quote": analysis["top_quote"],
            }
        ],
        "recommended_next_step": analysis["recommended_next_step"],
    }


def get_suggested_pairups(student_id: str | None = None):
    """
    Example logic:
    - compare student strengths vs another student's growth areas
    - use passport support needs + chat summary domain
    - prefer complementary pairing instead of same weakness pairing
    """
    students = [
        {
            "student_id": "oliver-bramwell",
            "strengths": ["creative_writing", "narrative_voice", "idea_generation"],
            "growth_areas": ["editing", "concision"],
        },
        {
            "student_id": "priya-n",
            "strengths": ["editing", "peer_feedback"],
            "growth_areas": ["creative_writing", "opening_ideas"],
        },
        {
            "student_id": "james-r",
            "strengths": ["maths_reasoning"],
            "growth_areas": ["written_explanations"],
        },
    ]

    target = next(
        (s for s in students if s["student_id"] == (student_id or "oliver-bramwell")),
        students[0],
    )

    candidate_pairs = []

    for other in students:
        if other["student_id"] == target["student_id"]:
            continue

        support_match = list(
            set(target["strengths"]) & set(other["growth_areas"])
        )
        reverse_match = list(
            set(other["strengths"]) & set(target["growth_areas"])
        )

        score = len(support_match) * 0.6 + len(reverse_match) * 0.4

        if score > 0:
            candidate_pairs.append(
                {
                    "student_a_id": other["student_id"],
                    "student_b_id": target["student_id"],
                    "match_score": round(min(score, 1.0), 2),
                    "reason": (
                        f"{target['student_id']} can support {other['student_id']} in "
                        f"{', '.join(support_match)} while receiving support in "
                        f"{', '.join(reverse_match) if reverse_match else 'reflection and feedback'}."
                    ),
                    "learning_balance": {
                        "a_receives": support_match or ["peer modelling"],
                        "b_receives": reverse_match or ["clarifying feedback"],
                    },
                    "task_prompt": "Work together on one draft: one generates ideas, one improves precision.",
                }
            )

    candidate_pairs = sorted(
        candidate_pairs,
        key=lambda x: x["match_score"],
        reverse=True,
    )

    return {
        "generated_at": datetime.now(tz=timezone.utc),
        "focus_area": "adaptive_peer_support",
        "pairs": candidate_pairs[:3],
    }


def get_learning_preferences(student_id: str):
    """Get learning preferences and signals for a student."""
    passport = StudentPassport.objects.filter(student_id=student_id).first()
    if passport:
        return {
            "student_id": student_id,
            "summary": {
                "preferred_mode": passport.preferred_mode or "Guidance with open-ended extension",
                "strengths": ["narrative voice", "meta-thinking"],
                "growth_areas": ["editing", "concision"],
                "language_adaptation": (
                    "Standard language with rich vocabulary welcome"
                ),
                "working_style": [
                    "responds well to challenge",
                    "benefits from probing questions",
                ],
                "support_needs": passport.support_needs or [
                    "clear structure",
                    "check-ins",
                    "task chunking",
                ],
            },
            "passport_signals": {
                "access_arrangements": passport.access_arrangements or [],
                "declared_needs": passport.declared_needs or [],
            },
            "chat_signals": {
                "recent_interests": [
                    "story structure",
                    "imagery",
                    "character perspective",
                ],
                "engagement_style": "curious, exploratory",
            },
        }

    return {
        "student_id": student_id,
        "summary": {
            "preferred_mode": "Guidance with open-ended extension",
            "strengths": ["narrative voice", "meta-thinking"],
            "growth_areas": ["editing", "concision"],
            "language_adaptation": (
                "Standard language with rich vocabulary welcome"
            ),
            "working_style": [
                "responds well to challenge",
                "benefits from probing questions",
            ],
            "support_needs": [
                "clear structure",
                "check-ins",
                "task chunking",
            ],
        },
        "passport_signals": {
            "access_arrangements": [
                "typing",
                "25% extra time",
                "reader",
                "prompt",
                "rest breaks",
                "reading ruler",
            ],
            "declared_needs": ["dyslexia", "ADHD"],
        },
        "chat_signals": {
            "recent_interests": [
                "story structure",
                "imagery",
                "character perspective",
            ],
            "engagement_style": "curious, exploratory",
        },
    }


# ---------------------------------------------------------------------------
# New Views
# ---------------------------------------------------------------------------


from rest_framework.decorators import api_view
from rest_framework import status as http_status
from .serializers import (
    ChatSummarySerializer,
    ClassSummarySerializer,
    SuggestedPairUpsSerializer,
    LearningPreferencesSerializer,
)


@api_view(["GET"])
def chat_summary(request, student_id):
    """GET /api/v1/metrics/chat-summary/<student_id>/"""
    data = get_chat_summary(student_id)
    serializer = ChatSummarySerializer(data)
    return Response(serializer.data, status=http_status.HTTP_200_OK)


@api_view(["GET"])
def suggested_pair_ups(request):
    """GET /api/v1/metrics/pair-ups/"""
    student_id = request.query_params.get("student_id")
    data = get_suggested_pairups(student_id)
    serializer = SuggestedPairUpsSerializer(data)
    return Response(serializer.data, status=http_status.HTTP_200_OK)


@api_view(["GET"])
def learning_preferences_summary(request, student_id):
    """GET /api/v1/metrics/learning-preferences/<student_id>/"""
    data = get_learning_preferences(student_id)
    serializer = LearningPreferencesSerializer(data)
    return Response(serializer.data, status=http_status.HTTP_200_OK)