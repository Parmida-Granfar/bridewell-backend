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

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage
from .nlp_utils import (
    LIVE_WINDOW_MINUTES,
    compute_behavior_mix,
    compute_cognitive_load,
    compute_engagement_timeline,
    compute_topic_wrestling,
)
from .serializers import (
    BehaviorMixSerializer,
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


def _to_dict(msg: ChatMessage) -> dict:
    """Convert a ChatMessage ORM object to a plain dict for nlp_utils."""
    return {
        "text": msg.text,
        "sender_type": msg.sender_type,
        "student_id": msg.student_id,
        "timestamp": msg.timestamp,
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class CognitiveLoadView(APIView):
    """
    GET /api/v1/metrics/cognitive-load/

    Returns a list of per-student cognitive-load scores computed from
    sentence complexity and response-delay proxies over the last 60 minutes.

    Response schema
    ---------------
    [
        {"student_id": "TOM",  "score": 0.82},
        {"student_id": "PRIYA","score": 0.64},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset()

        # Group messages by student_id
        messages_by_student: dict[str, list[dict]] = defaultdict(list)
        for msg in qs:
            messages_by_student[msg.student_id].append(_to_dict(msg))

        data = compute_cognitive_load(dict(messages_by_student))

        serializer = CognitiveLoadItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class TopicWrestlingView(APIView):
    """
    GET /api/v1/metrics/topic-wrestling/

    Extracts noun phrases from student messages that contain confusion /
    help-seeking signals. Returns a ranked frequency list.

    Query params
    ------------
    top_n (int, default 10): Maximum number of topics to return.

    Response schema
    ---------------
    [
        {"topic": "comparing fractions", "count": 14},
        {"topic": "opening sentences",   "count": 9},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        top_n = int(request.query_params.get("top_n", 10))
        top_n = max(1, min(top_n, 50))  # clamp to [1, 50]

        qs = _live_queryset().filter(sender_type=ChatMessage.SenderType.STUDENT)
        student_messages = [_to_dict(msg) for msg in qs]

        data = compute_topic_wrestling(student_messages, top_n=top_n)

        serializer = TopicWrestlingItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class EngagementTimelineView(APIView):
    """
    GET /api/v1/metrics/engagement-timeline/

    Builds a 30-minute time-series of engagement scores using message
    frequency and student-to-AI word-count ratio as proxies.

    Query params
    ------------
    window   (int, default 30): Rolling window in minutes (max 60).
    bucket   (int, default 2):  Bucket size in minutes (min 1).

    Response schema
    ---------------
    [
        {"time": "13:58", "score": 0.72},
        {"time": "14:00", "score": 0.80},
        ...
    ]
    """

    def get(self, request: Request) -> Response:
        window = int(request.query_params.get("window", 30))
        bucket = int(request.query_params.get("bucket", 2))

        # Clamp to sensible ranges
        window = max(5, min(window, LIVE_WINDOW_MINUTES))
        bucket = max(1, min(bucket, window))

        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window)
        qs = ChatMessage.objects.filter(timestamp__gte=cutoff).order_by("timestamp")
        messages = [_to_dict(msg) for msg in qs]

        data = compute_engagement_timeline(
            messages,
            window_minutes=window,
            bucket_minutes=bucket,
        )

        serializer = EngagementTimelineItemSerializer(data=data, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class BehaviorMixView(APIView):
    """
    GET /api/v1/metrics/behavior-mix/

    Classifies every student message from the last 60 minutes into one of
    four intent buckets and returns raw counts.

    Response schema
    ---------------
    {
        "deep_questions":  42,
        "scaffolded":      28,
        "off_topic":       18,
        "answer_seeking":  12
    }
    """

    def get(self, request: Request) -> Response:
        qs = _live_queryset().filter(sender_type=ChatMessage.SenderType.STUDENT)
        student_messages = [_to_dict(msg) for msg in qs]

        data = compute_behavior_mix(student_messages)

        serializer = BehaviorMixSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)
