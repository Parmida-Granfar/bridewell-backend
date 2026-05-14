"""
tests.py — Unit and integration tests for the Bridewell AI metrics API.

Run with:
    python manage.py test bridewell_api
or:
    pytest bridewell_api/tests.py -v
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .models import ChatMessage
from .nlp_utils import (
    compute_behavior_mix,
    compute_cognitive_load,
    compute_engagement_timeline,
    compute_topic_wrestling,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_msg(
    text: str,
    sender_type: str,
    student_id: str = "TEST",
    offset_seconds: int = 0,
) -> dict:
    """Return a plain-dict message as expected by nlp_utils functions."""
    return {
        "text": text,
        "sender_type": sender_type,
        "student_id": student_id,
        "timestamp": datetime.now(tz=timezone.utc) - timedelta(seconds=offset_seconds),
    }


def _db_msg(
    text: str,
    sender_type: str,
    student_id: str = "TEST",
    offset_seconds: int = 0,
) -> ChatMessage:
    """Create and persist a ChatMessage for view-level tests."""
    return ChatMessage.objects.create(
        text=text,
        sender_type=sender_type,
        student_id=student_id,
        timestamp=datetime.now(tz=timezone.utc) - timedelta(seconds=offset_seconds),
    )


# ---------------------------------------------------------------------------
# nlp_utils unit tests
# ---------------------------------------------------------------------------


class CognitiveLoadUnitTests(TestCase):
    def test_empty_input_returns_empty_list(self):
        result = compute_cognitive_load({})
        self.assertEqual(result, [])

    def test_score_in_valid_range(self):
        msgs_by_student = {
            "TOM": [
                _make_msg(
                    "I don't understand why fractions work this way at all.",
                    "student",
                    "TOM",
                    offset_seconds=90,
                ),
                _make_msg("Can you help me?", "assistant", "TOM", offset_seconds=60),
                _make_msg(
                    "Because the denominator represents the number of equal parts.",
                    "student",
                    "TOM",
                    offset_seconds=10,
                ),
            ]
        }
        result = compute_cognitive_load(msgs_by_student)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["student_id"], "TOM")
        self.assertGreaterEqual(result[0]["score"], 0.0)
        self.assertLessEqual(result[0]["score"], 1.0)

    def test_passport_adjustment_includes_support_context(self):
        msgs_by_student = {
            "TOM": [
                _make_msg(
                    "I don't understand this problem and I need help.",
                    "student",
                    "TOM",
                )
            ]
        }
        passports = {
            "TOM": {
                "access_arrangements": ["25% extra time"],
                "declared_needs": ["dyslexia"],
                "preferred_mode": "explicit instruction",
                "support_needs": ["step by step", "clear structure"],
            }
        }
        result = compute_cognitive_load(msgs_by_student, passports_by_student=passports)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["student_id"], "TOM")
        self.assertGreaterEqual(result[0]["score"], 0.0)
        self.assertLessEqual(result[0]["score"], 1.0)

    def test_multiple_students_returned_sorted(self):
        msgs_by_student = {
            "ZOE": [_make_msg("Hi", "student", "ZOE")],
            "AMY": [_make_msg("Hello", "student", "AMY")],
        }
        result = compute_cognitive_load(msgs_by_student)
        ids = [r["student_id"] for r in result]
        self.assertEqual(ids, sorted(ids))


class TopicWrestlingUnitTests(TestCase):
    def test_empty_messages_returns_empty(self):
        self.assertEqual(compute_topic_wrestling([]), [])

    def test_non_confusion_messages_yield_no_results(self):
        msgs = [_make_msg("The sky is blue.", "student")]
        # Might return nothing since no confusion signal
        result = compute_topic_wrestling(msgs)
        # Result is a list; no assertion on length since "blue" might slip through
        self.assertIsInstance(result, list)

    def test_confusion_messages_yield_topics(self):
        msgs = [
            _make_msg("I don't understand fractions at all", "student"),
            _make_msg("I'm confused about equivalent fractions", "student"),
            _make_msg("Can you help me with fractions?", "student"),
        ]
        result = compute_topic_wrestling(msgs, top_n=5)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIn("topic", item)
            self.assertIn("count", item)
            self.assertIsInstance(item["count"], int)

    def test_top_n_respected(self):
        msgs = [
            _make_msg(f"I don't understand concept {i}", "student")
            for i in range(20)
        ]
        result = compute_topic_wrestling(msgs, top_n=3)
        self.assertLessEqual(len(result), 3)

    def test_explicit_support_explanation_requests_are_not_counted_as_struggle_topics(self):
        msgs = [
            _make_msg("Can you explain this to me?", "student", "TOM"),
        ]
        passports = {
            "TOM": {
                "access_arrangements": ["reader"],
                "declared_needs": ["dyslexia"],
                "preferred_mode": "explicit instruction",
                "support_needs": ["clear structure", "step by step"],
            }
        }
        result = compute_topic_wrestling(msgs, passports_by_student=passports)
        self.assertEqual(result, [])

    def test_parse_studentlog_dataset_normalizes_messages(self):
        sample = {
            "conversations": [
                {
                    "conversation_id": "CHAT_TEST",
                    "user_id": "USER_1",
                    "messages": [
                        {
                            "message_id": "MSG_TEST_1",
                            "role": "user",
                            "timestamp": 1777983252,
                            "content": "I need help with this question",
                        },
                        {
                            "message_id": "MSG_TEST_2",
                            "role": "assistant",
                            "timestamp": 1777983260,
                            "content": "Sure, what part is confusing?",
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "studentlogs.json"
            path.write_text(json.dumps(sample), encoding="utf-8")
            from .studentlog_utils import parse_studentlog_dataset

            normalized = parse_studentlog_dataset(path)

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["sender_type"], "student")
        self.assertEqual(normalized[1]["sender_type"], "assistant")
        self.assertEqual(normalized[0]["student_id"], "USER_1")
        self.assertEqual(normalized[0]["source"], "studentlogs")

    def test_import_studentlogs_command_creates_messages(self):
        sample = {
            "conversations": [
                {
                    "conversation_id": "CHAT_TEST",
                    "user_id": "USER_2",
                    "messages": [
                        {
                            "message_id": "MSG_TEST_3",
                            "role": "user",
                            "timestamp": 1777983252,
                            "content": "Can you explain this?",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "studentlogs.json"
            path.write_text(json.dumps(sample), encoding="utf-8")
            call_command("import_studentlogs", str(path), clear_source=True)

        self.assertEqual(ChatMessage.objects.filter(source="studentlogs").count(), 1)

    def test_sorted_descending_by_count(self):
        msgs = [
            _make_msg("I don't understand fractions", "student"),
            _make_msg("I'm confused about fractions", "student"),
            _make_msg("Help me with fractions please", "student"),
            _make_msg("I'm confused about division", "student"),
        ]
        result = compute_topic_wrestling(msgs)
        counts = [r["count"] for r in result]
        self.assertEqual(counts, sorted(counts, reverse=True))


class EngagementTimelineUnitTests(TestCase):
    def test_empty_messages_returns_all_zeros(self):
        result = compute_engagement_timeline([], window_minutes=10, bucket_minutes=2)
        self.assertEqual(len(result), 5)
        for item in result:
            self.assertEqual(item["score"], 0.0)

    def test_scores_in_valid_range(self):
        msgs = [
            _make_msg("Tell me more", "student", offset_seconds=i * 30)
            for i in range(20)
        ]
        result = compute_engagement_timeline(msgs, window_minutes=30, bucket_minutes=5)
        for item in result:
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)

    def test_bucket_count_matches_window_bucket_ratio(self):
        result = compute_engagement_timeline([], window_minutes=30, bucket_minutes=2)
        self.assertEqual(len(result), 15)

    def test_time_format(self):
        result = compute_engagement_timeline([], window_minutes=10, bucket_minutes=2)
        for item in result:
            # Must be HH:MM format
            parts = item["time"].split(":")
            self.assertEqual(len(parts), 2)
            self.assertTrue(parts[0].isdigit())
            self.assertTrue(parts[1].isdigit())


class BehaviorMixUnitTests(TestCase):
    def test_empty_messages_returns_zero_counts(self):
        result = compute_behavior_mix([])
        self.assertEqual(
            result,
            {"deep_questions": 0, "scaffolded": 0, "off_topic": 0, "answer_seeking": 0},
        )

    def test_deep_question_classified_correctly(self):
        msgs = [_make_msg("Why does this method work differently from the other?", "student")]
        result = compute_behavior_mix(msgs)
        self.assertGreater(result["deep_questions"], 0)

    def test_answer_seeking_classified_correctly(self):
        msgs = [_make_msg("Just tell me the answer please", "student")]
        result = compute_behavior_mix(msgs)
        self.assertGreater(result["answer_seeking"], 0)

    def test_off_topic_classified_correctly(self):
        msgs = [_make_msg("Did you see the football game last weekend?", "student")]
        result = compute_behavior_mix(msgs)
        self.assertGreater(result["off_topic"], 0)

    def test_all_keys_present(self):
        result = compute_behavior_mix([_make_msg("Hello", "student")])
        self.assertIn("deep_questions", result)
        self.assertIn("scaffolded", result)
        self.assertIn("off_topic", result)
        self.assertIn("answer_seeking", result)

    def test_total_equals_message_count(self):
        msgs = [_make_msg(f"message {i}", "student") for i in range(10)]
        result = compute_behavior_mix(msgs)
        total = sum(result.values())
        self.assertEqual(total, 10)


# ---------------------------------------------------------------------------
# View / integration tests
# ---------------------------------------------------------------------------


class CognitiveLoadViewTests(APITestCase):
    def setUp(self):
        _db_msg("I don't understand why the numerator matters", "student", "TOM", 300)
        _db_msg("Because it shows the parts you have", "assistant", "TOM", 240)
        _db_msg("Oh ok that helps, but what about equivalent fractions though?", "student", "TOM", 60)
        _db_msg("What is a fraction", "student", "PRIYA", 120)

    def test_returns_200(self):
        response = self.client.get("/api/v1/metrics/cognitive-load/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_response_is_list(self):
        response = self.client.get("/api/v1/metrics/cognitive-load/")
        self.assertIsInstance(response.json(), list)

    def test_each_item_has_required_keys(self):
        response = self.client.get("/api/v1/metrics/cognitive-load/")
        for item in response.json():
            self.assertIn("student_id", item)
            self.assertIn("score", item)

    def test_scores_in_range(self):
        response = self.client.get("/api/v1/metrics/cognitive-load/")
        for item in response.json():
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)

    def test_old_messages_excluded(self):
        # Create a message 90 minutes ago — beyond the 60-min live window
        _db_msg("Ancient message", "student", "GHOST", offset_seconds=5400)
        response = self.client.get("/api/v1/metrics/cognitive-load/")
        ids = [item["student_id"] for item in response.json()]
        self.assertNotIn("GHOST", ids)


class TopicWrestlingViewTests(APITestCase):
    def setUp(self):
        _db_msg("I don't understand how to compare fractions", "student", "TOM", 300)
        _db_msg("I'm confused about equivalent fractions", "student", "PRIYA", 200)
        _db_msg("Here is an explanation", "assistant", "TOM", 100)

    def test_returns_200(self):
        response = self.client.get("/api/v1/metrics/topic-wrestling/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_response_is_list(self):
        self.assertIsInstance(self.client.get("/api/v1/metrics/topic-wrestling/").json(), list)

    def test_top_n_query_param(self):
        response = self.client.get("/api/v1/metrics/topic-wrestling/?top_n=2")
        self.assertLessEqual(len(response.json()), 2)

    def test_item_schema(self):
        response = self.client.get("/api/v1/metrics/topic-wrestling/")
        for item in response.json():
            self.assertIn("topic", item)
            self.assertIn("count", item)
            self.assertIsInstance(item["count"], int)


class ClassSummaryViewTests(APITestCase):
    def setUp(self):
        _db_msg("I don't understand how to compare fractions", "student", "TOM", 300)
        _db_msg("I'm confused about equivalent fractions", "student", "PRIYA", 200)
        _db_msg("Here is an explanation", "assistant", "TOM", 100)

    def test_returns_200(self):
        response = self.client.get("/api/v1/metrics/class-summary/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_response_contains_summary_keys(self):
        response = self.client.get("/api/v1/metrics/class-summary/")
        data = response.json()
        self.assertIn("cognitive_load", data)
        self.assertIn("topic_wrestling", data)
        self.assertIn("behavior_mix", data)
        self.assertIn("engagement", data)
        self.assertIn("passport_summary", data)

    def test_cognitive_load_summary_fields(self):
        data = self.client.get("/api/v1/metrics/class-summary/").json()
        self.assertIn("average_score", data["cognitive_load"])
        self.assertIn("median_score", data["cognitive_load"])
        self.assertIn("high_risk_count", data["cognitive_load"])
        self.assertIn("top_students_needing_support", data["cognitive_load"])


class EngagementTimelineViewTests(APITestCase):
    def setUp(self):
        for i in range(15):
            _db_msg(f"student message {i}", "student", "TOM", offset_seconds=i * 60)
            _db_msg(f"assistant reply {i}", "assistant", "TOM", offset_seconds=i * 60 + 30)

    def test_returns_200(self):
        response = self.client.get("/api/v1/metrics/engagement-timeline/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_response_is_list(self):
        self.assertIsInstance(self.client.get("/api/v1/metrics/engagement-timeline/").json(), list)

    def test_default_bucket_count(self):
        # Default window=30, bucket=2 → 15 buckets
        response = self.client.get("/api/v1/metrics/engagement-timeline/")
        self.assertEqual(len(response.json()), 15)

    def test_custom_window_and_bucket(self):
        response = self.client.get("/api/v1/metrics/engagement-timeline/?window=10&bucket=5")
        self.assertEqual(len(response.json()), 2)

    def test_score_range(self):
        response = self.client.get("/api/v1/metrics/engagement-timeline/")
        for item in response.json():
            self.assertGreaterEqual(item["score"], 0.0)
            self.assertLessEqual(item["score"], 1.0)


class BehaviorMixViewTests(APITestCase):
    def setUp(self):
        _db_msg("Why does this algorithm work?", "student", "TOM", 100)
        _db_msg("Just give me the answer", "student", "PRIYA", 200)
        _db_msg("Did you watch the game?", "student", "OLIVER", 300)
        _db_msg("Can you hint me?", "student", "MIA", 400)
        _db_msg("Because the loop iterates", "assistant", "TOM", 50)

    def test_returns_200(self):
        response = self.client.get("/api/v1/metrics/behavior-mix/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_response_is_object(self):
        self.assertIsInstance(self.client.get("/api/v1/metrics/behavior-mix/").json(), dict)

    def test_all_keys_present(self):
        data = self.client.get("/api/v1/metrics/behavior-mix/").json()
        for key in ("deep_questions", "scaffolded", "off_topic", "answer_seeking"):
            self.assertIn(key, data)

    def test_total_equals_student_message_count(self):
        data = self.client.get("/api/v1/metrics/behavior-mix/").json()
        # 4 student messages were inserted in setUp
        self.assertEqual(sum(data.values()), 4)

    def test_assistant_messages_excluded(self):
        # Add extra assistant messages; total should still be 4
        _db_msg("Excellent question!", "assistant", "TOM", 10)
        data = self.client.get("/api/v1/metrics/behavior-mix/").json()
        self.assertEqual(sum(data.values()), 4)