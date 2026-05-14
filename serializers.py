"""
serializers.py — DRF serializers for each metrics endpoint.

These are *output* serializers: they validate the shape of the data
returned by nlp_utils before it leaves the API.
"""

from rest_framework import serializers


class CognitiveLoadItemSerializer(serializers.Serializer):
    """Single student cognitive-load entry."""

    student_id = serializers.CharField()
    score = serializers.FloatField(min_value=0.0, max_value=1.0)


class TopicWrestlingItemSerializer(serializers.Serializer):
    """A topic students are wrestling with and its mention count."""

    topic = serializers.CharField()
    count = serializers.IntegerField(min_value=0)


class EngagementTimelineItemSerializer(serializers.Serializer):
    """Single time-bucket engagement data point."""

    time = serializers.CharField(help_text="HH:MM in the server's local time")
    score = serializers.FloatField(min_value=0.0, max_value=1.0)


class BehaviorMixSerializer(serializers.Serializer):
    """Distribution of student behaviour types across all messages."""

    deep_questions = serializers.IntegerField(min_value=0)
    scaffolded = serializers.IntegerField(min_value=0)
    off_topic = serializers.IntegerField(min_value=0)
    answer_seeking = serializers.IntegerField(min_value=0)
