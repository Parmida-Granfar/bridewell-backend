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


class TopStudentRiskSerializer(serializers.Serializer):
    student_id = serializers.CharField()
    score = serializers.FloatField(min_value=0.0, max_value=1.0)


class CognitiveLoadSummarySerializer(serializers.Serializer):
    average_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    median_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    high_risk_count = serializers.IntegerField(min_value=0)
    warning_risk_count = serializers.IntegerField(min_value=0)
    total_students = serializers.IntegerField(min_value=0)
    top_students_needing_support = TopStudentRiskSerializer(many=True)


class EngagementOverviewSerializer(serializers.Serializer):
    average_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    active_students = serializers.IntegerField(min_value=0)
    bucket_count = serializers.IntegerField(min_value=0)
    highest_bucket_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    lowest_bucket_score = serializers.FloatField(min_value=0.0, max_value=1.0)


class PassportSummarySerializer(serializers.Serializer):
    students_with_passports = serializers.IntegerField(min_value=0)
    total_students = serializers.IntegerField(min_value=0)
    common_access_arrangements = serializers.ListField(child=serializers.CharField())
    common_declared_needs = serializers.ListField(child=serializers.CharField())


class ClassSummarySerializer(serializers.Serializer):
    cognitive_load = CognitiveLoadSummarySerializer()
    topic_wrestling = TopicWrestlingItemSerializer(many=True)
    behavior_mix = BehaviorMixSerializer()
    engagement = EngagementOverviewSerializer()
    passport_summary = PassportSummarySerializer()


# -----------------------------
# New Serializers
# -----------------------------


class EvidenceSerializer(serializers.Serializer):
    source = serializers.CharField()
    message_id = serializers.CharField()
    quote = serializers.CharField()


class SignalsSerializer(serializers.Serializer):
    engagement = serializers.FloatField()
    confidence = serializers.FloatField()
    support_need = serializers.CharField()


class ChatSummarySerializer(serializers.Serializer):
    student_id = serializers.CharField()
    generated_at = serializers.DateTimeField()
    summary = serializers.CharField()
    highlights = serializers.ListField(child=serializers.CharField())
    signals = SignalsSerializer()
    evidence = EvidenceSerializer(many=True)
    recommended_next_step = serializers.CharField()


class LearningBalanceSerializer(serializers.Serializer):
    a_receives = serializers.ListField(child=serializers.CharField())
    b_receives = serializers.ListField(child=serializers.CharField())


class PairSerializer(serializers.Serializer):
    student_a_id = serializers.CharField()
    student_b_id = serializers.CharField()
    match_score = serializers.FloatField()
    reason = serializers.CharField()
    learning_balance = LearningBalanceSerializer()
    task_prompt = serializers.CharField()


class SuggestedPairUpsSerializer(serializers.Serializer):
    generated_at = serializers.DateTimeField()
    focus_area = serializers.CharField()
    pairs = PairSerializer(many=True)


class SummaryBlockSerializer(serializers.Serializer):
    preferred_mode = serializers.CharField()
    strengths = serializers.ListField(child=serializers.CharField())
    growth_areas = serializers.ListField(child=serializers.CharField())
    language_adaptation = serializers.CharField()
    working_style = serializers.ListField(child=serializers.CharField())
    support_needs = serializers.ListField(child=serializers.CharField())


class PassportSignalsSerializer(serializers.Serializer):
    access_arrangements = serializers.ListField(child=serializers.CharField())
    declared_needs = serializers.ListField(child=serializers.CharField())


class ChatSignalsSerializer(serializers.Serializer):
    recent_interests = serializers.ListField(child=serializers.CharField())
    engagement_style = serializers.CharField()


class LearningPreferencesSerializer(serializers.Serializer):
    student_id = serializers.CharField()
    summary = SummaryBlockSerializer()
    passport_signals = PassportSignalsSerializer()
    chat_signals = ChatSignalsSerializer()