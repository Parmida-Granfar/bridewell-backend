"""
models.py — ChatMessage model for Bridewell AI.
"""

from django.db import models


class ChatMessage(models.Model):
    """
    Represents a single message exchanged between a student and the AI
    assistant (backed by GPT-4o-mini or similar).
    """

    class SenderType(models.TextChoices):
        STUDENT = "student", "Student"
        ASSISTANT = "assistant", "Assistant"

    class ActionType(models.TextChoices):
        CHAT = "chat", "Chat"
        REPHRASE = "rephrase", "Rephrase"
        SIMPLIFY = "simplify", "Simplify"
        HINT = "hint", "Hint"

    # Core fields
    text = models.TextField()
    sender_type = models.CharField(
        max_length=16,
        choices=SenderType.choices,
        db_index=True,
    )
    student_id = models.CharField(
        max_length=64,
        db_index=True,
        help_text="Identifies the student in the session. "
                  "May be anonymised (e.g. 'TOM', 'PRIYA').",
    )
    timestamp = models.DateTimeField(db_index=True)

    # Optional metadata
    session_id = models.CharField(max_length=64, blank=True, db_index=True)
    year_group = models.CharField(max_length=16, blank=True)
    source = models.CharField(
        max_length=64,
        blank=True,
        help_text="Optional origin of the message, e.g. 'studentlogs'.",
        db_index=True,
    )
    source_id = models.CharField(
        max_length=128,
        blank=True,
        help_text="Optional original message ID from the source log.",
    )
    action_type = models.CharField(
        max_length=16,
        choices=ActionType.choices,
        default="chat",
        db_index=True,
        help_text="Type of interaction: chat, rephrase, simplify, hint.",
    )

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["timestamp", "sender_type"]),
            models.Index(fields=["student_id", "timestamp"]),
        ]

    def __str__(self) -> str:
        return (
            f"[{self.timestamp:%Y-%m-%d %H:%M}] "
            f"{self.student_id} ({self.sender_type}): {self.text[:60]}"
        )


class StudentPassport(models.Model):
    """Structured student passport data extracted from support documentation."""

    student_id = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Student identifier used to join passport data with chat logs.",
    )
    access_arrangements = models.JSONField(default=list, blank=True)
    declared_needs = models.JSONField(default=list, blank=True)
    preferred_mode = models.CharField(max_length=128, blank=True)
    support_needs = models.JSONField(default=list, blank=True)
    raw_text = models.TextField(blank=True)
    source_file = models.CharField(max_length=256, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["student_id"])]

    def __str__(self) -> str:
        return f"Passport for {self.student_id}"
