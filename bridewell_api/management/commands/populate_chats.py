"""
Populate ChatMessage table with dummy data.
Run: python manage.py populate_chats
"""

from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand

from bridewell_api.models import ChatMessage


class Command(BaseCommand):
    help = "Populate ChatMessage table with dummy data"

    def handle(self, *args, **options):
        # Clear existing messages
        ChatMessage.objects.all().delete()
        
        now = datetime.now(tz=timezone.utc)

        messages = [
            {"text": "I don't understand why fractions work this way", "sender_type": "student", "student_id": "TOM"},
            {"text": "Because the denominator shows how many equal parts the whole is divided into", "sender_type": "assistant", "student_id": "TOM"},
            {"text": "Oh I get it now! But what about equivalent fractions?", "sender_type": "student", "student_id": "TOM"},
            {"text": "I'm confused about how to compare fractions", "sender_type": "student", "student_id": "PRIYA"},
            {"text": "Can you help me with division please?", "sender_type": "student", "student_id": "ZOE"},
            {"text": "Why does this algorithm work differently?", "sender_type": "student", "student_id": "AMY"},
            {"text": "Just tell me the answer please", "sender_type": "student", "student_id": "OLIVER"},
            {"text": "Did you see the football game last weekend?", "sender_type": "student", "student_id": "MIA"},
            {"text": "Here's a hint: think of fractions as parts of a pizza", "sender_type": "assistant", "student_id": "ZOE"},
            {"text": "Thanks! That helps a lot", "sender_type": "student", "student_id": "ZOE"},
        ]

        for i, msg in enumerate(messages):
            ChatMessage.objects.create(
                text=msg["text"],
                sender_type=msg["sender_type"],
                student_id=msg["student_id"],
                timestamp=now - timedelta(minutes=i*5)
            )

        self.stdout.write(self.style.SUCCESS(f"Created {len(messages)} dummy messages"))