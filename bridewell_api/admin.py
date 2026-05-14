"""
Admin configuration for bridewell_api.
"""

from django.contrib import admin

from .models import ChatMessage


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'sender_type', 'timestamp', 'text_preview')
    list_filter = ('sender_type', 'year_group', 'timestamp')
    search_fields = ('student_id', 'text', 'session_id')
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)

    @staticmethod
    def text_preview(obj: ChatMessage) -> str:
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text