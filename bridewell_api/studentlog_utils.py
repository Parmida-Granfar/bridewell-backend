"""Utilities for importing normalized student log datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_timestamp(timestamp: Any) -> datetime:
    if timestamp is None:
        raise ValueError("Missing timestamp in student log message")
    ts = int(timestamp)
    if ts > 10**12:
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _normalize_role(role: str) -> str:
    return "student" if role.lower() == "user" else "assistant"


def _normalize_message(
    message: dict[str, Any],
    conversation_id: str,
    student_id: str,
) -> dict[str, Any]:
    return {
        "text": message.get("content", message.get("text", "")),
        "sender_type": _normalize_role(message.get("role", "user")),
        "student_id": student_id,
        "timestamp": _normalize_timestamp(message.get("timestamp") or message.get("created_at")),
        "session_id": conversation_id,
        "source": "studentlogs",
        "source_id": message.get("message_id", ""),
    }


def parse_studentlog_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        candidates = list(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON file found in student log directory: {path}")
        path = candidates[0]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "conversations" in payload:
        return _parse_conversation_dataset(payload["conversations"])

    if "message_index" in payload:
        return _parse_flat_message_index(payload["message_index"])

    raise ValueError("Unrecognized student log dataset structure")


def _parse_conversation_dataset(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for conversation in conversations:
        conversation_id = conversation.get("conversation_id") or ""
        student_id = conversation.get("user_id") or conversation.get("student_id") or "unknown"
        for message in conversation.get("messages", []):
            normalized.append(_normalize_message(message, conversation_id, student_id))
    return normalized


def _parse_flat_message_index(message_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in message_index:
        student_id = message.get("user_id") or message.get("student_id") or message.get("conversation_id") or "unknown"
        conversation_id = message.get("conversation_id", "")
        normalized.append(_normalize_message(message, conversation_id, student_id))
    return normalized
