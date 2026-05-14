"""Passport document parsing utilities."""

import re
from pathlib import Path
from typing import Any

KNOWN_FIELD_MAP = {
    "student id": "student_id",
    "pupil id": "student_id",
    "name": "student_id",
    "student": "student_id",
    "preferred mode": "preferred_mode",
    "preferred_mode": "preferred_mode",
    "access arrangements": "access_arrangements",
    "access_arrangement": "access_arrangements",
    "access arrangement": "access_arrangements",
    "declared needs": "declared_needs",
    "needs": "declared_needs",
    "support needs": "support_needs",
    "support need": "support_needs",
    "learning support": "support_needs",
    "support": "support_needs",
}

LIST_FIELDS = {"access_arrangements", "declared_needs", "support_needs"}

SPLIT_PATTERN = re.compile(r"[,;•\n\r]+")


def parse_passport_docx(docx_path: str | Path) -> dict[str, Any]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required to parse passport documents. "
            "Install it with `pip install python-docx`."
        ) from exc

    path = Path(docx_path)
    document = Document(path)
    lines = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    parsed = _extract_sections(lines)
    normalized = _normalize_parsed_passport(parsed)
    normalized["raw_text"] = "\n".join(lines)
    normalized["source_file"] = str(path)
    return normalized


def _extract_sections(lines: list[str]) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    current_key: str | None = None

    for line in lines:
        split_match = re.split(r"\s*:\s*", line, maxsplit=1)
        if len(split_match) == 2:
            raw_key, raw_value = split_match
            key = _normalize_key(raw_key)
            if key:
                value = raw_value.strip()
                if key in LIST_FIELDS:
                    sections[key] = _split_list(value)
                else:
                    sections[key] = value
                current_key = key if not value and key in LIST_FIELDS else None
                continue

        key = _normalize_key(line)
        if key in LIST_FIELDS:
            sections.setdefault(key, [])
            current_key = key
            continue

        if current_key and isinstance(sections.get(current_key), list):
            sections[current_key].extend(_split_list(line))
        else:
            current_key = None

    return sections


def _normalize_key(raw_key: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", raw_key.strip().lower()).strip()
    return KNOWN_FIELD_MAP.get(normalized)


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    items = [item.strip() for item in SPLIT_PATTERN.split(value) if item.strip()]
    return items


def _normalize_parsed_passport(parsed: dict[str, Any]) -> dict[str, Any]:
    student_id = parsed.get("student_id")
    if not student_id:
        student_id = parsed.get("name") or parsed.get("pupil") or ""

    return {
        "student_id": student_id,
        "access_arrangements": parsed.get("access_arrangements", []),
        "declared_needs": parsed.get("declared_needs", []),
        "preferred_mode": parsed.get("preferred_mode", ""),
        "support_needs": parsed.get("support_needs", []),
    }
