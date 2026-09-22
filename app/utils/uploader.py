"""Robust extraction of subtitle uploader/author usernames from provider payloads.

Different providers expose the uploader under different shapes:
- SubDL:      ``author`` / ``uploader`` (string)
- SubSource:  ``contributors``: [{ "id", "displayname" }] (only ``uploaderId`` otherwise)
- OpenSubtitles: ``uploader``: { "name", ... } (or legacy ``author``)

This helper normalizes all of them into a clean display name and deliberately
ignores bare numeric ids (e.g. ``uploaderId``) since those are not usernames.
"""

from typing import Any

# Direct string / name-bearing keys (checked in priority order).
_UPLOADER_STRING_KEYS: tuple[str, ...] = (
    "author",
    "uploader",
    "username",
    "user_name",
    "userName",
    "uploader_name",
    "uploaderName",
    "displayname",
    "display_name",
    "displayName",
    "nickname",
    "nickName",
    "added_by",
    "addedBy",
    "uploaded_by",
    "uploadedBy",
    "submitted_by",
    "submittedBy",
    "created_by",
    "createdBy",
    "contributor",
    "owner",
)

# Lists of contributor/user objects (e.g. SubSource ``contributors``).
_CONTRIBUTOR_LIST_KEYS: tuple[str, ...] = (
    "contributors",
    "contributor",
    "uploaders",
    "authors",
    "users",
)

# Nested uploader objects.
_NESTED_UPLOADER_KEYS: tuple[str, ...] = ("user", "uploader", "author", "profile", "account")

# Keys within a nested object that carry the human-readable name.
_NAME_KEYS: tuple[str, ...] = (
    "displayname",
    "display_name",
    "displayName",
    "username",
    "user_name",
    "userName",
    "nickname",
    "nickName",
    "name",
)


def _coerce_uploader_value(value: Any, depth: int = 0) -> str:
    """Coerce a value of unknown shape into a clean, non-numeric uploader name."""
    if value is None or depth > 3:
        return ""

    if isinstance(value, str):
        cleaned = value.strip()
        # Bare numeric ids (e.g. uploaderId) are not usernames.
        return "" if not cleaned or cleaned.isdigit() else cleaned

    if isinstance(value, dict):
        for key in _NAME_KEYS:
            if key in value:
                name = _coerce_uploader_value(value[key], depth + 1)
                if name:
                    return name
        for key in _NESTED_UPLOADER_KEYS:
            if key in value:
                name = _coerce_uploader_value(value[key], depth + 1)
                if name:
                    return name
        return ""

    if isinstance(value, list | tuple | set):
        for entry in value:
            name = _coerce_uploader_value(entry, depth + 1)
            if name:
                return name

    return ""


def extract_uploader(item: Any) -> str:
    """
    Extract a subtitle uploader/author username from a provider item mapping.

    Returns an empty string when no human-readable name is available (e.g. when
    only a numeric ``uploaderId`` is present).
    """
    if not isinstance(item, dict):
        return ""

    # 1. Contributor/user object lists (e.g. SubSource ``contributors``).
    for key in _CONTRIBUTOR_LIST_KEYS:
        if key in item:
            name = _coerce_uploader_value(item[key])
            if name:
                return name

    # 2. Direct name-bearing keys.
    for key in _UPLOADER_STRING_KEYS:
        if key in item:
            name = _coerce_uploader_value(item[key])
            if name:
                return name

    # 3. Nested uploader objects.
    for key in _NESTED_UPLOADER_KEYS:
        if key in item:
            name = _coerce_uploader_value(item[key])
            if name:
                return name

    return ""
