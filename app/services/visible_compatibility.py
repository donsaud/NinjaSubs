"""Presentation helpers; compatibility and policy live exclusively in release_scoring."""
from typing import Any

from app.services.release_scoring import CONTRACT_VERSION

VISIBLE_COMPATIBILITY_VERSION = CONTRACT_VERSION


def value(item: Any, field: str, default: Any = None) -> Any:
    return item.get(field, default) if isinstance(item, dict) else getattr(item, field, default)


def exact_flag(item: Any) -> bool:
    return value(item, "is_hash_match") is True or value(item, "matched_by_hash") is True


def displayed_score(number: Any) -> int:
    try:
        return max(0, min(100, int(round(float(number)))))
    except (TypeError, ValueError, OverflowError):
        return 0


def presentation_key(item: Any) -> tuple:
    """Use the exact DisplayPolicy execution key; never URL/arrival order."""
    key = value(item, "weighted_key", ())
    if key:
        return tuple(key)
    from app.services.release_scoring.policies import DisplayPolicy
    from app.services.release_scoring_integration import matches
    results = matches([item], {})
    return DisplayPolicy((value(item, "lang", "ara"),)).key(results[0])
