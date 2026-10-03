"""AutoSync is language-agnostic end to end.

The orchestrator used to reject any subtitle whose language did not start with
``ar``. That was a product-scope decision for an Arabic-only add-on, not a
technical requirement: alass aligns any two subtitle tracks, and a reference
routinely differs in language from the target it times.

These tests pin the resulting contract:

* any language with a known code enters the pipeline;
* only a *missing* language is skipped, because there is nothing to align;
* the reference may be in a different language from the target;
* the reference language never determines the output language.
"""

from __future__ import annotations

import pytest

from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference

_SPACING = (2870, 4310, 1990, 5380, 2410, 3620, 4790, 2180)


def _ts(ms: int) -> str:
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _cues(cues: int, offset_ms: int = 0) -> list[tuple[int, int]]:
    out = []
    cursor = 4000 + offset_ms
    for i in range(cues):
        step = _SPACING[i % len(_SPACING)]
        out.append((cursor, cursor + step - 400))
        cursor += step
    return out


def _srt(cues: int, offset_ms: int, label: str) -> str:
    blocks = []
    for i, (start, end) in enumerate(_cues(cues, offset_ms), 1):
        blocks.append(
            f"{i}\n{_ts(start)} --> {_ts(end)}\n"
            f"{label} line {i} carrying enough words to look like dialogue\n"
        )
    return "\n".join(blocks)


def _bytes(text: str) -> bytes:
    return text.encode("utf-8")


class CountingStrategy:
    """Records that it was consulted; returns a reference so alass would run."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.calls = 0

    name = "counting"
    validates_target = True

    async def resolve_with_provenance(self, query, *, update_validator=None):
        self.calls += 1
        if self.text is None:
            return ResolvedReference(None)
        return ResolvedReference(self.text, kind="hash")


class CapturingSync:
    """Stands in for SubtitleSyncService; records the argv roles."""

    def __init__(self, result: str | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result

    async def sync_async(self, target_srt, reference_srt, **kwargs):
        self.calls.append(
            {"target": target_srt, "reference": reference_srt, **kwargs}
        )
        return self._result


@pytest.fixture
def orch(monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)

    def build(strategy_text: str | None = None, sync: CapturingSync | None = None):
        strategy = CountingStrategy(strategy_text)
        service = sync or CapturingSync()
        o = SyncOrchestrator(
            hash_reference_strategy=strategy, sync_service=service
        )
        return o, strategy, service

    return build


@pytest.mark.parametrize("lang", ["ara", "eng", "spa", "fra", "deu", "zho", "und"])
@pytest.mark.asyncio
async def test_any_known_language_enters_the_pipeline(orch, lang):
    o, strategy, _ = orch()
    payload = _bytes(_srt(80, 0, "target"))
    await o.evaluate_and_sync(
        payload, {"imdb_id": "tt1", "lang": lang, "video_hash": "abc"}, "t", True
    )
    assert strategy.calls == 1, f"language {lang!r} was wrongly skipped"


@pytest.mark.asyncio
async def test_only_a_missing_language_is_skipped(orch):
    o, strategy, _ = orch()
    payload = _bytes(_srt(80, 0, "target"))
    for meta in ({"lang": ""}, {"lang": None}, {}):
        await o.evaluate_and_sync(payload, {"imdb_id": "tt1", **meta}, "t", True)
    assert strategy.calls == 0


@pytest.mark.asyncio
async def test_a_reference_in_another_language_still_reaches_alass(orch):
    """An English reference must time a Spanish target perfectly well."""
    reference = _srt(80, 0, "english reference")
    target = _bytes(_srt(80, 2337, "spanish target"))
    o, strategy, sync = orch(reference, CapturingSync(target.decode()))

    await o.evaluate_and_sync(
        target,
        {"imdb_id": "tt1", "lang": "spa", "video_hash": "abc"},
        "t",
        True,
    )

    assert strategy.calls == 1
    assert len(sync.calls) == 1
    call = sync.calls[0]
    # Roles are what matter: reference in slot 1, the user's own track in slot 2.
    assert "english reference" in call["reference"]
    assert "spanish target" in call["target"]
    assert call["reference"] != call["target"]


@pytest.mark.asyncio
async def test_the_reference_language_never_becomes_the_output(orch):
    """An English reference aligned to an Arabic target must not emit English."""
    reference = _srt(80, 0, "english reference")
    target = _bytes(_srt(80, 2337, "arabic target"))
    o, _strategy, sync = orch(reference, CapturingSync(target.decode()))

    out = await o.evaluate_and_sync(
        target,
        {"imdb_id": "tt1", "lang": "ara", "video_hash": "abc"},
        "t",
        True,
    )

    assert len(sync.calls) == 1
    # The ALASS target -- and therefore the output -- is the user's subtitle.
    assert sync.calls[0]["target"].startswith("1\n00:00:06,337")
    assert "arabic target line 1" in sync.calls[0]["target"]
    assert "english reference line 1" not in sync.calls[0]["target"]
    assert b"english reference" not in out


@pytest.mark.asyncio
async def test_server_and_user_gates_still_apply_for_any_language(orch):
    """Removing the language gate must not weaken the other two gates."""
    from app.config import settings

    o, strategy, _ = orch()
    payload = _bytes(_srt(80, 0, "target"))
    meta = {"imdb_id": "tt1", "lang": "eng", "video_hash": "abc"}

    settings.ENABLE_SUBTITLE_SYNC = False
    assert await o.evaluate_and_sync(payload, meta, "t", True) == payload
    assert strategy.calls == 0

    settings.ENABLE_SUBTITLE_SYNC = True
    assert await o.evaluate_and_sync(payload, meta, "t", False) == payload
    assert strategy.calls == 0
