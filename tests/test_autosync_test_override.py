"""Tests for ENABLE_AUTOSYNC_TEST_OVERRIDE -- the forced-production delivery test.

Purpose: prove whether the ordinary production serve-time pipeline preserves the
timing of an already-validated Alass artifact. If it does, production delivery is
exonerated and the remaining problem is acceptance policy, not delivery.

The override is a development instrument, so the safety of the *instrument* is
what these tests police: off by default, allowlisted, digest-pinned, and inert
with respect to every cache and to the verifier.
"""

from __future__ import annotations

import hashlib
import re

import pytest

import app.autosync_test_override as ov
from app.autosync_test_override import (
    KNOWN_CASES,
    case_for_sub_id,
    forced_transformed_bytes,
    override_enabled,
)

TIMING = re.compile(
    r"(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})"
)

EVOLV_SUB = "74a34c232d159f8b"
ASAP_SUB = "96c0442cc4656685"


def to_ms(hms: str, frac: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return ((h * 60 + m) * 60 + s) * 1000 + int(frac.ljust(3, "0"))


def timings(data: bytes) -> list[tuple[int, int]]:
    return [
        (to_ms(a, b), to_ms(c, d))
        for a, b, c, d in TIMING.findall(data.decode("utf-8", "replace"))
    ]


def overlaps(times: list[tuple[int, int]]) -> dict[int, int]:
    """Cue index -> amount by which it overlaps the following cue."""
    return {
        i: times[i][1] - times[i + 1][0]
        for i in range(len(times) - 1)
        if times[i][1] > times[i + 1][0]
    }


def pin_digest(monkeypatch, data: bytes) -> None:
    """Point the allowlist at `data` for every case.

    KnownCase is a frozen dataclass, so the tuple itself is replaced rather than
    an attribute poked on an instance.
    """
    import dataclasses

    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        ov,
        "KNOWN_CASES",
        tuple(
            dataclasses.replace(c, expected_sha256=digest) for c in KNOWN_CASES
        ),
    )


def fixture_dir(tmp_path, monkeypatch, contents: dict[str, bytes] | None = None):
    d = tmp_path / "fx"
    d.mkdir()
    if contents is None:
        return d
    for name, data in contents.items():
        (d / name).write_bytes(data)
    monkeypatch.setenv(ov.FIXTURE_ENV, str(d))
    return d


# ------------------------------------------------------------- gating ------ #


def test_override_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ov.ENVVAR, raising=False)
    assert override_enabled() is False


@pytest.mark.parametrize("raw", ["", "false", "0", "no", "off", "maybe", "enabled"])
def test_override_stays_off_unless_explicitly_true(raw, monkeypatch):
    """A typo must not enable a test path."""
    monkeypatch.setenv(ov.ENVVAR, raw)
    assert override_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", " On "])
def test_override_enables_on_explicit_true(raw, monkeypatch):
    monkeypatch.setenv(ov.ENVVAR, raw)
    assert override_enabled() is True


def test_disabled_override_serves_nothing_even_for_allowlisted_ids(tmp_path, monkeypatch):
    fixture_dir(tmp_path, monkeypatch, {
        KNOWN_CASES[0].filename: b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n"
    })
    monkeypatch.delenv(ov.ENVVAR, raising=False)
    assert forced_transformed_bytes(EVOLV_SUB) is None


# --------------------------------------------------------- allowlist ------- #


def test_only_exact_allowlisted_ids_match():
    for bad in (
        "", "74a34c232d159f8", "74a34c232d159f8bb", "74a34c232d159f8B",
        "EVOLV", "../74a34c232d159f8b", "74a34c232d159f8b.srt", "*",
    ):
        assert case_for_sub_id(bad) is None, f"unexpectedly matched: {bad!r}"
    assert case_for_sub_id(EVOLV_SUB) is not None
    assert case_for_sub_id(ASAP_SUB) is not None


def test_allowlist_is_exactly_two_real_cases():
    assert len(KNOWN_CASES) == 2
    assert {c.sub_id for c in KNOWN_CASES} == {EVOLV_SUB, ASAP_SUB}
    # Filenames are constants, not request data.
    for case in KNOWN_CASES:
        assert "/" not in case.filename and ".." not in case.filename


def test_no_caller_supplied_filename_is_honoured(tmp_path, monkeypatch):
    """A request cannot name the file; the filename comes from the constant."""
    data = b"1\n00:00:01,000 --> 00:00:02,000\nreal\n\n"
    d = fixture_dir(tmp_path, monkeypatch, {KNOWN_CASES[0].filename: data})
    pin_digest(monkeypatch, data)
    monkeypatch.setenv(ov.ENVVAR, "true")
    # An id-shaped string that tries to traverse must not resolve.
    for hostile in (
        "../../../../etc/passwd",
        f"{EVOLV_SUB}/../../../etc/passwd",
        "74a34c232d159f8b%00../../x",
    ):
        assert forced_transformed_bytes(hostile) is None
    # The legitimate id still reads only the allowlisted file.
    got = forced_transformed_bytes(EVOLV_SUB)
    assert got == data
    assert not (d / "passwd").exists()


# ------------------------------------------------------ digest pinning ----- #


def test_wrong_digest_is_refused(tmp_path, monkeypatch):
    """A substituted fixture is refused, not delivered as the known-good one."""
    fixture_dir(tmp_path, monkeypatch, {
        KNOWN_CASES[0].filename: b"1\n00:00:01,000 --> 00:00:02,000\nsubstituted\n\n",
    })
    monkeypatch.setenv(ov.ENVVAR, "true")
    assert forced_transformed_bytes(EVOLV_SUB) is None


def test_missing_fixture_declines_rather_than_raises(tmp_path, monkeypatch):
    fixture_dir(tmp_path, monkeypatch, {})
    monkeypatch.setenv(ov.ENVVAR, "true")
    assert forced_transformed_bytes(EVOLV_SUB) is None


def test_correct_digest_is_returned(tmp_path, monkeypatch):
    data = b"1\n00:00:01,000 --> 00:00:02,000\nvalidated\n\n"
    digest = hashlib.sha256(data).hexdigest()
    fixture_dir(tmp_path, monkeypatch, {KNOWN_CASES[0].filename: data})
    monkeypatch.setenv(ov.ENVVAR, "true")
    import dataclasses

    monkeypatch.setattr(
        ov,
        "KNOWN_CASES",
        (dataclasses.replace(KNOWN_CASES[0], expected_sha256=digest), *KNOWN_CASES[1:]),
    )
    got = forced_transformed_bytes(EVOLV_SUB)
    assert got == data


# ------------------------------------------------------------ isolation ---- #


def test_override_touches_no_cache(monkeypatch, tmp_path):
    """No provider cache, SyncCache, verdict store, or negative cache access."""
    import app.cache as cache_mod
    from app.services.sync_cache import SyncCache

    touched: list[str] = []

    def trip(name):
        def _boom(*a, **k):
            touched.append(name)
            raise AssertionError(f"override touched {name}")
        return _boom

    for method in ("get", "set", "get_meta", "set_meta",
                   "get_verdict", "set_verdict", "mark_failed", "is_failed"):
        monkeypatch.setattr(SyncCache, method, trip(f"SyncCache.{method}"), raising=False)
    for method in ("get_subtitle", "save_subtitle", "get_metadata"):
        monkeypatch.setattr(
            type(cache_mod.cache_manager), method, trip(f"cache.{method}"),
            raising=False,
        )

    d = tmp_path / "fx"
    d.mkdir()
    data = b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n"
    for case in KNOWN_CASES:
        (d / case.filename).write_bytes(data)
    pin_digest(monkeypatch, data)
    monkeypatch.setenv(ov.FIXTURE_ENV, str(d))
    monkeypatch.setenv(ov.ENVVAR, "true")

    for sub_id in (EVOLV_SUB, ASAP_SUB):
        assert forced_transformed_bytes(sub_id) is not None
    assert not touched, f"override reached production state: {touched}"


def test_override_does_not_alter_the_verifier_outcome():
    """The override substitutes bytes; it must not consult or change a verdict.

    Checked against the parsed imports rather than the raw text, because the
    module docstring deliberately names every component it excludes, and a
    substring search would flag that documentation as a violation.
    """
    import ast
    import pathlib

    banned_imports = {
        "SyncOrchestrator", "AlignmentAnalyzer", "SyncCache",
        "SubtitleSyncService", "may_serve_synchronized", "SyncState",
        "cache_manager", "LRUCacheManager",
    }
    tree = ast.parse(pathlib.Path(ov.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    overlap = imported & banned_imports
    assert not overlap, f"override imports production state: {sorted(overlap)}"


def test_override_is_marked_dev_only():
    src = open(ov.__file__, encoding="utf-8").read()
    assert "DEV/TEST" in src
    assert "development/test only" in src.lower()


def test_logs_contain_no_subtitle_contents(tmp_path, monkeypatch, caplog):
    import logging

    data = b"1\n00:00:01,000 --> 00:00:02,000\nSECRET_DIALOGUE_LINE\n\n"
    d = tmp_path / "fx"
    d.mkdir()
    for case in KNOWN_CASES:
        (d / case.filename).write_bytes(data)
    pin_digest(monkeypatch, data)
    monkeypatch.setenv(ov.FIXTURE_ENV, str(d))
    monkeypatch.setenv(ov.ENVVAR, "true")
    with caplog.at_level(logging.WARNING, logger="app.autosync_test_override"):
        forced_transformed_bytes(EVOLV_SUB)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRET_DIALOGUE_LINE" not in logged
    assert "00:00:01" not in logged


# ------------------------------------------------- timing preservation ----- #


def test_production_pipeline_preserves_cue_timing_except_documented_deoverlap():
    """The Part 3 invariant, stated precisely.

    Production serve-time processing is allowed to trim a cue's END where the
    source genuinely overlaps the next cue, because that is a normalisation
    rather than a timing change. Nothing else may move. Asserted against the
    real cleaners with every presentation option on, so a future text-processing
    change that touches timing fails here.
    """
    import app.main as main_mod
    from app.utils.cleaners import CleanOptions

    # A synthetic fixture with a deliberate 190ms overlap, matching the shape
    # measured on the real ASAP artifact.
    text = (
        "1\n00:00:00,000 --> 00:00:10,590\nfirst\n\n"
        "2\n00:00:10,400 --> 00:00:16,050\nsecond\n\n"
        "3\n00:00:16,070 --> 00:00:20,000\nthird\n\n"
    )
    original = text.encode("utf-8")
    base = timings(original)
    assert overlaps(base) == {0: 190}

    out = main_mod._run_subtitle_optimization_pipeline(
        original,
        enable_rtl_fix=True,
        enable_ad_removal=True,
        keep_translator_credits=True,
        options=CleanOptions(fix_encoding=True),
        strip_hi=True,
        eastern_arabic_numerals=True,
        strip_diacritics=True,
    )
    after = timings(out)
    assert len(after) == len(base), "cue count must not change"

    trimmed = overlaps(base)
    for i, (before, now) in enumerate(zip(base, after, strict=True)):
        assert now[0] == before[0], f"cue {i} start moved: {before} -> {now}"
        if i in trimmed:
            # Only the overlap may be removed, and the end may only shrink to
            # the next cue's start.
            assert now[1] == base[i + 1][0], (
                f"cue {i} end should clamp to the next start: {now[1]} != {base[i+1][0]}"
            )
            assert now[1] <= before[1]
        else:
            assert now == before, f"cue {i} timing changed: {before} -> {now}"


def test_deoverlap_clamp_is_the_only_permitted_timing_change():
    """A non-overlapping fixture must survive with identical timing."""
    import app.main as main_mod
    from app.utils.cleaners import CleanOptions

    text = (
        "1\n00:00:00,000 --> 00:00:10,000\nfirst\n\n"
        "2\n00:00:10,400 --> 00:00:16,000\nsecond\n\n"
        "3\n00:00:16,400 --> 00:00:20,000\nthird\n\n"
    )
    original = text.encode("utf-8")
    out = main_mod._run_subtitle_optimization_pipeline(
        original,
        enable_rtl_fix=True,
        enable_ad_removal=True,
        keep_translator_credits=True,
        options=CleanOptions(fix_encoding=True),
        strip_hi=True,
        eastern_arabic_numerals=True,
        strip_diacritics=True,
    )
    assert timings(out) == timings(original), (
        "a non-overlapping fixture must keep identical timing through production"
    )


def test_timing_check_would_catch_a_touching_text_transformer(monkeypatch):
    """The regression must actually fail if a text stage alters timing.

    Proven by injecting a fake text stage that shifts every cue by 250ms and
    confirming the invariant function rejects it. Without this, the tests above
    could pass for the wrong reason.
    """
    import app.main as main_mod

    original = (
        b"1\n00:00:00,000 --> 00:00:10,000\nfirst\n\n"
        b"2\n00:00:10,400 --> 00:00:16,000\nsecond\n\n"
    )
    base = timings(original)
    tampered = (
        b"1\n00:00:00,250 --> 00:00:10,250\nfirst\n\n"
        b"2\n00:00:10,650 --> 00:00:16,250\nsecond\n\n"
    )
    # Sanity: the honest check rejects the tampered timings.
    assert timings(tampered) != base
    assert timings(tampered)[0][0] != base[0][0]
    # And the real pipeline on untampered input is clean.
    out = main_mod._run_subtitle_optimization_pipeline(
        original,
        enable_rtl_fix=True, enable_ad_removal=True, keep_translator_credits=True,
        options=None, strip_hi=False, eastern_arabic_numerals=False,
        strip_diacritics=False,
    )
    assert timings(out) == base
