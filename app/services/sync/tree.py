"""Explicit decision tree for reference selection.

Replaces additive scoring heuristics with an auditable verdict:

``Exact Team Match`` > ``Source/Edition Tier`` > ``Abort``.

Edition candidates are ranked by a multi-tier hierarchy so a reference is still
found when the release group differs:

* Tier 0 — exact source (``webdl`` -> ``webdl``); BluRay and REMUX share the
  retail-disc master and count as one family;
* Tier 1 — cross-format (a ``webdl`` stream anchored by a BluRay/REMUX
  reference, or a shared explicit ``23.976`` fps fingerprint);
* Tier 2 — relaxed fallback for an unknown source on one side (relaxed mode).

DVD/CAM/screener references, mismatched episodes/seasons, and explicit cut
conflicts are always rejected. Resolution is cross-compatible within one source
family (1080p BluRay shares the retail-disc master with 720p BluRay). In strict
mode (default) an unknown source on one side aborts; relaxed mode treats it as a
wildcard while still rejecting direct conflicts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from app.services.sync.matching import (
    _codec_kind,
    _edition_tags,
    _release_group,
    _resolution,
    _season_number,
    _source_kind,
    _sources_compatible,
    candidate_episode_number,
    looks_like_season_pack,
    reference_group_rank,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime cycle
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

DecisionKind = Literal["team", "edition", "abort"]


@dataclass(frozen=True)
class ReferenceDecision:
    """The tree's verdict for one provider's candidate list."""

    kind: DecisionKind
    release: Any | None
    reason: str


def _release_name(release: Any) -> str:
    return str(getattr(release, "release_name", "") or "")


def _team_candidates(
    candidates: list[Any], target_group: str | None
) -> list[Any]:
    """Candidates carrying the target's exact release group (case-insensitive).

    The group may sit anywhere in the candidate name (``...-CHD`` or
    ``...CHD.BluRay...``); whole-token equality keeps the match exact while
    covering non-trailing placements.
    """
    if not target_group:
        return []
    wanted = target_group.lower()
    matches = []
    for rel in candidates:
        rel_name = _release_name(rel)
        if (_release_group(rel_name) or "").lower() == wanted:
            matches.append(rel)
            continue
        tokens = set(re.findall(r"[A-Za-z0-9]+", rel_name.lower()))
        if wanted in tokens:
            matches.append(rel)
    return matches


def _affinity_key(
    rel_name: str,
    target_source: str | None,
    target_res: str | None,
) -> tuple[int, int]:
    """Tie-break among team matches: same source, then same resolution."""
    source = _source_kind(rel_name)
    resolution = _resolution(rel_name)
    return (
        0 if (target_source and source == target_source) else 1,
        0 if (target_res and resolution == target_res) else 1,
    )


# Sources that can never anchor a reference: they are off-master (different
# framings/runtimes) or low quality, so no amount of source-tier climbing helps.
_INCOMPATIBLE_SOURCES = frozenset({"dvd", "cam"})
# Cross-format anchoring: a WEB-DL stream may be aligned to a retail-disc
# (BluRay/REMUX) reference — alass anchors the constant offset across editions.
_CROSS_FORMAT_TARGET_SOURCES = frozenset({"webdl"})
_CROSS_FORMAT_REFERENCE_SOURCES = frozenset({"bluray", "remux"})
_23976_REGEX = re.compile(r"(?i)(?<!\d)23[.,]976(?!\d)")
_SCREENER_REGEX = re.compile(r"(?i)\bscreener\b")


def _is_23976(name: str | None) -> bool:
    """True when a release name declares a 23.976 fps frame rate."""
    return bool(_23976_REGEX.search(name or ""))


def _is_screener(name: str | None) -> bool:
    return bool(_SCREENER_REGEX.search(name or ""))


def _cross_format_ok(
    target_name: str | None,
    target_source: str | None,
    candidate_name: str | None,
    candidate_source: str | None,
) -> bool:
    """Tier-3 compatibility: WEB-DL anchored by a retail disc, or shared 23.976 fps.

    alass resolves a *constant* offset, so a WEB-DL episode can be timed against
    a BluRay/REMUX master of the same content (and vice versa for the fps case).
    """
    if (
        target_source in _CROSS_FORMAT_TARGET_SOURCES
        and candidate_source in _CROSS_FORMAT_REFERENCE_SOURCES
    ):
        return True
    return _is_23976(target_name) and _is_23976(candidate_name)


def _edition_rank(rel_name: str, target_res: str | None) -> tuple[int, int]:
    """Order edition matches: exact resolution first, then UHD/remux carriers.

    A 2160p target must prefer 2160p/REMUX references over legacy releases
    that merely pass the edition gates (UHD remasters carry different intro
    bumpers than old BluRays). Stable sort keeps provider order otherwise.
    """
    resolution = _resolution(rel_name)
    exact = 0 if (target_res and resolution == target_res) else 1
    tokens = set(re.findall(r"[A-Za-z0-9]+", rel_name.lower()))
    uhd = (
        0
        if (resolution in ("2160p", "4k") or "uhd" in tokens or "remux" in tokens)
        else 1
    )
    return (exact, uhd)


def _edition_sort_key(
    rel_name: str,
    target_source: str | None,
    target_res: str | None,
    episode: int | None,
) -> tuple[int, int, int, int, int, int]:
    """Rank an edition candidate for the *best reference*, not just validity.

    Preference: known source matching the target (BluRay/REMUX family) over
    unknown; scene retail encode over micro-rip/repack (bumpers preserved, so
    the timeline matches the disc master); single episode over a
    whole-season/complete pack; explicitly tagged matching episode over
    untagged; exact resolution; then UHD/REMUX.
    """
    candidate_source = _source_kind(rel_name)
    if target_source and candidate_source:
        source_rank = 0 if _sources_compatible(target_source, candidate_source) else 2
    elif target_source:
        source_rank = 1  # unknown source: usable but less certain
    else:
        source_rank = 0
    group_rank = reference_group_rank(rel_name)
    pack = 1 if looks_like_season_pack(rel_name) else 0
    explicit_episode = (
        0
        if (episode is not None and not pack and candidate_episode_number(rel_name) == episode)
        else 1
    )
    exact_res, uhd = _edition_rank(rel_name, target_res)
    return (source_rank, group_rank, pack, explicit_episode, exact_res, uhd)


def _edition_tier(
    rel_name: str,
    *,
    target_name: str | None,
    target_source: str | None,
    target_codec: str | None,
    target_res: str | None,
    target_tags: frozenset[str],
    episode: int | None,
    strict: bool,
) -> int | None:
    """Edition-compatibility tier for a candidate, or ``None`` to reject it.

    Tier 0 = exact source (BluRay <-> REMUX share the retail-disc master);
    Tier 1 = cross-format (a WEB-DL target anchored by a retail-disc reference,
    or a shared 23.976 fps fingerprint); Tier 2 = relaxed fallback (unknown
    source on one side). Hard-incompatible sources (DVD/CAM/screener), wrong
    episodes, and explicit cut conflicts are rejected outright.
    """
    if episode is not None:
        # Season packs are treated as untagged (sliceable), so a loose "Season
        # 1" -> episode 1 mis-parse cannot exclude them from later episodes.
        candidate_episode = candidate_episode_number(rel_name)
        if candidate_episode is not None and candidate_episode != episode:
            return None
    if strict and _edition_tags(rel_name) != target_tags:
        # Different cuts (US vs International, theatrical vs Extended) have
        # different timings: any tag mismatch aborts in strict mode.
        return None
    if _is_screener(rel_name) or _is_screener(target_name):
        return None

    candidate_source = _source_kind(rel_name)
    if candidate_source in _INCOMPATIBLE_SOURCES or target_source in _INCOMPATIBLE_SOURCES:
        return None  # DVD/CAM never serve as a reference

    same_family = (
        bool(target_source)
        and bool(candidate_source)
        and _sources_compatible(target_source, candidate_source)
    )
    cross_format = (
        bool(target_source)
        and bool(candidate_source)
        and _cross_format_ok(target_name, target_source, rel_name, candidate_source)
    )
    compatible = same_family or cross_format
    if target_source and candidate_source:
        if not compatible:
            return None
        tier = 0 if same_family else 1
    elif target_source or candidate_source:
        if strict:
            return None  # fail closed on a one-sided unknown
        tier = 2
    else:
        tier = 2

    # Within a compatible pair a codec/resolution mismatch cannot contradict the
    # master (x264/x265 and 720p/1080p share subtitle timing). Explicit
    # conflicts still fail, and resolution must match outside the 1080p/720p pair.
    candidate_codec = _codec_kind(rel_name)
    if target_codec and candidate_codec and candidate_codec != target_codec:
        if not compatible:
            return None
    elif strict and (target_codec or candidate_codec) and not compatible:
        return None
    candidate_res = _resolution(rel_name)
    if target_res and candidate_res:
        if candidate_res != target_res and not (
            compatible and {target_res, candidate_res} <= {"1080p", "720p"}
        ):
            return None
    elif strict and (target_res or candidate_res) and not compatible:
        return None
    return tier


def decide(
    candidates: list[Any],
    query: ReferenceQuery,
    *,
    strict: bool = True,
    provider_name: str = "?",
) -> ReferenceDecision:
    """Filter by season/episode, then apply the team > edition > abort tree."""
    season = query.season
    episode = query.episode
    target_filename = query.target_filename

    pool = list(candidates)
    if season is not None:
        season_matched = [
            rel for rel in pool if _season_number(_release_name(rel)) == season
        ]
        if not season_matched:
            return ReferenceDecision(
                "abort", None, f"no candidate carries season {season}"
            )
        pool = season_matched

    if episode is not None:
        # Drop only explicitly mismatched episodes. Season-confirmed but
        # untagged candidates (retail season packs) stay in the pool: an
        # explicitly-tagged wrong-source single must not shadow them, since
        # _select_zip_member slices packs to the requested episode while the
        # edition check below rejects the wrong source.
        pool = [
            rel
            for rel in pool
            if candidate_episode_number(_release_name(rel)) in (None, episode)
        ]
        if not pool:
            return ReferenceDecision(
                "abort", None, f"no candidate carries S{season}E{episode}"
            )

    # Same-source priority: a candidate whose known source conflicts with a
    # known target source can never win (edition rejects the conflict, team
    # needs the group), so it sinks below same-family and unknown-source
    # candidates. Stable sort preserves provider order within each band.
    target_source = _source_kind(target_filename)
    if target_source:
        pool.sort(
            key=lambda rel: 0
            if _sources_compatible(target_source, _source_kind(_release_name(rel)))
            else 1
        )

    logger.info(
        "[reference] %s evaluated %d candidate(s) for %r:",
        provider_name,
        len(pool),
        target_filename,
    )
    for rel in pool[:15]:
        rel_name = _release_name(rel)
        logger.info(
            "[reference]   - %r source=%s season=%s group=%s",
            rel_name,
            _source_kind(rel_name),
            _season_number(rel_name),
            _release_group(rel_name),
        )

    target_group = _release_group(target_filename)
    target_tags = _edition_tags(target_filename)
    team = sorted(
        _team_candidates(pool, target_group),
        key=lambda rel: _affinity_key(
            _release_name(rel),
            _source_kind(target_filename),
            _resolution(target_filename),
        ),
    )
    # A shared group does not excuse a conflicting cut: an Extended stream
    # must never align against the same group's Theatrical release.
    vetoed = []
    kept = []
    for rel in team:
        if not _sources_compatible(target_source, _source_kind(_release_name(rel))):
            vetoed.append(rel)
            continue
        candidate_tags = _edition_tags(_release_name(rel))
        if candidate_tags == target_tags:
            kept.append(rel)
            continue
        if strict or (candidate_tags and target_tags):
            vetoed.append(rel)
            continue
        kept.append(rel)
    vetoed_ids = set()
    for rel in vetoed:
        vetoed_ids.add(id(rel))
        logger.info(
            "[reference] vetoing team candidate %r: source/cut conflict (%s vs %s)",
            _release_name(rel),
            sorted(_edition_tags(_release_name(rel))),
            sorted(target_tags),
        )
    # Explicit cut conflicts leave the pool in every mode (like source
    # conflicts): a vetoed release must not resurface via edition wildcards.
    pool = [rel for rel in pool if id(rel) not in vetoed_ids]
    if kept:
        best = kept[0]
        return ReferenceDecision(
            "team",
            best,
            f"exact release-group match ({target_group}) on {_release_name(best)!r}",
        )

    target_codec = _codec_kind(target_filename)
    target_res = _resolution(target_filename)
    if not any((target_group, target_source, target_codec, target_res)):
        if strict:
            return ReferenceDecision(
                "abort", None, "target carries no edition info to confirm against"
            )
        # Relaxed fallback: an uninformative target (e.g. an obfuscated debrid
        # hash) carries nothing to confirm against, so instead of aborting take
        # the best same-season/episode candidate and let the timeline gate and
        # shift guardrails reject a genuinely wrong reference downstream.
        # Explicit single-episode releases are preferred over whole-season /
        # "complete" packs (a raw pack cannot be sliced to one episode).
        pool.sort(
            key=lambda rel: _edition_sort_key(
                _release_name(rel), target_source, target_res, episode
            )
        )
        if not pool:
            return ReferenceDecision("abort", None, "no candidate to fall back to")
        best = pool[0]
        return ReferenceDecision(
            "edition",
            best,
            f"relaxed fallback for uninformative target on {_release_name(best)!r}",
        )
    edition: list[tuple[int, Any]] = []
    for rel in pool:
        tier = _edition_tier(
            _release_name(rel),
            target_name=target_filename,
            target_source=target_source,
            target_codec=target_codec,
            target_res=target_res,
            target_tags=target_tags,
            episode=episode,
            strict=strict,
        )
        if tier is not None:
            edition.append((tier, rel))
    # Best tier first (exact source > cross-format > relaxed), then the usual
    # quality tie-breaks within a tier.
    edition.sort(
        key=lambda item: (
            item[0],
            _edition_sort_key(_release_name(item[1]), target_source, target_res, episode),
        )
    )
    if edition:
        tier, best = edition[0]
        return ReferenceDecision(
            "edition",
            best,
            f"source/codec/edition match (tier {tier}) on {_release_name(best)!r}",
        )
    return ReferenceDecision(
        "abort", None, "no candidate proves the same edition as the target"
    )
