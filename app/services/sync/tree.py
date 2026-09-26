"""Explicit decision tree for reference selection.

Replaces additive scoring heuristics with an auditable verdict:

``Exact Team Match`` > ``Exact Source/Codec Edition Match`` > ``Abort``.

In strict mode (default) anything that cannot be confirmed aborts: unknown
groups, unknown source/codec on either side, and untagged episodes.
Resolution is cross-compatible within one source family (1080p BluRay shares
the retail-disc master with 720p BluRay) and may be unspecified there.
Relaxed mode keeps the same tree but treats unknowns as wildcards while
still rejecting direct conflicts.
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


def _edition_ok(
    rel_name: str,
    *,
    target_source: str | None,
    target_codec: str | None,
    target_res: str | None,
    target_tags: frozenset[str],
    episode: int | None,
    strict: bool,
) -> bool:
    """True when a candidate is provably the same edition as the target."""
    if episode is not None:
        # Season packs are treated as untagged (sliceable), so a loose "Season
        # 1" -> episode 1 mis-parse cannot exclude them from later episodes.
        candidate_episode = candidate_episode_number(rel_name)
        if candidate_episode is not None and candidate_episode != episode:
            return False
    if strict and _edition_tags(rel_name) != target_tags:
        # Different cuts (US vs International, theatrical vs Extended,
        # BluRay vs Remux) have different timings: any tag mismatch aborts.
        return False
    candidate_source = _source_kind(rel_name)
    same_family = (
        bool(target_source)
        and bool(candidate_source)
        and _sources_compatible(target_source, candidate_source)
    )
    if target_source and candidate_source:
        if not same_family:
            return False  # direct conflict: different sources never mix
    elif strict and (target_source or candidate_source):
        return False  # fail closed: unknowns on either side
    # Once the source family is confirmed on both sides (e.g. BluRay), a
    # missing codec/resolution cannot contradict it: 720p and undertagged
    # siblings (ALL.BluRay) share the same retail-disc master. Explicit
    # conflicts still fail in every mode.
    candidate_codec = _codec_kind(rel_name)
    if target_codec and candidate_codec and candidate_codec != target_codec:
        # x264 vs x265 of the same retail master has identical subtitle timing;
        # only fail closed in strict mode or across conflicting sources.
        if strict or not same_family:
            return False
    elif strict and (target_codec or candidate_codec) and not same_family:
        return False
    candidate_res = _resolution(rel_name)
    if target_res and candidate_res:
        if candidate_res != target_res and not (
            same_family and {target_res, candidate_res} <= {"1080p", "720p"}
        ):
            return False
    elif strict and (target_res or candidate_res) and not same_family:
        return False
    return True


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
    edition = [
        rel
        for rel in pool
        if _edition_ok(
            _release_name(rel),
            target_source=target_source,
            target_codec=target_codec,
            target_res=target_res,
            target_tags=target_tags,
            episode=episode,
            strict=strict,
        )
    ]
    edition.sort(
        key=lambda rel: _edition_sort_key(
            _release_name(rel), target_source, target_res, episode
        )
    )
    if edition:
        best = edition[0]
        return ReferenceDecision(
            "edition",
            best,
            f"source/codec/edition match on {_release_name(best)!r}",
        )
    return ReferenceDecision(
        "abort", None, "no candidate proves the same edition as the target"
    )
