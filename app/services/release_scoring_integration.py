"""Production adapters for the sole weighted release authority; no networking."""
import hashlib
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from app.services.release_parsing.normalize import canonical_record
from app.services.release_parsing.rules import INPUT_FIELDS
from app.services.release_scoring.engine import evaluate
from app.services.release_scoring.policies import DisplayPolicy, ReferencePolicy, rank
from app.services.release_scoring.profiles import candidate_profiles, target_profile

MISSING_ID = "<missing-provider-id>"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def provider_id(release: Any) -> str:
    for field in ("provider_id", "subtitle_id", "file_id", "id"):
        value = get(release, field)
        if value is not None and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(value)):
            return str(value)
    # Decode a provider ID from the existing local proxy route, never use a URL as identity.
    if str(get(release, "provider", "")).casefold() == "opensubtitles":
        route = re.fullmatch(r"/sub/opensubtitles/(\d+)\.(?:srt|ass|ssa|vtt)", str(get(release, "download_url", "")))
        if route:
            return route[1]
    if str(get(release, "provider", "")).casefold() == "subsource":
        route = re.fullmatch(r"(?:/api/v1)?/subtitles/(\d+)/download", urlsplit(str(get(release, "download_url", ""))).path)
        if route:
            return route[1]
    return MISSING_ID


def _name_identity(name: str, existing: dict) -> dict:
    """Reuse established anime/worded episode parsing, never legacy scoring."""
    from app.services.subtitle_matcher import extract_metadata
    parsed = canonical_record({"filename": name})
    extra = {}
    meta = extract_metadata(name)
    episodes = meta.get("episodes") or set()
    # Absolute-episode inference applies only to names carrying no release year.
    # This mirrors the authoritative parser's own guard, so legacy anime parsing
    # can never contradict a year-bearing movie identity ("Movie.2024.Release.03"
    # is a sequence number, not episode 3).
    year_bearing = re.search(r"\b(?:19|20)\d{2}\b", name) is not None
    # A name the authoritative parser resolved as an episode span is a batch.
    # Legacy parsing must not reintroduce its first number as a single episode.
    batch = parsed.value("scope") == "season_pack" and parsed.value("episode") is None
    if not year_bearing and not batch:
        for field in ("season", "episode"):
            if parsed.value(field) is None and field not in existing and meta.get(field) is not None and (field != "episode" or len(episodes) <= 1):
                extra[field] = meta[field]
        if len(episodes) == 1 and parsed.value("episode") is None and "episode" not in existing:
            extra["episode"] = next(iter(episodes))
    if extra.get("episode") is not None or extra.get("season") is not None:
        extra["media_type"] = "series"
        if parsed.value("title") is None and meta.get("title"):
            # Legacy anime parsing can leave technical WEB tokens in its title.
            # Use only the independently parsed prefix preceding the episode.
            number = meta.get("absolute_episode") or meta.get("episode")
            boundary = re.search(rf"(?i)(?:^|[ ._\-#])(?:E|EP|episode)?0*{number}(?!\d)", name) if number is not None else None
            prefix = name[:boundary.start()].strip(" ._-") if boundary else ""
            extra["title"] = extract_metadata(prefix).get("title") if prefix else meta["title"]
    if meta.get("edition") and not any(parsed.value(f) for f in ("cut", "version", "variant_labels")):
        extra["edition"] = meta["edition"]
    return extra


def record(release: Any) -> dict:
    provider = str(get(release, "provider", "unknown")).casefold()
    if provider not in {"subdl", "subsource", "opensubtitles"}:
        provider = "unknown"
    metadata = {}
    for field in INPUT_FIELDS - {"hash_verified", "moviehash", "video_size", "imdb_verified"}:
        value = get(release, field)
        if isinstance(value, str | int | bool) and value != "":
            metadata[field] = value
    name = str(get(release, "release_name", ""))
    metadata.update(_name_identity(name, metadata))
    names = get(release, "release_names", ())
    if not isinstance(names, list | tuple):
        names = ()
    names = sorted({name, *(n for n in names if isinstance(n, str))})
    pid = provider_id(release)
    # Missing-ID records have a deterministic metadata identity, not an arrival index.
    identity = (provider, pid, names, metadata, str(get(release, "lang", ""))) if pid == MISSING_ID else (provider, pid)
    return {"id": digest(identity), "provider": provider, "provider_id": pid,
            "release_names": names, "metadata": metadata,
            "language": str(get(release, "lang", "")).lower() if re.fullmatch(r"[A-Za-z]{2,3}", str(get(release, "lang", ""))) else "unknown",
            "hearing_impaired": get(release, "hearing_impaired") is True or bool(re.search(r"(?i)(?:^|[ ._\[(-])(?:SDH|HI|CC)(?:[ ._\])-]|$)", name))}


def target(value: Any) -> dict:
    metadata = {}
    for field in INPUT_FIELDS - {"hash_verified", "moviehash", "imdb_verified", "scope", "edition"}:
        val = get(value, field)
        if isinstance(val, str | int | bool) and val != "":
            metadata[field] = val
    media_type = str(get(value, "media_type", "movie")).lower()
    metadata["media_type"] = "series" if media_type in {"series", "tv", "anime"} or get(value, "season") is not None or get(value, "episode") is not None else "movie"
    if media_type == "unknown":
        metadata.pop("media_type", None)
    filename = get(value, "target_filename") or get(value, "filename") or ""
    metadata.update(_name_identity(str(filename), metadata))
    parsed = canonical_record({"filename": str(filename)})
    metadata = {k: v for k, v in metadata.items() if k not in {"title", "year"} or parsed.value(k) is None}
    if re.fullmatch(r"tt[0-9]+", str(metadata.get("imdb_id", ""))):
        metadata["imdb_verified"] = True
    if valid_hash(get(value, "video_hash"), get(value, "video_size")):
        metadata.update(moviehash=str(get(value, "video_hash")).lower(), video_size=get(value, "video_size"), hash_verified=True)
    result: dict[str, Any] = {"filename": str(filename), "metadata": metadata}
    languages = get(value, "display_languages")
    if isinstance(languages, list | tuple) and len(languages) <= 20 and all(
            isinstance(v, str) and re.fullmatch(r"[A-Za-z]{2,3}", v) for v in languages):
        result["display_languages"] = list(languages)
        result["exclude_hi"] = get(value, "exclude_hi") is True
        if get(value, "hi_preference") in {"neutral", "prefer", "exclude"}:
            result["hi_preference"] = get(value, "hi_preference")
    return result


def valid_hash(video_hash: Any, size: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{16}", str(video_hash or ""))
                and str(size or "").isdecimal() and int(size) > 0)


logger = logging.getLogger(__name__)
_selected_member: ContextVar[list[str] | None] = ContextVar("weighted_selected_member", default=None)


def set_value(item: Any, key: str, value: Any) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def _provider_evidence(release: Any, raw: dict, filename: str | None = None) -> None:
    item = record(release)
    details = raw.get("feature_details", {})
    if not isinstance(details, dict):
        details = {}
    extra = dict(item["metadata"])
    for field in ("source", "release_group", "cut", "version", "edition", "variant_labels",
                  "resolution", "video_codec", "audio_codec", "year", "season", "episode"):
        value = raw.get(field)
        if isinstance(value, str | int) and len(str(value)) <= 256:
            extra[field] = value
    # OpenSubtitles feature identity is observed provider evidence, not the search's IMDb.
    if item["provider"] == "opensubtitles":
        feature_type = str(details.get("feature_type", "")).casefold()
        is_episode = feature_type != "movie" and (
            feature_type in {"episode", "tv", "series"}
            or any(details.get(f) is not None for f in ("season_number", "episode_number"))
            or any(extra.get(f) is not None for f in ("season", "episode"))
            or canonical_record({**item, "filename": filename or ""}).value("media_type") == "series"
        )
        if is_episode:
            # Episode IMDb is not comparable to a request's series IMDb. Only
            # explicit provider parent/series identity may populate this field.
            extra.pop("imdb_id", None)
            for field in ("parent_imdb_id", "series_imdb_id"):
                imdb = str(details.get(field) or "")
                if re.fullmatch(r"(?:tt)?[0-9]+", imdb):
                    extra["imdb_id"] = imdb if imdb.startswith("tt") else "tt" + imdb
                    item["imdb_authoritative"] = True
                    break
        else:
            imdb = str(details.get("imdb_id") or "")
            if imdb.isdecimal():
                extra["imdb_id"] = "tt" + imdb
                item["imdb_authoritative"] = True
        if feature_type in {"episode", "tv", "series"}:
            item["provider_media_type"] = "series"
        elif feature_type == "movie":
            item["provider_media_type"] = "movie"
        if is_episode and details.get("year") is not None:
            item["year_identity_scope"] = "episode"
        for field, provider_field in (("year", "year"), ("season", "season_number"), ("episode", "episode_number")):
            value = details.get(provider_field)
            if isinstance(value, str | int) and len(str(value)) <= 256:
                extra[field] = value
    names = item["release_names"]
    if isinstance(filename, str) and len(filename) <= 2048:
        names = sorted(set(names + [filename]))
    item = {**item, "metadata": extra, "release_names": names}
    set_value(release, "release_evidence", [*get(release, "release_evidence", []), item])



def provider_evidence(release: Any, raw: dict, filename: str | None = None) -> None:
    try:
        _provider_evidence(release, raw, filename)
    except Exception:
        # Malformed optional metadata must not discard a valid provider result.
        pass


def hash_proof(release: Any, params: dict, literal_match: Any) -> None:
    # Called only at the actual provider hash query result boundary.
    if get(release, "provider") != "opensubtitles" or literal_match is not True or not valid_hash(params.get("moviehash"), params.get("moviebytesize")):
        return
    records = get(release, "release_evidence", []) or [record(release)]
    updated = []
    for item in records:
        metadata = {**item["metadata"], "moviehash": str(params["moviehash"]).lower(),
                    "video_size": params["moviebytesize"], "hash_verified": True}
        updated.append({**item, "metadata": metadata, "hash_provenance": "opensubtitles_hash_query_literal_true"})
    set_value(release, "release_evidence", updated)


@contextmanager
def extracting_member():
    selected: list[str] = []
    token = _selected_member.set(selected)
    try:
        yield selected
    finally:
        _selected_member.reset(token)


def member_selected(name: str) -> None:
    selected = _selected_member.get()
    if selected is not None:
        selected.append(name)


def records_for(release: Any, member: str | None = None) -> list[dict]:
    evidence = get(release, "release_evidence", []) or [record(release)]
    records = []
    for item in evidence:
        metadata = dict(item["metadata"])
        if item.get("provider_media_type") in {"series", "movie"}:
            metadata["media_type"] = item["provider_media_type"]
        if item.get("year_identity_scope") == "episode":
            metadata.pop("year", None)
        entry = {**item, "metadata": metadata}
        if member:
            parsed = canonical_record({"filename": member})
            parent = canonical_record(item)
            from app.services.sync.matching import _member_episode_number
            member_episode = parsed.value("episode") or _member_episode_number(member)
            member_season = parsed.value("season") or parent.value("season")
            independently_known = bool(member_episode is not None and member_season is not None
                                       and not parsed.field("season").conflict and not parsed.field("episode").conflict
                                       and not parent.field("season").conflict)
            entry.update(archive_member=member, archive_member_verified=(independently_known or parent.value("media_type") == "movie"))
            if independently_known:
                entry["archive_member_identity"] = {"season": member_season, "episode": member_episode}
        records.append(entry)
    return records


def matches(releases: list, video: Any, *, member: str | None = None) -> tuple:
    target_record = target(video)
    records = [record for release in releases for record in records_for(release, member)]
    return tuple(evaluate(target_profile(target_record), c) for c in candidate_profiles(records))


def ranked_pairs(releases: list, video: Any, policy: DisplayPolicy | ReferencePolicy) -> list[tuple[Any, Any]]:
    results = matches(releases, video)
    objects: dict[str, list[Any]] = {}
    for release in releases:
        for item in records_for(release):
            objects.setdefault(item["id"], []).append(release)
    ordered = []
    for result in rank(results, policy):
        # Duplicate records merge evidence; representative choice is deterministic
        # and never a semantic ranking input. URLs are not candidate identities.
        group = [obj for alias in result.candidate.aliases for obj in objects.get(alias, ())]
        representative = min(group, key=lambda obj: (str(get(obj, "release_name", "")), str(get(obj, "download_url", ""))))
        merged_records = {digest(item): item for obj in group for item in records_for(obj)}
        set_value(representative, "release_evidence", [merged_records[k] for k in sorted(merged_records)])
        ordered.append((representative, result))
    return ordered


def reference_order(releases: list, video: Any) -> list:
    ordered, _ = discovery_pairs(releases, video, log_phase="discovery")
    return [release for release, _ in ordered]


def discovery_pairs(
    releases: list,
    video: Any,
    *,
    policy: DisplayPolicy | ReferencePolicy | None = None,
    log_phase: str | None = None,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Return policy-ordered and policy-ineligible release/result pairs.

    This is the same evaluation and deterministic policy used by
    :func:`ranked_pairs`, without mutating provider release objects. Planning
    can therefore explain or reserve an attempt for a weighted result while the
    authoritative score, order, and eligibility remain unchanged.
    """
    selected_policy = policy if policy is not None else ReferencePolicy(discovery=True)
    results = matches(releases, video)
    objects: dict[str, list[Any]] = {}
    for release in releases:
        for item in records_for(release):
            objects.setdefault(item["id"], []).append(release)
    ordered = []
    selected_identities: set[str] = set()
    for result in rank(results, selected_policy):
        # Duplicate records merge evidence; representative choice is deterministic
        # and never a semantic ranking input. URLs are not candidate identities.
        group = [obj for alias in result.candidate.aliases for obj in objects.get(alias, ())]
        representative = min(group, key=lambda obj: (str(get(obj, "release_name", "")), str(get(obj, "download_url", ""))))
        ordered.append((representative, result))
        selected_identities.add(result.candidate.identity)
        if log_phase and len(ordered) <= 5:
            log_reference(result, log_phase)
    # Policy-ineligible results are also returned so planning can explain why a
    # provider response did not enter the eligible order. They are never used
    # for selection here; the caller preserves ReferencePolicy order and applies
    # the same eligibility test before planning an attempt.
    ineligible = []
    for result in sorted(results, key=lambda item: (-item.score, -item.score_without_hash, item.candidate.identity)):
        if result.candidate.identity in selected_identities:
            continue
        group = [obj for alias in result.candidate.aliases for obj in objects.get(alias, ())]
        if not group:
            continue
        representative = min(group, key=lambda obj: (str(get(obj, "release_name", "")), str(get(obj, "download_url", ""))))
        ineligible.append((representative, result))
    return ordered, ineligible


def reference_resolvable(release: Any, video: Any) -> bool:
    """Ready metadata or a known season pack resolvable by actual extraction."""
    for result in matches([release], video):
        if result.ready_reference_eligible:
            return True
        if (result.discovery_eligible and result.candidate.unresolved_pack
                and result.candidate.fact("source").value is not None
                and result.candidate.fact("season").value is not None):
            return True
    return False


def reference_ready(release: Any, video: Any, member: str | None = None) -> bool:
    results = matches([release], video, member=member)
    if not results:
        return False
    for result in results:
        log_reference(result, "after_extraction" if member else "ready")
    return any(ReferencePolicy().eligible(result) for result in results)


def resolved_member_identity(release: Any, video: Any, member: str | None = None):
    """Season/episode a READY reference actually proves, else ``None``.

    Mirrors :func:`reference_ready` and returns the identity only for a result
    the reference policy would accept. This is what separates an extracted and
    re-scored episode member from a season pack that merely could not be
    resolved yet: the pack has no proven episode, so nothing is returned and no
    artifact may be stored against the video.
    """
    proof = reference_proof(release, video, member)
    if proof is None:
        return None
    return proof["season"], proof["episode"]


def reference_proof(release: Any, video: Any, member: str | None = None) -> dict | None:
    """Return the proven identity and weighted score for a READY reference.

    The same result object whose eligibility authorizes the download supplies
    every retained fact. An unresolved pack has no proven episode and therefore
    has no proof.
    """
    results = matches([release], video, member=member)
    if not results:
        return None
    for result in results:
        log_reference(result, "after_extraction" if member else "ready")
    for result in results:
        if not ReferencePolicy().eligible(result):
            continue
        season = _proven_member_number(result.candidate.fact("season").value)
        episode = _proven_member_number(result.candidate.fact("episode").value)
        if get(video, "episode") is not None and episode is None:
            # A season pack has no proven episode until its requested member is
            # extracted and independently re-scored.
            continue
        return {
            "season": season,
            "episode": episode,
            "score": result.score,
            "score_without_hash": result.score_without_hash,
            "percentage": result.percentage,
        }
    return None


def _proven_member_number(value: Any) -> int | str | None:
    """Normalize a proven season/episode number for cache binding.

    Canonical parser values are strings such as ``"8"`` while query metadata
    commonly uses integers. Normalizing only decimal values preserves strict
    identity matching without changing any non-numeric identity.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value


def log_attempt(release: Any, video: Any) -> None:
    """Best-effort correlation of real attempts; never changes execution."""
    try:
        for result in matches([release], video)[:1]:
            log_reference(result, "attempt")
    except Exception:
        pass


def log_reference(result: Any, phase: str) -> None:
    try:
        logger.info("[weighted-reference] %s", json.dumps({
            "safe_id": digest(result.candidate.identity)[:16], "provider": result.candidate.provider,
            "language": result.candidate.language, "phase": phase, "score": result.ranking_score,
            "metadata_score": result.score, "metadata_percentage": result.percentage,
            "source_priority_points": result.source_priority_points,
            "percentage": result.ranking_percentage, "matches": [e.field for e in result.evidence if e.state.value == "match"],
            "mismatches": [e.field for e in result.evidence if e.state.value == "mismatch"],
            "ready": result.ready_reference_eligible, "rejections": result.rejection_reasons,
            "readiness": result.readiness_reasons}, sort_keys=True))
    except Exception:
        pass  # Explainability cannot affect subtitle delivery.


def metadata_record(value: Any) -> dict:
    if isinstance(value, str):
        return {"filename": value, "metadata": _name_identity(value, {})}
    if not isinstance(value, dict):
        return record(value)
    aliases = {"group": "release_group", "codec": "video_codec", "audio": "audio_codec", "service": "service_disc_family"}
    metadata = {aliases.get(k, k): v for k, v in value.items() if aliases.get(k, k) in INPUT_FIELDS and v is not None}
    name = value.get("raw_filename") or value.get("release_name") or ""
    # A historical extract_metadata dictionary is a projection of its own
    # filename, not independent provider evidence. Prefer the exact parser's
    # identity when that filename independently supplies it.
    parsed = canonical_record({"filename": name})
    if parsed.value("title") is not None:
        metadata.pop("title", None)
    metadata.update(_name_identity(name, metadata))
    return {"filename": name, "metadata": metadata}


def compare_metadata(video: Any, candidate: Any):
    t = metadata_record(video)
    c = metadata_record(candidate)
    c.update(id="metadata", provider="unknown", provider_id="metadata", language="ara")
    result = evaluate(target_profile(t), candidate_profiles([c])[0])
    # Existing anime absolute/range identity guard is eligibility, never score.
    from dataclasses import replace

    from app.services.subtitle_matcher import (
        extract_metadata,
        hard_compatibility_filter,
        is_anime_content,
    )
    def legacy(value):
        return extract_metadata(value) if isinstance(value, str) else value if isinstance(value, dict) else extract_metadata(get(value, "release_name", ""))
    target_meta, candidate_meta = legacy(video), legacy(candidate)
    if is_anime_content(target_meta, {}):
        accepted, reason, _ = hard_compatibility_filter(target_meta, candidate_meta)
        if not accepted:
            result = replace(result, rejection_reasons=tuple(sorted(set(result.rejection_reasons) | {"anime_identity_conflict"})))
    return result


def display_order(video_filename: str | None, releases: list, *, preferred_languages=None,
                  exclude_sdh=False, season=None, episode=None, title=None, year=None,
                  hi_preference="neutral", imdb_id=None, media_type=None, video_hash=None, video_size=None,
                  discard_mismatches=True) -> list:
    video = {"filename": video_filename, "season": season, "episode": episode, "title": title,
             "year": year, "imdb_id": imdb_id, "media_type": media_type or
             ("series" if season is not None or episode is not None or canonical_record({"filename": video_filename or ""}).value("media_type") == "series" else "movie" if video_filename or title or imdb_id else "unknown"),
             "video_hash": video_hash, "video_size": video_size}
    languages = tuple(preferred_languages) if preferred_languages is not None else tuple(sorted({get(s, "lang", "ara") for s in releases}))
    policy = DisplayPolicy(languages, exclude_sdh or hi_preference == "exclude", True if hi_preference == "prefer" else None)
    ordered = []
    # Preserve existing anime/absolute/range identity eligibility independently
    # of weighted release scoring. This filter never supplies ranking points.
    from app.services.subtitle_matcher import (
        extract_metadata,
        hard_compatibility_filter,
        is_anime_content,
    )
    target_meta = extract_metadata(video_filename or "")
    if season is not None:
        target_meta["season"] = season
    if episode is not None:
        target_meta["episode"] = episode
        target_meta["episodes"] = {episode}
    if is_anime_content(target_meta, {}):
        releases = [r for r in releases if hard_compatibility_filter(target_meta, extract_metadata(get(r, "release_name", "")))[0]]
    for release, result in ranked_pairs(releases, video, policy):
        from app.services.subtitle_matcher import is_debug_ranking_enabled
        if is_debug_ranking_enabled():
            logger.info("[ranking] weighted safe_id=%s score=%s percentage=%.2f", digest(result.candidate.identity)[:16], result.ranking_score, result.ranking_percentage)
        set_value(release, "score", round(result.ranking_percentage))
        set_value(release, "match_percentage", round(result.ranking_percentage))
        set_value(release, "compatibility", result)
        set_value(release, "weighted_key", policy.key(result))
        from app.models import MatchTier
        states = {e.field: e.state.value for e in result.evidence}
        set_value(release, "match_tier", MatchTier.HASH if result.verified_hash else MatchTier.EXACT
                  if states.get("release_group") == "match" and states.get("source") == "match"
                  else MatchTier.SOURCE_FAMILY if states.get("source") == "match" else MatchTier.CLOSE)
        ordered.append(release)
    return ordered
