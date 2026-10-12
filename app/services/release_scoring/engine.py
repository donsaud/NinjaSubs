"""Independent comparison and additive compatibility scoring."""
from .model import (
    Contribution,
    MatchEvidence,
    MatchResult,
    MatchState,
    SubtitleCandidateProfile,
    TargetVideoProfile,
)

# Numeric behavioral facts from upstream patched defaults; no GPL implementation.
EPISODE_WEIGHTS = (("title", 160), ("year", 90), ("season", 30), ("episode", 30),
                   ("source", 25), ("release_group", 20), ("audio_codec", 1),
                   ("resolution", 1), ("video_codec", 1), ("service_disc_family", 1))
MOVIE_WEIGHTS = (("title", 60), ("year", 40), ("source", 30), ("edition", 30),
                 ("release_group", 15), ("audio_codec", 1), ("resolution", 1),
                 ("video_codec", 1), ("service_disc_family", 1))
SOURCE_FAMILIES = {"bluray": "disc_hd", "remux": "disc_hd", "bdrip": "disc_hd",
                   "web_dl": "web", "webrip": "web", "hdtv": "broadcast", "dvdrip": "disc_sd",
                   "sdtv": "broadcast", "dsr": "broadcast", "satrip": "broadcast",
                   "cam": "telesync", "hdcam": "telesync", "ts": "telesync", "tc": "telesync",
                   "hdts": "telesync", "telesync": "telesync", "telecine": "telesync",
                   "scr": "screener", "screener": "screener", "dvdscr": "screener"}
REFERENCE_MINIMUM_PERCENT = 70


def _compare(target: TargetVideoProfile, candidate: SubtitleCandidateProfile, field: str) -> MatchEvidence:
    left, right = target.fact(field), candidate.fact(field)
    relation = "direct"
    if left.conflict or right.conflict:
        state = MatchState.CONFLICT
    elif left.value is None or right.value is None:
        state = MatchState.UNKNOWN
    elif left.value == right.value:
        state = MatchState.MATCH
    elif (field == "source" and SOURCE_FAMILIES.get(left.value) is not None
          and SOURCE_FAMILIES.get(left.value) == SOURCE_FAMILIES.get(right.value)):
        state, relation = MatchState.MATCH, "source_family"
    else:
        state = MatchState.MISMATCH
    return MatchEvidence(field, state, left, right, relation)


def evaluate(target: TargetVideoProfile, candidate: SubtitleCandidateProfile) -> MatchResult:
    fields = sorted(set(dict(target.facts)) | set(dict(candidate.facts)))
    evidence = tuple(_compare(target, candidate, field) for field in fields)
    states = {e.field: e.state for e in evidence}
    is_series = target.fact("media_type").value == "series"
    weights = EPISODE_WEIGHTS if is_series else MOVIE_WEIGHTS
    # A request uses one common denominator, determined only by target facts.
    # Unavailable target facts cannot reward or penalize particular candidates.
    def comparable(field: str) -> bool:
        if field == "edition":
            return any(target.fact(f).value is not None for f in ("cut", "version", "variant_labels"))
        if field == "title" and target.imdb_authoritative and target.fact("imdb_id").value:
            return True
        return target.fact(field).value is not None

    maximum = max(1, sum(weight for field, weight in weights if comparable(field)))
    verified_identity = (target.imdb_authoritative and candidate.imdb_authoritative
                         and states.get("imdb_id") is MatchState.MATCH)
    rejections = []
    for field in ("media_type", "season", "episode"):
        if states.get(field) in {MatchState.CONFLICT, MatchState.MISMATCH}:
            rejections.append(field + "_" + states[field].value)
    if states.get("imdb_id") in {MatchState.CONFLICT, MatchState.MISMATCH} and (
            target.imdb_authoritative and candidate.imdb_authoritative):
        rejections.append("imdb_id_" + states["imdb_id"].value)
    if not verified_identity:
        for field in (("title",) if is_series else ("title", "year")):
            if states.get(field) in {MatchState.MISMATCH, MatchState.CONFLICT}:
                rejections.append(field + "_" + states[field].value)
    for field in ("cut", "version"):
        if states.get(field) in {MatchState.CONFLICT, MatchState.MISMATCH}:
            # Extended and unrated are non-exclusive; named cuts are exclusive.
            if field == "cut" or states[field] is MatchState.CONFLICT:
                rejections.append("cut_version_conflict")
    for left, right in ((target, candidate), (candidate, target), (target, target), (candidate, candidate)):
        if left.fact("cut").value == "theatricalcut" and right.fact("version").value is not None:
            rejections.append("cut_version_conflict")
    verified_hash = target.hash_authoritative and candidate.hash_authoritative
    hash_state = states.get("moviehash", MatchState.UNKNOWN)
    size_state = states.get("video_size", MatchState.UNKNOWN)
    if verified_hash and (hash_state in {MatchState.MISMATCH, MatchState.CONFLICT}
                          or size_state in {MatchState.MISMATCH, MatchState.CONFLICT}):
        rejections.append("verified_moviehash_conflict")
    exact_hash = verified_hash and hash_state is MatchState.MATCH and size_state is MatchState.MATCH
    evidence = tuple(MatchEvidence(e.field, e.state if verified_hash else MatchState.UNKNOWN,
                                    e.target, e.candidate, "verified_query" if verified_hash else "unverified")
                     if e.field == "moviehash" else e for e in evidence)
    contributions = []
    for field, weight in weights:
        earned = states.get(field) is MatchState.MATCH
        basis = "comparable_match" if earned else "no_match_credit"
        if field in {"title", "year"} and verified_identity:
            earned, basis = True, "verified_independent_imdb_identity"
        elif field == "release_group" and states.get("source") is not MatchState.MATCH:
            earned, basis = False, "group_requires_source_compatibility"
        elif field == "edition":
            edition_states = [states.get(f) for f in ("cut", "version", "variant_labels")
                          if target.fact(f).value is not None]
            earned = bool(edition_states) and all(state is MatchState.MATCH for state in edition_states)
            basis = "explicit_comparable_edition" if earned else "unknown_or_different_edition"
        if not comparable(field):
            earned, basis = False, "target_fact_unknown"
        contributions.append(Contribution(field, weight, weight if earned else 0, basis))
    without_hash = sum(c.points for c in contributions)
    hash_weight = 359 if is_series else 179
    contributions.append(Contribution("moviehash", hash_weight, hash_weight if exact_hash else 0,
                                      "verified_query" if exact_hash else "no_hash_proof"))
    score = without_hash + (hash_weight if exact_hash else 0)
    readiness = []
    if not verified_identity and states.get("title") is not MatchState.MATCH:
        readiness.append("content_identity_unresolved")
    if states.get("media_type") is not MatchState.MATCH:
        readiness.append("media_type_unresolved")
    if not is_series and not verified_identity and states.get("year") is not MatchState.MATCH:
        readiness.append("movie_year_unresolved")
    if is_series and any(states.get(f) is not MatchState.MATCH for f in ("season", "episode")):
        readiness.append("episode_unresolved")
    if target.fact("source").value is None or candidate.fact("source").value is None:
        readiness.append("source_unresolved")
    if states.get("source") is MatchState.CONFLICT:
        readiness.append("source_conflict")
    if candidate.unresolved_pack:
        readiness.append("season_pack_unresolved")
    if candidate.archive_members and not candidate.independently_resolved_member:
        readiness.append("archive_member_unresolved")
    if score * 100 < maximum * REFERENCE_MINIMUM_PERCENT:
        readiness.append("reference_score_below_minimum")
    return MatchResult(candidate, evidence, tuple(contributions), score, without_hash, maximum,
                       tuple(sorted(set(rejections))), tuple(sorted(set(readiness))), exact_hash)
