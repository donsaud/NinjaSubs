"""Run the mandatory safety mutations and prove the suite catches each one.

Phase 21 of the finalisation pass: a test that stays green after the safety
behaviour it guards is deliberately removed is not a test. This tool removes
each guard in turn, in a throwaway copy of the working tree, and requires the
suite to FAIL. A guard whose removal leaves the suite green is reported as
UNPROTECTED, which is a release blocker.

It never edits the working tree: every mutation is applied to a temporary copy
and the original is left untouched. No network, no provider credentials.

    python tools/run_safety_mutations.py
    python tools/run_safety_mutations.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One deliberately broken safety behaviour."""

    name: str
    file: str
    old: str
    new: str
    #: Why removing this must break the suite.
    guards: str


MUTATIONS: tuple[Mutation, ...] = (
    # --- MovieHash input-boundary validation ------------------------------
    #
    # MovieHash is OPTIONAL acceleration, not a synchronisation dependency, and
    # hashless synchronization is a first-class supported path. These three
    # protect the *other* half of that contract: when a client DOES supply a
    # hash, only a well-formed one may be treated as identity evidence. A
    # malformed value forwarded into ``params["moviehash"]`` spends a metered
    # OpenSubtitles call on a query that cannot match while the log reads like
    # an exact-identity lookup.
    Mutation(
        name="video-hash-validation-removed",
        file="app/services/ranking.py",
        old="    return candidate if _VIDEO_HASH_RE.match(candidate) else None",
        new="    return candidate  # MUTATION: any truthy value becomes identity evidence",
        guards=(
            "a malformed client-supplied videoHash must be treated as unavailable, "
            "never forwarded as exact-identity evidence"
        ),
    ),
    Mutation(
        name="video-hash-format-not-enforced",
        file="app/services/ranking.py",
        old='_VIDEO_HASH_RE = re.compile(r"^[0-9a-f]{16}$")',
        new='_VIDEO_HASH_RE = re.compile(r"^[0-9a-zA-Z]{1,64}$")  # MUTATION',
        guards=(
            "MovieHash is exactly 16 hex characters: a wrong-length or non-hex "
            "value is not a MovieHash and must not reach hash reference logic"
        ),
    ),
    Mutation(
        name="video-hash-normalization-skipped",
        file="app/services/ranking.py",
        old="    candidate = value.strip().lower()",
        new="    candidate = value.strip()  # MUTATION: case/whitespace not normalised",
        guards=(
            "a hash differing only by case or padding must normalise to one "
            "canonical form, so the same video cannot split into two identities"
        ),
    ),
    Mutation(
        name="target-fingerprint-binding-removed",
        file="app/services/sync_cache.py",
        old='        prefix = f"final_sub:{imdb}:{season_ep}:{fp}:{sid}"',
        new='        prefix = f"final_sub:{imdb}:"  # MUTATION',
        guards="a payload cached for one video must never serve another target",
    ),
    Mutation(
        name="negative-verdict-served-as-verified",
        file="app/main.py",
        old=(
            "    if state not in (SyncState.VERIFIED_SYNCED.value, "
            "SyncState.VERIFIED_RESYNCED.value):"
        ),
        new="    if False:  # MUTATION",
        guards="REJECTED/UNVERIFIED/PROBABLE_SYNC must never yield a served artifact",
    ),
    Mutation(
        name="stale-engine-version-accepted",
        file="app/services/sync_cache.py",
        old='        if int(data.get("engine_version") or 0) != SYNC_VERDICT_ENGINE_VERSION:',
        new="        if False:  # MUTATION",
        guards="a verdict written by an older engine must not be reused",
    ),
    Mutation(
        name="divergent-target-identity",
        file="app/main.py",
        old=(
            '        "video_hash": meta.get("video_hash"),\n'
            '        "video_size": meta.get("video_size"),\n'
            "    }"
        ),
        new="        # MUTATION: identity fields dropped\n    }",
        guards="reuse and serve paths must derive the same target identity",
    ),
    Mutation(
        name="strict-timestamp-grammar-restored",
        file="app/services/subtitle_matcher.py",
        old=(
            r'r"((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*'
            r'((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})"'
        ),
        new=(
            r'r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*'
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"'
        ),
        guards="MM:SS,mmm targets must parse, or the verifier sees an empty target",
    ),
    Mutation(
        name="alass-budget-process-wide-again",
        file="app/services/sync/orchestrator.py",
        old="            if alass_attempted >= self._alass_candidate_limit:",
        new=(
            '            if self._metrics["alass_runs"] >= self._alass_candidate_limit:'
            "  # MUTATION"
        ),
        guards=(
            "ALASS_CANDIDATE_LIMIT is a per-request allowance; a process-wide "
            "counter starves every request after the first few"
        ),
    ),
    Mutation(
        name="reference-cache-target-binding-removed",
        file="app/services/sync/cache.py",
        old='        return sorted(self.root.glob(f"{query.cache_stem}_*.srt"))',
        new='        return sorted(self.root.glob("*_*.srt"))  # MUTATION',
        guards="a reference proven for one target must not satisfy another",
    ),
    Mutation(
        name="unverified-payload-served-from-cache",
        file="app/services/sync/orchestrator.py",
        old="                if is_reusable_verified(stored_state, stored_verification):",
        new="                if True:  # MUTATION",
        guards=(
            "a payload hit is not a verification: an UNVERIFIED/UNKNOWN or "
            "rejected Alass artifact must never be re-served as a finished "
            "synchronization, or it gets silently promoted on the next request"
        ),
    ),
    Mutation(
        name="resync-verdict-served-without-artifact",
        file="app/services/sync/orchestrator.py",
        old=(
            "                and (remembered_state != SyncState.VERIFIED_RESYNCED.value"
            " or resync_artifact_available)"
        ),
        new="                and True  # MUTATION",
        guards=(
            "a verified_resynced verdict means alass re-timed the subtitle, so "
            "its transformed artifact is the answer; without it the branch "
            "serves the unsynchronized original under a verified-resync label"
        ),
    ),
    Mutation(
        name="unverified-sync-output-served",
        file="app/services/sync/orchestrator.py",
        old="""                serve_synchronized = may_serve_synchronized(
                    evaluation.sync_state.value, evaluation.verification.value
                )""",
        new="                serve_synchronized = True  # MUTATION",
        guards=(
            "the analyzer is the authority on synchronization trust: an output "
            "it marks UNVERIFIED/UNKNOWN or REJECTED must never replace the "
            "original subtitle, or a mangled alignment reaches the user "
            "(real Whiplash case: up to +293s displacement, served anyway)"
        ),
    ),
    Mutation(
        name="compatible-release-fallback-hidden-from-user",
        file="app/main.py",
        old="        display_label = _with_fallback_disclosure(display_label, sub_id)",
        new="        display_label = display_label  # MUTATION",
        guards=(
            "when a compatible fallback release is served instead of the "
            "requested candidate, the listing must disclose it; otherwise the "
            "user is told release A while release B's bytes are delivered"
        ),
    ),
    Mutation(
        name="sync-cache-reuse-fallback-disclosure-lost",
        file="app/main.py",
        old="""                    served_release_name=str(
                        (meta_record or {}).get("served_release_name") or ""
                    ).strip(),""",
        new='                    served_release_name="",  # MUTATION',
        guards=(
            "reusing previously verified fallback bytes must still disclose "
            "which release they came from, or the listing attributes them to "
            "the requested candidate after a cache hit"
        ),
    ),
    Mutation(
        name="metadata-persists-raw-credentials",
        file="app/cache.py",
        old="            safe, _ = sanitize_metadata(metadata)",
        new="            safe = dict(metadata)  # MUTATION",
        guards=(
            "provider API keys and bearer tokens must never be written to "
            "_meta/*.json; the download path re-resolves them per request"
        ),
    ),
    Mutation(
        name="subs-cache-overwritten-with-sync-artifact",
        file="app/main.py",
        old="""            if synced_content and synced_content != cached_content:""",
        new="""            if synced_content and synced_content != cached_content:
                await cache_manager.save_subtitle(target_id, synced_content)  # MUTATION""",
        guards=(
            "the disk subtitle cache holds provider bytes; writing a "
            "synchronization artifact back makes the next request re-analyze "
            "that artifact as the source subtitle against the same reference, "
            "a circular verification that can promote UNKNOWN to VERIFIED"
        ),
    ),
    # --- Large Offset Evidence Gate ---------------------------------------- #
    Mutation(
        name="large-offset-evidence-requirement-removed",
        file="app/services/sync/large_offset.py",
        # Must stay syntactically valid: a mutation that only breaks the import
        # proves nothing, because the suite dies during collection instead of a
        # guard asserting the behaviour.
        old=(
            "    if assessment.structural_similarity is None or (\n"
            "        assessment.structural_similarity < "
            "LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY\n"
            "    ):\n"
            "        assessment.reason_codes.append(REASON_STRUCTURAL_MISMATCH)\n"
            "        return assessment\n"
        ),
        new=(
            "    if assessment.structural_similarity is None and (\n"
            "        assessment.structural_similarity < "
            "LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY\n"
            "    ):\n"
            "        assessment.reason_codes.append(REASON_STRUCTURAL_MISMATCH)\n"
            "        return assessment\n"
        ),
        guards=(
            "a large offset must be corroborated by timing structure, not one cue; "
            "this mutation also relaxes the similarity floor, so it fails on any "
            "real assertion rather than on a syntax error"
        ),
    ),
    Mutation(
        name="large-offset-first-cue-only-accepted",
        file="app/services/sync/large_offset.py",
        old="    if len(offsets) < LARGE_OFFSET_MIN_ANCHORS:",
        new="    if len(offsets) < 1:  # MUTATION",
        guards="several distant regions must agree; one first-cue comparison is not evidence",
    ),
    Mutation(
        name="large-offset-dispersion-ignored",
        file="app/services/sync/large_offset.py",
        old="    if (assessment.offset_dispersion_ms or 0.0) > LARGE_OFFSET_MAX_DISPERSION_MS:",
        new="    if False:  # MUTATION",
        guards="a drifting or re-cut timeline must not pass as a constant offset",
    ),
    Mutation(
        name="large-offset-ceiling-removed",
        file="app/services/sync/large_offset.py",
        old="    if abs(seed_offset_ms) > max_offset_ms:",
        new="    if False:  # MUTATION",
        guards="the large-offset path must stay bounded; no offset is unlimited",
    ),
    Mutation(
        name="large-offset-accepted-without-evidence",
        file="app/services/sync/orchestrator.py",
        old="                if large_offset is not None and large_offset.accepted:",
        new="                if True:  # MUTATION",
        guards="a cue-sanity rejection must never be bypassed without evidence",
    ),
    # --- Measured movement seed -------------------------------------------- #
    Mutation(
        name="movement-seed-from-large-offset-gate",
        file="app/services/sync/alignment.py",
        old="""        seed = measure_movement_seed(
            target_cues,
            synced_cues,""",
        new="""        seed = measure_movement_seed(  # MUTATION: no cue arrays
            [],
            [],""",
        guards=(
            "the movement seed must come from the two cue arrays, never from a "
            "pre-existing estimate handed in by the caller"
        ),
    ),
    Mutation(
        name="movement-seed-correlation-floor-removed",
        file="app/services/sync/movement_seed.py",
        old="    if correlation < MIN_CORRELATION:",
        new="    if False:  # MUTATION",
        guards=(
            "a displacement measured on unrelated tracks must not be trusted; the "
            "seed needs real correlation before it can re-centre the pairing"
        ),
    ),
    Mutation(
        name="movement-seed-peak-margin-removed",
        file="app/services/sync/movement_seed.py",
        old="    if margin < MIN_PEAK_MARGIN:",
        new="    if False:  # MUTATION",
        guards=(
            "an ambiguous correlation with several comparable peaks must be "
            "refused rather than resolved arbitrarily"
        ),
    ),
    Mutation(
        name="movement-seed-half-split-stability-removed",
        file="app/services/sync/movement_seed.py",
        old="    if disagreement is None or disagreement > MAX_HALF_SPLIT_DISAGREEMENT_MS:",
        new="    if False:  # MUTATION",
        guards=(
            "a drifting or multi-jump timeline must not be summarised by a single "
            "misleading global displacement"
        ),
    ),
    Mutation(
        name="movement-seed-coarse-fine-agreement-removed",
        file="app/services/sync/movement_seed.py",
        old="    if abs(lag_bins - coarse_bins) > FINE_RADIUS_BINS:",
        new="    if False:  # MUTATION",
        guards=(
            "a refined lag outside the coarse winner's basin means the fine scan "
            "latched onto a peak the coarse pass never favoured"
        ),
    ),
    Mutation(
        name="movement-seed-support-floor-removed",
        file="app/services/sync/movement_seed.py",
        old="    if ratio < MIN_SEED_SUPPORT:",
        new="    if False:  # MUTATION",
        guards=(
            "a seed that cannot pair cues at the local radius is not evidence and "
            "must not re-centre the movement pass"
        ),
    ),
    Mutation(
        name="movement-seed-bypasses-movement-validation",
        file="app/services/sync/alignment.py",
        old="""        if seed is not None:
            evaluation.reasons.append(seed.explain())""",
        # A confident seed must not exempt the result from the movement gate.
        # The mutation this replaces only edited the "seed is not None"
        # condition, which is a semantic no-op and could never be caught --
        # exactly the sort of mutation that reports UNPROTECTED forever.
        new="""        if seed is None and len(movement_pairs) < MIN_CUES_FOR_PERCENTILES:""",
        guards=(
            "a confident seed is only a search centre; the movement statistics must "
            "still decide the verdict"
        ),
    ),
    Mutation(
        name="post-alass-ceiling-widened-globally",
        file="app/services/sync/alignment.py",
          old="        max_plausible_offset_ms: int = MAX_PLAUSIBLE_OFFSET_MS,\n    ) -> SubtitleEvaluation:\n        \"\"\"Compare pre/post cues",
          new="        max_plausible_offset_ms: int = 180_000,  # MUTATION\n    ) -> SubtitleEvaluation:\n        \"\"\"Compare pre/post cues",
        guards="the widened magnitude ceiling must be opt-in, never the default",
    ),
    Mutation(
        name="pairing-unmatched-reporting-removed",
        file="app/services/sync/alignment.py",
        old="        unmatched_before=unmatched_before,",
        new="        unmatched_before=[],  # MUTATION",
        guards="cues with no counterpart must be reported, never silently dropped",
    ),
    Mutation(
        name="unmatched-cue-manufactured-as-boundary-residual",
        file="app/services/sync/alignment.py",
        old="            unmatched_before.append(index)",
        new="            matches.append(CueMatch(  # MUTATION\n                before_index=index,\n                after_index=0,\n                position_ms=start,\n                delta_ms=float(tolerance_ms),\n            ))\n            continue",
        guards="an unmatched cue must never become a residual at the tolerance edge",
    ),
    Mutation(
        name="pairing-unmatched-cannot-rescue-bad-alignment",
        file="app/services/sync/alignment.py",
        old="        if len(residual_pairs) < MIN_CUES_FOR_PERCENTILES:",
        new="        if False:  # MUTATION",
        guards="excluding unmatched cues must not let a bad alignment verify",
    ),
    Mutation(
        name="large-offset-rejection-consumes-budget",
        file="app/services/sync/orchestrator.py",
        old="                    candidates_rejected += 1",
        new="                    alass_attempted += 1  # MUTATION\n                    candidates_rejected += 1",
        guards="a reference refused before alass must not spend the alass run budget",
    ),
    # --- Subtitle format router (ASS/SSA content vs extension) --------------- #
    Mutation(
        name="ass-conversion-moved-after-verification",
        file="app/main.py",
        old=(
            "    if is_ass_subtitle(payload):\n"
            "        if not convert_ass:\n"
            "            # alass emits SRT; keep the user's native ASS styling preference.\n"
            "            return payload\n"
            "        payload = convert_ass_to_srt_bytes(payload, apply_rtl=False)\n"
        ),
        new=(
            "    if False:  # MUTATION: conversion left to the response builder,\n"
            "        # which runs AFTER the sync attempt, so the SRT-only verifier\n"
            "        # sees raw ASS again and reports 0 cues (ab461b8781aafd89).\n"
            "        if not convert_ass:\n"
            "            return payload\n"
            "        payload = convert_ass_to_srt_bytes(payload, apply_rtl=False)\n"
        ),
        guards=(
            "conversion must happen BEFORE sync: the verifier parser is SRT-only, so "
            "raw ASS yields 0 cues and the real 562-cue artifact is rejected on "
            "container format alone"
        ),
    ),
    Mutation(
        name="native-ass-preservation-guard-removed",
        file="app/main.py",
        old=(
            "        if not convert_ass:\n"
            "            # alass emits SRT; keep the user's native ASS styling preference.\n"
            "            return payload\n"
        ),
        new=(
            "        if False:  # MUTATION\n"
            "            return payload\n"
        ),
        guards=(
            "a user who preserves native ASS must get the provider bytes back "
            "byte-identically; converting just to satisfy the sync path destroys "
            "{\\pos} and styling for no benefit"
        ),
    ),
    Mutation(
        name="format-router-bypassed-ass-reaches-sync",
        file="app/main.py",
        old=(
            "    if is_ass_subtitle(payload):\n"
            "        if not convert_ass:\n"
            "            # alass emits SRT; keep the user's native ASS styling preference.\n"
            "            return payload\n"
            "        payload = convert_ass_to_srt_bytes(payload, apply_rtl=False)\n"
        ),
        new="    # MUTATION: router bypassed, raw payload forwarded as-is\n",
        guards=(
            "the router is the only thing standing between ASS/SSA and the SRT-only "
            "sync path; every sync entry point must route through it"
        ),
    ),
    Mutation(
        name="ass-detected-by-index-line-not-content",
        file="app/main.py",
        old="    if is_ass_subtitle(payload):",
        new=(
            "    if payload.lstrip().startswith(b\"1\\n\"):  # MUTATION: a naive SRT\n"
            "        # sniff instead of content-based format detection\n"
        ),
        guards=(
            "ASS is routinely stored under an .srt name, so detection must be by "
            "content; a leading-index sniff classifies ASS-under-.srt as SRT and "
            "reproduces the 562-vs-0 contradiction"
        ),
    ),
    # --- STREMIO_PLAYBACK_WITNESS (dev-only isolated fixture delivery) ------ #
    Mutation(
        name="witness-mutable-by-default",
        file="app/playback_witness.py",
        old='    raw = os.getenv("ENABLE_STREMIO_PLAYBACK_WITNESS", "").strip().lower()\n'
            '    if raw in ("true", "1", "yes", "on"):\n        return True',
        new=(
            "    return True  # MUTATION: witness always on\n"
        ),
        guards=(
            "the witness mounts a file-delivery endpoint; it must require an "
            "explicit opt-in and be absent from a default deployment entirely"
        ),
    ),
    Mutation(
        name="witness-digest-check-removed",
        file="app/playback_witness.py",
        old="        if actual.lower() != digest.lower():",
        new="        if False:  # MUTATION",
        guards=(
            "a URL that advertises one digest must never serve different bytes; "
            "dropping the check lets a changed fixture masquerade as the "
            "previously-validated artifact"
        ),
    ),
    Mutation(
        name="witness-serves-wrong-fixture",
        file="app/playback_witness.py",
        old="    for fx in WITNESS_FIXTURES:\n        if fx.key.lower() == key.lower():\n            return fx",
        new="    for fx in WITNESS_FIXTURES:  # MUTATION: returns the first fixture\n        return fx",
        guards=(
            "each witness must serve its own artifact under its own label; "
            "returning the first one would file the ASAP bytes under the EVOLV "
            "label and silently invalidate the comparison"
        ),
    ),
    Mutation(
        name="witness-logs-subtitle-contents",
        file="app/playback_witness.py",
        old=(
            '        logger.info(\n'
            '            "[witness] served %s bytes=%d sha256=%s",\n'
            '            fixture.key,\n'
            '            len(data),\n'
            '            actual[:16],\n'
            '        )'
        ),
        new=(
            '        logger.info("[witness] served %s bytes=%d sha256=%s body=%r",\n'
            '                     fixture.key, len(data), actual[:16], data[:200])'
        ),
        guards="subtitle text must never reach a log line; only name, size, digest",
    ),
    # --- Witness Stremio request shape --------------------------------------- #
    # Without the {type}/{id}/{extra} route, the production catch-all
    # /{config}/subtitles/... swallows the request and the aggregator answers it.
    Mutation(
        name="witness-stremio-route-removed",
        file="app/playback_witness.py",
        old=(
            '    @router.api_route(\n'
            '        "/subtitles/{media_type}/{media_id}/{extra:path}.json",\n'
            '        methods=["GET", "HEAD", "OPTIONS"],\n'
            '        response_model=SubtitlesResponse,\n'
            '    )\n'
            '    @router.api_route(\n'
            '        "/subtitles/{media_type}/{media_id}/{extra:path}",\n'
            '        methods=["GET", "HEAD", "OPTIONS"],\n'
            '        response_model=SubtitlesResponse,\n'
            '    )\n'
        ),
        new=(
            '    # MUTATION: the real Stremio request shape is no longer routed.\n'
            '    # Both variants are removed, because the :path form alone would\n'
            '    # still swallow the ".json" suffix and hide the regression.\n'
        ),
        guards=(
            "Stremio requests /subtitles/{type}/{id}/{extra}.json; without this "
            "route the production catch-all treats the witness mount prefix as a "
            "user config string and the aggregator answers instead of the witness"
        ),
    ),
    Mutation(
        name="witness-ignores-media-type",
        file="app/playback_witness.py",
        old=(
            '    media_type = _decode(media_type).strip().lower()\n'
            "    if media_type not in _ALLOWED_TYPES:\n"
            "        return None"
        ),
        new="    media_type = _decode(media_type).strip().lower()  # MUTATION: unchecked",
        guards=(
            "the witness serves one fixed episode; accepting any media type would "
            "turn it into a general unvetted subtitle source"
        ),
    ),
    Mutation(
        name="witness-id-decoded-incorrectly",
        file="app/playback_witness.py",
        old='    stem = _decode(raw_id).split(".")[0].strip()',
        new=(
            "    stem = (raw_id or \"\").split(\":\")[0]  # MUTATION: no decoding, and\n"
            "    # the season/episode are discarded so any id matches"
        ),
        guards=(
            "the id arrives percent-encoded; failing to decode it, or comparing "
            "only the imdb part, would let the wrong episode match the witness"
        ),
    ),
    Mutation(
        name="witness-target-check-dropped",
        file="app/playback_witness.py",
        old=(
            "    if (imdb_id, season, episode) != (TARGET_IMDB, TARGET_SEASON, TARGET_EPISODE):\n"
            "        return None"
        ),
        new="    # MUTATION: any target accepted",
        guards=(
            "the witness is one fixed playback fixture, not a subtitle source; "
            "without this check any imdb id, season or episode would be served"
        ),
    ),
    Mutation(
        name="witness-extra-used-to-select-a-file",
        file="app/playback_witness.py",
        old=(
            "        if \"/\" in value or \"\\\\\" in value or \"..\" in value:\n"
            '            logger.warning("[witness] ignoring unsafe value for extra key %r", key)\n'
            "            continue\n"
        ),
        new="        # MUTATION: traversal strings accepted into metadata\n",
        guards=(
            "extra is attacker-controlled; a '..' or separator must never survive "
            "into anything derived from it"
        ),
    ),
    Mutation(
        name="witness-extra-keys-unfiltered",
        file="app/playback_witness.py",
        old="        if key not in _SAFE_EXTRA_KEYS:\n            continue",
        new="        # MUTATION: any key accepted and logged",
        guards=(
            "extra is attacker-controlled, so an unfiltered key/value would let a "
            "caller inject a credential or a log control sequence"
        ),
    ),
    Mutation(
        name="witness-fixture-allowlist-removed",
        file="app/playback_witness.py",
        old=(
            "    for fx in WITNESS_FIXTURES:\n"
            "        if fx.key.lower() == key.lower():\n"
            "            return fx\n"
            "    raise HTTPException(status_code=404, detail=\"unknown witness fixture\")"
        ),
        new=(
            "    return WITNESS_FIXTURES[0]  # MUTATION: allowlist removed\n"
        ),
        guards=(
            "fixture selection must be an allowlist; otherwise any key could name "
            "a file and the ASAP bytes would be served under the EVOLV label"
        ),
    ),
    Mutation(
        name="witness-extra-length-cap-removed",
        file="app/playback_witness.py",
        old="    if len(decoded) > _MAX_EXTRA_LEN:",
        new="    if False:  # MUTATION",
        guards=(
            "extra is unbounded; without a cap a single request can force an "
            "arbitrarily large string to be decoded and scanned"
        ),
    ),
    Mutation(
        name="witness-routed-after-production-catch-all",
        file="app/main.py",
        old=(
            "if witness_enabled():\n"
            "    app.include_router(build_witness_router())"
        ),
        new=(
            "if witness_enabled():\n"
            "    # MUTATION: included from a deferred hook that runs after the\n"
            "    # production subtitle routes are registered, so the catch-all wins\n"
            "    app.router.late_witness_include = build_witness_router"
        ),
        guards=(
            "route matching is first-registration-wins: if the witness router is "
            "registered after /{config}/subtitles/{type}/{id}/{extra}, the "
            "production aggregator answers real Stremio requests again"
        ),
    ),
    # --- Diagnostic segmentation candidate must stay diagnostic and OFF ------- #
    Mutation(
        name="segmentation-candidate-wired-into-production",
        file="app/services/sync/alignment.py",
        old="from app.services.sync.structural import",
        new=(
            "from app.services.sync.segmentation_pairing import aggregated_pair  "
            "# MUTATION: a diagnostic candidate wired into the verifier\n"
            "from app.services.sync.structural import"
        ),
        guards=(
            "the segmentation-aware candidate is diagnostic only and was measured "
            "as unsafe; it must never become an input to a verification decision"
        ),
    ),
    Mutation(
        name="diagnostic-candidate-declares-an-acceptance-limit",
        file="app/services/sync/segmentation_pairing.py",
        old="TIMING = re.compile(",
        new=(
            "MAX_P95_MS_FOR_STABLE = 2000.0  # MUTATION: an acceptance limit in a\n"
            "# diagnostic module invites a later reviewer to wire it up\n"
            "TIMING = re.compile("
        ),
        guards=(
            "a candidate module must carry no acceptance threshold, or it becomes "
            "an attractive one-line change to production policy"
        ),
    ),
    # --- Forced-production delivery override must stay opt-in -------------- #
    Mutation(
        name="autosync-test-override-on-by-default",
        file="app/autosync_test_override.py",
        old='    raw = os.getenv(ENVVAR, "").strip().lower()\n    if raw in _TRUE:\n        return True',
        new=(
            "    return True  # MUTATION: a delivery override that is on by default\n"
        ),
        guards=(
            "the override substitutes an unverified artifact for production "
            "delivery; it must require an explicit opt-in and never be the default"
        ),
    ),
    Mutation(
        name="autosync-test-override-allowlist-removed",
        file="app/autosync_test_override.py",
        old=(
            "    for case in KNOWN_CASES:\n"
            "        if case.sub_id == sub_id:\n"
            "            return case\n"
            "    return None"
        ),
        new="    return KNOWN_CASES[0]  # MUTATION: any id resolves to a fixture",
        guards=(
            "the override must be restricted to the exact known subtitle ids; "
            "without the allowlist it would substitute a file for any request"
        ),
    ),
    Mutation(
        name="autosync-test-override-digest-check-removed",
        file="app/autosync_test_override.py",
        old="    if actual != case.expected_sha256:",
        new="    if False:  # MUTATION",
            guards=(
                "the fixture is digest-pinned so a substituted or regenerated file is "
                "refused rather than delivered as the validated artifact"
            ),
        ),
    # --- Piecewise shadow model (diagnostic, REJECTED) ---------------------- #
    # The model was measured as unsafe: damaged input scored p95 0 and legitimate
    # offsets were rejected. These mutations keep it unwired and keep the
    # findings that disqualified it from being quietly undone.
    Mutation(
        name="piecewise-shadow-wired-into-production",
        file="app/main.py",
        old="from app.models import Manifest, SubtitleItem, SubtitlesResponse, UserPreferences",
        new=(
            "from app.models import Manifest, SubtitleItem, SubtitlesResponse, UserPreferences\n"
            "from app.services.sync.piecewise_shadow_eval import evaluate_shadow  # MUTATION"
        ),
        guards=(
            "the shadow model is diagnostic only and was rejected as unsafe; it must "
            "never become an input to a production decision"
        ),
    ),
    Mutation(
        name="piecewise-shadow-decisions-look-like-verification",
        file="app/services/sync/piecewise_shadow_eval.py",
        old='    ACCEPT = "PIECEWISE_SHADOW_ACCEPT"',
        new='    ACCEPT = "VERIFIED_SYNCED"  # MUTATION: a shadow label that reads as a verdict',
        guards=(
            "a shadow decision must be unmistakable in a log; renaming it to a real "
            "verification state is how a diagnostic becomes a policy by accident"
        ),
    ),
    Mutation(
        name="piecewise-breakpoint-search-disabled",
        file="app/services/sync/piecewise_shadow.py",
        old="    if max_segments >= 2:\n        layouts += [[b] for b in candidates]",
        new=(
            "    if max_segments >= 2 and False:  # MUTATION: never split, so the\n"
            "        # model can no longer represent a discontinuity at all\n"
            "        layouts += [[b] for b in candidates]"
        ),
        guards=(
            "the whole point of the model is representing a discontinuity; a fit "
            "that can never place a breakpoint would misreport every piecewise case"
        ),
    ),
    Mutation(
        name="piecewise-complexity-penalty-removed",
        file="app/services/sync/piecewise_shadow.py",
        old="COMPLEXITY_PENALTY_MS = 1000.0",
        new="COMPLEXITY_PENALTY_MS = 0.0  # MUTATION: breakpoints become free",
        guards=(
            "without a penalty the model can explain any residual by chopping the "
            "timeline into pieces, which is how a fit stops meaning anything"
        ),
    ),
    Mutation(
        name="piecewise-completeness-folded-into-timing-label",
        file="app/services/sync/piecewise_shadow_eval.py",
        old="    if fit.penalized_p95 <= SHADOW_MAX_PENALIZED_P95_MS:",
        new=(
            "    completeness_threshold_ok = completeness.cue_count_ratio > 0.7  # MUTATION\n"
            "    if fit.penalized_p95 <= SHADOW_MAX_PENALIZED_P95_MS and completeness_threshold_ok:"
        ),
        guards=(
            "timing and completeness must stay separate verdicts; folding a "
            "completeness bound into the timing label is exactly the conflation "
            "the whole experiment concluded against"
        ),
    ),
    Mutation(
        name="piecewise-damaged-input-forced-acceptable",
        file="app/services/sync/piecewise_shadow_eval.py",
        old="SHADOW_MIN_PAIRS = 40",
        new="SHADOW_MIN_PAIRS = 0  # MUTATION: no minimum evidence to fit a field",
        guards=(
            "the abstention floor keeps the model from labelling thin evidence; "
            "removing it lets a handful of pairs produce a confident-looking label"
        ),
    ),
    # --- Anchor-guided correspondence (diagnostic) --------------------------- #
    Mutation(
        name="anchor-correspondence-wired-into-production",
        file="app/main.py",
        old="from app.services.sync.alignment import SubtitleEvaluation",
        new=(
            "from app.services.sync.anchor_correspondence import "
            "anchor_guided_correspondence  # MUTATION\n"
            "from app.services.sync.alignment import SubtitleEvaluation"
        ),
        guards=(
            "the anchor experiment is diagnostic; no production module may import it"
        ),
    ),
    Mutation(
        name="anchor-monotonicity-sign-inverted",
        file="app/services/sync/anchor_correspondence.py",
        old=(
            "        (merged[i + 1].start_ms - merged[i + 1].offset_ms)\n"
            "        > (merged[i].end_ms - 1 - merged[i].offset_ms)"
        ),
        new=(
            "        (merged[i + 1].start_ms + merged[i + 1].offset_ms)\n"
            "        > (merged[i].end_ms - 1 + merged[i].offset_ms)"
        ),
        guards=(
            "offset is target-minus-reference, so the time map must SUBTRACT it; "
            "inverting the sign falsely abstains on every legitimate large offset "
            "(this was a real bug found in the experiment)"
        ),
    ),
    Mutation(
        name="anchor-ambiguity-rejection-removed",
        file="app/services/sync/anchor_correspondence.py",
        old="AMBIGUITY_MS = 2000",
        new="AMBIGUITY_MS = 10**12  # MUTATION: every disagreement counts as agreement",
        guards=(
            "ambiguity must leave a cue unmatched rather than force the nearest "
            "candidate; without the margin the model pairs cues it cannot justify"
        ),
    ),
    Mutation(
        name="anchor-residual-condition-removed",
        file="app/services/sync/anchor_correspondence.py",
        old="    if p95 <= SHADOW_RESIDUAL_BOUND_MS:",
        new="    if True:  # MUTATION: correspondence count alone decides the label",
        guards=(
            "having correspondences is not having good ones; dropping the residual "
            "condition labelled every corrupted fixture GOOD, which is the exact "
            "false match the experiment was built to detect"
        ),
    ),
    Mutation(
        name="anchor-abstention-floor-removed",
        file="app/services/sync/anchor_correspondence.py",
        old="MIN_CORRESPONDENCES = 25",
        new="MIN_CORRESPONDENCES = 0  # MUTATION: label on no evidence at all",
        guards=(
            "no correspondence and poor correspondence must not collapse into one "
            "label; the floor keeps thin or absent evidence an abstention"
        ),
    ),
    Mutation(
        name="anchor-time-map-advance-check-removed",
        file="app/services/sync/anchor_correspondence.py",
        old="    if not mapping.time_map_increasing:",
        new="    if False:  # MUTATION: accept a mapping that folds back in time",
        guards=(
            "a decreasing offset is legitimate but a time map that does not advance "
            "is not; accepting it would let a closing-only offset match the head"
        ),
    ),
    # --- Interval correspondence (diagnostic) ------------------------------ #
    Mutation(
        name="interval-wired-into-production",
        file="app/main.py",
        old="from app.services.sync.alignment import SubtitleEvaluation",
        new=(
            "from app.services.sync.interval_correspondence import "
            "align_intervals  # MUTATION\n"
            "from app.services.sync.alignment import SubtitleEvaluation"
        ),
        guards="the interval experiment is diagnostic; no production module may import it",
    ),
    Mutation(
        name="interval-search-window-widened",
        file="app/services/sync/interval_correspondence.py",
        old="SEARCH_WINDOW_MS = 4000",
        new="SEARCH_WINDOW_MS = 400_000  # MUTATION: effectively unbounded",
        guards=(
            "the window is how far the model is willing to look; widening it turns "
            "a bounded local alignment back into the global nearest-start search "
            "that failed at large offsets"
        ),
    ),
    Mutation(
        name="interval-correspondence-ceiling-removed",
        file="app/services/sync/interval_correspondence.py",
        old="MAX_GROUP_CENTER_MS = 2500",
        new="MAX_GROUP_CENTER_MS = 2500_000_000  # MUTATION: believe anything",
        guards=(
            "the window is how far the model will look, the ceiling is how far it "
            "will believe; without the ceiling a cue displaced by seconds is paired "
            "against the window alone"
        ),
    ),
    # NOTE on the two guards deliberately NOT given mutations.
    #
    # ``MAX_GROUP`` and ``MIN_GROUP_OVERLAP_RATIO`` are retained as defence in
    # depth, but neither can be isolated by a single-point mutation: the objective
    # already forbids the match they forbid. Because SKIP_TARGET_COST (900) is
    # cheaper than MATCH_COST_PER_MS * a 2.5s displacement (2500), the aligner
    # already declines a weak pair regardless of the explicit rule, and because
    # GROUP_EXTRA_COST makes grouping expensive it already prefers several 1:1
    # groups over one oversized one. Removing either constant changed no result.
    #
    # They are kept because they make the intent explicit and they would bite if
    # the objective weights were ever retuned. They are deliberately NOT listed as
    # mutations, because a mutation that cannot be caught is not evidence of
    # anything, and reporting UNPROTECTED for a correctly redundant guard would
    # bury the real findings. The load-bearing guards here are the objective
    # weights -- interval-skip-made-free and interval-group-penalty-removed -- and
    # both are caught.
    Mutation(
        name="interval-anchor-offset-ignored",
        file="app/services/sync/interval_correspondence.py",
        old="        p = c - predict(c)",
        new="        p = c  # MUTATION: the accepted large offset is not used",
        guards=(
            "the whole point is that the accepted large-offset anchors place the "
            "search band; ignoring the offset makes a wrong-release target look "
            "locally alignable"
        ),
    ),
    Mutation(
        name="interval-group-span-unlimited",
        file="app/services/sync/interval_correspondence.py",
        old="    if _span(t_start, t_end) > GROUP_SPAN_MS or _span(r_start, r_end) > GROUP_SPAN_MS:",
        new="    if False:  # MUTATION: no bound on group span",
        guards="grouping must be bounded in time as well as in count",
    ),
    Mutation(
        name="interval-reference-consumption-removed",
        file="app/services/sync/interval_correspondence.py",
        old="                    if nxt < cost[i + kt][r_hi + 1]:\n"
            "                        cost[i + kt][r_hi + 1] = nxt\n"
            "                        back[i + kt][r_hi + 1] = (\"group\", i, j, kt, kr)",
        new="                    if nxt < cost[i + kt][r_hi]:\n"
            "                        cost[i + kt][r_hi] = nxt\n"
            "                        back[i + kt][r_hi] = (\"group\", i, j, kt, kr)",
        guards=(
            "each reference cue may be consumed once; failing to advance past it "
            "lets one reference cue absorb many target cues and destroys "
            "monotonicity"
        ),
    ),
    Mutation(
        name="interval-coverage-accounting-inflated",
        file="app/services/sync/interval_correspondence.py",
        old="        matched_reference=len(matched_reference),",
        new="        matched_reference=nr,  # MUTATION: assume full coverage",
        guards=(
            "reference coverage is the signal that exposes deletion, truncation "
            "and sparsity; assuming it is full reports every damaged input as "
            "fully aligned"
        ),
    ),
    Mutation(
        name="interval-aligned-coverage-bar-removed",
        file="app/services/sync/interval_correspondence.py",
        old="    if report.reference_coverage < ALIGNED_MIN_REFERENCE_COVERAGE:",
        new="    if False:  # MUTATION: anything with a match is 'aligned'",
        guards=(
            "deleting or truncating a target leaves the surviving cues perfectly "
            "aligned; without a coverage bar the model calls that aligned instead "
            "of reporting low reference coverage"
        ),
    ),
    Mutation(
        name="interval-deleted-content-forced-aligned",
        file="app/services/sync/interval_correspondence.py",
        old="    if surplus_r >= SURPLUS_MIN_FRACTION and surplus_r > surplus_t:",
        new="    if False:  # MUTATION: never report missing reference content",
        guards=(
            "a truncated or sparse target must be named as surplus reference "
            "rather than presented as a complete alignment"
        ),
    ),
    Mutation(
        name="interval-content-volume-observation-removed",
        file="app/services/sync/interval_correspondence.py",
        old="    if act > 1.0 + tol and cue > 1.0 + tol:",
        new="    if False:  # MUTATION: content excess never observed",
        guards=(
            "duplication is invisible to timing -- an adjacent duplicate is a legal "
            "split -- so total speech time is the only signal that exposes it; "
            "removing this hides duplication entirely"
        ),
    ),
    Mutation(
        name="interval-skip-made-free",
        file="app/services/sync/interval_correspondence.py",
        old="SKIP_TARGET_COST = 900.0",
        new="SKIP_TARGET_COST = 0.0  # MUTATION: skipping is free, so skip everything",
        guards=(
            "declaring a cue unmatchable must be cheaper than a poor match; making "
            "it free destroys the correspondence and the model reports nothing"
        ),
    ),
    Mutation(
        name="interval-group-penalty-removed",
        file="app/services/sync/interval_correspondence.py",
        old="GROUP_EXTRA_COST = 250.0",
        new="GROUP_EXTRA_COST = 0.0  # MUTATION: grouping becomes free",
        guards=(
            "grouping must cost something, or the model prefers grouping over "
            "one-to-one pairing whenever it lowers the residual"
        ),
    ),
    # --- Anchor grading and content integrity (diagnostic) ------------------ #
    Mutation(
        name="graded-anchor-variant-wired-into-production",
        file="app/main.py",
        old="from app.services.sync.alignment import SubtitleEvaluation",
        new=(
            "from app.services.sync.interval_correspondence import "
            "align_with_graded_anchors  # MUTATION\n"
            "from app.services.sync.alignment import SubtitleEvaluation"
        ),
        guards="the graded-anchor variant is diagnostic; no production module may import it",
    ),
    Mutation(
        name="anchor-grading-rejects-everything",
        file="app/services/sync/interval_correspondence.py",
        old="        if deviation > dispersion_limit_ms:",
        new="        if True:  # MUTATION: every anchor is an outlier",
        guards=(
            "anchor grading must keep the anchors the large-offset gate already "
            "accepted; rejecting them all leaves no band and the DP abstains on "
            "every real large-offset case"
        ),
    ),
    Mutation(
        name="anchor-grading-accepts-everything",
        file="app/services/sync/interval_correspondence.py",
        old="        if deviation > dispersion_limit_ms:",
        new="        if False:  # MUTATION: no anchor is ever an outlier",
        guards=(
            "a grossly wrong anchor must be dropped rather than allowed to bend the "
            "band; accepting everything lets one bad region move the search centre"
        ),
    ),
    Mutation(
        name="anchor-drift-sign-reversed",
        file="app/services/sync/interval_correspondence.py",
        old="        slope = drift_ms_per_minute / 60000.0",
        new="        slope = -drift_ms_per_minute / 60000.0  # MUTATION: wrong direction",
        guards=(
            "drift is measured, not guessed; reversing its sign points the band the "
            "wrong way along the timeline and destroys a drifting alignment"
        ),
    ),
    Mutation(
        name="anchor-drift-ignored-entirely",
        file="app/services/sync/interval_correspondence.py",
        old="        slope = drift_ms_per_minute / 60000.0",
        new="        slope = 0.0  # MUTATION: assume a pure constant shift",
        guards=(
            "the gate measures drift because the shift is not constant; flattening "
            "it reproduces the piecewise-constant model that invented breakpoints "
            "out of EVOLV's smooth ramp"
        ),
    ),
    Mutation(
        name="anchor-contradiction-check-removed",
        file="app/services/sync/interval_correspondence.py",
        old="        contradictory = residual_spread > float(window_ms)",
        new="        contradictory = False  # MUTATION: trust anchors that disagree",
        guards=(
            "anchors that disagree about the trend would place the band where no "
            "anchor is; the model must abstain rather than align against them"
        ),
    ),
    Mutation(
        name="anchor-two-pass-refinement-removed",
        file="app/services/sync/interval_correspondence.py",
        old="    for _ in range(2):",
        new="    for _ in range(1):  # MUTATION: seed pass only, no refinement",
        guards=(
            "the gate re-measures against the median its own anchors produced so "
            "one badly-placed cue cannot decide the outcome; sampling once changes "
            "the measured drift and the anchor set"
        ),
    ),
    Mutation(
        name="anchor-cue-bounding-removed",
        file="app/services/sync/interval_correspondence.py",
        old="    bounded = _bounded(_as_cues(t))",
        new="    bounded = _as_cues(t)  # MUTATION: sample every cue",
        guards=(
            "the gate samples at most LARGE_OFFSET_MAX_SAMPLED_CUES cues; sampling "
            "more changes which cues are measured and moves the reported drift"
        ),
    ),
    Mutation(
        name="content-integrity-wired-into-production",
        file="app/main.py",
        old="from app.services.sync.alignment import SubtitleEvaluation",
        new=(
            "from app.services.sync.content_integrity import "
            "assess_content_integrity  # MUTATION\n"
            "from app.services.sync.alignment import SubtitleEvaluation"
        ),
        guards="the content-integrity shadow is diagnostic; no production module may import it",
    ),
    Mutation(
        name="content-integrity-good-without-text-evidence",
        file="app/services/sync/content_integrity.py",
        old='    if text_ev is not Evidence.USEFUL and report.verdict is ContentIntegrity.GOOD:',
        new='    if False:  # MUTATION: GOOD allowed without same-language evidence',
        guards=(
            "GOOD requires positive evidence that the content corresponds; with a "
            "different-language reference matching volumes prove nothing, so a "
            "same-language check that passes vacuously would certify any file"
        ),
    ),
    Mutation(
        name="content-integrity-suspect-suppressed",
        file="app/services/sync/content_integrity.py",
        old="        if (cue_ratio > 1.0) == (active_ratio > 1.0):\n            report.verdict = ContentIntegrity.SUSPECT",
        new="        if False:  # MUTATION: volume excess never reported",
        guards=(
            "duplication and deletion both move cue count and speech time together; "
            "suppressing the reading hides exactly the damage the module exists "
            "to expose, and duplication is invisible to timing"
        ),
    ),
    Mutation(
        name="content-integrity-segments-treated-as-volume-damage",
        file="app/services/sync/content_integrity.py",
        old="            report.verdict = ContentIntegrity.UNKNOWN\n            report.limits.append(\n                f\"cue count {cue_ratio:.2f}x but speech time {active_ratio:.2f}x \"",
        new="            report.verdict = ContentIntegrity.SUSPECT  # MUTATION\n            report.limits.append(\n                f\"cue count {cue_ratio:.2f}x but speech time {active_ratio:.2f}x \"",
        guards=(
            "a segmentation difference moves cue count and speech time in opposite "
            "directions; calling that content damage would reject every legitimately "
            "re-cut or re-segmented subtitle"
        ),
    ),
    Mutation(
        name="content-integrity-claims-timing-quality",
        file="app/services/sync/content_integrity.py",
        old="    timing_implied: bool = False",
        new="    timing_implied: bool = True  # MUTATION: content verdict implies timing",
        guards=(
            "a subtitle can be perfectly timed and still be the wrong content; the "
            "two verdicts must stay separable or GOOD TIMING hides BAD CONTENT"
        ),
    ),
    # ----------------------------------------------------------------------- #
    # Large Offset serving: the scoped exception must stay narrow
    # ----------------------------------------------------------------------- #
    Mutation(
        name="large-offset-mad-bound-removed",
        file="app/services/sync/large_offset_investigation.py",
        old="    if mad is None or mad > LARGE_OFFSET_MAX_MOVEMENT_MAD_MS:",
        new="    if False:  # MUTATION: any movement may be served",
        guards=(
            "the one measurement the exception is bounded by: a correction that did "
            "not move every cue by the same amount is not a constant shift, and "
            "nothing may be served in the original's place on its behalf"
        ),
    ),
    Mutation(
        name="large-offset-structural-floor-removed",
        file="app/services/sync/large_offset_investigation.py",
        old=(
            "    if structure is None or structure < LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY:"
        ),
        new="    if False:  # MUTATION: any structural agreement may be served",
        guards=(
            "structural disagreement says something about the correction itself, "
            "not about segmentation differences between releases, so it is never "
            "in scope for this path"
        ),
    ),
    Mutation(
        name="large-offset-rejection-scope-widened",
        file="app/services/sync/large_offset_investigation.py",
        old=(
            "    if rejection is not None and rejection is not RejectionReason.LOW_CONFIDENCE:"
        ),
        new="    if False:  # MUTATION: content/structure refusals ignored too",
        guards=(
            "only a confidence-only refusal is in scope; a verdict that refused on "
            "content, structure, cue loss or invalid timing binds, because those "
            "describe the correction rather than the release's segmentation"
        ),
    ),
    Mutation(
        name="large-offset-missing-evaluation-serves",
        file="app/services/sync/large_offset_investigation.py",
        old=(
            "    if evaluation is None:\n"
            "        investigation.serving_reason_codes.append(SERVE_REASON_NO_EVALUATION)\n"
            '        logger.info("large_offset.serving_denied %s", SERVE_REASON_NO_EVALUATION)\n'
            "        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)"
        ),
        new=(
            "    if evaluation is None:\n"
            "        # MUTATION: fail open without the analyzer's measurement\n"
            "        return _record_serving(\n"
            "            investigation,\n"
            "            LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET,\n"
            "        )"
        ),
        guards=(
            "without the analyzer's own measurement there is no way to tell a "
            "segmentation-driven refusal from a structural one, so the original "
            "must be served rather than guessed at"
        ),
    ),
    Mutation(
        name="large-offset-identity-precondition-removed",
        file="app/services/sync/large_offset_investigation.py",
        old="    if not investigation.eligible_for_alass:",
        new="    if False:  # MUTATION: identity precondition ignored",
        guards=(
            "reference-first same-episode identity is the precondition the whole "
            "design rests on; without it the path is an unconditional rescue"
        ),
    ),
    Mutation(
        name="large-offset-override-unconditional",
        file="app/services/sync/orchestrator.py",
        old=(
            "                if (\n"
            "                    serving_state\n"
            "                    is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET\n"
            "                ):"
        ),
        new=(
            "                if large_offset_inv is not None:  # MUTATION: override "
            "bypasses the decision"
        ),
        guards=(
            "reaching the large-offset path at all must not be enough: only the full "
            "decision (identity, valid output, constant movement, structural "
            "agreement, no structural refusal) may override the verifier's refusal"
        ),
    ),
    # ----------------------------------------------------------------------- #
    Mutation(
        name="low-dialogue-anchor-distribution-ignored",
        file="app/services/sync/reference.py",
        old="    if health.dialogue_regions < MIN_DIALOGUE_REGIONS:",
        new="    if False:  # MUTATION: cue count alone decides usability",
        guards=(
            "a reference whose dialogue is numerous but bunched into one stretch "
            "anchors nothing away from that stretch; loosening the coverage floor is "
            "only sound while the evidence still has to reach across the runtime"
        ),
    ),
)


#: Hard ceiling on a single pytest run inside the harness. The full host suite
#: takes ~50s, so this is generous -- but it exists so that a stalled child can
#: never block the matrix indefinitely. A timeout is reported as
#: ``MUTATION_TIMEOUT`` and is never counted as a catch.
MUTATION_RUN_TIMEOUT_SECONDS = 180

#: Hard ceiling on the whole matrix, so a systemic problem cannot run for hours
#: without the operator noticing. Individual runs are bounded by the value above.
MATRIX_TIMEOUT_SECONDS = 3600

#: Sentinel exit code meaning "the child had to be killed for exceeding its
#: timeout". Distinct from any real pytest exit code, so it can never be
#: confused with an assertion failure.
TIMEOUT_EXIT = -9


def _run(
    cmd: list[str],
    cwd: Path,
    timeout: int = MUTATION_RUN_TIMEOUT_SECONDS,
) -> tuple[int, str]:
    """Run a command with a hard timeout. Returns ``(exit_code, output)``.

    ``subprocess.run`` without ``timeout`` waits forever. That is what made an
    earlier run of this tool look hung: 90 mutations x (full copytree + full
    ~50s suite) is over 80 minutes of serial work with buffered output and no
    progress reporting, and nothing could interrupt it.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):  # pragma: no cover - platform dependent
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):  # pragma: no cover - platform dependent
            err = err.decode("utf-8", "replace")
        return TIMEOUT_EXIT, out + err + (
            f"\n[run_safety_mutations] TIMEOUT after {timeout}s: {' '.join(cmd)}\n"
        )
    return proc.returncode, proc.stdout + proc.stderr


def _apply(tree: Path, mutation: Mutation) -> bool:
    path = tree / mutation.file
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if mutation.old not in text:
        return False
    path.write_text(text.replace(mutation.old, mutation.new, 1), encoding="utf-8")
    return True


def _module_imported_by(text: str) -> str | None:
    """First ``app.services[...].X`` module named in a snippet, if any.

    Wiring mutations edit a production file to import the experiment, so the
    module the mutation is *about* is named in the replacement text rather than
    being the file being edited.
    """
    m = re.search(r"app\.services\.[a-z_]*\.?([a-z_]+)\s+import", text)
    return m.group(1) if m else None


def _subject_stem(mutation: Mutation) -> str:
    """Module a mutation is about, used to pick its guarding tests."""
    return (
        _module_imported_by(mutation.new)
        or _module_imported_by(mutation.old)
        or Path(mutation.file).stem
    )


#: Used only when no test file mentions the subject module. Deliberately small:
#: these are the tests that assert diagnostics stay unwired and that the model
#: never touches production state.
FALLBACK_GUARD_TESTS = (
    "tests/test_interval_correspondence.py",
    "tests/test_content_integrity_and_anchors.py",
)


def guard_tests(mutation: Mutation) -> list[str]:
    """The smallest intended regression test selection for one mutation.

    Running the whole repository suite per mutation is what made this tool
    unusable: ~50s x 90 mutations. Instead the tests that actually reference the
    module under mutation are selected. That is not a weakening -- the guarding
    test must mention the module in order to assert anything about it -- and the
    subset's baseline is verified independently below.
    """
    stem = _subject_stem(mutation)

    def mentioning(needle: str) -> list[str]:
        out = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            try:
                body = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - unreadable file
                continue
            if needle in body:
                out.append(str(path.relative_to(ROOT)))
        return out

    # Prefer tests that mention the *subject module*. Wiring mutations edit
    # app/main.py, which almost every test file references; matching on the
    # edited file alone would drag in ~47 files for a guard that only needs the
    # three that assert the experiment stays unwired.
    by_stem = mentioning(stem)
    if by_stem:
        return by_stem
    return mentioning(mutation.file) or list(FALLBACK_GUARD_TESTS)


def _failing_tests(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.startswith("FAILED")]


def _node_id(failed_line: str) -> str:
    """``FAILED path::test - reason`` -> ``path::test``."""
    return failed_line[len("FAILED ") :].split(" - ")[0].strip()


def verify_subset_baseline(
    tests: list[str], python: str
) -> tuple[set[str], str]:
    """Establish the baseline for exactly the tests this mutation will run.

    Returns the set of node ids that already fail in the unmutated tree, plus a
    note. Those are deselected from the mutated run, so a pre-existing failure can
    never be credited as a catch -- the original bug, which was that ``pytest -x``
    stopped on a host-only ``guessit`` failure and every guard sorting after it
    read as CAUGHT while green.

    Scoping the deselection to the subset rather than the whole suite is what
    makes this cheap: subsets are 1-3 files for most mutations, and the result is
    cached per distinct subset.
    """
    missing = [t for t in tests if not (ROOT / t).exists()]
    if missing:
        return set(), f"MISSING guard test file(s): {missing}"
    code, output = _run(
        [python, "-m", "pytest", "-q", "--no-header", *tests], ROOT
    )
    if code == 0:
        return set(), "green"
    if code == TIMEOUT_EXIT:
        return set(), "BASELINE_TIMEOUT"
    failed = {_node_id(t) for t in _failing_tests(output)}
    if not failed:
        return set(), "non-zero exit without a named failure"
    return failed, f"{len(failed)} pre-existing failure(s) in subset"


def baseline_failures(python: str) -> set[str]:
    """Deprecated global baseline scan. Retained for reference only.

    The matrix no longer calls this. It cost a full ~50s suite run and existed
    only to deselect three host-only `guessit` failures from every mutation.
    Per-subset baseline verification (:func:`verify_subset_baseline`) replaces
    it and cannot be fooled by ordering.
    """
    with tempfile.TemporaryDirectory(prefix="ninjasubs-mutbase-") as tmp:
        tree = Path(tmp) / "tree"
        shutil.copytree(
            ROOT,
            tree,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "subs_cache", "cache", ".pytest_cache"
            ),
        )
        code, output = _run(
            [python, "-m", "pytest", "-q", "--no-header", "-p", "no:randomly"],
            tree,
            timeout=MUTATION_RUN_TIMEOUT_SECONDS,
        )
        if code in (0, TIMEOUT_EXIT):
            return set()
        return {_node_id(t) for t in _failing_tests(output)}


def check(
    mutation: Mutation,
    python: str,
    subset_baseline: set[str] | None = None,
    baseline_detail: str = "",
) -> dict[str, object]:
    """Apply one mutation in a temp copy; the guard tests MUST fail, newly.

    ``subset_baseline`` holds node ids that already failed before the mutation.
    They are deselected from the mutated run, and any remaining failure is
    required to be new, so an unrelated pre-existing failure can never be
    credited as a catch.
    """
    result: dict[str, object] = {
        "mutation": mutation.name,
        "file": mutation.file,
        "guards": mutation.guards,
    }
    tests = guard_tests(mutation)
    result["guard_tests"] = tests
    baseline = set(subset_baseline or ())

    if baseline_detail.startswith(("MISSING", "BASELINE_TIMEOUT")):
        result.update(
            status="BASELINE_ERROR",
            detail=f"could not establish a baseline for {tests} ({baseline_detail})",
        )
        return result

    with tempfile.TemporaryDirectory(prefix="ninjasubs-mut-") as tmp:
        tree = Path(tmp) / "tree"
        # Copy the working tree without .git or the test cache, so a mutation
        # can never touch the real checkout or the real subs_cache.
        shutil.copytree(
            ROOT,
            tree,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "subs_cache", "cache", ".pytest_cache"
            ),
        )
        if not _apply(tree, mutation):
            result.update(
                status="SKIPPED", detail="anchor text not found; mutation is stale"
            )
            return result
        cmd = [python, "-m", "pytest", "-q", "-x", "--no-header"]
        for node in sorted(baseline):
            cmd += ["--deselect", node]
        cmd += tests
        code, output = _run(cmd, tree)

        # A timeout is NOT a catch. The child was killed for exceeding its bound,
        # which tells us nothing about whether any guard fired.
        if code == TIMEOUT_EXIT:
            result.update(
                status="MUTATION_TIMEOUT",
                detail=(
                    f"guard run exceeded {MUTATION_RUN_TIMEOUT_SECONDS}s and was "
                    "killed; not a catch"
                ),
            )
            return result

        # A mutation must be caught by a guard *asserting*, not by the module
        # failing to load. Collection and import failures kill the run for a
        # reason unrelated to any guard, and counting them as catches would let a
        # genuinely missing guard hide behind a crash. Each failure kind is named
        # so a false catch cannot be mistaken for a real one.
        if code == 0:
            result.update(
                status="UNPROTECTED",
                detail=(
                    f"guard tests stayed green after the safety behaviour was "
                    f"removed: {tests}"
                ),
            )
            return result

        if "SyntaxError" in output or "IndentationError" in output:
            result.update(
                status="ERROR",
                detail="mutation broke the module syntactically; not a guard catch",
            )
            return result
        if "ModuleNotFoundError" in output or "ImportError" in output:
            result.update(
                status="ERROR",
                detail="mutation broke an import; not a guard catch",
            )
            return result
        if "ERROR collecting" in output or "errors during collection" in output:
            result.update(
                status="ERROR",
                detail="mutation broke collection; not a guard catch",
            )
            return result

        failed = _failing_tests(output)
        fresh = [t for t in failed if _node_id(t) not in baseline]
        if code == 1 and fresh:
            result.update(
                status="CAUGHT",
                detail=f"{len(fresh)} guard test(s) failed as required",
                failing_tests=fresh[:5],
            )
        elif code == 1 and failed and not fresh:
            result.update(
                status="UNPROTECTED",
                detail=(
                    "the only failure was already failing before the mutation, so "
                    "no guard actually caught this"
                ),
                failing_tests=failed[:3],
            )
        elif code == 1:
            result.update(
                status="ERROR",
                detail="suite failed without naming a test; not a guard assertion",
            )
        else:
            result.update(
                status="ERROR",
                detail=f"pytest exited {code} without a named failure",
            )
        return result


def main() -> int:
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_path", help="write a machine-readable report")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--only",
        dest="only",
        help="run only mutations whose name contains this substring (checkpointing)",
    )
    args = ap.parse_args()

    selected = list(MUTATIONS)
    if args.only:
        selected = [m for m in MUTATIONS if args.only in m.name]
        if not selected:
            print(f"  no mutation name contains {args.only!r}")
            return 2

    print("=" * 74)
    print("SAFETY MUTATION MATRIX")
    print("=" * 74)
    print(
        f"  {len(selected)} mutation(s); guard subset per mutation; "
        f"per-run timeout {MUTATION_RUN_TIMEOUT_SECONDS}s; "
        f"matrix budget {MATRIX_TIMEOUT_SECONDS}s"
    )

    # Baseline verification, cached per distinct guard subset. A subset that is
    # not green before mutation is BASELINE_ERROR and is never scored, so an
    # unrelated failure cannot be credited as a catch.
    subset_cache: dict[tuple[str, ...], tuple[set[str], str]] = {}
    results = []
    for i, m in enumerate(selected, 1):
        if time.monotonic() - started > MATRIX_TIMEOUT_SECONDS:
            print(
                f"\n  MATRIX BUDGET EXCEEDED after {i - 1}/{len(selected)} "
                "mutations; stopping"
            )
            break
        key = tuple(guard_tests(m))
        if key not in subset_cache:
            subset_cache[key] = verify_subset_baseline(list(key), args.python)
        pre_existing, detail = subset_cache[key]
        r = check(m, args.python, pre_existing, detail)
        results.append(r)
        # Progress is flushed as it happens: a silent multi-minute run is what
        # made this tool impossible to supervise.
        print(
            f"  [{i:>3}/{len(selected)}] [{r['status']:<17}] "
            f"{m.name:<52} {len(key)} file(s)"
            f"{f', {len(pre_existing)} pre-existing' if pre_existing else ''}",
            flush=True,
        )

    width = max(len(str(r["mutation"])) for r in results)
    print("-" * 74)
    for r in results:
        print(f"  [{r['status']:<17}] {r['mutation']:<{width}}  {r['detail']}")
        if r.get("failing_tests"):
            for t in r["failing_tests"]:
                print(f"                 - {t}")

    caught = sum(1 for r in results if r["status"] == "CAUGHT")
    unprotected = [r["mutation"] for r in results if r["status"] == "UNPROTECTED"]
    skipped = [r["mutation"] for r in results if r["status"] == "SKIPPED"]
    errored = [r["mutation"] for r in results if r["status"] == "ERROR"]
    timed_out = [r["mutation"] for r in results if r["status"] == "MUTATION_TIMEOUT"]
    baseline_bad = [
        r["mutation"] for r in results if r["status"] == "BASELINE_ERROR"
    ]

    print("-" * 74)
    print(f"  caught {caught}/{len(results)}  in {time.monotonic() - started:.0f}s")
    if unprotected:
        print(f"  UNPROTECTED (release blocker): {', '.join(unprotected)}")
    if timed_out:
        print(f"  MUTATION_TIMEOUT (not a catch): {', '.join(timed_out)}")
    if baseline_bad:
        print(f"  BASELINE_ERROR (not scored): {', '.join(baseline_bad)}")
    if skipped:
        print(f"  stale anchors, review the mutation: {', '.join(skipped)}")
    if errored:
        print(f"  errors, review the mutation: {', '.join(errored)}")
    print("=" * 74)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "results": results,
                    "caught": caught,
                    "total": len(results),
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "run_timeout_s": MUTATION_RUN_TIMEOUT_SECONDS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  report written to {args.json_path}")

    # UNPROTECTED and MUTATION_TIMEOUT both block: a timeout proves nothing
    # about the guard, so it must never read as a pass.
    return 1 if (unprotected or timed_out or baseline_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
