"""
Subtitle Ranking Audit & Diagnostic Tool for NinjaSubs.

Provides deep inspection of the two-stage matcher:
1. Hard compatibility filter decisions (accepted/rejected and fatal desync reason).
2. Soft compatibility ranking score and detailed component breakdown:
   - title, season/episode, group, source, service, edition, fps, resolution, codec, audio, hash.
3. Ranking order across candidates with confidence and reasons.
4. Performance timing breakdown (parsing, filtering, scoring, sorting, total).
"""

import argparse
import os
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import SubtitleRelease
from app.services.subtitle_matcher import (
    WEIGHT_EXACT_HASH,
    calculate_compatibility,
    extract_metadata,
    hard_compatibility_filter,
    rank_subtitles,
)


def extract_score_components(reasons: list[str]) -> dict[str, int]:
    """
    Parse reasons list into structured attribute score components.
    Uses regex to extract signed integer points from reason strings like '(+40)' or '(-30)'.
    """
    components = {
        "title": 0,
        "season_episode": 0,
        "year": 0,
        "group": 0,
        "source": 0,
        "service": 0,
        "edition": 0,
        "fps": 0,
        "resolution": 0,
        "codec": 0,
        "audio": 0,
        "remux_repack": 0,
        "hash": 0,
    }

    point_regex = re.compile(r"\(([+-]?\d+)\)")

    for r in reasons:
        m = point_regex.search(r)
        pts = int(m.group(1)) if m else 0
        r_lower = r.lower()

        if "hash" in r_lower:
            components["hash"] += WEIGHT_EXACT_HASH
        elif "title" in r_lower:
            components["title"] += pts
        elif "season" in r_lower or "episode" in r_lower:
            components["season_episode"] += pts
        elif "year" in r_lower:
            components["year"] += pts
        elif "fansub" in r_lower or "group" in r_lower or "scene" in r_lower:
            components["group"] += pts
        elif "source" in r_lower or "cam" in r_lower or "disc" in r_lower or "web" in r_lower:
            components["source"] += pts
        elif (
            "platform" in r_lower
            or "streaming" in r_lower
            or "hmax" in r_lower
            or "amzn" in r_lower
            or "netflix" in r_lower
        ):
            components["service"] += pts
        elif "edition" in r_lower or "cut" in r_lower:
            components["edition"] += pts
        elif "fps" in r_lower:
            components["fps"] += pts
        elif "resolution" in r_lower:
            components["resolution"] += pts
        elif "codec" in r_lower or "hdr" in r_lower or "bit" in r_lower or "vision" in r_lower:
            components["codec"] += pts
        elif "audio" in r_lower or "atmos" in r_lower or "dts" in r_lower or "ddp" in r_lower:
            components["audio"] += pts
        elif "remux" in r_lower or "repack" in r_lower:
            components["remux_repack"] += pts

    return {k: v for k, v in components.items() if v != 0}


def audit_ranking(
    target_video: str,
    candidates: list[Any],
    preferred_languages: list[str] | None = None,
    discard_mismatches: bool = False,
    exclude_sdh: bool = False,
    expected_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute ranking audit for a target release against subtitle candidates.
    Measures timing metrics and outputs diagnostic details.
    """
    pref_langs = preferred_languages or ["ara"]

    t0 = time.perf_counter()

    # 1. Metadata parsing timing
    t_parse_start = time.perf_counter()
    v_meta = extract_metadata(target_video)
    cand_metas = []
    for c in candidates:
        name = getattr(c, "release_name", None) or (
            c.get("release_name") if isinstance(c, dict) else str(c)
        )
        cand_metas.append(extract_metadata(name or ""))
    t_parse_end = time.perf_counter()
    parse_time_ms = (t_parse_end - t_parse_start) * 1000

    # 2. Hard filter timing
    t_filt_start = time.perf_counter()
    hard_filter_results = []
    for cm in cand_metas:
        hard_filter_results.append(hard_compatibility_filter(v_meta, cm))
    t_filt_end = time.perf_counter()
    filter_time_ms = (t_filt_end - t_filt_start) * 1000

    # 3. Soft scoring timing
    t_score_start = time.perf_counter()
    compat_results = []
    for c, cm in zip(candidates, cand_metas, strict=False):
        is_h = bool(
            getattr(c, "is_hash_match", False)
            if hasattr(c, "is_hash_match")
            else (c.get("is_hash_match", False) if isinstance(c, dict) else False)
        )
        compat_results.append(calculate_compatibility(v_meta, cm, is_hash_match=is_h))
    t_score_end = time.perf_counter()
    score_time_ms = (t_score_end - t_score_start) * 1000

    # 4. Total rank_subtitles call timing
    t_rank_start = time.perf_counter()
    sub_objects = []
    for idx, c in enumerate(candidates, 1):
        if isinstance(c, SubtitleRelease):
            sub_objects.append(c)
        elif isinstance(c, dict):
            sub_objects.append(SubtitleRelease(**c))
        else:
            c_str = str(c)
            c_lang = "eng" if "english" in c_str.lower() or ".en." in c_str.lower() else "ara"
            sub_objects.append(
                SubtitleRelease(
                    release_name=c_str,
                    download_url=f"http://mock/{idx}.srt",
                    provider=f"mock_{idx}",
                    lang=c_lang,
                )
            )

    ranked = rank_subtitles(
        video_filename=target_video,
        subtitles=sub_objects,
        preferred_languages=pref_langs,
        discard_mismatches=discard_mismatches,
        exclude_sdh=exclude_sdh,
    )
    t_rank_end = time.perf_counter()
    total_time_ms = (t_rank_end - t0) * 1000
    sort_time_ms = (t_rank_end - t_rank_start) * 1000

    n_candidates = len(candidates)
    avg_per_cand_ms = total_time_ms / max(1, n_candidates)

    audit_records = []
    for rank_idx, item in enumerate(ranked, 1):
        rel_name = item.release_name
        compat = getattr(item, "compatibility", None) or calculate_compatibility(
            v_meta, rel_name, is_hash_match=item.is_hash_match
        )
        components = extract_score_components(compat.reasons)

        record = {
            "rank": rank_idx,
            "release": rel_name,
            "language": getattr(item, "lang", "ara"),
            "accepted": compat.accepted,
            "score": compat.score,
            "confidence": compat.confidence,
            "percentage": compat.percentage,
            "hash_match": compat.is_hash_match,
            "match_method": compat.match_method,
            "hard_reject_reason": compat.hard_reject_reason,
            "reasons": compat.reasons,
            "components": components,
        }
        audit_records.append(record)

    # Check for anomalies against expected ordering if provided
    anomalies = []
    if expected_order:
        actual_names = [r["release"] for r in audit_records]
        for exp_idx, exp_name in enumerate(expected_order):
            if exp_name in actual_names:
                act_idx = actual_names.index(exp_name)
                if act_idx != exp_idx:
                    anomalies.append(
                        {
                            "release": exp_name,
                            "expected_rank": exp_idx + 1,
                            "actual_rank": act_idx + 1,
                            "reason": f"Expected rank {exp_idx + 1} but got {act_idx + 1}",
                        }
                    )
            else:
                anomalies.append(
                    {
                        "release": exp_name,
                        "expected_rank": exp_idx + 1,
                        "actual_rank": None,
                        "reason": "Expected candidate was discarded or not in ranked output",
                    }
                )

    return {
        "target_video": target_video,
        "target_metadata": v_meta,
        "candidate_count": n_candidates,
        "ranked_count": len(ranked),
        "parse_time_ms": round(parse_time_ms, 3),
        "filter_time_ms": round(filter_time_ms, 3),
        "score_time_ms": round(score_time_ms, 3),
        "sort_time_ms": round(sort_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
        "avg_per_candidate_ms": round(avg_per_cand_ms, 3),
        "records": audit_records,
        "anomalies": anomalies,
    }


def format_audit_report(result: dict[str, Any]) -> str:
    """Format audit result into human-readable diagnostic text."""
    lines = []
    lines.append("=" * 80)
    lines.append("TARGET:")
    lines.append(f"  {result['target_video']}")
    lines.append("TIMING:")
    lines.append(
        f"  Candidates: {result['candidate_count']} | Total Time: {result['total_time_ms']} ms | Avg/Cand: {result['avg_per_candidate_ms']} ms"
    )
    lines.append(
        f"  Breakdown: parse={result['parse_time_ms']}ms, filter={result['filter_time_ms']}ms, score={result['score_time_ms']}ms, sort={result['sort_time_ms']}ms"
    )
    lines.append("-" * 80)

    for r in result["records"]:
        lines.append(f"{r['rank']}. {r['release']}")
        lines.append(f"   accepted={r['accepted']}")
        if not r["accepted"]:
            lines.append(f"   hard_reject_reason={r['hard_reject_reason']}")
        lines.append(f"   score={r['score']}")
        lines.append(f"   confidence={r['confidence']}")
        lines.append(f"   percentage={r['percentage']}%")
        lines.append(f"   hash_match={r['hash_match']}")
        lines.append(f"   match_method={r['match_method']}")
        if r["components"]:
            comp_str = ", ".join(f"{k}={v:+d}" for k, v in r["components"].items())
            lines.append(f"   components: [{comp_str}]")
        lines.append(f"   reasons: {r['reasons']}")
        lines.append("")

    if result["anomalies"]:
        lines.append("ANOMALIES DETECTED:")
        for a in result["anomalies"]:
            lines.append(f"  ! {a['release']}: {a['reason']}")
    else:
        lines.append("NO ANOMALIES DETECTED.")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit subtitle matching and ranking")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX",
        help="Target video filename",
    )
    parser.add_argument("--candidates", "-c", nargs="+", help="Candidate subtitle filenames")
    parser.add_argument(
        "--langs", "-l", nargs="+", default=["ara", "eng"], help="Preferred languages"
    )
    parser.add_argument("--discard", action="store_true", help="Discard rejected mismatches")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run 50+ candidate performance benchmark"
    )
    args = parser.parse_args()

    if args.benchmark:
        sources = ["WEB-DL", "WEBRip", "BluRay", "Remux", "HDTV"]
        groups = ["FLUX", "NTb", "CMRG", "SPARKS", "ION10", "PSA", "AVS", "MiNX"]
        resolutions = ["2160p", "1080p", "720p"]
        services = ["HMAX", "AMZN", "NF", "DSNP"]
        bench_candidates = []
        idx = 1
        for src in sources:
            for grp in groups:
                for res in resolutions:
                    svc = services[idx % len(services)]
                    bench_candidates.append(
                        f"House.of.the.Dragon.S02E01.{res}.{svc}.{src}.x264-{grp}.srt"
                    )
                    idx += 1
                    if len(bench_candidates) >= 60:
                        break
                if len(bench_candidates) >= 60:
                    break
            if len(bench_candidates) >= 60:
                break
        sample_candidates = bench_candidates
    else:
        sample_candidates = args.candidates or [
            "House of the Dragon S02E01 HMAX WEB-DL FLUX Arabic.srt",
            "House of the Dragon S02E01 AMZN WEB-DL NTb Arabic.srt",
            "House of the Dragon S02E02 HMAX WEB-DL Arabic.srt",
            "House of the Dragon S02E01 HMAX WEBRip Arabic.srt",
            "House of the Dragon S02E01 HMAX WEB-DL FLUX English.srt",
        ]

    audit_result = audit_ranking(
        target_video=args.target,
        candidates=sample_candidates,
        preferred_languages=args.langs,
        discard_mismatches=args.discard,
    )
    print(format_audit_report(audit_result))
