"""Deterministic generator for the synthetic golden synchronization dataset.

The fixtures are small text files, not media. A target fixture is the timing
model of an imagined video: its cue starts are that video's speech onsets. A
candidate fixture is a subtitle carrying whatever timing defect the case is
about. This is enough to exercise the real analyzer and the real verification
thresholds against a known-correct answer, and it commits no copyrighted media.

Ground truth is written by hand in the manifest below. Nothing in this file
consults the analyzer, the predictor or the trust model, and the generator
refuses to run if a case's declared failure mode does not match the defect it
actually applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURES = REPO / "tests" / "fixtures" / "golden_sync"
SUBTITLES = FIXTURES / "subtitles"

DIALOGUE = "I never thought I would find someone like you"


def _ts(ms: int) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _srt(starts: list[int], *, duration: int = 1500, body: str = DIALOGUE) -> str:
    return "\n".join(
        f"{i}\n{_ts(s)} --> {_ts(s + duration)}\n{body}\n" for i, s in enumerate(starts, 1)
    )


def onsets(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


def offset(starts: list[int], delta_ms: int) -> list[int]:
    return [s + delta_ms for s in starts]


def drift(starts: list[int], rate: float) -> list[int]:
    return [int(s * rate) for s in starts]


def collapse_every_third(starts: list[int]) -> list[int]:
    """Drop two of every three cues: a structurally similar but lossy subtitle."""
    kept: list[int] = []
    for index, start in enumerate(starts):
        if index % 3 != 2:
            kept.append(start)
    return kept


def silence_structure(last_ms: int, *, block: int, gap: int) -> list[int]:
    """A plausible target-video silence structure.

    Dialogue blocks separated by silences, which is the shape silencedetect
    reports for ordinary programme material. Declared as ground truth so the
    validator is handed the video's structure as a fact rather than inferring
    it from the subtitle it is judging.
    """
    landmarks: list[int] = []
    position = block
    while position < last_ms + block:
        landmarks.append(position)
        landmarks.append(position + gap)
        position += block + gap
    return landmarks


def activity_cue_starts(segments: list[tuple[str, int]]) -> list[int]:
    """Cue starts implied by an (kind, duration_ms) activity structure."""
    out: list[int] = []
    position = 0
    for kind, length in segments:
        if kind == "tone":
            out.append(position + 1_000)
            out.append(position + length - 500)
        position += length
    return out


def resplit(starts: list[int]) -> list[int]:
    """Same instants, different cue boundaries: timing-identical, structurally
    different. A structural comparator must not mistake this for a mismatch."""
    out: list[int] = []
    for start in starts:
        out.append(start)
    return sorted(set(out))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- case construction ----------------------------------------------------- #

BLURAY_NAME = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
WEBDL_NAME = "Dexter.S08E05.1080p.WEB-DL.DDP5.1.H.264-NTb.mkv"
WEBRIP_NAME = "Dexter.S08E05.720p.WEBRip.x264-GRP.mkv"
HDTV_NAME = "Dexter.S08E05.720p.HDTV.x264-KILLERS.mkv"
TV_NAME = "Dexter.S08E05.720p.HDTV.x264-2HD.mkv"

VOTEDUB = "The body of the subtitle is a translation, not the original line."


def build() -> dict:
    """Return the manifest, writing every subtitle fixture to disk."""
    SUBTITLES.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    files: dict[str, str] = {}

    def add(
        case_id: str,
        *,
        failure_mode: str,
        release_class: str,
        target_name: str,
        target_starts: list[int],
        candidates: list[tuple[str, str, str, dict, dict]],
        source: str = "objective",
        valid_reference_keys: list[str] | None = None,
        notes: str | None = None,
        video_landmarks: list[int] | None = None,
    ) -> None:
        target_text = _srt(target_starts)
        target_rel = f"subtitles/{case_id}__target.srt"
        files[target_rel] = target_text

        cand_entries: list[dict] = []
        truths: list[dict] = []
        for provider, release_name, payload, truth, alignment in candidates:
            rel = f"subtitles/{case_id}__{provider}__{release_name}.srt"
            files[rel] = payload
            cand_entries.append(
                {
                    "provider": provider,
                    "release_name": release_name,
                    "path": rel,
                    "sha256": _sha(payload),
                    "language": "eng",
                }
            )
            truths.append(
                {
                    "provider": provider,
                    "release_name": release_name,
                    "alignment": alignment,
                    **truth,
                }
            )

        cases.append(
            {
                "case_id": case_id,
                "failure_mode": failure_mode,
                "release_class": release_class,
                "annotation_source": source,
                "target": {
                    "path": target_rel,
                    "filename": target_name,
                    "sha256": _sha(target_text),
                    "duration_ms": target_starts[-1] + 5000,
                    "fingerprint_complete": target_name is not None,
                    **(
                        {
                            "video": {
                                "source": "declared",
                                "duration_ms": video_landmarks[-1] + 5000,
                                "audio_stream_count": 1,
                                "video_stream_count": 1,
                                "audio_landmarks": video_landmarks,
                                "video_landmarks": video_landmarks,
                            }
                        }
                        if video_landmarks
                        else {}
                    ),
                },
                "candidates": cand_entries,
                "ground_truth": truths,
                "valid_reference_keys": valid_reference_keys or [],
                **({"notes": notes} if notes else {}),
            }
        )

    base = onsets(60)
    # The target video's own silence structure, declared as ground truth. Only
    # the cases where an independent video witness exists carry it; the rest
    # leave the validator to abstain, which is the production situation.
    video_timeline = silence_structure(base[-1], block=24_000, gap=4_000)
    exact_name = "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt"
    other_name = "Dexter.S08E05.1080p.BluRay.x264-OtherGRP.srt"
    webdl_name = "Dexter.S08E05.1080p.WEB-DL.x264-NTb.srt"
    hdtv_name = "Dexter.S08E05.720p.HDTV.x264-KILLERS.srt"
    wrong_ep = "Dexter.S08E06.1080p.BluRay.x264-PiR8.srt"

    def rel(name: str) -> str:
        return name.rsplit(".", 1)[0] + ".srt"

    # Declared alignment outcomes. These are inputs to the verifier, written
    # down so a reader can see exactly what it was shown.
    A_CLEAN = {"mode": "as_target", "residual_ms": 0, "annotation_note": "Perfect alignment to the true timing."}
    A_JITTER = {"mode": "as_target", "residual_ms": 350, "annotation_note": "Successful aligner with ordinary residual jitter."}
    A_DRIFT = {"mode": "as_target_with_drift", "residual_ms": 300, "drift_per_minute_ms": 420.0, "annotation_note": "Aligned, but a rate mismatch leaves progressive drift."}
    A_LOSSY = {"mode": "as_target", "residual_ms": 300, "coverage": 0.67, "annotation_note": "Aligned, but a third of the cues were lost in the process."}
    A_WRONG_CUT = {"mode": "partial_misalign", "displaced_from_index": 30, "displaced_by_ms": 12000, "annotation_note": "Exit 0 with small residuals, but the back half belongs to a different cut. The signature of a wrong-cut alignment."}
    A_WRONG_EPISODE = {"mode": "partial_misalign", "displaced_from_index": 5, "displaced_by_ms": 9000, "annotation_note": "Exit 0, plausible early cues, then the timeline jumps to another episode."}
    A_NO_ALIGN = {"mode": "no_alignment", "annotation_note": "The aligner produced nothing usable."}

    # --- 1. same release, already synchronized: the easy positive ---------- #
    add(
        "same_release_already_synced",
        failure_mode="already_synced",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(base),
             {"content": "same_release", "cut": "same_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "Cue starts identical to the target timing model."},
             A_CLEAN),
        ],
        valid_reference_keys=[f"subdl|{rel(exact_name)}"],
    )

    # --- 2. constant offset: a classic hard positive ----------------------- #
    add(
        "same_release_constant_offset",
        failure_mode="offset",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(offset(base, 7_500)),
             {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "Uniform 7.5s shift, known and deliberate. Correct after re-alignment, incorrect before it."},
             A_JITTER),
        ],
    )

    # --- 3. FPS drift: aligned, but a rate mismatch survives ---------------- #
    add(
        "same_release_fps_drift",
        failure_mode="drift",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=onsets(60, start_ms=20_000),
        candidates=[
            ("subsource", rel(exact_name), _srt(drift(onsets(60, start_ms=20_000), 0.96)),
             {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "Cue starts stretched by 0.96, standing in for a 25/23.976 fps conversion."},
             A_DRIFT),
        ],
    )

    # --- 4. same content, different release, large stable offset ----------- #
    add(
        "different_release_large_offset",
        failure_mode="offset_different_release",
        release_class="webdl",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(webdl_name), _srt(offset(onsets(40, 30_000), 180_000)),
             {"content": "same_content_different_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "WEB-DL cut of the same episode with 3 minutes of opening material."},
             A_JITTER),
        ],
    )

    # --- 5. different cut: the hard negative that must not verify ---------- #
    add(
        "same_release_different_cut",
        failure_mode="different_cut",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(onsets(60, 2_000, start_ms=95_000)),
             {"content": "same_release", "cut": "different_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "incorrect",
              "annotation_note": ("Identical release name, episode code, group and resolution, but the segment "
                                   "starts 95s later: an alternate cut. Structurally near-identical and "
                                   "correctly synced to its OWN cut. Any verified claim against this target "
                                   "is a false positive.")},
             A_WRONG_CUT),
        ],
        notes="Same title, season, episode, release group and resolution; wrong cut.",
        video_landmarks=video_timeline,
    )

    # --- 6. wrong episode: a negative that looks perfect by metadata -------- #
    add(
        "wrong_episode",
        failure_mode="wrong_episode",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subsource", rel(wrong_ep), _srt(base),
             {"content": "different_content", "cut": "unknown", "original_sync": "already_synced",
              "resyncable": False, "final_alignment": "incorrect",
              "annotation_note": "S08E06 subtitle carrying S08E05 timing. Perfectly synchronized to the wrong episode."},
             A_WRONG_EPISODE),
        ],
    )

    # --- 7. credits-only: usable only as a timing anchor, not as content --- #
    add(
        "credits_only",
        failure_mode="credits_only",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(base, body="sync by someone"),
             {"content": "same_release", "cut": "same_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": ("Timing is exact, but every line is a credit. A legitimate "
                                  "anchor with no dialogue to align.")},
             A_CLEAN),
        ],
    )

    # --- 8. cue collapse: structurally similar, information lossy ----------- #
    add(
        "cue_collapse",
        failure_mode="cue_collapse",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(collapse_every_third(base)),
             {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "Two of every three cues dropped by the aligner."},
             A_LOSSY),
        ],
    )

    # --- 9. sparse dialogue: too few cues to justify a claim --------------- #
    add(
        "sparse_dialogue",
        failure_mode="sparse_dialogue",
        release_class="webrip",
        target_name=WEBRIP_NAME,
        target_starts=base,
        candidates=[
            ("subdl", "Dexter.S08E05.720p.WEBRip.x264-GRP.srt", _srt(onsets(6, 20_000)),
             {"content": "same_release", "cut": "same_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": ("Only six cues, all correct. Below any honest sample size, so no "
                                  "verified claim is defensible even though the timing is right.")},
             A_CLEAN),
        ],
    )

    # --- 10. malformed: unparseable subtitle ------------------------------- #
    add(
        "malformed_subtitle",
        failure_mode="malformed",
        release_class="unknown",
        target_name=HDTV_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(hdtv_name), "this file has no timestamps and no cue structure at all\n",
             {"content": "same_release", "cut": "unknown", "original_sync": "unknown",
              "resyncable": False, "final_alignment": "incorrect",
              "annotation_note": "Unparseable payload. Nothing can be aligned."},
             A_NO_ALIGN),
        ],
    )

    # --- 11. duplicate timing grid across providers ------------------------ #
    add(
        "duplicate_timing_grid",
        failure_mode="duplicate_grid",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(offset(base, 5_000)),
             {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "5s offset, shared with the second provider."},
             A_JITTER),
            ("subsource", rel(other_name), _srt(offset(base, 5_000), body=VOTEDUB),
             {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": ("Different provider, group and text, identical timing grid. "
                                  "Two records, one timing model.")},
             A_JITTER),
        ],
        valid_reference_keys=[f"subdl|{rel(exact_name)}", f"subsource|{rel(other_name)}"],
        notes="VALID_REFERENCE_SET: both candidates are equally valid anchors.",
    )

    # --- 12. the circular-error case --------------------------------------- #
    # Two providers present byte-different copies of ONE wrong timing model.
    # Consensus will be 100% and the alignment will be flawless. Only
    # independent ground truth says the answer is wrong.
    circular_starts = onsets(60, 2_000, start_ms=95_000)
    add(
        "circular_shared_wrong_timing",
        failure_mode="circular",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(circular_starts),
             {"content": "same_release", "cut": "different_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "incorrect",
              "annotation_note": ("Cut B timing, aligned perfectly to Cut B. Provider consensus will be "
                                  "100% and the residual will be 0, and the answer is still wrong for this "
                                  "target. This is the case that makes self-consistency insufficient.")},
             A_CLEAN),
            ("subsource", rel(other_name), _srt(circular_starts, body=VOTEDUB),
             {"content": "same_release", "cut": "different_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "incorrect",
              "annotation_note": "Byte-different provider copy of the same wrong grid."},
             A_CLEAN),
        ],
        notes=("Deliberate trap: self-consistent and wrong. Expected to look internally perfect while "
               "ground truth says incorrect."),
        video_landmarks=video_timeline,
    )

    # --- 13. vote-sub, identical timing, different text -------------------- #
    add(
        "vote_sub_same_timing",
        failure_mode="vote_sub",
        release_class="bluray",
        target_name=BLURAY_NAME,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(base, body=VOTEDUB),
             {"content": "same_release", "cut": "same_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": ("Vote-sub with entirely different text and identical timing. A text "
                                  "comparator must call this a match, not a mismatch.")},
             A_CLEAN),
        ],
    )

    # --- 14. different cue splitting, identical timing --------------------- #
    add(
        "resplit_identical_timing",
        failure_mode="resplit",
        release_class="webdl",
        target_name=WEBDL_NAME,
        target_starts=onsets(40, 2_500),
        candidates=[
            ("subdl", rel(webdl_name), _srt(onsets(40, 2_500), duration=900),
             {"content": "same_release", "cut": "same_cut", "original_sync": "already_synced",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": "Different cue boundaries, identical cue starts."},
             A_CLEAN),
        ],
    )

    # --- 15. incomplete fingerprint, known-good relationship ---------------- #
    add(
        "missing_fingerprint",
        failure_mode="missing_fingerprint",
        release_class="bluray",
        target_name=None,
        target_starts=base,
        candidates=[
            ("subdl", rel(exact_name), _srt(offset(base, 4_000)),
             {"content": "unknown", "cut": "unknown", "original_sync": "resyncable",
              "resyncable": True, "final_alignment": "correct",
              "annotation_note": ("Ground truth is known, but the request carries no filename, so no "
                                  "release-level evidence exists. Measures capability lost to incomplete "
                                  "Stremio metadata.")},
             {"mode": "as_target", "residual_ms": 300,
              "annotation_note": "Aligned correctly; the gap is metadata, not timing."}),
        ],
        notes="filename=None: the evaluator must not invent one.",
    )

    # --- 16-19. release classes with an ordinary successful alignment ------- #
    for case_id, fname, rc, rc_label in (
        ("webrip_offset", WEBRIP_NAME, "webrip", "WEBRip"),
        ("hdtv_offset", HDTV_NAME, "hdtv", "HDTV"),
        ("tv_offset", TV_NAME, "hdtv", "P2P TV"),
        ("webdl_offset", WEBDL_NAME, "webdl", "WEB-DL"),
    ):
        add(
            case_id,
            failure_mode="offset",
            release_class=rc,
            target_name=fname,
            target_starts=onsets(50, 2_400),
            candidates=[
                ("subsource", fname.rsplit(".", 1)[0] + ".srt", _srt(offset(onsets(50, 2_400), 3_200)),
                 {"content": "same_release", "cut": "same_cut", "original_sync": "resyncable",
                  "resyncable": True, "final_alignment": "correct",
                  "annotation_note": f"{rc_label} case with a 3.2s constant offset, aligned cleanly."},
                 A_JITTER),
            ],
        )

    # --- 20. second wrong cut: a recap difference, not a splice ------------- #
    # §21 requires a second structural failure so the validator is not fitted to
    # one pattern. Here the opening carries a longer recap, so the timeline
    # diverges early and re-converges later. A splice test alone would not see
    # it: the halves disagree in the opposite arrangement.
    standard_block = [("tone", 24_000), ("silence", 4_000)] * 12
    recap_block = [("tone", 34_000), ("silence", 6_000)] * 3
    tv_edit_segments = recap_block + [("tone", 24_000), ("silence", 4_000)] * 9
    add(
        "wrong_cut_tv_edit_recap",
        failure_mode="different_cut_tv_edit",
        release_class="hdtv",
        target_name=TV_NAME,
        target_starts=onsets(60),
        candidates=[
            (
                "subdl",
                "Dexter.S08E05.720p.HDTV.x264-2HD.srt",
                _srt(activity_cue_starts(tv_edit_segments)),
                {
                    "content": "same_release",
                    "cut": "different_cut",
                    "original_sync": "already_synced",
                    "resyncable": True,
                    "final_alignment": "incorrect",
                    "annotation_note": (
                        "TV edit with a longer recap. The timeline diverges through the "
                        "opening and re-converges afterwards, so no single offset "
                        "explains it. Deliberately a different failure shape from the "
                        "splice case, so the validator cannot be fitted to one pattern."
                    ),
                },
                A_WRONG_CUT,
            )
        ],
        video_landmarks=video_timeline,
        notes="Second wrong cut, different structural signature: recap, not splice.",
    )
    _ = standard_block  # documents the target's own rhythm for a reader

    manifest = {
        "dataset_version": "v1",
        "schema_version": 1,
        "engine_version": 1,
        "description": (
            "Synthetic golden synchronization dataset. Each target fixture is the "
            "speech-timing model of an imagined video; no media is committed and no "
            "label is derived from the system under test."
        ),
        "protocol": "docs/golden_sync_annotation.md",
        "cases": cases,
    }
    return manifest, files


def main() -> None:
    manifest, files = build()
    SUBTITLES.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = FIXTURES / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    (FIXTURES / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(files)} fixtures and {len(manifest['cases'])} cases to {FIXTURES}")


if __name__ == "__main__":
    main()
