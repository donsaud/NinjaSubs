"""Host-side Stremio validation helper for NinjaSubs.

Run this on the machine that has Stremio installed. It performs every check that
can be automated and prints the exact expectations for the part that cannot --
the human watching the subtitle.

It deliberately does NOT drive the GUI. It verifies the addon is healthy and
correctly configured, fetches the exact payload that would be served, prints its
hash, and prints the playback checklist. Watching a subtitle is a human judgement
and is not automatable here.

    python tools/validate_stremio_playback.py --imdb tt0773262 --season 8 --episode 4

Exit codes: 0 = every automated check passed, 1 = a check failed, 2 = the addon
is not usable yet (no credentials, not running).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("NINJASUBS_URL", "http://localhost:7000")
REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests" / "fixtures" / "real_media" / "dexter_s08e04_manifest.json"
CACHE = pathlib.Path(os.environ.get("NINJASUBS_CACHE_DIR", REPO / "subs_cache"))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _c(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{RESET}"


def fetch(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc).encode()


def check_addon(base: str) -> dict:
    print("=" * 78)
    print("STEP 1 — addon health and configuration")
    print("=" * 78)
    status, body = fetch(f"{base}/health")
    if status != 200:
        print(_c(f"  FAIL  {base}/health returned {status}", RED))
        print(_c("  Is the service running?  docker compose ps", DIM))
        return {"ok": False, "reason": "health"}
    try:
        health = json.loads(body)
    except Exception:  # noqa: BLE001
        print(_c("  FAIL  /health did not return JSON", RED))
        return {"ok": False, "reason": "health-parse"}

    print(f"  status        {health.get('status')}")
    keys = health.get("env_keys", {}) or {}
    configured = [k for k, v in keys.items() if v]
    missing = [k for k, v in keys.items() if not v]
    print(f"  providers set {configured or 'none'}")
    if not configured:
        print(_c("  FAIL  no provider credentials configured.", RED))
        print(_c("  Stremio would list ZERO subtitles. Add a key to .env:", DIM))
        print(_c("    SUBDL_KEY=...  or  SUBSOURCE_KEY=...  "
                 "or  OPENSUBTITLES_USERNAME/PASSWORD_HASH", DIM))
        return {"ok": False, "reason": "no-credentials", "health": health}
    if missing:
        print(_c(f"  note   not configured: {', '.join(missing)}", YELLOW))

    status, body = fetch(f"{base}/manifest.json")
    if status == 200:
        try:
            manifest = json.loads(body)
            catalogs = len(manifest.get("catalogs", []) or [])
            print(f"  catalog count {catalogs}")
            if catalogs == 0:
                print(_c("  FAIL  manifest advertises no catalogs.", RED))
                return {"ok": False, "reason": "no-catalogs", "health": health}
        except Exception:  # noqa: BLE001
            print(_c("  note   manifest was not JSON", YELLOW))
    return {"ok": True, "health": health, "configured": configured}


def load_expectations() -> dict:
    if not MANIFEST.is_file():
        print(_c(f"  note  manifest not found at {MANIFEST}", YELLOW))
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data.get("_derived", {})


def show_expectations(expected: dict) -> None:
    print()
    print("=" * 78)
    print("STEP 2 — what the API is expected to serve (from the committed manifest)")
    print("=" * 78)
    if not expected:
        print("  no manifest available; run tools/build_real_media_manifest.py first")
        return
    print(f"  video fingerprint  {expected.get('video_fingerprint', 'n/a')}")
    print(f"  video size         {expected.get('video_size', 'n/a')} bytes")
    print(f"  reference sha256   {expected.get('reference_sha256', 'n/a')}")
    print("  alass artifacts:")
    for sid, digest in (expected.get("alass_artifact_sha256") or {}).items():
        print(f"    {sid}  {digest}")
    print(f"  expected delivery  {expected.get('expected_delivery_category', 'n/a')}")
    print("  verifier state     UNVERIFIED (residual p95 above the limit)")
    print()
    print(_c("  The original subtitle is what you should see in Stremio. That is", DIM))
    print(_c("  correct current behaviour, not a bug. The Alass artifact exists but", DIM))
    print(_c("  the verifier does not yet accept it, so the original is served.", DIM))


def show_provider_bytes() -> None:
    print()
    print("=" * 78)
    print("STEP 3 — provider bytes currently cached (what should be served)")
    print("=" * 78)
    if not CACHE.is_dir():
        print(_c(f"  no cache directory at {CACHE}", YELLOW))
        return
    found = 0
    for path in sorted(CACHE.glob("*.srt")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {path.name:<26} {digest[:32]}…  {path.stat().st_size:>7} bytes")
        found += 1
        if found >= 8:
            break
    if not found:
        print("  (cache is empty)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NinjaSubs for real Stremio playback of Dexter S08E04."
    )
    parser.add_argument("--url", default=DEFAULT_BASE,
                        help=f"addon base URL (default {DEFAULT_BASE})")
    parser.add_argument("--imdb", default="tt0773262")
    parser.add_argument("--season", default="8")
    parser.add_argument("--episode", default="4")
    args = parser.parse_args()

    print(_c("NinjaSubs — Stremio playback validation helper", DIM))
    print(_c("This performs the automated checks. Watching the subtitle is yours.", DIM))
    print()

    result = check_addon(args.url.rstrip("/"))
    if not result.get("ok"):
        print()
        print(_c("STOP — fix the above before proceeding to playback.", RED))
        return 2

    show_expectations(load_expectations())
    show_provider_bytes()

    print()
    print("=" * 78)
    print("STEP 4 — manual playback checklist (you do this part)")
    print("=" * 78)
    print(f"""
  Target: Dexter · Season {args.season} · Episode {args.episode}  (tt0773262)
  Video:  Dexter.S08E04.Scar.Tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv
          sha256 c929b5f64845b1554de2580c4f91e44a6e70ab4eeca71449c4df97c404197e0f

  IMPORTANT — two different questions
    "What the addon served"  -> decided by the API, measured by sha256 (STEP 2/3).
    "What you see on screen" -> decided by you, recorded below.
    They can disagree without anything being broken.

  WHICH CASES TO TEST
    A. EVOLV  (74a34c232d159f8b)
       expect decision=ORIGINAL, served e82586e4cbacf6d3
       verifier UNVERIFIED -> the ORIGINAL is served, correctly
    B. ASAP   (96c0442cc4656685)
       expect decision=ORIGINAL, served 30246e82e06d3ee2
       verifier UNVERIFIED -> the ORIGINAL is served, correctly
    C. synthetic +96.5s  — ONLY if selectable in your environment
       expect decision=TRANSFORMED, served 9d9863c963947cb2
       this is the positive control; if not selectable, skip and say so

    A and B are NOT VERIFIED and must never be recorded as verified.

  For EACH case, play from the beginning and record PASS / PARTIAL / FAIL /
  UNDETERMINED at each position:

    [ ]  0%   (00:00)  opening scene / credits
    [ ] 10%   (05:00)  first opening dialogue
    [ ] 25%   (12:30)  first quarter
    [ ] 50%   (25:00)  midpoint
    [ ] 75%   (37:30)  third quarter
    [ ] 90%   (45:00)  late episode
    [ ] 100%  (50:00)  ending

  And once per case:

    [ ] opening scene specifically — it carries a ~96s displaced section that is
        NOT a uniform shift, so it may look different from the body
    [ ] a dialogue-heavy scene — does text track the speech line by line?
    [ ] seek backward ~5 min, then forward   — does the subtitle follow?
    [ ] pause 30 s, then resume             — does timing survive?
    [ ] restart playback from the beginning — same as first load?
    [ ] play again after a few minutes       — cache warm-up, any change?

  For each row record WHAT YOU SAW:
    in sync / early by ~Ns / late by ~Ns / no subtitle shown / text with no speech
    and any styling or conversion change you notice.

  Then record: what the addon served (ORIGINAL or TRANSFORMED) and whether the
  served sha256 matches STEP 2. Use UNDETERMINED rather than guessing.

  Full protocol and the per-case result sheet:
    docs/stremio_playback_validation.md
""")

    print("=" * 78)
    print("STEP 5 — report back")
    print("=" * 78)
    print("""
  Reply with, per case (A, B, and C if available):
    - the release name Stremio listed, and any fallback-release notice
    - each position's result and what you saw
    - the scene checks (opening, dialogue-heavy, seek, pause, restart, warm-up)
    - what the addon served: ORIGINAL or TRANSFORMED, and the served sha256
    - OVERALL: PASS / PARTIAL / FAIL / UNDETERMINED

  A PASS is engineering evidence that the delivered subtitle is watchable in sync.
  It is NOT a verification state — the production verifier still reports UNVERIFIED
  for EVOLV and ASAP, and that is correct for now.

  Without this, the rendered-player leg stays UNDETERMINED no matter what the logs
  show.
    """)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
