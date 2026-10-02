"""Preflight check for the STREMIO_PLAYBACK_WITNESS fixture mount.

Run this on the host BEFORE opening Stremio. It answers the only question that
matters at this point: does the witness service actually see the pre-validated
Alass fixtures, and are they still the exact bytes that were validated?

    python tools/check_witness_mount.py
    python tools/check_witness_mount.py --url http://localhost:7100

Why this exists
---------------
Docker creates a missing bind-mount source as an *empty directory* and starts
happily. So a wrong or unset WITNESS_FIXTURE_DIR produces an enabled service
that serves nothing, and the symptom -- a 503 or an empty subtitle list -- looks
like a broken service rather than a misconfigured path. This turns that into one
explicit line.

Exit codes
----------
    0  the witness can serve both fixtures, byte-exact
    1  a check failed
    2  the witness service is not running or not reachable
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

DEFAULT_URL = os.environ.get("WITNESS_URL", "http://localhost:7100")
REPO = pathlib.Path(__file__).resolve().parent.parent

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def _c(text: str, colour: str) -> str:
    return text if sys.stdout.isatty() else text


def fetch(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc).encode()


def host_fixture_dir() -> pathlib.Path:
    """The host directory compose will bind-mount, resolved the same way.

    Mirrors the compose default so the message names the path that was actually
    used rather than a guess.
    """
    explicit = os.environ.get("WITNESS_FIXTURE_DIR", "").strip()
    if explicit:
        return pathlib.Path(explicit)
    return REPO / "autosync-diagnostics"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help=f"witness base URL ({DEFAULT_URL})")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print("=" * 74)
    print("STREMIO_PLAYBACK_WITNESS -- fixture mount preflight")
    print("=" * 74)

    failures: list[str] = []

    # --- 1. the host directory the compose file will use -------------------- #
    print("\n1. host fixture directory")
    host_dir = host_fixture_dir()
    configured = bool(os.environ.get("WITNESS_FIXTURE_DIR", "").strip())
    print(f"   WITNESS_FIXTURE_DIR : "
          f"{os.environ.get('WITNESS_FIXTURE_DIR', '<unset, using compose default>')}")
    print(f"   resolved path       : {host_dir}")
    if not configured:
        note = (
            f"\n   NOTE: WITNESS_FIXTURE_DIR is not set, so Compose falls back to\n"
            f"         {host_dir}\n"
            f"   Docker creates a missing bind-mount source as an EMPTY directory,\n"
            f"   which is why the witness can be 'available' and serve nothing.\n"
            f"   Add this to .env and recreate the service:\n"
            f"      WITNESS_FIXTURE_DIR=<absolute path to the folder holding the\n"
            f"                            pre-validated Alass outputs>\n"
            f"      docker compose --profile witness up -d --force-recreate witness"
        )
        print(_c(note, YELLOW))
    if not host_dir.is_dir():
        print(_c(f"   FAIL  directory does not exist on the host: {host_dir}", RED))
        failures.append(f"host fixture directory missing: {host_dir}")
    elif not any(host_dir.iterdir()):
        print(_c(f"   FAIL  directory exists but is EMPTY: {host_dir}", RED))
        failures.append(f"host fixture directory is empty: {host_dir}")
    else:
        print(_c(f"   OK    {len(list(host_dir.iterdir()))} file(s) present", GREEN))

    # --- 2. is the service up ------------------------------------------------ #
    print("\n2. witness service")
    status, body = fetch(f"{base}/stremio-playback-witness/fixtures.json")
    if status != 200:
        print(_c(f"   FAIL  {base}/.../fixtures.json returned {status}", RED))
        print(_c("   Is it running?  docker compose --profile witness ps", DIM))
        return 2
    try:
        info = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        print(_c(f"   FAIL  response was not JSON ({exc})", RED))
        return 2
    print(_c("   OK    reachable", GREEN))
    print(f"   container fixture_dir : {info.get('fixture_dir')}")
    print(f"   mount_usable          : {info.get('mount_usable')}")
    if not info.get("mount_usable", False):
        print(_c(f"   FAIL  {info.get('mount_problem')}", RED))
        if info.get("reminder"):
            print(_c(f"   {info['reminder']}", YELLOW))
        failures.append("container reports the fixture mount is unusable")

    # --- 3. fixtures present and byte-exact ---------------------------------- #
    print("\n3. fixtures")
    for entry in info.get("fixtures", []):
        key = entry.get("key")
        if not entry.get("available"):
            print(_c(f"   {key:<6} UNAVAILABLE  {entry.get('detail')}", RED))
            failures.append(f"{key} unavailable")
            continue
        digest = entry.get("sha256", "")
        size = entry.get("bytes")
        matches = entry.get("matches_validated_digest")
        flag = "OK   " if matches else "FAIL "
        colour = GREEN if matches else RED
        print(_c(f"   {flag} {key:<6} {size:>6} bytes  {digest[:32]}...", colour))
        print(_c(f"          {entry.get('filename')}", DIM))
        if not matches:
            failures.append(
                f"{key} digest {digest[:16]} is not the validated artifact -- do not "
                f"use it, and do not substitute a regenerated file"
            )

    # --- 4. the real Stremio request shape ------------------------------------ #
    print("\n4. real Stremio-shaped discovery URL")
    # Exactly what a Stremio client sends. If this shape is not routed, the
    # production catch-all answers it and the witness is never consulted.
    real_path = (
        "/stremio-playback-witness/subtitles/series/"
        "tt0773262%3A8%3A4/"
        "videoSize%3D5414925805%26filename%3D"
        "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
        ".json"
    )
    code, real_body = fetch(f"{base}{real_path}")
    if code != 200:
        print(_c(f"   FAIL  real Stremio URL returned {code}", RED))
        failures.append(f"real Stremio-shaped request returned {code}")
    else:
        real_subs = json.loads(real_body).get("subtitles", [])
        labels = [s.get("lang", "") for s in real_subs]
        print(f"   OK    {code}, {len(real_subs)} candidate(s)")
        for want in ("EVOLV ALASS WITNESS", "ASAP ALASS WITNESS"):
            if any(want in x for x in labels):
                print(_c(f"   OK    contains TEST - {want}", GREEN))
            else:
                print(_c(f"   FAIL  missing TEST - {want}", RED))
                failures.append(f"real Stremio response missing {want}")

    # --- 5. served bytes over real HTTP ---------------------------------------- #
    print("\n5. served bytes (HTTP)")
    code, subs_body = fetch(
        f"{base}/stremio-playback-witness/subtitles/tt0773262:8:4.json"
    )
    if code != 200:
        print(_c(f"   FAIL  subtitle list returned {code}", RED))
        failures.append(f"subtitle list returned {code}")
    else:
        subs = json.loads(subs_body).get("subtitles", [])
        if not subs:
            print(_c("   FAIL  no witness subtitles offered", RED))
            failures.append("no witness subtitles offered")
        for item in subs:
            url = item["url"]
            code, payload = fetch(url)
            served = hashlib.sha256(payload).hexdigest()
            advertised = url.rsplit("/", 1)[-1].removesuffix(".srt")
            ok = code == 200 and served == advertised
            colour = GREEN if ok else RED
            print(_c(f"   {'OK   ' if ok else 'FAIL '} {item['id'].split(':')[1]:<6} "
                     f"{len(payload):>6} bytes  {served[:32]}...", colour))
            if not ok:
                failures.append(f"{item['id']} served bytes != URL digest (HTTP {code})")

    # --- verdict -------------------------------------------------------------- #
    print()
    print("=" * 74)
    if failures:
        for f in failures:
            print(_c(f"  FAIL  {f}", RED))
        print(_c("\n  Do NOT proceed to the Stremio playback test.", RED))
        print("=" * 74)
        return 1
    print(_c("  OK  the witness will serve both fixtures byte-for-byte.", GREEN))
    print(_c(f"      Add-on manifest: {base}/stremio-playback-witness/manifest.json", DIM))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
