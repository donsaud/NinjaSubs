"""One-time maintenance: strip legacy credentials from cached subtitle metadata.

The read-repair path in ``LRUCacheManager.get_metadata`` sanitizes an entry when
it is read, which leaves the migration dependent on cache access. This tool does
the same work once, up front, over every metadata file -- so a deployment is not
carrying plaintext provider credentials while nothing happens to read them.

Guarantees:

* only ``_meta/*.json`` is touched; subtitle payload bytes are never opened;
* only credential fields and credential-bearing URL parameters are removed;
* non-sensitive metadata is preserved verbatim;
* idempotent -- a second run reports zero changes;
* aggregate counts only; no secret value is ever printed;
* ``--dry-run`` reports without writing.

Usage::

    python tools/migrate_cache_metadata.py --dry-run
    python tools/migrate_cache_metadata.py
    python tools/migrate_cache_metadata.py --cache-dir /path/to/subs_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import sanitize_metadata  # noqa: E402

SECRET_FIELDS = ("subdl_key", "subsource_key", "opensubtitles_key")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache root containing _meta/. Defaults to CACHE_DIR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args(argv)

    from app.config import settings

    root = Path(args.cache_dir or settings.CACHE_DIR)
    meta_dir = root / "_meta"
    if not meta_dir.is_dir():
        print(f"no metadata directory at {meta_dir}")
        return 1

    files = sorted(meta_dir.glob("*.json"))
    scanned = 0
    unreadable = 0
    secret_fields_removed = 0
    urls_redacted = 0
    changed_files = 0
    payload_files_seen = 0

    for path in files:
        scanned += 1
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            unreadable += 1
            continue
        if not isinstance(data, dict):
            unreadable += 1
            continue

        safe, changed = sanitize_metadata(data)
        if not changed:
            continue
        changed_files += 1
        for field in SECRET_FIELDS:
            if field in data:
                secret_fields_removed += 1
        for url_field in ("download_url", "stream_url"):
            if data.get(url_field) != safe.get(url_field):
                urls_redacted += 1
        if not args.dry_run:
            # Atomic enough for a maintenance run: write via a sibling temp file
            # so an interrupted run cannot leave truncated JSON behind.
            tmp = path.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(safe), encoding="utf-8")
                tmp.replace(path)
            except OSError as exc:
                print(f"failed to rewrite {path.name}: {exc}")
                return 1

    # Payload files are counted only to report that they were left alone.
    payload_files_seen = len(list(root.glob("*.srt")))

    print(f"mode                : {'dry-run' if args.dry_run else 'write'}")
    print(f"cache dir           : {root}")
    print(f"metadata scanned    : {scanned}")
    print(f"unreadable/skipped  : {unreadable}")
    print(f"files changed       : {changed_files}")
    print(f"secret fields removed: {secret_fields_removed}")
    print(f"credential URLs redacted: {urls_redacted}")
    print(f"payload files untouched: {payload_files_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
