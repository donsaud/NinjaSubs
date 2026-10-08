"""Offline development benchmark; no providers, downloads or synchronization."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.services.sync import matching
from app.services.sync.external_strategy import _select_candidates_ranked
from app.services.sync.query import ReferenceQuery
from app.services.sync.release_metadata import SEMANTIC_VERSION, _parse_cached


def sample_names(count: int) -> list[str]:
    families = (
        "Film{n}.2020.Criterion.Collection.1080p.BluRay.x265-GRP",
        "Film{n}.2020.Directors.Cut.2160p.UHD.BluRay.REMUX-FraMeSToR",
        "Show{n}.S02E03.AMZN.1080p.WEB-DL.x264-GRP",
        "Show{n}.S02E03.720p.HDTV.x264-GRP",
        "Film{n}.2020.Open.Matte.Remastered.1080p.BluRay.x264-GRP",
        "Ambiguous Film {n}",
    )
    return [families[n % len(families)].format(n=n) for n in range(count)]


def benchmark() -> dict:
    rows = []
    for count in (10, 50, 100):
        names = sample_names(count)
        releases = [SimpleNamespace(release_name=name, provider="subdl", lang="en") for name in names]
        query = ReferenceQuery(imdb_id="tt0000000", target_filename=names[0])
        matching.guess_metadata.cache_clear()
        _parse_cached.cache_clear()
        with patch.object(matching, "_guessit", wraps=matching._guessit) as guess:
            start = time.perf_counter()
            for name in names:
                matching.parse_release_identity(name)
            cold = (time.perf_counter() - start) * 1000
            cold_calls = guess.call_count
            start = time.perf_counter()
            for name in names:
                matching.parse_release_identity(name)
            warm = (time.perf_counter() - start) * 1000
            warm_calls = guess.call_count - cold_calls
            matching.guess_metadata.cache_clear()
            _parse_cached.cache_clear()
            before = guess.call_count
            start = time.perf_counter()
            _select_candidates_ranked(releases, query)
            rank_cold = (time.perf_counter() - start) * 1000
            rank_calls = guess.call_count - before
            start = time.perf_counter()
            _select_candidates_ranked(releases, query)
            rank_warm = (time.perf_counter() - start) * 1000
            rows.append(dict(count=count, cold_ms=round(cold, 3), warm_ms=round(warm, 3),
                             ranking_cold_ms=round(rank_cold, 3), ranking_warm_ms=round(rank_warm, 3),
                             guessit_cold=cold_calls, guessit_warm=warm_calls, guessit_ranking=rank_calls,
                             cache=dict(_parse_cached.cache_info()._asdict())))
    return {"semantic_version": SEMANTIC_VERSION, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
