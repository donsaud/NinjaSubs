# Baseline test failures: classification

The suite reports three failures on a host checkout. All three were reproduced on
a clean tree and investigated. **None is a repository defect**, and none was
"fixed" — there is nothing in the code to fix.

## Summary

| test | cause | classification | action |
|---|---|---|---|
| `test_guessit_e2e.py::test_guessit_parses_complex_movie_metadata` | missing `guessit` | D — environment | install the declared dependency |
| `test_guessit_e2e.py::test_guessit_parses_tv_season_episode_metadata` | missing `guessit` | D — environment | same |
| `test_reference_target_bound.py::test_timing_families_are_distinct_for_distinct_releases` | missing `guessit` | D — environment | same |

## Evidence

`guessit` is a declared dependency:

```
requirements.txt:  guessit>=3.8.0
```

It is present in the container image at version 4.4.0, and absent from the host
Python environment. With it installed, all three pass:

```
$ docker exec ninjasubs pytest <the three tests> -q
......                                                                   [100%]
6 passed in 0.82s
```

## The `release_family_key` case is the same root cause, not a stale test

The failing assertion is:

```python
assert release_family_key(DEMAND) == release_family_key(DEMAND + ".en")
```

`release_family_key` calls `extract_metadata`, which depends on `guessit`. With
`guessit` absent the metadata is empty, the function falls through to its
filename-derived fallback, and the language suffix becomes part of the key:

```
                       without guessit              with guessit (4.4.0)
"DEMAND.srt"          'BluRay||demand'             'BluRay||demand'
"DEMAND.srt.en"       'BluRay||'                   'BluRay||demand'
```

So the test asserts correct intended behaviour, and it fails only because the
environment cannot reach the code path that implements it. With the dependency
present, a language suffix correctly does not create a new release family — which
is what the test is there to protect.

The assertion reflects intended current behaviour. It is not stale, and the
release-parsing logic was not touched.

## Recommendation

Leave all three open as environment failures. To run the suite green, install the
declared test requirements on the host:

```
pip install -r requirements.txt
```

No production change, no test change, and no release-parsing change is warranted.
