# Changelog

All notable changes to the **NinjaSubs** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Security
- **Configure page XSS**: prefilled API keys, host values and language options are HTML-escaped, and the inline preferences JSON can no longer close its `<script>` element.
- **Archive limits**: new `MAX_SUBTITLE_BYTES` (10 MB) and `MAX_ARCHIVE_BYTES` (50 MB). ZIP members are size-checked before and during inflation (zip bombs), oversized downloads are refused, and symlink members in RAR/7z/tar archives are never followed.
- **Access-log redaction** now also hides the config token on paths with no known route after it (`/<token>/`, `/<token>/favicon.ico`).
- `/cache/clear` is `POST`-only.

### Fixed
- **Requests**: a non-string credential in a token no longer causes a 500; `exclude_hi: "false"` parses as false.
- **Downloads**: HTML/JSON error pages are no longer accepted (or cached) as subtitles; ASS/SSA, MicroDVD, SAMI and UTF-16 downloads are; the OpenSubtitles proxy validates payloads before caching.
- **Event loop**: archive extraction and reference decoding run off the event loop.
- **Encoding**: legacy single-byte files use cp1256 only for Arabic/Persian/Urdu (or an unknown language) and cp1252 otherwise, so Latin-script subtitles are no longer garbled.
- **Rate limits**: SubDL no longer retries a season-pack query after a 429/401/403; SubSource searches honour their circuit breaker; a short `Retry-After` can no longer shorten an open breaker.
- **Matching**: `-2HD` release groups and `2005 - 05` years are not read as episodes; `S01E100`/`S01E1085` keep their season; movie titles such as `300.2006`, `21.Jump.Street.2012` and `Ocean's.11` are no longer parsed as episodes or seasons; release-group detection is deterministic and ignores title words (`Lost`, `Max`).
- **Languages**: SubSource results are labelled with the language they matched (Persian is no longer reported as Arabic) and matched on whole words (Malay ≠ Malayalam). Malformed provider items are skipped individually.
- **Text processing**: the ad cleaner no longer deletes ordinary Arabic dialogue containing "قناة", "تويتر" or a bare credit noun; text after a blank line inside a cue is kept; BOM-prefixed WebVTT stays WebVTT; ASS→SRT output is time-ordered with layered duplicates removed; the RTL fix recognises extended Arabic letters and presentation forms; HI stripping matches "باب" as a word and keeps dialogue dashes.
- **Sync**: each reference candidate starts from the subtitle as requested (a rejected candidate's pre-shift could make the next one serve the unshifted original as verified); NUL bytes are stripped before alass again; candidates deferred by the alass budget no longer trigger the 5-minute negative cache; source exclusions accumulate across reference tiers; per-request stream tokens no longer split the payload cache or coalescing; explicit sync requests do not join non-explicit flights; cues shifted entirely before zero are dropped instead of becoming zero-length; the listing "synced" badge requires a verified outcome; the rejected-candidate memo is bounded; a second cancellation no longer releases the alass slot early.
- **Deployment**: a fresh clone starts with the default compose file; placeholder `YOUR_..._HERE` keys are treated as unset.

### Removed
- Kitsu is no longer advertised in `idPrefixes`: Kitsu IDs were never resolved to IMDb, so every provider returned nothing for them.
- Dead SubtitleCat and YIFYSubtitles provider modules, their settings and cache-key dimensions (legacy token flags still parse), the unused `OPENSUBTITLES_USERNAME`/`OPENSUBTITLES_PASSWORD` variables, and a pre-delivery gate call that could never block.

---

## [1.1.0] - 2026-09-28

### Added
- **Auto-Sync Engine via `alass`**: Subtitle auto-synchronization against reference subtitles extracted from the playback stream or upstream providers.
- **Intelligent Reference Discovery**: Strategy-based candidate selection (`ExternalExactStrategy`, `SameProviderReferenceStrategy`) prioritizing exact release groups, matching sources (BluRay, WEB-DL), codecs, and streaming services.
- **Metadata Parsing with `guessit`**: Integrated `guessit` library as ground-truth metadata parser with an LRU cache (`@lru_cache(maxsize=2048)`), supplemented by regex engines for scene edge cases.
- **Percentage-Based Subtitle Ranking**: Subtitle candidate ranking based strictly on normalized `guessit` match percentage clamped between `0%` and `100%`.
- **In-Flight Coalescing**: Hoisted process-scoped `SyncOrchestrator` ensuring concurrent identical sync requests share execution without duplicate workload.
- **Sync Result Cache**: 24-hour persistent SQLite disk caching (`SyncCache`) for synchronized subtitle artifacts.
- **Root `manifest.json`**: Community-standard static manifest file for deployment and addon catalogs.

### Changed
- **Auto-Sync Enabled by Default**: Auto-sync feature is now enabled by default across server settings, container environment, user preferences, and configuration UI.
- **Strict Percentage Sorting**: Final subtitle list returned to Stremio is sorted strictly in descending order by `guessit` match percentage (highest matching candidate first), with score tie-breaking.
- **Clean ISO-639-2 Language Codes**: Ensured subtitle track language codes remain pristine ISO-639-2 codes (`ara`, `en`, etc.) across all provider responses.
- **Version Bump**: Bumped project version to `1.1.0` in `app/__init__.py`, `app/models.py`, `app/main.py`, `app/templates/configure.html`, and `manifest.json`.

### Fixed
- **Season Number Validation**: Enforced `val <= 99` guard to prevent release years (e.g. `2011` in `Hunter X Hunter.2011`) from being incorrectly parsed as TV season numbers.
- **Edition Mismatch Safeguard**: Rejection of conflicting editions (Theatrical vs Extended / Director's Cut / IMAX) during auto-sync candidate selection to prevent desynchronization.

---

## [1.0.0] - 2026-09-01

### Added
- **Multi-Provider Subtitle Aggregation**: Parallel querying of SubDL, SubSource, OpenSubtitles, YIFYSubtitles, and SubtitleCat.
- **Arabic Subtitle Optimization**: Context-aware Arabic RTL punctuation and bracket alignment, optional Tashkeel (diacritics) stripping, Eastern Arabic numeral conversion, and Hearing Impaired (HI) dialogue cleanup.
- **Two-Stage Matcher Architecture**: Stage 0 MovieHash deterministic short-circuit, Stage 1 hard exclusion filter, and Stage 2 soft scoring matrix.
- **Stateless Configuration UI**: Modern web configurator encoding all user options into URL-safe manifest tokens.
- **In-Memory Archive Extraction**: Safe, Zip-Slip-protected in-memory extraction for ZIP, RAR, and 7z archives.
