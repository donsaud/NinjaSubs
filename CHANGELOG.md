# Changelog

All notable changes to the **NinjaSubs** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
