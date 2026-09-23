# Manual Stremio UI Validation & Observability Guide

**Status**: 
- **Automated Validation**: **COMPLETE** (286 / 286 tests passing, 100% pass rate)
- **Live Provider / UI Validation**: **READY TO START**
- **Docker Container**: Up & Healthy on port 7000 (`/manifest.json` -> 200, `/diagnostics/ranking` -> 200)

---

## 1. Overview & Objectives

Automated unit, ranking audit, and integration tests have confirmed that the Two-Stage Subtitle Matcher correctly scores, partitions, and formats subtitles without any reordering in the FastAPI response path.

However, **synthetic tests cannot reproduce all real-world upstream metadata inconsistencies**, such as:
- Unofficial fansub tags or release group formatting on SubDL / SubSource / OpenSubtitles
- Misleading video stream filenames passed by torrent or debrid providers
- Subtitle titles containing unexpected scene suffixes, hex hashes, or advertising watermarks
- Unexpected FPS drift or encoding tags in real provider archives

This document provides:
1. Instructions for enabling **lightweight diagnostic observability** during testing.
2. A comprehensive **manual testing checklist** covering Movies, TV series, Anime, Multi-Language, and SDH.
3. A structured **test log template** to record live results, stream filenames, and potential anomalies.

---

## 2. Observability & Safe Diagnostic Mode

### Enabling Diagnostic Logging
To inspect real ranking decisions in live Docker container logs without exposing API keys or secrets:

1. In your environment or `.env` file, set:
   ```env
   NINJASUBS_DEBUG_RANKING=true
   ```
   *(Defaults to `false` in production for minimal log overhead).*

2. When enabled, each subtitle search outputs structured diagnostic logs:
   ```text
   [DEBUG_RANKING] === Ranking Evaluation for Target: 'House.of.the.Dragon.S02E01...' (Total Candidates: 14) ===
   [DEBUG_RANKING] DISCARDED: [subdl][ara] 'House.of.the.Dragon.S02E02...' | HardReject: Explicit episode mismatch (E1 vs E2)
   [DEBUG_RANKING] #1 [subdl][ara] 'House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX' | Score: 195 (100%) | Conf: 0.98 | Accepted: True | Method: filename | Hash: False | SourceMatch: True | ServiceMatch: True | EpMatch: True | SeMatch: True | FPS: unknown | Reject: None | Reasons: ['Title prefix match (+25)', 'TV Season 2 & Episode 1 match (+20)', ...]
   [DEBUG_RANKING] #2 [subsource][ara] 'House.of.the.Dragon.S02E01.1080p.AMZN.WEB-DL-FLUX' | Score: 105 (54%) | Conf: 0.98 | Accepted: True | Method: filename | Hash: False | SourceMatch: True | ServiceMatch: False | EpMatch: True | SeMatch: True | FPS: unknown | Reject: None | Reasons: [...]
   [DEBUG_RANKING] === End Ranking Evaluation ===
   ```

### Safe Diagnostic Endpoint
- **URL**: `http://localhost:7000/diagnostics/ranking`
- **Output**: Returns JSON with current active ranking weights, provider configuration status, and debug state.
- **Security Guarantee**: Never exposes API keys, tokens, or authorization headers.

---

## 3. How to Install & Configure in Stremio

1. Open your browser to the local addon configuration page:
   ```
   http://localhost:7000/configure
   ```
   *(Or your LAN IP e.g. `http://192.168.1.X:7000/configure` for Android TV / Nuvio).*

2. Configure your desired options:
   - **Preferred Languages**: Select Arabic (`ara`), English (`eng`), French (`fra`), etc.
   - **Exclude Hearing Impaired (HI)**: Toggle as needed for test cases.
   - **API Keys**: Enter your SubDL, SubSource, or OpenSubtitles API keys.

3. Click **Install Addon** to open Stremio, or copy the manifest URL and paste it into the Stremio search bar:
   ```
   http://localhost:7000/<encoded_config>/manifest.json
   ```

---

## 4. Manual Testing Checklist

### A. Movies Validation
- [ ] **M1. Dune: Part Two (2024)**
  - Stream: High-bitrate 2160p UHD Remux or 1080p BluRay stream.
  - Verify: Correct 2024 subtitles rank first; any 2023 subtitles or CAM releases appear at the bottom.
- [ ] **M2. Recent 1080p Movie (e.g. *Oppenheimer* or *The Batman*)**
  - Stream: 1080p BluRay or WEB-DL.
  - Verify: Same source family (BluRay vs WEB-DL) correctly prioritized; matching scene release group (e.g. FLUX, NTb) is top-ranked.
- [ ] **M3. 2160p / 4K UHD Movie**
  - Stream: 2160p UHD HDR/DV stream.
  - Verify: Subtitles matching 2160p/UHD rank above 1080p/720p HDTV.

---

### B. TV Series Validation
- [ ] **T1. House of the Dragon (Season 2 Episode 1)**
  - Stream: 1080p HMAX WEB-DL stream.
  - Verify: HMAX subtitles rank ahead of AMZN/Netflix subtitles; S02E02 or S01E01 subtitles **never appear**.
- [ ] **T2. Multi-Provider TV Episode (e.g. *The Last of Us* or *Succession*)**
  - Stream: Recent episode available across SubDL, SubSource, and OpenSubtitles.
  - Verify: No duplicate identical releases appear; top match has highest compatibility badge `[100%]`.
- [ ] **T3. Wrong Episode Elimination**
  - Stream: Any TV episode (e.g. Episode 3).
  - Verify: Only Episode 3 subtitles appear in the list. Subtitles for Episode 1, 2, or 4 are 100% absent.

---

### C. Anime Validation
- [ ] **A1. One Piece (Recent Standalone Episode, e.g. 1050+)**
  - Stream: Crunchyroll WEB-DL or Erai-raws / SubsPlease stream.
  - Verify: Episode number matches accurately; subtitles for adjacent episodes (e.g. 1051) are excluded.
- [ ] **A2. Multi-Episode Anime Batch / Range**
  - Stream: An episode covered by a multi-episode subtitle pack (e.g. `1049-1050`).
  - Verify: Multi-episode pack is accepted and selectable.
- [ ] **A3. Anime with Numbers in Title (e.g. *Mob Psycho 100*)**
  - Stream: Episode 1 or 2.
  - Verify: The number `100` in the title is not misclassified as the episode number.

---

### D. Multi-Language Partitioning
- [ ] **L1. Arabic Only Configured**
  - Verify: Only Arabic subtitles are returned.
- [ ] **L2. Arabic Preferred, English Secondary**
  - Verify: All Arabic subtitles are listed first (sorted highest to lowest score); all English subtitles follow afterwards (sorted highest to lowest score).
  - Check: An English subtitle with a higher compatibility score never jumps ahead of Arabic subtitles.

---

### E. Hearing Impaired (SDH) Filtering
- [ ] **S1. Exclude HI = ON**
  - Verify: No subtitles with `[SDH]`, `.sdh.`, or hearing impaired tags appear in the Stremio list.
- [ ] **S2. Exclude HI = OFF**
  - Verify: SDH subtitles are visible and correctly labeled with their compatibility percentage.

---

### F. Specific Compatibility Edge Cases
- [ ] **R1. Binary MovieHash Match**
  - Stream: Torrent/stream with OpenSubtitles hash match.
  - Verify: Exact hash match appears as #1 with `[100%]` badge.
- [ ] **R2. Streaming Platform Alignment**
  - Verify: HMAX matches HMAX, AMZN matches AMZN, NF matches NF.
- [ ] **R3. Source Hierarchy**
  - Verify: Remux/BluRay target prefers BluRay/Remux subtitles over WEBRip/HDTV.
- [ ] **R4. Repack / Proper Alignment**
  - Stream: A release marked `REPACK` or `PROPER`.
  - Verify: Matching `PROPER`/`REPACK` subtitles rank ahead of unaligned standard subtitles.
- [ ] **R5. Frame Rate (FPS) Drift**
  - Verify: 23.976fps streams display 23.976fps subtitles first, while 25fps PAL subtitles rank significantly lower.

---

## 5. Live Manual Test Log Sheet

Record real results in the table below during testing:

| Test ID | Media Title | Actual Stream Filename | Configured Languages | Exclude HI | First 3 Subtitle Titles Shown | Provider | Correct Order? (Y/N) | Notes / Anomalies Observed |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **M1** | Dune: Part Two | *(Fill from stream)* | `ara, eng` | ON | 1.<br>2.<br>3. | SubDL | | |
| **M2** | *(Recent 1080p)* | *(Fill from stream)* | `ara` | ON | 1.<br>2.<br>3. | | | |
| **T1** | House of the Dragon | *(Fill from stream)* | `ara, eng` | ON | 1.<br>2.<br>3. | | | |
| **T2** | *(Other series)* | *(Fill from stream)* | `ara` | ON | 1.<br>2.<br>3. | | | |
| **A1** | One Piece | *(Fill from stream)* | `ara` | ON | 1.<br>2.<br>3. | | | |
| **S1** | *(Test Movie/TV)* | *(Fill from stream)* | `ara` | OFF | 1.<br>2.<br>3. | | | |

---

## 6. Next Steps & Reporting

1. Perform manual tests using the checklist above on your Stremio desktop, web, or TV player.
2. If any subtitle ranks unexpectedly:
   - Check container logs (`docker logs ninjasubs`) with `NINJASUBS_DEBUG_RANKING=true`.
   - Note the exact stream filename and subtitle release name in the log sheet.
3. Report any observed discrepancies for precision calibration.
