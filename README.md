# NinjaSubs - Stremio Subtitle Addon

A lightweight, high-performance, self-hosted [Stremio](https://stremio.com) and [Nuvio](https://github.com) Addon microservice built with **Python 3.11+**, **FastAPI**, and **Uvicorn** (`uvloop`). It searches, unpacks, and serves native subtitles from **SubDL**, **SubSource**, and **OpenSubtitles** APIs, featuring **informative match badges**, automated **LRU disk cache management**, and **safe in-memory ZIP extraction** with Zip Slip defense and legacy Arabic transcoding.

---

## Key Features

1. **Strict Release-Name Display for Stremio & Nuvio**:
   - Uses a hybrid `lang` schema (`"ara | " + release_filename`) to force Stremio Desktop, Web, Mobile, and Android TV to show the complete release name in the subtitle dropdown.
   - Preserves the unmodified release filename in `title` (e.g. `House.of.the.Dragon.S02E01.1080p.WEB-DL.DDP5.1.Atmos.H.264-FLUX.srt`).
2. **Dual Provider Aggregation & Resilience**:
   - Simultaneously queries Subdl and Subsource in parallel using `asyncio.gather`.
   - If one provider fails or times out (strict 6.0s limit), results from the surviving provider are seamlessly delivered.
   - Automatically deduplicates matching releases across providers.
   - Built-in Cinemeta integration for resolving titles and release years when needed.
3. **Ultra-Low Memory Footprint (<100MB RAM)**:
   - Built with lightweight asynchronous standard libraries.
   - `httpx.AsyncClient` with connection pooling, keep-alive, and strict timeouts.
   - Unpacks `.zip` archives directly in RAM using `io.BytesIO` and `zipfile` (no intermediate zip files written to disk).
4. **Automated LRU Disk Cache Management**:
   - Cache directory `/app/cache` keeps unpacked `.srt` files and download metadata.
   - Automated cleanup enforces strict storage bounds (max **1 GB** or **500 files**), evicting the Least Recently Used (LRU) files.
5. **Security & Arabic Encoding Transcoding**:
   - **Zip Slip Protection**: Rejects directory traversal attempts (`../`, `..\`, absolute paths).
   - **Transcoding**: Automatically converts Arabic subtitles in `Windows-1256` (CP1256) or `ISO-8859-6` to clean `UTF-8` so subtitles display without corrupted characters.

---

## API Authentication & Per-User Configuration

The addon supports two flexible authentication modes:
1. **Server-Wide Environment Variables** (`.env` or `docker-compose.yml`):
   - `SUBDL_API_KEY`: Used as default for all requests to SubDL.
   - `SUBSOURCE_API_KEY`: Used as default for requests to SubSource.
   - `OPENSUBTITLES_API_KEY`: Used as default for requests to OpenSubtitles.
2. **Per-User Stateless Configuration via Web UI / Manifest URL**:
   - Access the configuration page at `http://<HOST_IP>:7000/configure` (or `/{config}/configure` to edit existing setup).
   - Enter your personal SubDL, SubSource, or OpenSubtitles API keys.
   - Choose your preferred languages and toggle **Exclude HI** (Hearing Impaired).
   - Click **Install Addon** (`stremio://...`) to install directly into Stremio, or **Copy Link** to copy the manifest URL.
   - All settings are serialized into a stateless URL-safe Base64 token embedded in `/{config}/manifest.json`.

---

## Stremio Protocol Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/configure` | `GET` | Responsive configuration UI (Subdl/Subsource keys, languages, exclude HI) |
| `/{config}/configure` | `GET` | Responsive configuration UI prefilled with user preferences |
| `/manifest.json` | `GET` | Server-wide default Stremio v3 manifest (`org.ninjasubs.addon`) |
| `/{config}/manifest.json` | `GET` | User-configured manifest with personal API keys |
| `/subtitles/{type}/{id}.json` | `GET` | Direct subtitle search (uses server-wide .env keys) |
| `/{config}/subtitles/{type}/{id}.json` | `GET` | User-configured subtitle search (uses personal API keys) |
| `/sub/{sub_id}.srt` | `GET` | Serves UTF-8 `.srt` file with `Cache-Control: public, max-age=86400` |
| `/{config}/sub/{sub_id}.srt` | `GET` | Serves UTF-8 `.srt` file with user-configured fallback |
| `/health` | `GET` | Health status, API key status, and live LRU disk cache statistics |

---

## Quick Start with Docker & Docker Compose

### 1. Configure Environment
Copy `.env.example` to `.env` and set your machine's LAN IP address so other devices on your home network (Smart TV, Android TV, iPad) can access the addon:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
BASE_URL=http://localhost:7000
SUBDL_API_KEY=your_optional_subdl_key
SUBSOURCE_API_KEY=your_optional_subsource_key
OPENSUBTITLES_API_KEY=your_optional_opensubtitles_key
```

### 2. Launch with Docker Compose
```bash
docker compose up -d --build
```

Check logs:
```bash
docker compose logs -f
```

The container runs as a non-root user (`appuser`, UID 10001) under strict memory limits (150M cap) and mounts `./subs_cache` for persistent storage.

---

## Local Development & Testing

### 1. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Run the Verification Suite
Execute the automated test suite:
```bash
pytest -v tests/
```

Test coverage includes:
- `tests/test_config_parser.py`: User configuration Base64 encoding/decoding, backward compatibility, and invalid payload fallbacks.
- `tests/test_parser.py`: Movie and series ID parsing, season/episode extraction, edge cases.
- `tests/test_stremio_contract.py`: Manifest schema, /configure UI rendering, CORS headers, provider fallback, and LRU eviction.
- `tests/test_unpacking.py`: In-memory ZIP extractor, Zip Slip defense, non-SRT filtering, CP1256 Arabic transcoding.

### 3. Start the Server Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7000 --reload
```

Open your browser at `http://localhost:7000/configure` to configure your API keys and install the addon.
