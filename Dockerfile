# Multi-stage lightweight Dockerfile for NinjaSubs Stremio Addon
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create non-root system user and prepare cache directory
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser && \
    mkdir -p /app/cache && \
    chown -R appuser:appuser /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the static `alass` subtitle synchronizer (Rust, pre-compiled linux64),
# ffprobe/ffmpeg for Tier-2 embedded subtitle track probing/extraction, and
# bsdtar (libarchive-tools) for RAR/7z/tar reference-subtitle archives.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates ffmpeg libarchive-tools && \
    curl -fsSL -o /usr/local/bin/alass \
        https://github.com/kaegi/alass/releases/download/v2.0.0/alass-linux64 && \
    chmod +x /usr/local/bin/alass && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY app/ /app/app/

# Ensure non-root ownership
RUN chown -R appuser:appuser /app

# Switch to non-root execution
USER appuser

# Expose Stremio Addon port
EXPOSE 7000

# Start server using uvloop for high concurrency under 100MB RAM footprint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000", "--loop", "uvloop"]
