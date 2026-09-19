# Multi-stage lightweight Dockerfile for Stremio Arabic Subtitles Addon
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
