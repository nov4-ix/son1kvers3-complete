# === PRODUCTION DOCKERFILE ===
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    pkg-config \
    # Audio libraries
    libsndfile1-dev \
    libfftw3-dev \
    librubberband-dev \
    # System utilities
    curl \
    git \
    # SSL and crypto
    libssl-dev \
    libffi-dev \
    # FFmpeg for audio processing
    ffmpeg \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r son1k && useradd -r -g son1k son1k

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p storage/uploads storage/output storage/models logs \
    && mkdir -p storage/uploads/ghost storage/output/ghost \
    && mkdir -p backend/data

# Set permissions
RUN chown -R son1k:son1k /app

# Switch to non-root user
USER son1k

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# === DEVELOPMENT DOCKERFILE ===
FROM base as development

# Development specific packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    ruff

# Switch back to root for development setup
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to son1k user
USER son1k

# Development command with hot reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# === MINIMAL DOCKERFILE (for production optimization) ===
FROM python:3.11-alpine as minimal

# Alpine packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    curl \
    ffmpeg \
    && apk add --no-cache --virtual .build-deps \
    build-base \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps

# Copy only necessary files
COPY src/ ./src/
COPY backend/data/ ./backend/data/

# Create user and directories
RUN adduser -D -s /bin/sh son1k \
    && mkdir -p storage logs \
    && chown -R son1k:son1k /app

USER son1k

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]