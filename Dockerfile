# Standardized isolated Docker container for Agent-Bench execution
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BENCH_ENVIRONMENT=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source code
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY datasets/ ./datasets/
COPY README.md ./

# Install package dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,openai,anthropic]"

# Create unprivileged user for secure sandboxed execution
RUN useradd -u 1000 -m appuser && \
    mkdir -p /app/data/runs && \
    chown -R appuser:appuser /app

USER appuser

# Default entrypoint runs the bench CLI
ENTRYPOINT ["bench"]
CMD ["--config-dir", "configs", "run-suite", "pix_basic_v1"]
