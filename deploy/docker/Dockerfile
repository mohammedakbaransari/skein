# deploy/docker/Dockerfile
# ===========================
# SKEIN Framework — Multi-stage Docker build
#
# Stage 1: builder  — installs dependencies, validates
# Stage 2: runtime  — minimal image, non-root, read-only FS
#
# Build:
#   docker build -f deploy/docker/Dockerfile -t skein-framework:latest .
#
# Run:
#   docker run -p 8080:8080 \
#     -e LLM_PROVIDER=anthropic \
#     -e LLM_API_KEY=sk-ant-... \
#     -e LLM_MODEL=claude-sonnet-4-6 \
#     skein-framework:latest
#
# Health check:
#   curl http://localhost:8080/health
#   curl http://localhost:8080/ready
#   curl http://localhost:8080/metrics

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into /install for copying to runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source code
COPY framework/ ./framework/
COPY agents/ ./agents/
COPY config/ ./config/
COPY data/ ./data/
COPY platform/ ./platform/
COPY scripts/ ./scripts/

# Validate imports — fail fast if any module has syntax errors
RUN PYTHONPATH=/build python3 -c "
import sys
sys.path.insert(0, '.')
modules = [
    'framework.core.types',
    'framework.core.registry',
    'framework.agents.base',
    'framework.resilience.retry',
    'framework.observability.logging',
    'framework.observability.metrics',
    'framework.orchestration.orchestrator',
    'framework.memory.store',
    'framework.governance.logger',
]
for m in modules:
    __import__(m)
    print(f'  OK  {m}')
print('Build validation passed')
"

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: create non-root user
RUN groupadd -r skein && useradd -r -g skein -u 1000 skein

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (no build tools, no source extras)
COPY --from=builder /build/framework ./framework
COPY --from=builder /build/agents ./agents
COPY --from=builder /build/config ./config
COPY --from=builder /build/data ./data
COPY --from=builder /build/platform ./platform

# Create runtime directories with correct ownership
RUN mkdir -p /app/logs/governance /app/logs/traces \
    && chown -R skein:skein /app/logs

# Drop to non-root user
USER skein

# Expose health port
EXPOSE 8080

# Environment defaults (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SKEIN_HEALTH_PORT=8080 \
    SKEIN_LOG_LEVEL=INFO \
    SKEIN_LOG_JSON=true \
    LLM_PROVIDER=ollama \
    LLM_MODEL=llama3.1 \
    LLM_BASE_URL=http://localhost:11434

# Health check (Docker native — Kubernetes uses its own probes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Entry point
CMD ["python3", "-m", "scripts.server"]
