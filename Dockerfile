# Optimized production image for W&B model registry
#
# This image does NOT bundle a model file. Instead, it loads models
# from Weights & Biases Model Registry at runtime, enabling:
# - Zero-downtime model updates (hot-reload)
# - Version tracking and rollback capability
# - Decoupled code deployment from model deployment
#
# Required env vars at runtime:
# - WANDB_API_KEY: W&B API key for registry access
# - USE_WANDB_REGISTRY=true: Enable registry mode
# - FINGRID_API_KEY: For real-time predictions

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for layer caching
COPY uv.lock pyproject.toml /app/

# Install ONLY production dependencies (no training, notebooks, or dev deps)
RUN uv sync --frozen --no-dev --no-install-project

# Copy only what's needed for serving (not notebooks, training scripts, etc.)
COPY app/ /app/app/
COPY ingestion/client.py /app/ingestion/client.py
COPY ingestion/processor.py /app/ingestion/processor.py
COPY ingestion/__init__.py /app/ingestion/__init__.py
COPY pipeline/__init__.py /app/pipeline/__init__.py
COPY pipeline/train.py /app/pipeline/train.py
COPY pipeline/features.py /app/pipeline/features.py

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# Default to W&B registry mode (can override at runtime)
ENV USE_WANDB_REGISTRY=true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run with optimized settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
