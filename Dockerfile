# Optimized production image
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
COPY ingestion/ /app/ingestion/
COPY models/__init__.py /app/models/__init__.py
COPY models/model.pkl /app/models/model.pkl

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run with optimized settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
