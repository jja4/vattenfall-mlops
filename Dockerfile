# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory to /app
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the lockfile and pyproject.toml first to leverage Docker cache
COPY uv.lock pyproject.toml /app/

# Install dependencies using uv
# --no-dev: Exclude development dependencies
# --frozen: Use the exact versions from uv.lock
# --no-install-project: We only want dependencies for now, not the project itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . /app

# Place the virtualenv in the path
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
