FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/names.db

WORKDIR /srv

# Install dependencies first so this layer is cached until the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY wsgi.py ./

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "--frozen", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "wsgi:app"]
