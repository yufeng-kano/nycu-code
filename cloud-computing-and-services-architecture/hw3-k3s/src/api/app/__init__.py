from __future__ import annotations

import os

from flask import Flask

from app.routes import api


def database_url_from_env() -> str:
    """Build the libpq URL from DATABASE_URL, or from the PG* variables set by the Deployment."""
    if url := os.environ.get(key="DATABASE_URL"):
        return url
    host = os.environ.get(key="PGHOST", default="db")
    port = os.environ.get(key="PGPORT", default="5432")
    user = os.environ.get(key="PGUSER", default="names")
    password = os.environ.get(key="PGPASSWORD", default="")
    dbname = os.environ.get(key="PGDATABASE", default="names")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def create_app(database_url: str | None = None) -> Flask:
    app = Flask(import_name=__name__)
    app.config["DATABASE_URL"] = database_url or database_url_from_env()
    app.register_blueprint(blueprint=api, url_prefix="/api")
    return app
