from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from app.routes import api


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path or os.environ.get("DB_PATH", "/data/names.db")
    Path(app.config["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    app.register_blueprint(api, url_prefix="/api")
    return app
