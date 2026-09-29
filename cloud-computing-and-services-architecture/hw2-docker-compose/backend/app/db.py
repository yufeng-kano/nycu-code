from __future__ import annotations

import sqlite3

from flask import current_app, g


def get_connection() -> sqlite3.Connection:
    """Return one connection per request, opened lazily and closed by teardown."""
    if "db" not in g:
        conn = sqlite3.connect(current_app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        # WAL lets readers proceed while another gunicorn worker is writing.
        conn.execute("PRAGMA journal_mode=WAL")
        g.db = conn
    connection: sqlite3.Connection = g.db
    return connection


def close_connection(_exc: BaseException | None = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()
