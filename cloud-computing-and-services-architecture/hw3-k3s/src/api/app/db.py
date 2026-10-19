from __future__ import annotations

import psycopg
from flask import current_app, g
from psycopg.rows import DictRow, dict_row


def get_connection() -> psycopg.Connection[DictRow]:
    """Return one connection per request, opened lazily and closed by teardown."""
    if "db" not in g:
        g.db = psycopg.connect(
            conninfo=current_app.config["DATABASE_URL"],
            row_factory=dict_row,
            connect_timeout=5,
        )
    connection: psycopg.Connection[DictRow] = g.db
    return connection


def close_connection(_exc: BaseException | None = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()
