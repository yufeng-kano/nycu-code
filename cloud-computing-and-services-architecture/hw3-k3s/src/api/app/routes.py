from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, jsonify, request
from psycopg import errors

from app.db import close_connection, get_connection

api = Blueprint(name="api", import_name=__name__)
api.teardown_app_request(f=close_connection)


def _name_from_request() -> str | None:
    """Read `name` from query string first, then from JSON body. Returns None if absent."""
    name = request.args.get(key="name")
    if name is None and request.is_json:
        body: Any = request.get_json(silent=True)
        if isinstance(body, dict):
            name = body.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None


@api.get("/health")
def health() -> tuple[Response, int]:
    """Readiness check: also proves the api tier can reach PostgreSQL through the `db` Service."""
    try:
        get_connection().execute(query="SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any DB failure means "not ready"
        return jsonify(status="error", database=str(exc)), 503
    return jsonify(status="ok", database="ok"), 200


@api.post("/add_name")
def add_name() -> tuple[Response, int]:
    name = _name_from_request()
    if name is None:
        return jsonify(error="name is required"), 400
    conn = get_connection()
    try:
        row = conn.execute(
            query="INSERT INTO names (name) VALUES (%s) RETURNING id",
            params=(name,),
        ).fetchone()
        conn.commit()
    except errors.UniqueViolation:
        conn.rollback()
        return jsonify(error=f"name '{name}' already exists"), 409
    assert row is not None
    return jsonify(id=row["id"], name=name), 201


@api.get("/get_names")
def get_names() -> Response:
    rows = get_connection().execute(query="SELECT id, name FROM names ORDER BY id").fetchall()
    return jsonify(names=[{"id": row["id"], "name": row["name"]} for row in rows])


@api.delete("/remove_name")
def remove_name() -> tuple[Response, int]:
    name = _name_from_request()
    if name is None:
        return jsonify(error="name is required"), 400
    conn = get_connection()
    cursor = conn.execute(query="DELETE FROM names WHERE name = %s", params=(name,))
    conn.commit()
    if cursor.rowcount == 0:
        return jsonify(error=f"name '{name}' not found"), 404
    return jsonify(deleted=name), 200
