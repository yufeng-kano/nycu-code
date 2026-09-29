from __future__ import annotations

import sqlite3
from typing import Any

from flask import Blueprint, Response, jsonify, request

from app.db import close_connection, get_connection

api = Blueprint("api", __name__)
api.teardown_app_request(close_connection)


def _name_from_request() -> str | None:
    """Read `name` from query string first, then from JSON body. Returns None if absent."""
    name = request.args.get("name")
    if name is None and request.is_json:
        body: Any = request.get_json(silent=True)
        if isinstance(body, dict):
            name = body.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None


@api.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@api.post("/add_name")
def add_name() -> tuple[Response, int]:
    name = _name_from_request()
    if name is None:
        return jsonify({"error": "name is required"}), 400
    conn = get_connection()
    try:
        cursor = conn.execute("INSERT INTO names (name) VALUES (?)", (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": f"name '{name}' already exists"}), 409
    return jsonify({"id": cursor.lastrowid, "name": name}), 201


@api.get("/get_names")
def get_names() -> Response:
    rows = get_connection().execute("SELECT id, name FROM names ORDER BY id").fetchall()
    return jsonify({"names": [{"id": row["id"], "name": row["name"]} for row in rows]})


@api.delete("/remove_name")
def remove_name() -> tuple[Response, int]:
    name = _name_from_request()
    if name is None:
        return jsonify({"error": "name is required"}), 400
    conn = get_connection()
    cursor = conn.execute("DELETE FROM names WHERE name = ?", (name,))
    conn.commit()
    if cursor.rowcount == 0:
        return jsonify({"error": f"name '{name}' not found"}), 404
    return jsonify({"deleted": name}), 200
