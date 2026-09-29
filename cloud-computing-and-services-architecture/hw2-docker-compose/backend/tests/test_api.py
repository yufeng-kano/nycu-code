from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from app import create_app

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[FlaskClient]:
    db_path = tmp_path / "names.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())
    app = create_app(db_path=str(db_path))
    app.testing = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client: FlaskClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_add_then_list(client: FlaskClient) -> None:
    resp = client.post("/api/add_name", json={"name": "Alice"})
    assert resp.status_code == 201
    assert resp.get_json()["name"] == "Alice"

    resp = client.get("/api/get_names")
    assert resp.status_code == 200
    assert [row["name"] for row in resp.get_json()["names"]] == ["Alice"]


def test_add_rejects_empty_and_duplicate(client: FlaskClient) -> None:
    assert client.post("/api/add_name", json={"name": "   "}).status_code == 400
    assert client.post("/api/add_name", json={}).status_code == 400
    assert client.post("/api/add_name", json={"name": "Bob"}).status_code == 201
    assert client.post("/api/add_name", json={"name": "Bob"}).status_code == 409


def test_remove_by_query_and_by_body(client: FlaskClient) -> None:
    client.post("/api/add_name", json={"name": "Carol"})
    client.post("/api/add_name", json={"name": "Dave"})

    assert client.delete("/api/remove_name?name=Carol").status_code == 200
    assert client.delete("/api/remove_name", json={"name": "Dave"}).status_code == 200
    assert client.delete("/api/remove_name?name=Carol").status_code == 404
    assert client.delete("/api/remove_name").status_code == 400
    assert client.get("/api/get_names").get_json() == {"names": []}
