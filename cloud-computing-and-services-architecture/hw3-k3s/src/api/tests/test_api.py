from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from flask.testing import FlaskClient

from app import create_app

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"

# Tests run against a real PostgreSQL, e.g.
#   docker run --rm -d -p 5432:5432 -e POSTGRES_PASSWORD=test postgres:17-alpine
#   TEST_DATABASE_URL=postgresql://postgres:test@localhost:5432/postgres uv run pytest
TEST_DATABASE_URL = os.environ.get(key="TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    condition=not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set",
)


@pytest.fixture
def client() -> Iterator[FlaskClient]:
    assert TEST_DATABASE_URL is not None
    with psycopg.connect(conninfo=TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute(query="DROP TABLE IF EXISTS names")
        conn.execute(query=SCHEMA_PATH.read_text())
    app = create_app(database_url=TEST_DATABASE_URL)
    app.testing = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client: FlaskClient) -> None:
    resp = client.get(path="/api/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "database": "ok"}


def test_add_then_list(client: FlaskClient) -> None:
    resp = client.post(path="/api/add_name", json={"name": "Alice"})
    assert resp.status_code == 201
    assert resp.get_json()["name"] == "Alice"

    resp = client.get(path="/api/get_names")
    assert resp.status_code == 200
    assert [row["name"] for row in resp.get_json()["names"]] == ["Alice"]


def test_add_rejects_empty_and_duplicate(client: FlaskClient) -> None:
    assert client.post(path="/api/add_name", json={"name": "   "}).status_code == 400
    assert client.post(path="/api/add_name", json={}).status_code == 400
    assert client.post(path="/api/add_name", json={"name": "Bob"}).status_code == 201
    assert client.post(path="/api/add_name", json={"name": "Bob"}).status_code == 409


def test_remove_by_query_and_by_body(client: FlaskClient) -> None:
    client.post(path="/api/add_name", json={"name": "Carol"})
    client.post(path="/api/add_name", json={"name": "Dave"})

    assert client.delete(path="/api/remove_name?name=Carol").status_code == 200
    assert client.delete(path="/api/remove_name", json={"name": "Dave"}).status_code == 200
    assert client.delete(path="/api/remove_name?name=Carol").status_code == 404
    assert client.delete(path="/api/remove_name").status_code == 400
    assert client.get(path="/api/get_names").get_json() == {"names": []}
