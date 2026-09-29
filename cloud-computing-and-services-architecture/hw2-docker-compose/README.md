# Homework #2: 3-Tier Web Application with Docker Compose

Add, list and remove names through a web page. Four containers:

| Container      | Image / stack                          | Role                                              |
| -------------- | -------------------------------------- | ------------------------------------------------- |
| `hw2-proxy`    | nginx                                  | Public entry point on port 8080, routes `/` and `/api` |
| `hw2-frontend` | nginx serving a Vite + Vue 3 + TS build | Static single-page UI                            |
| `hw2-backend`  | Flask 3 on gunicorn, managed with uv   | JSON API, reads/writes the SQLite file            |
| `hw2-db`       | alpine + sqlite3                       | Applies `schema.sql` to the shared volume         |

## Run

```sh
docker compose up --build
```

Open http://localhost:8080. Stop with `docker compose down`, add `-v` to drop the data volume.

## API

| Method | Path                     | Body / query           | Success | Errors   |
| ------ | ------------------------ | ---------------------- | ------- | -------- |
| POST   | `/api/add_name`          | `{"name": "Alice"}`    | 201     | 400, 409 |
| GET    | `/api/get_names`         |                        | 200     |          |
| DELETE | `/api/remove_name`       | `?name=Alice` or JSON  | 200     | 400, 404 |
| GET    | `/api/health`            |                        | 200     |          |

## Local development

```sh
# backend
cd backend
uv sync
sqlite3 /tmp/names.db < ../db/schema.sql
DB_PATH=/tmp/names.db uv run flask --app wsgi run --port 8000
uv run ruff check . && uv run mypy && uv run pytest

# frontend (Vite dev server proxies /api to :8000)
cd frontend
npm install
npm run dev
```

See `REPORT.md` for the design discussion.
