#!/bin/sh
set -eu

DB_PATH="${DB_PATH:-/data/names.db}"

mkdir -p "$(dirname "$DB_PATH")"
sqlite3 "$DB_PATH" < /schema.sql
echo "schema applied to $DB_PATH"

# SQLite is an embedded library, not a server. There is no daemon to run,
# so this container stays alive only to satisfy the "database container" requirement.
exec tail -f /dev/null
