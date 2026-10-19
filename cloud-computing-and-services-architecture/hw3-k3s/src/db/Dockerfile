FROM postgres:17-alpine

# The official entrypoint runs these scripts once, when it initialises an empty data directory.
# On later starts (the PVC already holds data) they are skipped, so existing rows are kept.
COPY schema.sql /docker-entrypoint-initdb.d/01-schema.sql

# Credentials (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB) come from a Kubernetes Secret.
EXPOSE 5432
