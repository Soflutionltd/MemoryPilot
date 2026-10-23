#!/usr/bin/env bash
# Runs the server mode's Postgres tests against a throwaway pgvector
# container (Docker), listening on 127.0.0.1 only. The password is
# generated once and kept in ~/.cache/memorypilot-dev (local, dev only).
#
#   scripts/test-postgres.sh          start the container if needed, run the tests
#   scripts/test-postgres.sh --stop   remove the container
set -euo pipefail

CONTAINER="mp-pg-test"
IMAGE="pgvector/pgvector:pg17"
PORT="55432"
PASS_FILE="$HOME/.cache/memorypilot-dev/pg.pass"

if [ "${1:-}" = "--stop" ]; then
  docker rm -f "$CONTAINER" >/dev/null
  echo "Removed $CONTAINER"
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running (start Docker Desktop)." >&2
  exit 1
fi

if [ ! -f "$PASS_FILE" ]; then
  mkdir -p "$(dirname "$PASS_FILE")"
  (umask 077 && openssl rand -hex 16 >"$PASS_FILE")
fi

if [ -z "$(docker ps -q -f name="^${CONTAINER}$")" ]; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" -p "127.0.0.1:${PORT}:5432" \
    -e POSTGRES_PASSWORD="$(cat "$PASS_FILE")" -e POSTGRES_DB=memorypilot "$IMAGE" >/dev/null
fi
for _ in $(seq 1 60); do
  docker exec "$CONTAINER" pg_isready -U postgres -d memorypilot >/dev/null 2>&1 && break
  sleep 1
done

export MEMORYPILOT_TEST_PG_URL="postgresql://postgres:$(cat "$PASS_FILE")@127.0.0.1:${PORT}/memorypilot"
export MEMORYPILOT_VAULT=off
cd "$(dirname "$0")/.."
cargo test -p memorypilot-server -- --include-ignored
