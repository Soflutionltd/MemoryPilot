#!/usr/bin/env bash
# Runs the server mode's Postgres tests against a throwaway pgvector
# container (Docker), listening on 127.0.0.1 only, TLS enabled. The password
# and the test CA are generated once and kept in ~/.cache/memorypilot-dev
# (local, dev only).
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

# TLS: a throwaway CA and a server certificate for "localhost" only, so the
# tests can check that a missing CA or another host name is refused.
TLS_DIR="$HOME/.cache/memorypilot-dev/tls"
if [ ! -f "$TLS_DIR/server.crt" ]; then
  mkdir -p "$TLS_DIR"
  (
    umask 077
    cd "$TLS_DIR"
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj "/CN=MemoryPilot test CA" \
      -keyout ca.key -out ca.crt 2>/dev/null
    openssl req -newkey rsa:2048 -nodes -subj "/CN=localhost" -keyout server.key -out server.csr 2>/dev/null
    printf "subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n" >server.ext
    openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 3650 \
      -extfile server.ext -out server.crt 2>/dev/null
  )
fi
if [ "$(docker exec -u postgres "$CONTAINER" psql -d memorypilot -tAc 'SHOW ssl')" != "on" ]; then
  docker cp "$TLS_DIR/server.crt" "$CONTAINER:/var/lib/postgresql/server.crt"
  docker cp "$TLS_DIR/server.key" "$CONTAINER:/var/lib/postgresql/server.key"
  docker exec -u root "$CONTAINER" sh -c \
    'chown postgres:postgres /var/lib/postgresql/server.* && chmod 600 /var/lib/postgresql/server.key'
  for setting in "ssl_cert_file = '/var/lib/postgresql/server.crt'" \
    "ssl_key_file = '/var/lib/postgresql/server.key'" "ssl = on"; do
    docker exec -u postgres "$CONTAINER" psql -q -d memorypilot -c "ALTER SYSTEM SET $setting" >/dev/null
  done
  docker exec -u postgres "$CONTAINER" psql -q -d memorypilot -c "SELECT pg_reload_conf()" >/dev/null
  sleep 1
fi

export MEMORYPILOT_TEST_PG_TLS_CA="$TLS_DIR/ca.crt"
export MEMORYPILOT_TEST_PG_URL="postgresql://postgres:$(cat "$PASS_FILE")@127.0.0.1:${PORT}/memorypilot"
export MEMORYPILOT_VAULT=off
cd "$(dirname "$0")/.."
cargo test -p memorypilot-server -- --include-ignored
