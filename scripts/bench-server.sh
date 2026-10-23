#!/usr/bin/env bash
# Isolation, expiry and latency on the live REST API. Needs the test Postgres
# (scripts/test-postgres.sh starts it). Does not call an LLM.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
PASS_FILE="$HOME/.cache/memorypilot-dev/pg.pass"
if [ -z "${MEMORYPILOT_TEST_PG_URL:-}" ]; then
  if [ -f "$PASS_FILE" ]; then
    MEMORYPILOT_TEST_PG_URL="postgresql://postgres:$(cat "$PASS_FILE")@127.0.0.1:55432/memorypilot"
  else
    echo "Run scripts/test-postgres.sh first (it starts the test Postgres)." >&2
    exit 1
  fi
fi
export MEMORYPILOT_PG_URL="$MEMORYPILOT_TEST_PG_URL"
export MEMORYPILOT_PG_TLS=off
export MEMORYPILOT_VAULT=off
export MEMORYPILOT_EMBEDDER=off
export MEMORYPILOT_BIND=127.0.0.1:0
bin="$root/target/debug/memorypilot-server"
[ -x "$bin" ] || cargo build -q -p memorypilot-server
log="$(mktemp)"
"$bin" serve >"$log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; rm -f "$log"' EXIT
for _ in $(seq 1 50); do
  if grep -q "listening on http://" "$log"; then
    break
  fi
  sleep 0.1
done
base="$(grep -o 'listening on http://[^ ]*' "$log" | tail -1 | awk '{print $3}')"
tenant="$("$bin" create-tenant "bench-$$")"
space="$("$bin" create-space "$tenant" app)"
key="$("$bin" create-key "$tenant" "$space" bench read,write,subjects | awk '/^key:/{print $2}')"
other="$("$bin" create-tenant "bench-other-$$")"
ospace="$("$bin" create-space "$other" app)"
okey="$("$bin" create-key "$other" "$ospace" bench read,write | awk '/^key:/{print $2}')"

curl -fsS -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
  -d '{"content":"The stand-up is at nine","kind":"fact"}' "$base/v1/memories" >/dev/null
curl -fsS -H "Authorization: Bearer $key" -H "X-MemoryPilot-Subject: alice" \
  -H "Content-Type: application/json" \
  -d '{"content":"Alice prefers green tea","kind":"preference"}' "$base/v1/memories" >/dev/null
# expired
curl -fsS -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
  -d '{"content":"Old stripe proxy","expires_at":"2020-01-01T00:00:00Z"}' "$base/v1/memories" >/dev/null

alice="$(curl -fsS -H "Authorization: Bearer $key" -H "X-MemoryPilot-Subject: alice" "$base/v1/memories")"
echo "$alice" | grep -q "green tea"
echo "$alice" | grep -q "stand-up"
other_list="$(curl -fsS -H "Authorization: Bearer $okey" "$base/v1/memories")"
if echo "$other_list" | grep -q "green tea"; then
  echo "other tenant saw alice" >&2
  exit 1
fi
search_expired="$(curl -fsS -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
  -d '{"query":"Old stripe proxy","limit":5}' "$base/v1/search")"
if echo "$search_expired" | grep -q "Old stripe"; then
  echo "expired memory still in search" >&2
  exit 1
fi

python3 - <<PY
import json, time, urllib.request
base = "$base"
key = "$key"
def search():
    req = urllib.request.Request(
        base + "/v1/search",
        data=json.dumps({"query": "stand-up time", "limit": 5}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()
search()
started = time.perf_counter()
for _ in range(20):
    search()
elapsed = (time.perf_counter() - started) / 20 * 1000
print(f"search_avg_ms {elapsed:.1f}")
if elapsed > 250:
    raise SystemExit(f"search too slow: {elapsed:.1f} ms")
print("isolation, expiry and latency: ok")
PY
