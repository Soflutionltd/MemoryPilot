#!/usr/bin/env bash
# Unit tests for the TypeScript, Python and Dart REST clients.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

echo "== TypeScript"
(cd "$root/sdks/typescript" && node --test test/client.test.js)

echo "== Python"
(cd "$root/sdks/python" && PYTHONPATH=src python3 -m unittest tests.test_client -v)

echo "== Dart"
(cd "$root/sdks/dart" && dart pub get && dart test)
