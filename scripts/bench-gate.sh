#!/bin/sh
# Refactoring gate: the deterministic benchmarks must give the same results
# before and after a change.
#
#   scripts/bench-gate.sh snapshot          freeze a copy of ~/.MemoryPilot/memory.db
#   scripts/bench-gate.sh run BINARY NAME   run the benchmarks into $WORK/NAME
#   python3 scripts/bench_gate.py $WORK CANDIDATE BASE1 BASE2 BASE3
#
# Runs happen on the frozen copy, in an isolated HOME, with the vault, the
# network connectors and Git verification off. Run the baseline binary three
# times: what varies between those runs is ignored by bench_gate.py.
set -u
WORK="${WORK:-/tmp/mp-bench}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
umask 077
mkdir -p "$WORK"

case "${1:-}" in
snapshot)
	mkdir -p "$WORK/snapshot"
	sqlite3 "$HOME/.MemoryPilot/memory.db" ".backup $WORK/snapshot/memory.db"
	cp "$HOME/.MemoryPilot/memory.db.ann.usearch" "$HOME/.MemoryPilot/memory.db.ann.usearch.keys.json" "$WORK/snapshot/" 2>/dev/null
	echo "snapshot in $WORK/snapshot"
	;;
run)
	BIN="$2"
	OUT="$WORK/$3"
	[ -f "$WORK/snapshot/memory.db" ] || { echo "no snapshot: run '$0 snapshot' first"; exit 1; }
	mkdir -p "$OUT"
	bench() {
		NAME="$1"
		shift
		H=$(mktemp -d "$WORK/home.XXXX")
		mkdir -p "$H/.MemoryPilot" "$H/.cache"
		cp "$WORK"/snapshot/memory.db* "$H/.MemoryPilot/"
		ln -s "$HOME/.cache/fastembed" "$H/.cache/fastembed"
		ln -s "$HOME/.fastembed_cache" "$H/.fastembed_cache"
		(cd "$REPO" && HOME="$H" MEMORYPILOT_VAULT=off MEMORYPILOT_CONNECTORS=off MEMORYPILOT_VERIFY=off \
			"$BIN" "$@" >"$OUT/$NAME.json" 2>"$OUT/$NAME.err")
		echo "$NAME exit=$?"
		rm -rf "$H"
	}
	bench recall --benchmark-recall
	bench search --benchmark-search
	bench fr --benchmark-fr
	bench episodic --benchmark-episodic
	bench longmemeval --benchmark-longmemeval benchmarks/longmemeval_s_cleaned.json --limit 60
	;;
*)
	sed -n '2,13p' "$0"
	exit 1
	;;
esac
