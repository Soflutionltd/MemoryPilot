"""Compare a benchmark run with baseline runs (see scripts/bench-gate.sh).

Usage: python3 scripts/bench_gate.py WORK CANDIDATE BASE1 BASE2 [BASE3...]
Exit 1 when the candidate differs on anything the baseline runs agree on.
"""
import json
import re
import sys

NAMES = ["recall", "search", "fr", "episodic", "longmemeval"]
# Timings, plus two fields that vary between runs of the same binary
# (tail order of selected ids, cluster coherence of tied neighbours).
IGNORED = re.compile(
    r"(ms|latency|elapsed|duration|seconds|_at|timestamp|time|selected_memory_ids|cluster_coherence)$"
)


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not IGNORED.search(k)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            diff(a.get(key), b.get(key), f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{index}]", out)
    elif a != b:
        out.append((path, str(a)[:90], str(b)[:90]))
    return out


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    work, candidate, bases = sys.argv[1], sys.argv[2], sys.argv[3:]
    failed = False
    for name in NAMES:
        runs = [strip(json.load(open(f"{work}/{base}/{name}.json"))) for base in bases]
        noisy = {path for run in runs[1:] for path, _, _ in diff(runs[0], run)}
        current = strip(json.load(open(f"{work}/{candidate}/{name}.json")))
        real = [d for d in diff(runs[0], current) if d[0] not in noisy]
        print(f"{name}: {len(noisy)} unstable path(s) ignored, {len(real)} difference(s)")
        for path, before, after in real[:6]:
            print("   ", path, "|", before, "|", after)
        failed |= bool(real)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
