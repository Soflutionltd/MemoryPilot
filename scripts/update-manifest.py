#!/usr/bin/env python3
"""Builds or edits latest.json, the manifest read by the automatic updater
(src/update.rs). Release tooling only: nothing here runs on user machines.

  update-manifest.py new VERSION DIST_DIR [--percent N] [--full-after-hours H]
  update-manifest.py edit MANIFEST pause|resume|full

The manifest is printed on stdout; scripts/publish-update-manifest.sh signs it.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

ASSETS = (
    "MemoryPilot-aarch64-apple-darwin.tar.gz",
    "MemoryPilot-x86_64-unknown-linux-gnu.tar.gz",
    "MemoryPilot-aarch64-unknown-linux-gnu.tar.gz",
)
VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new(args: argparse.Namespace) -> dict:
    if not VERSION.match(args.version):
        sys.exit(f"version must be X.Y.Z, got {args.version!r}")
    if not 0 <= args.percent <= 100:
        sys.exit("--percent must be between 0 and 100")
    if args.full_after_hours < 0:
        sys.exit("--full-after-hours must not be negative")
    dist = pathlib.Path(args.dist)
    missing = [name for name in ASSETS if not (dist / name).is_file()]
    if missing:
        sys.exit(f"missing archives in {dist}: {', '.join(missing)}")
    now = int(time.time())
    return {
        "version": args.version,
        "published_at": now,
        "paused": False,
        "rollout": {
            "percent": args.percent,
            "full_at": now + args.full_after_hours * 3600,
        },
        "assets": {name: sha256(dist / name) for name in ASSETS},
    }


def edit(args: argparse.Namespace) -> dict:
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    if args.action == "pause":
        manifest["paused"] = True
    elif args.action == "resume":
        manifest["paused"] = False
    else:
        manifest["paused"] = False
        manifest["rollout"] = {"percent": 100, "full_at": int(time.time())}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("new")
    build.add_argument("version")
    build.add_argument("dist")
    build.add_argument("--percent", type=int, default=10)
    build.add_argument("--full-after-hours", type=int, default=24)
    change = commands.add_parser("edit")
    change.add_argument("manifest")
    change.add_argument("action", choices=("pause", "resume", "full"))
    args = parser.parse_args()
    manifest = new(args) if args.command == "new" else edit(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
