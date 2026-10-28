#!/bin/sh
# Publishes the signed automatic update manifest (latest.json) of a release.
# Run it once the three archives are on the release: Linux ones from CI,
# the macOS one from scripts/release-macos.sh.
#
#   scripts/publish-update-manifest.sh v4.12.0 [--percent 10] [--full-after-hours 24]
#   scripts/publish-update-manifest.sh v4.12.0 --pause | --resume | --full
#
# --pause stops the rollout (installed copies keep the version they have),
# --resume restarts it, --full serves every installation now. Servers read
# releases/latest/download/latest.json, so the tag must be the latest release.
#
# The minisign secret key is read from 1Password (op-sof) into a private
# temporary file, never printed, and the signature is checked against the
# public key compiled into src/update.rs before anything is uploaded.
set -eu

REPO="Soflutionltd/MemoryPilot"
KEY_REF="op://Soflution/77adwexi24pfdmhekerilaxtxm/notesPlain"
ASSETS="MemoryPilot-aarch64-apple-darwin.tar.gz MemoryPilot-x86_64-unknown-linux-gnu.tar.gz MemoryPilot-aarch64-unknown-linux-gnu.tar.gz"

die() {
	echo "✗ $*" >&2
	exit 1
}

TAG="${1:-}"
case "$TAG" in
v[0-9]*) shift ;;
*) die "usage: $0 vX.Y.Z [--percent N] [--full-after-hours H] | --pause | --resume | --full" ;;
esac

ACTION="new"
PERCENT=10
HOURS=24
while [ $# -gt 0 ]; do
	case "$1" in
	--percent) PERCENT="${2:?--percent needs a value}"; shift 2 ;;
	--full-after-hours) HOURS="${2:?--full-after-hours needs a value}"; shift 2 ;;
	--pause) ACTION="pause"; shift ;;
	--resume) ACTION="resume"; shift ;;
	--full) ACTION="full"; shift ;;
	*) die "unknown option: $1" ;;
	esac
done

command -v minisign >/dev/null || die "minisign missing: brew install minisign"
command -v op-sof >/dev/null || die "op-sof missing"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC_KEY="$(sed -n 's/^const UPDATE_PUBLIC_KEY: &str = "\(.*\)";$/\1/p' "$ROOT/src/update.rs")"
[ -n "$PUBLIC_KEY" ] || die "UPDATE_PUBLIC_KEY not found in src/update.rs"

umask 077
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

if [ "$ACTION" = "new" ]; then
	echo "▸ download ${TAG} archives"
	mkdir "$WORK/dist"
	for asset in $ASSETS; do
		gh release download "$TAG" --repo "$REPO" --dir "$WORK/dist" --pattern "$asset" --pattern "$asset.sha256" ||
			die "$asset is not on ${TAG} yet: wait for the CI build and scripts/release-macos.sh"
	done
	(cd "$WORK/dist" && for asset in $ASSETS; do shasum -a 256 -c "$asset.sha256" >/dev/null || exit 1; done) ||
		die "an archive does not match its published .sha256"
	python3 "$ROOT/scripts/update-manifest.py" new "${TAG#v}" "$WORK/dist" \
		--percent "$PERCENT" --full-after-hours "$HOURS" >"$WORK/latest.json"
else
	echo "▸ ${ACTION} ${TAG} rollout"
	mkdir "$WORK/current"
	gh release download "$TAG" --repo "$REPO" --dir "$WORK/current" --pattern latest.json --pattern latest.json.minisig ||
		die "no manifest on ${TAG}: publish it first"
	minisign -Vq -P "$PUBLIC_KEY" -m "$WORK/current/latest.json" -x "$WORK/current/latest.json.minisig" ||
		die "the manifest on ${TAG} is not signed by our key: not re-signing it"
	python3 "$ROOT/scripts/update-manifest.py" edit "$WORK/current/latest.json" "$ACTION" >"$WORK/latest.json"
fi

echo "▸ sign"
op-sof read "$KEY_REF" >"$WORK/update.key"
printf '\n' >>"$WORK/update.key"
minisign -S -s "$WORK/update.key" -m "$WORK/latest.json" -x "$WORK/latest.json.minisig" \
	-t "MemoryPilot ${TAG} update manifest" >/dev/null </dev/null
rm -f "$WORK/update.key"
minisign -Vq -P "$PUBLIC_KEY" -m "$WORK/latest.json" -x "$WORK/latest.json.minisig" ||
	die "signature does not verify with the public key of src/update.rs"

echo "▸ upload"
gh release upload "$TAG" "$WORK/latest.json" "$WORK/latest.json.minisig" --repo "$REPO" --clobber
echo "✓ ${TAG} manifest published (${ACTION})"
