#!/bin/sh
# Installs the working tree's build in place of ~/.local/bin/MemoryPilot,
# signed with the Developer ID like the release asset.
#
#   scripts/install-local.sh
#
# The Keychain trusts a program by its signature: an ad hoc build is a new
# program for it after every change and asks again for the vault key. Signed
# by the team with the MemoryPilot identifier, every build is the same program,
# so one "Always Allow" lasts. The binary it replaces is kept as
# .MemoryPilot-previous.bak, and the HTTP service is restarted on the new one.
set -eu

IDENTITY="Developer ID Application: Soflution ltd (TB8CC687M3)"
TEAM="TB8CC687M3"
DEST="${MEMORYPILOT_BIN:-$HOME/.local/bin/MemoryPilot}"
SERVICE="com.soflution.memorypilot-http"

die() { echo "install-local: $*" >&2; exit 1; }

cd "$(dirname "$0")/.."
[ "$(uname -s)" = "Darwin" ] || die "macOS only"
security find-identity -v -p codesigning | grep -q "$IDENTITY" || die "signing identity missing: $IDENTITY"

echo "▸ build"
cargo build --release --locked --features http --bin MemoryPilot

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp target/release/MemoryPilot "$STAGE/MemoryPilot"

echo "▸ sign"
codesign --force --options runtime --identifier MemoryPilot --sign "$IDENTITY" "$STAGE/MemoryPilot"
codesign --verify --strict "$STAGE/MemoryPilot"
codesign -d -r- "$STAGE/MemoryPilot" 2>&1 | grep -q "subject.OU\] = ${TEAM}" || die "designated requirement is not tied to team ${TEAM}"

echo "▸ install ${DEST}"
mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ]; then
	cp -p "$DEST" "$(dirname "$DEST")/.MemoryPilot-previous.bak"
fi
cp "$STAGE/MemoryPilot" "$DEST.new"
mv -f "$DEST.new" "$DEST"

if launchctl print "gui/$(id -u)/${SERVICE}" >/dev/null 2>&1; then
	echo "▸ restart ${SERVICE}"
	launchctl kickstart -k "gui/$(id -u)/${SERVICE}"
fi
"$DEST" --version
