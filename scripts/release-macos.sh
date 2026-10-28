#!/bin/sh
# Signed and notarized macOS release asset (Apple silicon).
#
#   scripts/release-macos.sh
#
# Builds the tag of the workspace version (vX.Y.Z, already pushed and checked
# out), signs it with the Developer ID of the login Keychain, notarizes it with
# the App Store Connect API key, uploads MemoryPilot-aarch64-apple-darwin.tar.gz
# (+ .sha256) to the GitHub release, then points the Homebrew tap at it.
#
# macOS lets a program read its Keychain item without a prompt only if it is
# signed by the team that created the item: an unsigned build asks for the
# Keychain password after every update. The CI (release.yml) builds Linux only
# so that it never replaces this asset with an unsigned one.
set -eu

IDENTITY="Developer ID Application: Soflution ltd (TB8CC687M3)"
TEAM="TB8CC687M3"
NOTARY_KEY_ID="${NOTARY_KEY_ID:-77ZLGAGK7W}"
NOTARY_KEY="${NOTARY_KEY:-$HOME/.appstoreconnect/private_keys/AuthKey_${NOTARY_KEY_ID}.p8}"
NOTARY_ISSUER="${NOTARY_ISSUER:-ab5aeb0f-f6b3-43e8-bf3f-5ed0dc325471}"
REPO="Soflutionltd/MemoryPilot"
TAP="Soflutionltd/homebrew-memorypilot"
TARGET="aarch64-apple-darwin"
ASSET="MemoryPilot-${TARGET}.tar.gz"

die() { echo "release-macos: $*" >&2; exit 1; }

cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || die "run on an Apple silicon Mac"
[ -f "$NOTARY_KEY" ] || die "notarization key missing: $NOTARY_KEY"
security find-identity -v -p codesigning | grep -q "$IDENTITY" || die "signing identity missing: $IDENTITY"

VERSION="$(sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p' Cargo.toml)"
[ -n "$VERSION" ] || die "no [workspace.package] version in Cargo.toml"
TAG="v${VERSION}"
git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null || die "tag ${TAG} is not on origin: push it first"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "${TAG}^{commit}")" ] || die "HEAD is not ${TAG}: check out the tag"
git diff --quiet HEAD || die "uncommitted changes: the asset must be the tag, nothing else"

echo "▸ build ${TAG} (${TARGET})"
cargo build --release --locked --features http --bin MemoryPilot --target "$TARGET"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "target/${TARGET}/release/MemoryPilot" "$STAGE/MemoryPilot"

echo "▸ sign"
codesign --force --options runtime --timestamp --identifier MemoryPilot --sign "$IDENTITY" "$STAGE/MemoryPilot"
codesign --verify --strict "$STAGE/MemoryPilot"
codesign -d -r- "$STAGE/MemoryPilot" 2>&1 | grep -q "subject.OU\] = ${TEAM}" || die "designated requirement is not tied to team ${TEAM}"

echo "▸ notarize (a few minutes)"
ditto -c -k "$STAGE/MemoryPilot" "$STAGE/notary.zip"
xcrun notarytool submit "$STAGE/notary.zip" \
	--key "$NOTARY_KEY" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER" \
	--wait --output-format json >"$STAGE/notary.json"
STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$STAGE/notary.json")"
if [ "$STATUS" != "Accepted" ]; then
	ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("id",""))' "$STAGE/notary.json")"
	die "notarization ${STATUS:-failed} (submission ${ID}): xcrun notarytool log ${ID} --key ... for details"
fi

echo "▸ package"
tar -czf "$STAGE/$ASSET" -C "$STAGE" MemoryPilot
(cd "$STAGE" && shasum -a 256 "$ASSET" >"$ASSET.sha256")
BIN_SHA="$(cut -d' ' -f1 "$STAGE/$ASSET.sha256")"

echo "▸ upload to ${REPO} ${TAG}"
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
	gh release create "$TAG" --repo "$REPO" --verify-tag --title "MemoryPilot ${TAG}" --generate-notes
fi
gh release upload "$TAG" "$STAGE/$ASSET" "$STAGE/$ASSET.sha256" --repo "$REPO" --clobber

echo "▸ Homebrew tap"
curl -fsSL "https://github.com/${REPO}/archive/refs/tags/${TAG}.tar.gz" -o "$STAGE/source.tar.gz"
SRC_SHA="$(shasum -a 256 "$STAGE/source.tar.gz" | cut -d' ' -f1)"
curl -fsSL "https://github.com/${REPO}/releases/download/${TAG}/${ASSET}" -o "$STAGE/check.tar.gz"
[ "$(shasum -a 256 "$STAGE/check.tar.gz" | cut -d' ' -f1)" = "$BIN_SHA" ] || die "uploaded asset differs from the signed one"
gh repo clone "$TAP" "$STAGE/tap" -- --quiet --depth 1
sh scripts/brew-formula.sh "$VERSION" "$BIN_SHA" "$SRC_SHA" >"$STAGE/tap/Formula/memorypilot.rb"
git -C "$STAGE/tap" add Formula/memorypilot.rb
if git -C "$STAGE/tap" diff --cached --quiet; then
	echo "  formula already at ${TAG}"
else
	git -C "$STAGE/tap" commit --quiet -m "memorypilot ${VERSION}: signed binary on Apple silicon"
	git -C "$STAGE/tap" push --quiet origin HEAD
fi

echo "✓ ${TAG}: signed, notarized, uploaded; Homebrew installs the signed binary"

echo "▸ automatic update manifest"
PUBLISHED="$(gh release view "$TAG" --repo "$REPO" --json assets -q '.assets[].name')"
if echo "$PUBLISHED" | grep -qx "MemoryPilot-x86_64-unknown-linux-gnu.tar.gz" &&
	echo "$PUBLISHED" | grep -qx "MemoryPilot-aarch64-unknown-linux-gnu.tar.gz"; then
	sh scripts/publish-update-manifest.sh "$TAG"
else
	echo "  Linux archives not on ${TAG} yet: once CI is done, run scripts/publish-update-manifest.sh ${TAG}"
fi
