#!/bin/sh
# Prints the Homebrew formula for a release (used by release-macos.sh).
#
#   scripts/brew-formula.sh VERSION BINARY_SHA256 SOURCE_SHA256
#
# Apple silicon gets the signed binary of the release (no Keychain prompt
# after updates). Intel Macs and Linux build from source: ONNX Runtime has no
# prebuilt for x86_64-apple-darwin, and Linux has no Keychain.
set -eu
[ $# -eq 3 ] || { sed -n '2,8p' "$0"; exit 1; }
VERSION="$1"
BIN_SHA="$2"
SRC_SHA="$3"
BASE="https://github.com/Soflutionltd/MemoryPilot"

cat <<EOF
class Memorypilot < Formula
  desc "Local memory layer for AI agents (MCP server)"
  homepage "${BASE}"
  url "${BASE}/archive/refs/tags/v${VERSION}.tar.gz"
  version "${VERSION}"
  sha256 "${SRC_SHA}"
  license :cannot_represent # Soflution Source Available License

  head do
    url "${BASE}.git", branch: "main"
    depends_on "rust" => :build
  end

  on_macos do
    on_arm do
      url "${BASE}/releases/download/v${VERSION}/MemoryPilot-aarch64-apple-darwin.tar.gz"
      sha256 "${BIN_SHA}"
    end
    on_intel do
      depends_on "rust" => :build
    end
  end

  on_linux do
    depends_on "rust" => :build
  end

  def install
    if OS.mac? && Hardware::CPU.arm? && build.stable?
      bin.install "MemoryPilot"
    else
      system "cargo", "install", *std_cargo_args(path: "."), "--features", "http", "--bin", "MemoryPilot"
    end
  end

  test do
    assert_match "memorypilot", shell_output("#{bin}/MemoryPilot --version")
  end
end
EOF
