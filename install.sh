#!/bin/bash
set -euo pipefail

BINARY_NAME="MemoryPilot"
INSTALL_DIR="$HOME/.local/bin"
BINARY_PATH="$INSTALL_DIR/$BINARY_NAME"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

configured=()
skipped=()

info()  { printf "${BOLD}▸${NC} %s\n" "$1"; }
ok()    { printf "${GREEN}✓${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}⚠${NC} %s\n" "$1"; }
fail()  { printf "${RED}✗${NC} %s\n" "$1"; }

# ─── JSON helper: upsert a key into an MCP config file ────────────────────────
# Usage: upsert_mcp_config <file> <root_key> <server_json>
upsert_mcp_config() {
    local file="$1" root_key="$2" server_json="$3"

    if [ -f "$file" ]; then
        python3 -c "
import json, sys
path = sys.argv[1]
root = sys.argv[2]
entry = json.loads(sys.argv[3])
with open(path, 'r') as f:
    data = json.load(f)
if root not in data:
    data[root] = {}
data[root].update(entry)
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
" "$file" "$root_key" "$server_json"
    else
        mkdir -p "$(dirname "$file")"
        python3 -c "
import json, sys
root = sys.argv[1]
entry = json.loads(sys.argv[2])
data = {root: entry}
with open(sys.argv[3], 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
" "$root_key" "$server_json" "$file"
    fi
}

# ─── Step 1: Build or locate binary ───────────────────────────────────────────

printf "\n${BOLD}MemoryPilot Installer${NC}\n"
printf "━━━━━━━━━━━━━━━━━━━━━\n\n"

if [ -x "$BINARY_PATH" ]; then
    info "Binary found at $BINARY_PATH — skipping build"
else
    if ! command -v cargo &>/dev/null; then
        fail "cargo not found. Install Rust: https://rustup.rs"
        exit 1
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/Cargo.toml" ]; then
        info "Building MemoryPilot (release)..."
        cd "$SCRIPT_DIR"
        cargo build --release --quiet
        mkdir -p "$INSTALL_DIR"
        cp "target/release/$BINARY_NAME" "$BINARY_PATH"
    elif [ -f "./Cargo.toml" ]; then
        info "Building MemoryPilot (release)..."
        cargo build --release --quiet
        mkdir -p "$INSTALL_DIR"
        cp "target/release/$BINARY_NAME" "$BINARY_PATH"
    else
        fail "Cargo.toml not found. Run this script from the MemoryPilot repo."
        exit 1
    fi

    chmod +x "$BINARY_PATH"
    if [ "$(uname)" = "Darwin" ]; then
        xattr -cr "$BINARY_PATH" 2>/dev/null || true
    fi
    ok "Built and installed to $BINARY_PATH"
fi

if ! command -v python3 &>/dev/null; then
    fail "python3 is required for JSON config management"
    exit 1
fi

printf "\n${BOLD}Detecting IDEs...${NC}\n\n"

# ─── Step 2: Configure each IDE ──────────────────────────────────────────────

STDIO_CONFIG="{\"$BINARY_NAME\": {\"command\": \"$BINARY_PATH\"}}"
VSCODE_CONFIG="{\"$BINARY_NAME\": {\"type\": \"stdio\", \"command\": \"$BINARY_PATH\"}}"

# Cursor (global)
CURSOR_CONFIG="$HOME/.cursor/mcp.json"
if [ -d "$HOME/.cursor" ]; then
    upsert_mcp_config "$CURSOR_CONFIG" "mcpServers" "$STDIO_CONFIG"
    ok "Cursor  → $CURSOR_CONFIG"
    configured+=("Cursor")
else
    skipped+=("Cursor")
fi

# VS Code (global)
VSCODE_MCP="$HOME/.vscode/mcp.json"
if [ -d "$HOME/.vscode" ]; then
    upsert_mcp_config "$VSCODE_MCP" "servers" "$VSCODE_CONFIG"
    ok "VS Code → $VSCODE_MCP"
    configured+=("VS Code")
else
    skipped+=("VS Code")
fi

# Claude Desktop (macOS)
if [ "$(uname)" = "Darwin" ]; then
    CLAUDE_DESKTOP="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    if [ -d "$HOME/Library/Application Support/Claude" ]; then
        upsert_mcp_config "$CLAUDE_DESKTOP" "mcpServers" "$STDIO_CONFIG"
        ok "Claude Desktop → $CLAUDE_DESKTOP"
        configured+=("Claude Desktop")
    else
        skipped+=("Claude Desktop")
    fi
else
    skipped+=("Claude Desktop (macOS only)")
fi

# Windsurf
WINDSURF_CONFIG="$HOME/.codeium/windsurf/mcp_config.json"
if [ -d "$HOME/.codeium/windsurf" ]; then
    upsert_mcp_config "$WINDSURF_CONFIG" "mcpServers" "$STDIO_CONFIG"
    ok "Windsurf → $WINDSURF_CONFIG"
    configured+=("Windsurf")
else
    skipped+=("Windsurf")
fi

# Claude Code (CLI)
if command -v claude &>/dev/null; then
    claude mcp add "$BINARY_NAME" -- "$BINARY_PATH" 2>/dev/null && \
        ok "Claude Code → claude mcp add" || \
        warn "Claude Code → failed (may already exist)"
    configured+=("Claude Code")
else
    skipped+=("Claude Code")
fi

# Codex (CLI)
if command -v codex &>/dev/null; then
    codex mcp add "$BINARY_NAME" -- "$BINARY_PATH" 2>/dev/null && \
        ok "Codex → codex mcp add" || \
        warn "Codex → failed (may already exist)"
    configured+=("Codex")
else
    skipped+=("Codex")
fi

# ─── Step 3: Summary ─────────────────────────────────────────────────────────

printf "\n━━━━━━━━━━━━━━━━━━━━━\n"
printf "${BOLD}Summary${NC}\n\n"
printf "  Binary: ${GREEN}$BINARY_PATH${NC}\n"
printf "  DB:     ~/.MemoryPilot/memory.db\n\n"

if [ ${#configured[@]} -gt 0 ]; then
    printf "  ${GREEN}Configured:${NC}\n"
    for ide in "${configured[@]}"; do
        printf "    ✓ %s\n" "$ide"
    done
fi

if [ ${#skipped[@]} -gt 0 ]; then
    printf "  ${YELLOW}Not detected:${NC}\n"
    for ide in "${skipped[@]}"; do
        printf "    - %s\n" "$ide"
    done
fi

printf "\n  Restart your IDE(s) to activate MemoryPilot.\n"
printf "  All configured IDEs share the same memory.\n\n"
