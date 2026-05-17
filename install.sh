#!/bin/bash
set -euo pipefail

BINARY_NAME="MemoryPilot"
INSTALL_DIR="$HOME/.local/bin"
BINARY_PATH="$INSTALL_DIR/$BINARY_NAME"
REPO_OWNER="Soflutionltd"
REPO_NAME="MemoryPilot"
REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}"

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

# ─── Standalone helpers: detect target + fetch pre-built binary ──────────────

detect_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Darwin)
            case "$arch" in
                arm64) echo "aarch64-apple-darwin" ;;
                x86_64) echo "x86_64-apple-darwin" ;;
                *) return 1 ;;
            esac
            ;;
        Linux)
            case "$arch" in
                x86_64) echo "x86_64-unknown-linux-gnu" ;;
                aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

try_download_prebuilt() {
    local target="$1" asset_url
    if ! command -v curl &>/dev/null; then
        return 1
    fi
    asset_url="${REPO_URL}/releases/latest/download/MemoryPilot-${target}.tar.gz"
    info "Trying pre-built binary for ${target}..."
    local tmp
    tmp="$(mktemp -d)"
    if curl -fsSL "$asset_url" -o "$tmp/MemoryPilot.tar.gz" 2>/dev/null; then
        tar -xzf "$tmp/MemoryPilot.tar.gz" -C "$tmp"
        if [ -f "$tmp/$BINARY_NAME" ]; then
            mkdir -p "$INSTALL_DIR"
            mv "$tmp/$BINARY_NAME" "$BINARY_PATH"
            chmod +x "$BINARY_PATH"
            if [ "$(uname)" = "Darwin" ]; then
                xattr -cr "$BINARY_PATH" 2>/dev/null || true
            fi
            rm -rf "$tmp"
            return 0
        fi
    fi
    rm -rf "$tmp"
    return 1
}

build_from_source() {
    if ! command -v cargo &>/dev/null; then
        fail "cargo not found and no pre-built binary available."
        fail "Install Rust from https://rustup.rs and re-run, or grab"
        fail "a release manually from ${REPO_URL}/releases/latest"
        exit 1
    fi
    local src_dir="$1"
    info "Building MemoryPilot from source (release + HTTP)..."
    (cd "$src_dir" && cargo build --release --features http --quiet)
    mkdir -p "$INSTALL_DIR"
    cp "$src_dir/target/release/$BINARY_NAME" "$BINARY_PATH"
    chmod +x "$BINARY_PATH"
    if [ "$(uname)" = "Darwin" ]; then
        xattr -cr "$BINARY_PATH" 2>/dev/null || true
    fi
}

clone_and_build() {
    if ! command -v git &>/dev/null; then
        fail "git not found. Install git or run the in-repo install."
        exit 1
    fi
    local clone_dir
    clone_dir="$(mktemp -d)/MemoryPilot"
    info "Cloning ${REPO_URL}..."
    git clone --depth 1 --quiet "$REPO_URL" "$clone_dir"
    build_from_source "$clone_dir"
}

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
    # Three install paths, in order of preference:
    #   1. Pre-built release binary (fastest, no Rust required).
    #   2. In-repo cargo build (when run from a clone).
    #   3. Standalone clone + cargo build (curl | bash from internet).
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
    target="$(detect_target || true)"

    if [ -n "$target" ] && try_download_prebuilt "$target"; then
        ok "Installed pre-built binary to $BINARY_PATH"
    elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/Cargo.toml" ]; then
        build_from_source "$SCRIPT_DIR"
        ok "Built from source ($SCRIPT_DIR) → $BINARY_PATH"
    elif [ -f "./Cargo.toml" ]; then
        build_from_source "$PWD"
        ok "Built from source ($PWD) → $BINARY_PATH"
    else
        warn "No pre-built binary for this platform and no local checkout found."
        info "Falling back to clone + build (requires git + cargo)..."
        clone_and_build
        ok "Built from cloned source → $BINARY_PATH"
    fi
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

# Gemini CLI (standard MCP format)
GEMINI_CONFIG="$HOME/.gemini/settings.json"
if [ -d "$HOME/.gemini" ] || command -v gemini &>/dev/null; then
    upsert_mcp_config "$GEMINI_CONFIG" "mcpServers" "$STDIO_CONFIG"
    ok "Gemini CLI → $GEMINI_CONFIG"
    configured+=("Gemini CLI")
else
    skipped+=("Gemini CLI")
fi

# OpenCode (standard MCP format)
OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json"
if [ -d "$HOME/.config/opencode" ] || command -v opencode &>/dev/null; then
    upsert_mcp_config "$OPENCODE_CONFIG" "mcp" "$STDIO_CONFIG"
    ok "OpenCode → $OPENCODE_CONFIG"
    configured+=("OpenCode")
else
    skipped+=("OpenCode")
fi

# Cline (VS Code extension — auto-detect global storage on macOS / Linux)
CLINE_MAC="$HOME/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
CLINE_LIN="$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
if [ -d "$(dirname "$CLINE_MAC")" ]; then
    upsert_mcp_config "$CLINE_MAC" "mcpServers" "$STDIO_CONFIG"
    ok "Cline → $CLINE_MAC"
    configured+=("Cline")
elif [ -d "$(dirname "$CLINE_LIN")" ]; then
    upsert_mcp_config "$CLINE_LIN" "mcpServers" "$STDIO_CONFIG"
    ok "Cline → $CLINE_LIN"
    configured+=("Cline")
else
    skipped+=("Cline")
fi

# Roo Code (VS Code extension)
ROO_MAC="$HOME/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/cline_mcp_settings.json"
ROO_LIN="$HOME/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/cline_mcp_settings.json"
if [ -d "$(dirname "$ROO_MAC")" ]; then
    upsert_mcp_config "$ROO_MAC" "mcpServers" "$STDIO_CONFIG"
    ok "Roo Code → $ROO_MAC"
    configured+=("Roo Code")
elif [ -d "$(dirname "$ROO_LIN")" ]; then
    upsert_mcp_config "$ROO_LIN" "mcpServers" "$STDIO_CONFIG"
    ok "Roo Code → $ROO_LIN"
    configured+=("Roo Code")
else
    skipped+=("Roo Code")
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
printf "  ${BOLD}For ChatGPT Desktop:${NC}\n"
printf "    Run: MemoryPilot --http 7437\n"
printf "    Then add http://localhost:7437/mcp as a custom connector.\n\n"
