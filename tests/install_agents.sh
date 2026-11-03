#!/bin/bash
# install.sh in a throwaway HOME: without a TTY only Claude Code gets the
# rules and skills; --agents-tools picks others; --uninstall-agents undoes.
# Uses the binary already built in this checkout (placed where install.sh
# finds it, so nothing is downloaded or compiled).
#
#   cargo build --release && bash tests/install_agents.sh
#   MEMORYPILOT_TEST_BINARY=/path/to/MemoryPilot bash tests/install_agents.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${MEMORYPILOT_TEST_BINARY:-$REPO/target/release/MemoryPilot}"
REAL_HOME="$(/usr/bin/python3 -c 'import os, pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')"

TEST_HOME="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "$TEST_HOME"' EXIT
export HOME="$TEST_HOME"
# No real claude / codex: install.sh would register the MCP server with them.
export PATH="$TEST_HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin"
unset CLAUDE_CONFIG_DIR CODEX_HOME GEMINI_CLI_HOME OPENCODE_CONFIG_DIR MEMORYPILOT_DB

if [ "$HOME" = "$REAL_HOME" ] || [ "$(cd "$HOME" && pwd -P)" = "$(cd "$REAL_HOME" && pwd -P)" ]; then
    echo "refusing to run: HOME is the real home ($REAL_HOME)" >&2
    exit 1
fi
case "$HOME" in
    "$REAL_HOME"|"$REAL_HOME"/*) echo "refusing to run: HOME is inside the real home" >&2; exit 1 ;;
esac
for tool in claude codex; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "refusing to run: '$tool' is on PATH ($(command -v "$tool"))" >&2
        exit 1
    fi
done
if [ ! -x "$BINARY" ]; then
    echo "missing $BINARY: run cargo build --release first" >&2
    exit 1
fi

failures=0
check() {
    if eval "$2"; then
        echo "ok   - $1"
    else
        echo "FAIL - $1"
        failures=$((failures + 1))
    fi
}
run_install() {
    bash "$REPO/install.sh" "$@" </dev/null >"$TEST_HOME/install.log" 2>&1 || {
        cat "$TEST_HOME/install.log"
        echo "FAIL - install.sh $* exited with $?"
        exit 1
    }
}

mkdir -p "$HOME/.local/bin" "$HOME/bin" "$HOME/.claude" "$HOME/.cursor" "$HOME/.codex" "$HOME/.gemini"
cp "$BINARY" "$HOME/.local/bin/MemoryPilot"
printf '# My rules\n' > "$HOME/.claude/CLAUDE.md"
RULES="$HOME/.MemoryPilot/agents/rules/AGENTS.md"
MANIFEST="$HOME/.MemoryPilot/agents/.manifest.json"
tools_in_manifest() { /usr/bin/python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["tools"]))' "$MANIFEST"; }
count_hooks() { /usr/bin/python3 -c '
import json, sys
config = json.load(open(sys.argv[1]))
print(sum(1 for g in config["hooks"].get(sys.argv[2], []) for h in g.get("hooks", [g]) if "MemoryPilot" in h.get("command", "")))
' "$@"; }

# 1. No TTY, no option: Claude Code only.
run_install
check "agents folder created with an example-only AGENTS.md" '[ -f "$RULES" ] && head -1 "$RULES" | grep -q "^<!--" && [ -d "$HOME/.MemoryPilot/agents/skills" ] && [ -d "$HOME/.MemoryPilot/agents/vendor" ]'
check "skills and vendor are empty" '[ -z "$(find "$HOME/.MemoryPilot/agents/skills" "$HOME/.MemoryPilot/agents/vendor" -mindepth 1)" ]'
check "manifest lists Claude Code only" '[ "$(tools_in_manifest)" = "claude" ]'
check "CLAUDE.md kept and imports the rules once" '[ "$(head -1 "$HOME/.claude/CLAUDE.md")" = "# My rules" ] && [ "$(grep -c "^@$RULES\$" "$HOME/.claude/CLAUDE.md")" = 1 ]'
check "Codex not wired" '[ ! -e "$HOME/.codex/AGENTS.md" ]'
check "Gemini not wired" '[ ! -e "$HOME/.gemini/GEMINI.md" ]'
check "Cursor rule not generated" '[ ! -e "$HOME/.cursor/rules/agents.mdc" ]'
check "no ~/.agents/skills" '[ ! -e "$HOME/.agents" ]'
check "hooks installed (SessionStart, Stop, SessionEnd)" '[ "$(count_hooks "$HOME/.claude/settings.json" SessionStart)" = 1 ] && [ "$(count_hooks "$HOME/.claude/settings.json" SessionEnd)" = 1 ] && [ "$(count_hooks "$HOME/.codex/hooks.json" Stop)" = 1 ]'
check "notice printed" 'grep -q "rules/AGENTS.md" "$TEST_HOME/install.log" && grep -q "Wired: Claude Code" "$TEST_HOME/install.log"'

# 2. --yes again: same result, no duplicates.
run_install --yes
check "--yes: still Claude Code only" '[ "$(tools_in_manifest)" = "claude" ]'
check "--yes: no duplicate import" '[ "$(grep -c "^@$RULES\$" "$HOME/.claude/CLAUDE.md")" = 1 ]'
check "--yes: no duplicate hooks" '[ "$(count_hooks "$HOME/.claude/settings.json" SessionStart)" = 1 ] && [ "$(count_hooks "$HOME/.claude/settings.json" SessionEnd)" = 1 ]'

# 3. Explicit choice.
run_install --yes --agents-tools=claude,codex
check "--agents-tools adds Codex" '[ "$(tools_in_manifest)" = "claude,codex" ] && grep -q "MemoryPilot agents: begin" "$HOME/.codex/AGENTS.md"'

# 4. Uninstall through install.sh.
run_install --uninstall-agents --tools codex
check "--uninstall-agents --tools codex" '[ ! -e "$HOME/.codex/AGENTS.md" ] && [ "$(tools_in_manifest)" = "claude" ]'
run_install --uninstall-agents
check "--uninstall-agents restores CLAUDE.md" '[ "$(cat "$HOME/.claude/CLAUDE.md")" = "# My rules" ]'
check "--uninstall-agents keeps the folder" '[ -f "$RULES" ]'

if [ "$failures" -gt 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed (HOME=$TEST_HOME, removed on exit)"
