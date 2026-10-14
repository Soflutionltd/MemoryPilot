#!/bin/zsh
# MemoryPilot remote access (iPhone, claude.ai web, Windows) through a
# Cloudflare named tunnel. The server only listens on 127.0.0.1 and every
# request needs the token from ~/.MemoryPilot/http.token.
#
#   scripts/remote-access.sh server            # LaunchAgent: MemoryPilot --http 7437
#   cloudflared tunnel login                   # once, in the browser (your Cloudflare account)
#   scripts/remote-access.sh tunnel memory.example.com
#   scripts/remote-access.sh url               # prints the connector URL (contains the token)
#   scripts/remote-access.sh stop              # unload both agents
set -e
BIN="$HOME/.local/bin/MemoryPilot"
PORT=7437
LA="$HOME/Library/LaunchAgents"
SERVER_PLIST="$LA/com.soflution.memorypilot-http.plist"
TUNNEL_PLIST="$LA/com.soflution.memorypilot-tunnel.plist"
TUNNEL_NAME="memorypilot"
mkdir -p "$LA" "$HOME/.MemoryPilot/logs"

case "$1" in
server)
  cat > "$SERVER_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.soflution.memorypilot-http</string>
  <key>ProgramArguments</key><array><string>$BIN</string><string>--http</string><string>$PORT</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>$HOME/.MemoryPilot/logs/http.log</string>
  <key>StandardOutPath</key><string>$HOME/.MemoryPilot/logs/http.log</string>
</dict></plist>
EOF
  launchctl bootout gui/$(id -u) "$SERVER_PLIST" 2>/dev/null || true
  launchctl bootstrap gui/$(id -u) "$SERVER_PLIST"
  sleep 3
  curl -s "http://127.0.0.1:$PORT/health" | head -5
  ;;
tunnel)
  HOST="$2"
  [ -z "$HOST" ] && { echo "usage: $0 tunnel <hostname>"; exit 1; }
  [ -f "$HOME/.cloudflared/cert.pem" ] || { echo "Run 'cloudflared tunnel login' first."; exit 1; }
  cloudflared tunnel list | grep -q " $TUNNEL_NAME " || cloudflared tunnel create "$TUNNEL_NAME"
  TID=$(cloudflared tunnel list | awk -v n="$TUNNEL_NAME" '$2==n {print $1}')
  cloudflared tunnel route dns "$TUNNEL_NAME" "$HOST" || true
  cat > "$HOME/.cloudflared/memorypilot.yml" <<EOF
tunnel: $TID
credentials-file: $HOME/.cloudflared/$TID.json
ingress:
  - hostname: $HOST
    service: http://127.0.0.1:$PORT
  - service: http_status:404
EOF
  echo "$HOST" > "$HOME/.MemoryPilot/remote.host"
  CF=$(command -v cloudflared)
  cat > "$TUNNEL_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.soflution.memorypilot-tunnel</string>
  <key>ProgramArguments</key><array><string>$CF</string><string>tunnel</string><string>--config</string><string>$HOME/.cloudflared/memorypilot.yml</string><string>run</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>$HOME/.MemoryPilot/logs/tunnel.log</string>
  <key>StandardOutPath</key><string>$HOME/.MemoryPilot/logs/tunnel.log</string>
</dict></plist>
EOF
  launchctl bootout gui/$(id -u) "$TUNNEL_PLIST" 2>/dev/null || true
  launchctl bootstrap gui/$(id -u) "$TUNNEL_PLIST"
  sleep 8
  curl -s "https://$HOST/health" | head -5
  ;;
url)
  HOST=$(cat "$HOME/.MemoryPilot/remote.host" 2>/dev/null || echo "<hostname>")
  echo "https://$HOST/mcp/$(cat "$HOME/.MemoryPilot/http.token")"
  ;;
stop)
  launchctl bootout gui/$(id -u) "$TUNNEL_PLIST" 2>/dev/null || true
  launchctl bootout gui/$(id -u) "$SERVER_PLIST" 2>/dev/null || true
  echo stopped
  ;;
*)
  sed -n 2,10p "$0"
  ;;
esac
