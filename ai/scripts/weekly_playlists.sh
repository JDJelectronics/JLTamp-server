#!/bin/bash
# Build the weekly per-user AI playlists: 🧬 Jouw DNA Mix and 🔮 Ontdekking van
# de Week, for every user.
#
# It calls the engine's own /ai/weekly endpoint with the admin token, so all
# the per-user logic (mint a token per user, build from their taste, create in
# their library) stays in one place rather than being duplicated here. The
# endpoint starts the build in the background and returns at once; this script
# then polls GET /ai/weekly until it is done and logs the outcome per user.
#
# Cron example (Monday 04:00):
#   0 4 * * 1 /home/USER/jltamp/ai/scripts/weekly_playlists.sh >> ~/jltamp-ai/weekly.log 2>&1
#
# macOS-proof: launchd starts with a bare PATH and BSD date has no `date -Is`,
# so the script sets both itself.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_URL="${AI_URL:-http://127.0.0.1:5000}"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# The venv's python, not whatever python3 launchd happens to find.
PY="$BASE_DIR/.venv/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
field() { "$PY" -c "import sys,json; print(json.load(sys.stdin).get('$1',''))" 2>/dev/null || true; }

# The admin token: log in with the service account from .env and use that.
set -a
# shellcheck disable=SC1091
source "$BASE_DIR/.env" 2>/dev/null || true
set +a
TOKEN=$(curl -s -m 20 -X POST "$JLTAMP_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$JLTAMP_EMAIL\",\"password\":\"$JLTAMP_PASSWORD\"}" | field token)

if [ -z "$TOKEN" ]; then
    echo "$(stamp) ERROR: could not obtain admin token"
    exit 1
fi

# The AI may still be reading the library (a restart takes a few minutes).
for _ in $(seq 1 30); do
    READY=$(curl -s -m 10 "$AI_URL/ai/health" | field ai_ready)
    [ "$READY" = "True" ] && break
    echo "$(stamp) AI not ready yet, waiting ..."
    sleep 60
done

echo "$(stamp) building weekly playlists ..."
curl -s -m 30 -X POST "$AI_URL/ai/weekly" -H "X-Plex-Token: $TOKEN"
echo

# Every user one after the other: allow the whole hour.
for _ in $(seq 1 120); do
    sleep 30
    STATE=$(curl -s -m 20 "$AI_URL/ai/weekly" -H "X-Plex-Token: $TOKEN" || true)
    STATUS=$(printf '%s' "$STATE" | field status)
    if [ -n "$STATUS" ] && [ "$STATUS" != "running" ]; then
        printf '%s' "$STATE" | "$PY" -m json.tool
        echo "$(stamp) done ($STATUS)"
        [ "$STATUS" = "done" ] && exit 0
        exit 1
    fi
done
echo "$(stamp) ERROR: still running after an hour"
exit 1
