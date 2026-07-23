#!/bin/bash
# Build the weekly per-user AI playlists. Meant to run from cron once a week.
#
# It calls the engine's own /ai/weekly endpoint with the admin token, so all
# the per-user logic (mint a token per user, build from their taste, create in
# their library) stays in one place rather than being duplicated here.
#
# Cron example (Monday 04:00):
#   0 4 * * 1 /home/USER/jltamp/ai/scripts/weekly_playlists.sh >> ~/jltamp-ai/weekly.log 2>&1
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_URL="${AI_URL:-http://127.0.0.1:5000}"

# The admin token: log in with the service account from .env and use that.
source "$BASE_DIR/.env" 2>/dev/null || true
TOKEN=$(curl -s -m 20 -X POST "$JLTAMP_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$JLTAMP_EMAIL\",\"password\":\"$JLTAMP_PASSWORD\"}" \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))")

if [ -z "$TOKEN" ]; then
    echo "$(date -Is) ERROR: could not obtain admin token"
    exit 1
fi

echo "$(date -Is) building weekly playlists ..."
curl -s -m 300 -X POST "$AI_URL/ai/weekly" -H "X-Plex-Token: $TOKEN" \
    | python3 -m json.tool
echo "$(date -Is) done"
