#!/bin/bash
# Measure BPM/energy for tracks added since the last run, and fold the result
# into the file the engine reads.
#
# analyze_audio.py is already incremental — it skips every track it has a
# measurement for — so this is just the scheduled wrapper around it. Without a
# schedule the gap only grows: of 74,950 tracks in the library, 63,102 had a
# measured tempo, and the ~11,800 without one can never appear in a wind-down
# (no tempo, no curve) nor earn an audio boost for "gym" or "slapen".
#
# The engine notices the new file by itself (it re-reads it when the mtime
# changes), so nothing needs restarting afterwards.
#
# WHERE TO RUN THIS. Two routes, and the script picks whichever is available:
#
#   Local files (~0.6 s/track) — only where the NAS is mounted, which is
#   your-server, not the Jetson. Point MUSIC_PATH_MAP at the mounts.
#
#   Streaming over the API (~2.9 s/track) — works anywhere, including the
#   Jetson, at the cost of downloading each track. Set AUDIO_ALLOW_STREAM=1.
#   The first run has a backlog to clear (~9 h for 11,800 tracks); every run
#   after that only sees newly added music and finishes in minutes.
#
# Cron example — nightly at 03:00, before the weekly playlists at 04:00:
#   0 3 * * * AUDIO_ALLOW_STREAM=1 /home/USER/jltamp/ai/scripts/refresh_features.sh >> ~/jltamp-ai/features.log 2>&1
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Where this machine can see the music. JLTamp reports the paths its own
# container knows (/music/mp3/...), which have to be mapped to the mounts here.
export MUSIC_PATH_MAP="${MUSIC_PATH_MAP:-/music/mp3:/path/to/your/music,/music/flac:/path/to/your/flac}"
WORKERS="${WORKERS:-2}"
LOCK="/tmp/jltamp-analyze.lock"

# A long run must not have the next night's run pile on top of it: two
# analysers writing the same file is how progress gets lost.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -Is) an analysis run is still going — skipping this one."
    exit 0
fi

# Is the music actually here? The directory *existing* proves nothing: on the
# Jetson /path/to/your/music exists and is empty, because nothing is mounted there.
# An empty mount would make every track "unreadable" and the run would look
# like a success that happened to find no work.
local_music=0
IFS=',' read -ra PAIRS <<< "$MUSIC_PATH_MAP"
for pair in "${PAIRS[@]}"; do
    dest="${pair#*:}"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        local_music=1
    fi
done

if [ "$local_music" -eq 1 ]; then
    echo "$(date -Is) reading the music locally ($MUSIC_PATH_MAP)"
elif [ "${AUDIO_ALLOW_STREAM:-}" = "1" ]; then
    echo "$(date -Is) no local music — streaming over the API (slower)"
else
    echo "$(date -Is) ERROR: no music at $MUSIC_PATH_MAP and AUDIO_ALLOW_STREAM is not set."
    echo "  Run this where the NAS is mounted (your-server), or set"
    echo "  AUDIO_ALLOW_STREAM=1 to fetch each track over the API instead."
    exit 1
fi

python3 scripts/analyze_audio.py --workers "$WORKERS"

# Only meaningful when the run was sharded; harmless otherwise.
if compgen -G "data/track_features.shard*.json" > /dev/null; then
    python3 scripts/merge_features.py
fi

echo "$(date -Is) done"
