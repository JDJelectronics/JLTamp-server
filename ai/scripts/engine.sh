#!/bin/bash
# Start/stop the engine by pidfile.
#
# Managing it with `pkill -f app.main` is a trap: the controlling shell's own
# command line contains that string, so pgrep matches the shell and it kills
# itself. A pidfile names exactly one process.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PIDFILE="logs/engine.pid"
mkdir -p logs

case "${1:-status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    # -u: unbuffered. To a file (not a tty) Python block-buffers stdout, so the
    # log stayed empty for minutes while the engine ran fine — every diagnosis
    # read a blank file. Line-buffered output makes the log trustworthy.
    setsid nohup python3 -u -m app.main > logs/engine.log 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 2
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null && echo "started (pid $(cat "$PIDFILE"))" \
      || { echo "failed to start:"; tail -5 logs/engine.log; exit 1; }
    ;;
  stop)
    [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
    kill "$(cat "$PIDFILE")" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped"
    ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    ;;
esac
