"""HTTP surface — the exact endpoints frontend/src/services/aiService.ts calls.

The app discovers this service by probing port 5000 across the tailnet/LAN and
then talks to /health, /whoami, /ai/playlist, /ai/status and /ai/suggest. Those
paths and response shapes are a contract with a shipped Android build: changing
them breaks installed apps, so they stay as they are.

What did change is who may call them. The previous version bound 0.0.0.0 with
`CORS(*)` and no authentication, which let any page in any browser on the
network drive it.
"""
from __future__ import annotations

import faulthandler
import signal
import subprocess
import threading as _threading
import time as _time

# SIGUSR1 dumps every thread's stack to stderr — the definitive
# way to see where a hang actually is, no root needed.
faulthandler.register(signal.SIGUSR1, all_threads=True)

from flask import Flask, jsonify, request
from flask_cors import CORS

from . import config
from .engine import Engine
from .jobs import JobManager

app = Flask(__name__)

# Wildcard CORS would undo the API key for browser clients, so origins must be
# named explicitly. With none configured, cross-origin requests are refused and
# only native (non-browser) clients can reach the service.
if config.CORS_ORIGINS:
    CORS(app, origins=config.CORS_ORIGINS)

engine = Engine()
jobs = JobManager(max_jobs=config.MAX_JOBS, timeout=config.JOB_TIMEOUT_SEC)

# Discovery has to work before the client can authenticate, so /health and
# /whoami stay open. They expose no library data.
OPEN_PATHS = {"/health", "/ai/health", "/whoami"}


@app.before_request
def _authenticate():
    # When every caller must present their own JLTamp token, that token *is*
    # the authentication and the endpoints check it themselves. Demanding a
    # shared key on top would mean shipping a second secret to every client —
    # one the app has no way to know.
    if config.REQUIRE_USER_TOKEN:
        return None
    if not config.API_KEY or request.path in OPEN_PATHS or request.method == "OPTIONS":
        return None
    supplied = (request.headers.get("X-AI-Key")
                or request.args.get("key")
                or "")
    if supplied != config.API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return None


# Cached, because computing it forks a subprocess. Forking from a request
# thread while another thread runs numpy (a weekly build, say) is the classic
# multithreaded-fork deadlock: the child inherits a held malloc lock and hangs,
# and the parent blocks forever in _execute_child. /health is called often and
# the address never changes, so fork once and reuse — the hot path never forks.
_ts_ip: str | None = None
_ts_ip_at = 0.0
_ts_lock = _threading.Lock()
_TS_TTL = 3600


def _tailscale_ip() -> str | None:
    global _ts_ip, _ts_ip_at
    now = _time.time()
    if _ts_ip is not None and now - _ts_ip_at < _TS_TTL:
        return _ts_ip
    # One fork at a time, and never while holding up other requests longer
    # than necessary — a stale value is fine until the refresh completes.
    if not _ts_lock.acquire(blocking=False):
        return _ts_ip
    try:
        out = subprocess.check_output(["/usr/bin/tailscale", "ip", "-4"],
                                      text=True, timeout=5)
        _ts_ip = next((l.strip() for l in out.splitlines()
                       if l.strip().startswith("100.")), None)
        _ts_ip_at = now
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        _ts_lock.release()
    return _ts_ip


# Two paths, one handler. When the service is reached directly, /health is the
# natural spot and shipped app builds already call it. When it is proxied under
# a JLTamp hostname only /ai/* is routed here, so /health there would hit
# JLTamp's own server instead — hence the alias.
@app.get("/health")
@app.get("/ai/health")
def health():
    data = engine.health()
    data["tailscale_ip"] = _tailscale_ip()
    data["jobs"] = jobs.stats()
    return jsonify(data)


@app.get("/whoami")
def whoami():
    return jsonify({"tailscale_ip": _tailscale_ip()})


@app.get("/ai/suggest")
def suggest():
    return jsonify({"suggestions": [
        "chille herfstavond met zachte piano",
        "energieke workout met stevige beats",
        "focus-flow voor een lange codeersessie",
        "gezellige vrijdagavond met vrienden",
        "melancholische regenachtige zondag",
        "opzwepende feestset voor laat op de avond",
        "rustige zondagochtend met koffie",
        "nostalgische jaren 80 hits om mee te zingen",
    ]})


def _caller_token() -> str:
    """The caller's own JLTamp session token, however the app sent it.

    aiService.ts already holds this token to talk to JLTamp; reusing it here
    means the AI needs no separate credential, and can act as the person who
    actually asked rather than as one shared account.
    """
    auth = request.headers.get("Authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return (request.headers.get("X-Plex-Token")
            or bearer
            or request.args.get("X-Plex-Token", "")).strip()


@app.post("/ai/playlist")
def playlist():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is leeg"}), 400

    token = _caller_token()
    if config.REQUIRE_USER_TOKEN:
        # Reachable from the internet: every caller must prove who they are,
        # and JLTamp is the authority on that.
        ctx = engine.user_context(token)
        if not ctx:
            return jsonify({"error": "ongeldig of ontbrekend JLTamp-token"}), 401

    # The duplicate guard is per user: two people asking for "chill muziek"
    # want two playlists, in their own libraries.
    guard = f"{token[:12]}:{prompt.lower()}"

    running = jobs.existing(guard)
    if running:
        return jsonify({"status": "already_running", "job_id": running,
                        "poll_interval": 1000, "timeout": config.JOB_TIMEOUT_SEC * 1000})

    if jobs.full():
        return jsonify({"error": "server busy, probeer het zo opnieuw"}), 429

    job_id = jobs.submit(guard, lambda _g: engine.handle(prompt, token))
    return jsonify({"status": "processing", "job_id": job_id,
                    "poll_interval": 1000, "timeout": config.JOB_TIMEOUT_SEC * 1000})


@app.post("/ai/weekly")
def weekly():
    """Build the weekly per-user playlists now. Admin token required — this
    acts on every user's library, so it must not be open to any caller."""
    token = _caller_token()
    ctx = engine.user_context(token)
    if not ctx or not (ctx["user"].get("isAdmin") or ctx["user"].get("is_admin")):
        return jsonify({"error": "admin token required"}), 403
    only = request.args.get("user_id", type=int)
    results = engine.generate_weekly(only_user_id=only)
    return jsonify({"results": results})


@app.get("/ai/status")
def status():
    job = jobs.get(request.args.get("job_id", ""))
    if not job:
        return jsonify({"error": "job not found"}), 404
    # The app reads the playlist name off the top level of a finished job.
    if job.get("status") == "done" and isinstance(job.get("result"), dict):
        job = {**job, **job["result"]}
    return jsonify(job)


def main() -> None:
    if not config.API_KEY:
        print("⚠️  AI_API_KEY is not set — this service is unauthenticated. "
              "Fine on a private tailnet, risky on a shared LAN.")
    engine.start()
    print(f"🎧 JLTamp AI on {config.HOST}:{config.PORT} → {config.JLTAMP_URL}")

    # Flask's built-in server is single-process and explicitly not for
    # production. Waitress is a real WSGI server and pure Python, so it
    # installs on the Jetson without a build step.
    try:
        from waitress import serve
    except ImportError:
        print("⚠️  waitress not installed — falling back to Flask's development "
              "server. Install it with: pip3 install waitress")
        app.run(host=config.HOST, port=config.PORT, threaded=True)
        return
    serve(app, host=config.HOST, port=config.PORT, threads=8,
          # Prompt jobs return immediately; only /health should ever be slow.
          channel_timeout=120, ident="JLTamp AI")


if __name__ == "__main__":
    main()
