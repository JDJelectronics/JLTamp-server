"""JLTamp Music Server — entrypoint.

Wires the Plex-compatible + multi-user routers together, initialises the SQLite
DB, seeds the admin account, and (optionally) serves the web UI. Scanning is
now MANUAL (Plex-style): libraries are added and scanned from the admin UI, so
there is no auto-scan on startup.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from . import config, backup
from .db import init_db
from .deps import require_admin, seed_admin
from .scanner import scan_all, scan_state, request_stop
from .routers import auth, media, stream, artwork, playlists, libraries, admin_users, likes, uploads, history, stats, mix, session

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jltamp")

VERSION = "0.3.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: prepare the DB, repair known data issues, guarantee the owner
    # admin, and begin periodic consistent DB snapshots for the backup.
    init_db()
    _repair_years()
    seed_admin()
    backup.start()
    log.info("═" * 60)
    log.info(" JLTamp Music Server (multi-user) v%s", VERSION)
    log.info("  name        : %s", config.SERVER_NAME)
    log.info("  server id   : %s", config.SERVER_ID)
    log.info("  music dir   : %s (read-only)", config.MUSIC_DIR)
    log.info("  data dir    : %s", config.DATA_DIR)
    log.info("  admin       : %s", config.ADMIN_EMAIL)
    log.info("  registration: %s", "open" if config.OPEN_REGISTRATION else "invite-only")
    log.info("═" * 60)
    yield
    # Shutdown: one last consistent snapshot so a clean restart/redeploy always
    # leaves the freshest possible backup copy.
    backup.make_snapshot()


# Interactive docs and the OpenAPI schema are OFF: on a public server they would
# hand an anonymous visitor a full map of every endpoint and request body. Set
# JLTAMP_DEV=1 locally to turn them back on.
_DEV = os.environ.get("JLTAMP_DEV", "").strip().lower() in ("1", "true", "yes", "on")
app = FastAPI(
    title="JLTamp Music Server", version=VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _DEV else None,
    redoc_url="/redoc" if _DEV else None,
    openapi_url="/openapi.json" if _DEV else None,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

for r in (auth.router, media.router, stream.router, artwork.router,
          playlists.router, libraries.router, admin_users.router, likes.router,
          uploads.router, history.router, stats.router, mix.router,
          session.router):
    app.include_router(r)


def _repair_years() -> None:
    """Older scans stored a compact date tag ("20180721") as the year, so those
    albums/tracks sort and filter as the year twenty-million. The scanner now
    parses the year properly; this repairs rows already in the database so the
    fix applies without a full rescan. Idempotent — a no-op once clean."""
    from .db import engine

    with engine.begin() as conn:
        fixed = 0
        for table in ("albums", "tracks"):
            # Leading 4 digits — "20180721" and "201807" both yield 2018, which
            # dividing by a fixed power of ten would not.
            res = conn.exec_driver_sql(
                f"UPDATE {table} "
                f"SET year = CAST(SUBSTR(CAST(year AS TEXT), 1, 4) AS INTEGER) "
                f"WHERE year IS NOT NULL AND year > 9999"
            )
            fixed += res.rowcount or 0
            # Whatever is still not a plausible year is not a year.
            conn.exec_driver_sql(
                f"UPDATE {table} SET year = NULL "
                f"WHERE year IS NOT NULL AND (year < 1000 OR year > 2999)"
            )
        if fixed:
            log.info("Repaired %d rows with a date stored as the year", fixed)


# ── health check (for the container HEALTHCHECK / uptime monitors) ───────────
@app.get("/healthz")
def healthz():
    """Liveness + DB reachability. Unauthenticated on purpose so a container
    healthcheck / uptime monitor can hit it, but it leaks nothing beyond ok/db."""
    from .db import engine
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:
        db_ok = False
    status = "ok" if db_ok else "degraded"
    return JSONResponse({"status": status, "db": db_ok, "version": VERSION},
                        status_code=200 if db_ok else 503)


# ── admin: manual scan of every library ──────────────────────────────────────
@app.post("/admin/rescan")
def admin_rescan(_: None = Depends(require_admin), full: bool = False):
    threading.Thread(target=scan_all, kwargs={"full": full}, daemon=True).start()
    return {"ok": True, "state": scan_state()}


@app.post("/admin/rescan/stop")
def admin_rescan_stop(_: None = Depends(require_admin)):
    """Stop a running scan cleanly — keeps everything scanned so far."""
    return {"ok": True, "state": request_stop()}


@app.get("/admin/status")
def admin_status(_: None = Depends(require_admin)):
    return scan_state()


# ── admin: loudness analysis (volume normalisation) ──────────────────────────
@app.post("/admin/analyze-loudness")
def admin_analyze_loudness(_: None = Depends(require_admin), limit: int | None = None):
    """Measure loudness (ffmpeg) for tracks without a gain yet, and store it so
    the app can level every track to the same volume. Resumable + stoppable."""
    from . import loudness
    return {"ok": True, "state": loudness.analyze_async(limit=limit)}


@app.post("/admin/analyze-loudness/stop")
def admin_analyze_loudness_stop(_: None = Depends(require_admin)):
    from . import loudness
    return {"ok": True, "state": loudness.request_stop()}


@app.get("/admin/analyze-loudness/status")
def admin_analyze_loudness_status(_: None = Depends(require_admin)):
    from . import loudness
    return loudness.state()


# ── API root descriptor (kept at /api so the web UI can own "/") ─────────────
def _descriptor():
    return {
        "server": "JLTamp Music Server",
        "version": VERSION,
        "name": config.SERVER_NAME,
        "machineIdentifier": config.SERVER_ID,
        # Public, unauthenticated: the login screen needs to know whether to offer
        # a "create account" panel at all, and it has to ask before signing in.
        "openRegistration": config.OPEN_REGISTRATION,
        "status": scan_state(),
    }


@app.get("/api")
def api_root():
    return _descriptor()


# ── Web UI (Expo web export), served last so API routes win ──────────────────
# Privacy policy — a real page at /privacy (declared before the SPA catch-all
# below so it isn't swallowed by the app shell). Used for the Play Store listing.
_PRIVACY = Path(__file__).parent / "static" / "privacy.html"
_DELETE_ACCOUNT = Path(__file__).parent / "static" / "account-deletion.html"


@app.get("/privacy")
def _privacy():
    return FileResponse(_PRIVACY, media_type="text/html")


@app.get("/delete-account")
def _delete_account():
    return FileResponse(_DELETE_ACCOUNT, media_type="text/html")


WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))
_INDEX = WEB_DIR / "index.html"

if WEB_DIR.is_dir() and _INDEX.exists():
    # Serve hashed static assets directly.
    for sub in ("_expo", "assets", "static"):
        d = WEB_DIR / sub
        if d.is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=d), name=sub)

    @app.get("/")
    def _root_index():
        return FileResponse(_INDEX)

    @app.get("/{full_path:path}")
    def _spa(full_path: str):
        """SPA fallback: serve the requested static file if it exists, else the
        app shell (index.html) so client-side routes (e.g. /invite/xxx) work.
        API routes are declared above, so they are matched before this."""
        candidate = (WEB_DIR / full_path).resolve()
        try:
            if candidate.is_file() and WEB_DIR.resolve() in candidate.parents:
                return FileResponse(candidate)
        except Exception:
            pass
        return FileResponse(_INDEX)
else:
    @app.get("/")
    def root():
        return JSONResponse(_descriptor())
