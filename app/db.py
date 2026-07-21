"""SQLite via SQLAlchemy 2.0 (synchronous). SQLite keeps the server a single
self-contained container — no external database service to run or depend on.

Handlers are declared `def` (not `async def`) so FastAPI runs them in a thread
pool, which is the correct place for blocking SQLite/file/ffmpeg work.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import DB_PATH


class Base(DeclarativeBase):
    pass


engine = create_engine(
    f"sqlite:///{DB_PATH}",
    # SQLite + threadpool: allow use across threads; WAL for concurrent reads
    # while a scan writes. `timeout` = busy-timeout: a writer WAITS up to N
    # seconds for the lock instead of instantly raising "database is locked"
    # (critical while the background scan/enrichment is writing heavily).
    connect_args={"check_same_thread": False, "timeout": 60},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _add_missing_columns(conn) -> None:
    """create_all() creates missing TABLES but never adds a column to a table
    that already exists — so a new field on an existing model would break every
    query against a live database. SQLite's ADD COLUMN is non-destructive (it
    backfills the default), so we bring existing tables up to the model here.

    Only ever ADDS. Never drops or rewrites a column: data is the user's.
    """
    for table in Base.metadata.sorted_tables:
        existing = {
            row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")')
        }
        if not existing:
            continue  # table doesn't exist yet — create_all will make it
        for col in table.columns:
            if col.name in existing:
                continue
            ddl = col.type.compile(engine.dialect)
            default = ""
            if col.default is not None and getattr(col.default, "is_scalar", False):
                value = col.default.arg
                default = f" DEFAULT {value!r}" if isinstance(value, str) else f" DEFAULT {value}"
            conn.exec_driver_sql(
                f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}{default}'
            )
            log_added = f"{table.name}.{col.name}"
            print(f"[db] migrated: added column {log_added}")


def init_db() -> None:
    # Import models so they're registered on Base before create_all.
    from . import models  # noqa: F401

    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        conn.exec_driver_sql("PRAGMA busy_timeout=60000")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _add_missing_columns(conn)


def get_session():
    """FastAPI dependency — yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
