"""Instellingen van de server die je in Beheer aanpast in plaats van in de
omgeving van de container.

Wat in docker-compose staat blijft de beginwaarde; zet je het in Beheer, dan
wint dat. Zo kost een andere servernaam geen herstart.
"""
from __future__ import annotations

import time

from . import config
from .db import SessionLocal
from .models import Meta

_CACHE: dict[str, tuple[float, str]] = {}


def _meta(sleutel: str) -> str | None:
    nu = time.time()
    hit = _CACHE.get(sleutel)
    if hit and nu - hit[0] < 30:
        return hit[1] or None
    db = SessionLocal()
    try:
        rij = db.get(Meta, sleutel)
        waarde = (rij.value if rij else "") or ""
    finally:
        db.close()
    _CACHE[sleutel] = (nu, waarde)
    return waarde or None


def zet_meta(sleutel: str, waarde: str) -> None:
    db = SessionLocal()
    try:
        rij = db.get(Meta, sleutel)
        if not rij:
            rij = Meta(key=sleutel)
            db.add(rij)
        rij.value = waarde
        db.commit()
    finally:
        db.close()
    _CACHE.pop(sleutel, None)


def servernaam() -> str:
    return _meta("server_naam") or config.SERVER_NAME
