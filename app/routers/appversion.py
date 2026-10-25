"""Welke versie van de app minimaal nodig is.

Waarom dit bestaat: in de winkel staat een versie die je niet meer wilt hebben —
omdat er een fout in zit die je gerepareerd hebt, of omdat de server iets vraagt
wat die versie niet kan. Zonder dit blijft iemand daar maanden op zitten, en
meldt hij een probleem dat allang weg is. Precies wat er nu gebeurde met een
gebruiker op build 6.

De regel staat in de database (`meta`), per platform, en is met één verzoek te
veranderen. De app vraagt hem op bij het opstarten.

**Wat het bewust NIET doet:** iemand buitensluiten omdat de server even niet
bereikbaar was. De app blokkeert alleen als hij een antwoord heeft gekregen én
daaruit blijkt dat hij te oud is. Geen antwoord betekent doorwerken.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, HTTPException

from ..db import SessionLocal
from ..deps import require_admin, optional_user
from ..models import Meta, User

router = APIRouter(tags=["app"])

META_SLEUTEL = "app_min_version"

# Waar de app te halen is. Staat hier en niet in de app, zodat een verhuizing
# naar een ander adres geen nieuwe app-versie kost.
STORE = {
    "ios": "https://apps.apple.com/app/id6807401608",
    "android": "https://play.google.com/store/apps/details?id=com.jltamp.app",
}


def _lees(db) -> dict:
    rij = db.get(Meta, META_SLEUTEL)
    if not rij or not rij.value:
        return {}
    try:
        uit = json.loads(rij.value)
        return uit if isinstance(uit, dict) else {}
    except Exception:
        return {}


@router.get("/app/requirement")
def eis(platform: str = "", user: User | None = Depends(optional_user)):
    """Wat de app moet hebben om te mogen draaien.

    Zonder ingestelde eis komt er `minBuild: 0` terug — dan blokkeert er niets.
    Bewust ook zonder inloggen te bevragen: iemand die niet meer langs het
    inlogscherm komt omdat zijn versie te oud is, moet dat wél te horen krijgen.
    """
    db = SessionLocal()
    try:
        alles = _lees(db)
        p = (platform or "").strip().lower()
        eis = alles.get(p) or {}
        return {
            "platform": p,
            # Het buildnummer is de scheidsrechter: dat loopt altijd op, ook als
            # het versienummer blijft staan.
            "minBuild": int(eis.get("minBuild") or 0),
            "minVersion": str(eis.get("minVersion") or ""),
            "message": str(eis.get("message") or ""),
            "storeUrl": str(eis.get("storeUrl") or STORE.get(p) or ""),
        }
    finally:
        db.close()


@router.post("/app/requirement")
def zet_eis(
    platform: str = Body(..., embed=True),
    minBuild: int = Body(0, embed=True),
    minVersion: str = Body("", embed=True),
    message: str = Body("", embed=True),
    storeUrl: str = Body("", embed=True),
    user: User = Depends(require_admin),
):
    """Zet de ondergrens voor één platform. Alleen de beheerder.

    Voorbeeld: `{"platform":"ios","minBuild":7,"minVersion":"1.0.6"}` — daarmee
    kan niemand meer met build 6 verder.
    """
    p = (platform or "").strip().lower()
    if p not in ("ios", "android", "web"):
        raise HTTPException(400, "platform moet ios, android of web zijn")
    db = SessionLocal()
    try:
        alles = _lees(db)
        alles[p] = {
            "minBuild": int(minBuild or 0),
            "minVersion": (minVersion or "").strip(),
            "message": (message or "").strip()[:300],
            "storeUrl": (storeUrl or "").strip() or STORE.get(p, ""),
        }
        waarde = json.dumps(alles, ensure_ascii=False)
        rij = db.get(Meta, META_SLEUTEL)
        if rij:
            rij.value = waarde
        else:
            db.add(Meta(key=META_SLEUTEL, value=waarde))
        db.commit()
        return {"ok": True, **alles[p], "platform": p}
    finally:
        db.close()
