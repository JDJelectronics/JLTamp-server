"""Verlanglijstje — wat er gezocht werd en niet in de bibliotheek stond.

Je merkt pas dat een nummer ontbreekt op het moment dat je het zoekt, en precies
dan ben je het een minuut later weer kwijt. Elke zoekopdracht die niets oplevert
wordt daarom onthouden (zie `noteer_misser`, aangeroepen vanuit de zoekroute in
media.py), en hier weer uitgeleverd.

De lijst wordt bij het uitlezen opnieuw tegen de bibliotheek gehouden: staat het
inmiddels wél in de collectie, dan verdwijnt de regel meteen én uit de database.
Er is dus geen afvinkknop, en dus ook niets om te vergeten af te vinken.

Wat hier NIET gebeurt: muziek ophalen. De server haalt niets van buiten binnen —
dit is een boodschappenlijstje, geen winkel.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, delete

from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..models import MissedSearch, Track, Album, Artist, User

router = APIRouter(tags=["wishlist"])

# Onder de drie tekens is het geen zoekopdracht maar een halve toetsaanslag.
MIN_LENGTE = 3
MAX_LENGTE = 200
# Binnen dit venster geldt een langere zoekopdracht als voortzetting van een
# kortere: "boh" gevolgd door "bohemian" is één zoekactie, geen twee.
SAMENVOEG_VENSTER = 300      # seconden
# Meer dan dit per gebruiker heeft geen zin; de oudste vallen af.
MAX_PER_GEBRUIKER = 200


def _norm(q: str) -> str:
    return " ".join((q or "").strip().lower().split())[:MAX_LENGTE]


def noteer_misser(db, user_id: int, query: str) -> None:
    """Onthoud een zoekopdracht die niets opleverde. Faalt nooit hardop: dit mag
    een zoekopdracht van de gebruiker niet in de weg zitten."""
    try:
        norm = _norm(query)
        if len(norm) < MIN_LENGTE:
            return
        nu = int(time.time())

        # Al eens precies zo gezocht? Dan alleen de teller en de tijd bij.
        bestaand = db.execute(
            select(MissedSearch).where(
                MissedSearch.user_id == user_id, MissedSearch.query_norm == norm
            )
        ).scalar_one_or_none()
        if bestaand:
            bestaand.hits += 1
            bestaand.last_at = nu
            db.commit()
            return

        # Typen gebeurt letter voor letter, en elke letter is een zoekopdracht.
        # Staat er van kort geleden een regel waarvan de ene tekst het begin van
        # de andere is, dan is dat dezelfde zoekactie: werk die bij naar de
        # langste tekst in plaats van er een tweede naast te zetten.
        recent = db.execute(
            select(MissedSearch)
            .where(MissedSearch.user_id == user_id,
                   MissedSearch.last_at >= nu - SAMENVOEG_VENSTER)
            .order_by(MissedSearch.last_at.desc())
            .limit(25)
        ).scalars()
        for r in recent:
            if norm.startswith(r.query_norm) or r.query_norm.startswith(norm):
                if len(norm) > len(r.query_norm):
                    r.query = query.strip()[:MAX_LENGTE]
                    r.query_norm = norm
                r.last_at = nu
                db.commit()
                return

        db.add(MissedSearch(user_id=user_id, query=query.strip()[:MAX_LENGTE],
                            query_norm=norm, hits=1, created_at=nu, last_at=nu))
        db.commit()

        # Bijhouden dat het geen bodemloze put wordt.
        aantal = db.execute(
            select(func.count(MissedSearch.id)).where(MissedSearch.user_id == user_id)
        ).scalar() or 0
        if aantal > MAX_PER_GEBRUIKER:
            oudste = db.execute(
                select(MissedSearch.id)
                .where(MissedSearch.user_id == user_id)
                .order_by(MissedSearch.last_at.asc())
                .limit(aantal - MAX_PER_GEBRUIKER)
            ).scalars().all()
            if oudste:
                db.execute(delete(MissedSearch).where(MissedSearch.id.in_(oudste)))
                db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


def _staat_er_inmiddels(db, allowed: list[int] | None, norm: str) -> bool:
    """Levert de zoekopdracht nu wél iets op? Zelfde vergelijking als de
    zoekroute, zodat de lijst niet iets blijft tonen dat allang binnen is.

    `allowed` is None voor een beheerder en betekent dan ALLE bibliotheken — niet
    geen enkele. Zonder dat onderscheid zou de lijst van een beheerder nooit
    opschonen.
    """
    like = f"%{norm}%"
    for model, veld in ((Track, Track.title), (Album, Album.title), (Artist, Artist.name)):
        vraag = select(model.id).where(func.lower(veld).like(like))
        if allowed is not None:
            vraag = vraag.where(model.library_id.in_(allowed))
        if db.execute(vraag.limit(1)).scalar_one_or_none() is not None:
            return True
    return False


@router.get("/search/missing")
def lijst(alles: bool = Query(False, description="admin: van alle gebruikers"),
          user: User = Depends(require_user)):
    """Wat deze gebruiker zocht en niet vond. Regels die inmiddels wél iets
    opleveren worden hier opgeruimd in plaats van getoond."""
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)
        vraag = select(MissedSearch).order_by(MissedSearch.last_at.desc())
        if not (alles and user.is_admin):
            vraag = vraag.where(MissedSearch.user_id == user.id)
        rijen = list(db.execute(vraag.limit(500)).scalars())

        namen: dict[int, str] = {}
        if alles and user.is_admin and rijen:
            for u in db.execute(
                select(User).where(User.id.in_(list({r.user_id for r in rijen})))
            ).scalars():
                namen[u.id] = u.display_name or u.email

        uit, opruimen = [], []
        for r in rijen:
            if _staat_er_inmiddels(db, allowed, r.query_norm):
                opruimen.append(r.id)
                continue
            regel = {"id": r.id, "query": r.query, "hits": r.hits,
                     "createdAt": r.created_at, "lastAt": r.last_at}
            if alles and user.is_admin:
                regel["user"] = namen.get(r.user_id, str(r.user_id))
            uit.append(regel)

        if opruimen:
            db.execute(delete(MissedSearch).where(MissedSearch.id.in_(opruimen)))
            db.commit()

        return {"size": len(uit), "items": uit}
    finally:
        db.close()


@router.delete("/search/missing/{item_id}")
def verwijder(item_id: int, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        rij = db.get(MissedSearch, item_id)
        if not rij:
            raise HTTPException(status_code=404, detail="Niet gevonden")
        if rij.user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="Niet van jou")
        db.delete(rij)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.delete("/search/missing")
def leeg(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        db.execute(delete(MissedSearch).where(MissedSearch.user_id == user.id))
        db.commit()
        return {"ok": True}
    finally:
        db.close()
