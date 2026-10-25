"""Storingsmeldingen van de app.

Waarom dit bestaat: een gebruiker meldde dat de app op zijn iPhone 17 vastliep,
en dat toestel is niet te leen. De logboeken van iOS staan op de telefoon zelf,
achter drie schermen in de instellingen — dat kun je iemand één keer vragen, niet
elke keer. Dus vertelt de app het voortaan zelf.

Wat er binnenkomt: een regel met het toestel, de versie, wat voor soort storing
het was en de foutmelding. Géén inhoud — geen nummers, geen zoekopdrachten, geen
namen van anderen. Wat je eraan hebt is de combinatie: "op iPhone 17 met versie
1.0.5 blijft de JS-draad twaalf seconden hangen op het beginscherm".

Drie soorten meldingen:

  fout      een JavaScript-fout die het scherm omver haalde (de ErrorBoundary)
  vastloper de JS-draad stond langer dan tien seconden stil — dát is wat iemand
            "de app loopt vast" noemt
  handmatig de gebruiker drukte zelf op "probleem melden"

Bewust simpel gehouden: één tabel, en per gebruiker blijven de laatste vijftig
regels staan. Het is een gereedschap om een vraag te beantwoorden, geen
meetsysteem.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Body, Depends
from sqlalchemy import delete, select

from ..db import SessionLocal
from ..deps import require_admin, require_user
from ..models import DiagReport, User

router = APIRouter(tags=["diag"])

# Meer dan dit per gebruiker heeft geen zin; de oudste vallen af.
MAX_PER_GEBRUIKER = 50
MAX_TEKST = 4000


@router.post("/diag/report")
def meld(
    kind: str = Body("fout", embed=True),
    message: str = Body("", embed=True),
    stack: str = Body("", embed=True),
    app_version: str = Body("", embed=True),
    platform: str = Body("", embed=True),
    model: str = Body("", embed=True),
    os_version: str = Body("", embed=True),
    scherm: str = Body("", embed=True),
    extra: str = Body("", embed=True),
    at: int = Body(0, embed=True),
    user: User = Depends(require_user),
):
    """Neem een melding aan. Faalt nooit hardop: een storingsmelding mag nooit
    zelf een storing worden."""
    db = SessionLocal()
    try:
        r = DiagReport(
            user_id=user.id,
            kind=(kind or "fout")[:32],
            message=(message or "")[:MAX_TEKST],
            stack=(stack or "")[:MAX_TEKST],
            app_version=(app_version or "")[:64],
            platform=(platform or "")[:32],
            model=(model or "")[:64],
            os_version=(os_version or "")[:64],
            scherm=(scherm or "")[:128],
            extra=(extra or "")[:MAX_TEKST],
            # De tijd van het toestel als die meekomt (de melding kan van een
            # vorige sessie zijn), anders die van nu.
            at=int(at) if at else int(time.time()),
            created_at=int(time.time()),
        )
        db.add(r)
        db.commit()

        # Opruimen: alles boven de vijftig van deze gebruiker weg.
        oud = db.execute(
            select(DiagReport.id)
            .where(DiagReport.user_id == user.id)
            .order_by(DiagReport.id.desc())
            .offset(MAX_PER_GEBRUIKER)
        ).scalars().all()
        if oud:
            db.execute(delete(DiagReport).where(DiagReport.id.in_(oud)))
            db.commit()
        return {"ok": True, "id": r.id}
    finally:
        db.close()


@router.get("/diag/reports")
def lees(limit: int = 50, kind: str = "", user: User = Depends(require_admin)):
    """Wat er gemeld is, nieuwste eerst. Alleen voor de beheerder."""
    limit = max(1, min(int(limit or 50), 200))
    db = SessionLocal()
    try:
        q = select(DiagReport).order_by(DiagReport.id.desc()).limit(limit)
        if kind:
            q = q.where(DiagReport.kind == kind)
        rijen = db.execute(q).scalars().all()
        return {"items": [
            {
                "id": r.id, "at": r.at, "kind": r.kind, "userId": r.user_id,
                "app": r.app_version, "platform": r.platform, "model": r.model,
                "os": r.os_version, "scherm": r.scherm,
                "message": r.message, "stack": r.stack, "extra": r.extra,
            }
            for r in rijen
        ]}
    finally:
        db.close()
