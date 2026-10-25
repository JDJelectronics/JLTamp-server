"""De stand van de server in één oogopslag, voor het tabblad Overzicht in Beheer.

Eén verzoek dat alles teruggeeft wat de beheerder wil weten zonder op de server
in te loggen: draaien de diensten (scheiden, stem, AI), hoe vol is de schijf, wie
is er nu verbonden en wie luistert of speelt samen, en wat meldde de app de laatste tijd aan fouten en vastlopers.

Alleen lezen. Alles met een korte time-out: een dienst die hangt mag dit scherm
niet laten hangen.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request

from fastapi import APIRouter, Depends
from sqlalchemy import select

from ..db import SessionLocal
from ..deps import require_admin
from ..models import User, DiagReport
from .. import config

router = APIRouter()

_GESTART = time.time()


def _schijf(pad) -> dict | None:
    try:
        d = shutil.disk_usage(str(pad))
        return {"totaal": d.total, "vrij": d.free, "gebruikt": d.used}
    except Exception:
        return None


def _ai() -> bool:
    url = os.environ.get("JLTAMP_AI_URL", "http://host.docker.internal:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


@router.get("/admin/overzicht")
def status(user: User = Depends(require_admin)):
    nu = time.time()

    # ── Diensten ──
    diensten = []
    try:
        from .karaoke import beschikbaarheid
        k = beschikbaarheid()
        diensten.append({"id": "scheiden", "ok": bool(k.get("beschikbaar")),
                         "detail": "band" if k.get("band") else (k.get("reden") or "")})
    except Exception:
        diensten.append({"id": "scheiden", "ok": False, "detail": "fout"})
    try:
        from .presentator import _stem_dienst_klaar
        diensten.append({"id": "stem", "ok": bool(_stem_dienst_klaar()), "detail": ""})
    except Exception:
        diensten.append({"id": "stem", "ok": False, "detail": "fout"})
    diensten.append({"id": "ai", "ok": _ai(), "detail": ""})

    # ── Wie is er ──
    db = SessionLocal()
    try:
        namen = {u.id: (u.display_name or u.email.split("@")[0]) for u in db.execute(select(User)).scalars()}
        fotos = {u.id: bool(getattr(u, "thumb_path", None)) for u in db.execute(select(User)).scalars()}

        verbonden = []
        try:
            from .session import PRESENCE, MANAGER
            from .radio import radio_gebruikers
            radio_ids = {g["user_id"] for g in radio_gebruikers()}
            gezien = set()
            for p in PRESENCE.peers.values():
                sleutel = (p.user_id, p.device)
                if sleutel in gezien:
                    continue
                gezien.add(sleutel)
                verbonden.append({"userId": p.user_id, "naam": namen.get(p.user_id, p.name),
                                  "toestel": p.device or "", "foto": fotos.get(p.user_id, False),
                                  "radio": p.user_id in radio_ids})
            sessies = []
            for s in MANAGER.sessions.values():
                if not s.participants:
                    continue
                sessies.append({
                    "host": namen.get(s.host_user_id, "?"),
                    "aantal": len({q.user_id for q in s.participants.values()}),
                    "nummer": (s.track or {}).get("title") if s.track else None,
                    "speelt": bool(s.is_playing),
                })
        except Exception:
            sessies = []

        spellen = []
        try:
            from .spel import SPELLEN
            for s in SPELLEN.values():
                if s.voorbij():
                    continue
                spellen.append({"host": s.naam, "soort": s.soort, "fase": s.fase,
                                "spelers": len([p for p in s.spelers.values() if not p.vertrokken])})
        except Exception:
            pass

        # ── Meldingen uit de app (laatste 14 dagen) ──
        grens = nu - 14 * 86400
        meldingen = [
            {"at": r.at, "soort": r.kind, "toestel": r.model, "platform": r.platform,
             "app": r.app_version, "tekst": (r.message or "")[:160],
             "scherm": r.scherm or "", "naam": namen.get(r.user_id, "") if r.user_id else ""}
            for r in db.execute(select(DiagReport).where(DiagReport.at >= grens)
                                .order_by(DiagReport.id.desc()).limit(25)).scalars()
        ]
    finally:
        db.close()

    return {
        "uptime": int(nu - _GESTART),
        "diensten": diensten,
        "schijf": {"data": _schijf(config.DATA_DIR), "muziek": _schijf("/music/mp3")},
        "verbonden": verbonden,
        "sessies": sessies,
        "spellen": spellen,
        "meldingen": meldingen,
    }

