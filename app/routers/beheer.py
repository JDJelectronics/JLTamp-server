"""Beheer: alles wat de beheerder in de app moet kunnen zien en doen.

Vier onderdelen, elk voor een eigen tabblad in Beheer:

- Live: wie luistert er samen, wie speelt een spel, waar loopt een feest — en
  dat ook kunnen stoppen.
- Activiteit: hoeveel er geluisterd wordt, wat het meest, door wie, en wat er
  gezocht werd zonder dat het er was.
- Meldingen: de fout- en vastlopermeldingen uit de app, met opruimen.
- Onderhoud: de minimale appversie, de Top 40, het luidheidsmeten en de
  database.

Plus per gebruiker de ingelogde apparaten, met afmelden.

Alles alleen voor beheerders. Wat stopt of verwijdert, doet dat alleen op
uitdrukkelijk verzoek; de app vraagt eerst om bevestiging.
"""
from __future__ import annotations

import hashlib
import os
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import delete, func, select

from .. import config
from ..db import SessionLocal
from ..deps import require_admin
from ..models import (DiagReport, MissedSearch, PlayEvent, Session as AuthSession,
                      Track, User)

router = APIRouter(tags=["beheer"])


def _namen(db) -> dict[int, str]:
    return {u.id: (u.display_name or u.email.split("@")[0])
            for u in db.execute(select(User)).scalars()}


# ── Live ────────────────────────────────────────────────────────────────────
@router.get("/admin/live")
def live(user: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        namen = _namen(db)
    finally:
        db.close()

    sessies = []
    try:
        from .session import MANAGER
        for s in list(MANAGER.sessions.values()):
            if not s.participants:
                continue
            sessies.append({
                "code": s.code,
                "host": namen.get(s.host_user_id, "?"),
                "hostId": s.host_user_id,
                "sinds": int((s.created_at or 0) / 1000),
                "nummer": (s.track or {}).get("title") if s.track else None,
                "artiest": ((s.track or {}).get("grandparentTitle")
                            or (s.track or {}).get("artist")) if s.track else None,
                "speelt": bool(s.is_playing),
                "deelnemers": [{"naam": p.name, "userId": p.user_id, "toestel": p.device or "",
                                "host": p.is_host} for p in s.participants.values()],
            })
    except Exception:
        pass

    spellen = []
    try:
        from .spel import SPELLEN
        for s in list(SPELLEN.values()):
            if s.voorbij():
                continue
            spellen.append({
                "token": s.token, "host": s.naam, "soort": s.soort, "fase": s.fase,
                "ronde": s.ronde, "rondes": s.rondes, "sinds": int(s.gestart),
                "spelers": [p.bijnaam for p in s.spelers.values() if not p.vertrokken],
            })
    except Exception:
        pass

    feesten = []
    try:
        from .feest import FEESTEN
        for f in list(FEESTEN.values()):
            if f.voorbij():
                continue
            feesten.append({
                "token": f.token, "host": namen.get(f.host_user_id, f.naam),
                "sinds": int(f.gestart), "gasten": len(f.gasten),
                "aanvragen": len(f.aanvragen),
                "open": len([a for a in f.aanvragen.values() if a.status == "open"]),
            })
    except Exception:
        pass

    return {"sessies": sessies, "spellen": spellen, "feesten": feesten}


@router.post("/admin/live/sessie/{code}/stop")
async def stop_sessie(code: str, user: User = Depends(require_admin)):
    """Een samen-luisteren-sessie beëindigen: iedereen eruit, sessie weg."""
    from .session import MANAGER
    s = MANAGER.get(code)
    if not s:
        raise HTTPException(404, "Sessie niet gevonden")
    for p in list(s.participants.values()):
        try:
            await p.ws.close(code=4000, reason="beëindigd door beheerder")
        except Exception:
            pass
    MANAGER.remove(code)
    return {"ok": True}


@router.post("/admin/live/spel/{token}/stop")
def stop_spel(token: str, user: User = Depends(require_admin)):
    from .spel import SPELLEN, _SLOT
    with _SLOT:
        s = SPELLEN.get(token)
        if not s:
            raise HTTPException(404, "Spel niet gevonden")
        s.gestopt = time.time()
    return {"ok": True}


@router.post("/admin/live/feest/{token}/stop")
def stop_feest(token: str, user: User = Depends(require_admin)):
    from .feest import FEESTEN, _SLOT
    with _SLOT:
        f = FEESTEN.get(token)
        if not f:
            raise HTTPException(404, "Feest niet gevonden")
        f.gestopt = time.time()
    return {"ok": True}


# ── Activiteit ──────────────────────────────────────────────────────────────
@router.get("/admin/activiteit")
def activiteit(dagen: int = Query(30), user: User = Depends(require_admin)):
    dagen = max(1, min(int(dagen or 30), 365))
    sinds = int(time.time()) - dagen * 86400
    db = SessionLocal()
    try:
        namen = _namen(db)
        per_dag = db.execute(
            select(func.strftime("%Y-%m-%d", PlayEvent.started_at, "unixepoch", "localtime").label("d"),
                   func.count(PlayEvent.id), func.sum(PlayEvent.listened_ms))
            .where(PlayEvent.started_at >= sinds).group_by("d").order_by("d")).all()
        # Elke dag erin, ook zonder luisteren, zodat de grafiek geen gaten heeft.
        bekend = {d: (n, ms or 0) for d, n, ms in per_dag}
        dagen_lijst = []
        for i in range(dagen - 1, -1, -1):
            d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            n, ms = bekend.get(d, (0, 0))
            dagen_lijst.append({"dag": d, "plays": n, "minuten": int(ms // 60000)})

        def top(kolom, extra=None, limit=10):
            cols = [kolom] + ([extra] if extra is not None else [])
            q = (select(*cols, func.count(PlayEvent.id).label("n"))
                 .where(PlayEvent.started_at >= sinds, kolom != "")
                 .group_by(*cols).order_by(func.count(PlayEvent.id).desc()).limit(limit))
            return db.execute(q).all()

        nummers = [{"titel": t, "artiest": a, "plays": n}
                   for t, a, n in top(PlayEvent.track_title, PlayEvent.artist_name)]
        artiesten = [{"naam": a, "plays": n} for a, n in top(PlayEvent.artist_name)]
        luisteraars = [
            {"userId": uid, "naam": namen.get(uid, "?"), "plays": n, "minuten": int((ms or 0) // 60000)}
            for uid, n, ms in db.execute(
                select(PlayEvent.user_id, func.count(PlayEvent.id), func.sum(PlayEvent.listened_ms))
                .where(PlayEvent.started_at >= sinds).group_by(PlayEvent.user_id)
                .order_by(func.count(PlayEvent.id).desc()).limit(10)).all()
        ]
        toestellen = [{"naam": d, "plays": n} for d, n in top(PlayEvent.device, limit=8)]
        totaal = db.execute(select(func.count(PlayEvent.id), func.sum(PlayEvent.listened_ms),
                                   func.count(func.distinct(PlayEvent.user_id)))
                            .where(PlayEvent.started_at >= sinds)).one()

        # Wat er gezocht werd en niet gevonden: samengevoegd over gebruikers.
        gemist = []
        for q, n, laatst, wie in db.execute(
                select(MissedSearch.query_norm, func.sum(MissedSearch.hits),
                       func.max(MissedSearch.last_at), func.group_concat(MissedSearch.user_id))
                .group_by(MissedSearch.query_norm)
                .order_by(func.max(MissedSearch.last_at).desc()).limit(40)).all():
            ids = sorted({int(x) for x in (wie or "").split(",") if x})
            gemist.append({"zoek": q, "keer": int(n or 0), "laatst": int(laatst or 0),
                           "wie": [namen.get(i, "?") for i in ids][:4]})
        return {
            "dagen": dagen,
            "totaal": {"plays": int(totaal[0] or 0), "minuten": int((totaal[1] or 0) // 60000),
                       "luisteraars": int(totaal[2] or 0)},
            "perDag": dagen_lijst, "nummers": nummers, "artiesten": artiesten,
            "luisteraars": luisteraars, "toestellen": toestellen, "gemist": gemist,
        }
    finally:
        db.close()


@router.post("/admin/gemist/weg")
def gemist_weg(zoek: str = Body("", embed=True), user: User = Depends(require_admin)):
    """Een gemiste zoekopdracht weghalen, bij iedereen. Leeg = de hele lijst."""
    db = SessionLocal()
    try:
        q = delete(MissedSearch)
        if zoek:
            q = q.where(MissedSearch.query_norm == zoek)
        n = db.execute(q).rowcount
        db.commit()
        return {"weg": n}
    finally:
        db.close()


# ── Meldingen uit de app ────────────────────────────────────────────────────
@router.get("/admin/meldingen")
def meldingen(soort: str = Query(""), platform: str = Query(""), limit: int = Query(100),
              user: User = Depends(require_admin)):
    limit = max(1, min(int(limit or 100), 300))
    db = SessionLocal()
    try:
        namen = _namen(db)
        q = select(DiagReport).order_by(DiagReport.id.desc())
        if soort:
            q = q.where(DiagReport.kind == soort)
        if platform:
            q = q.where(DiagReport.platform == platform)
        rijen = db.execute(q.limit(limit)).scalars().all()
        tellen = dict(db.execute(select(DiagReport.kind, func.count(DiagReport.id))
                                 .group_by(DiagReport.kind)).all())
        return {
            "tellen": tellen,
            "items": [{
                "id": r.id, "at": r.at, "soort": r.kind, "naam": namen.get(r.user_id, "") if r.user_id else "",
                "app": r.app_version, "platform": r.platform, "toestel": r.model, "os": r.os_version,
                "scherm": r.scherm, "tekst": r.message, "stack": (r.stack or "")[:4000],
                "extra": (r.extra or "")[:2000],
            } for r in rijen],
        }
    finally:
        db.close()


@router.delete("/admin/meldingen/{mid}")
def melding_weg(mid: int, user: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        r = db.get(DiagReport, mid)
        if not r:
            raise HTTPException(404, "Niet gevonden")
        db.delete(r)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/admin/meldingen/opruimen")
def meldingen_opruimen(ouderDan: int = Body(30, embed=True), user: User = Depends(require_admin)):
    """Meldingen ouder dan zoveel dagen weg. 0 = alles."""
    grens = int(time.time()) - max(0, int(ouderDan)) * 86400
    db = SessionLocal()
    try:
        q = delete(DiagReport)
        if ouderDan > 0:
            q = q.where(DiagReport.at < grens)
        n = db.execute(q).rowcount
        db.commit()
        return {"weg": n}
    finally:
        db.close()


# ── Onderhoud ───────────────────────────────────────────────────────────────
@router.get("/admin/onderhoud")
def onderhoud(user: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        from .appversion import _lees
        eis = _lees(db)
        zonder_gain = db.query(func.count(Track.id)).filter(Track.gain_db.is_(None)).scalar() or 0
        alle = db.query(func.count(Track.id)).scalar() or 0
    finally:
        db.close()
    try:
        from .. import loudness
        luid = loudness.state()
    except Exception:
        luid = {}
    try:
        from .charts import _lees_lijst_stand, FEED_TOP40
        top40 = _lees_lijst_stand(FEED_TOP40)
    except Exception:
        top40 = {}
    try:
        db_grootte = os.path.getsize(config.DB_PATH)
    except Exception:
        db_grootte = 0
    return {
        "appEis": {p: eis.get(p) or {} for p in ("ios", "android")},
        "luidheid": {**luid, "zonder": int(zonder_gain), "alle": int(alle)},
        "top40": top40,
        "database": {"grootte": db_grootte},
    }


# ── Ingelogde apparaten per gebruiker ───────────────────────────────────────
#
# Een sessie is een sleutel (het token); die geven we nooit terug. Wel een
# korte afgeleide id waarmee je één sessie kunt aanwijzen om af te melden.
# De sessies van de AI-dienst blijven buiten beeld: die afmelden breekt de
# weeklijsten van die persoon.
def _sid(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


@router.get("/admin/users/{uid}/apparaten")
def apparaten(uid: int, user: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        rijen = db.execute(select(AuthSession).where(AuthSession.user_id == uid,
                                                     AuthSession.label.notin_(("ai-service", "bekijk-als")))
                           .order_by(AuthSession.last_seen_at.desc())).scalars().all()
        return {"items": [{"id": _sid(r.token), "soort": r.label or "", "sinds": r.created_at or 0,
                           "gezien": r.last_seen_at or 0} for r in rijen]}
    finally:
        db.close()


@router.post("/admin/users/{uid}/afmelden")
def afmelden(uid: int, id: str = Body("", embed=True), admin: User = Depends(require_admin)):
    """Eén apparaat afmelden (id), of zonder id: overal. De AI-dienst blijft."""
    db = SessionLocal()
    try:
        rijen = db.execute(select(AuthSession).where(AuthSession.user_id == uid,
                                                     AuthSession.label.notin_(("ai-service", "bekijk-als")))).scalars().all()
        n = 0
        for r in rijen:
            if id and _sid(r.token) != id:
                continue
            db.delete(r)
            n += 1
        db.commit()
        return {"afgemeld": n}
    finally:
        db.close()


# ── Inrichting ──────────────────────────────────────────────────────────────
@router.post("/admin/inrichting")
def inrichting(serverNaam: str = Body("", embed=True), user: User = Depends(require_admin)):
    """De naam van de server (in de app, in de mails). Leeg = terug naar de standaard."""
    from ..instellingen import zet_meta, servernaam
    zet_meta("server_naam", (serverNaam or "").strip()[:60])
    return {"serverNaam": servernaam(), "standaard": config.SERVER_NAME}


# ── Bericht aan wie er online is ────────────────────────────────────────────
#
# Een melding die meteen als venster opduikt bij iedereen die de app open heeft
# (via de presence-verbinding). Met `blijf` blijft hij zoveel minuten staan, en
# krijgt ook wie in die tijd de app opent hem nog te zien — één keer per
# toestel, de app onthoudt het id.
import json as _json  # noqa: E402
import secrets as _secrets  # noqa: E402

BERICHT_SLEUTEL = "mededeling"


def huidig_bericht() -> dict | None:
    from ..instellingen import _meta
    try:
        b = _json.loads(_meta(BERICHT_SLEUTEL) or "null")
    except Exception:
        return None
    if not b or (b.get("tot") or 0) < time.time():
        return None
    return b


@router.get("/admin/bericht")
def lees_bericht(user: User = Depends(require_admin)):
    return {"bericht": huidig_bericht()}


@router.post("/admin/bericht")
async def stuur_bericht(titel: str = Body("", embed=True), tekst: str = Body(..., embed=True),
                        blijf: int = Body(0, embed=True), user: User = Depends(require_admin)):
    tekst = (tekst or "").strip()[:500]
    if not tekst:
        raise HTTPException(400, "Tekst is leeg")
    from ..instellingen import zet_meta
    b = {"id": _secrets.token_hex(6), "titel": (titel or "").strip()[:80], "tekst": tekst,
         "van": user.display_name or user.email.split("@")[0],
         "at": int(time.time()), "tot": int(time.time()) + max(0, min(int(blijf or 0), 7 * 1440)) * 60}
    zet_meta(BERICHT_SLEUTEL, _json.dumps(b, ensure_ascii=False) if blijf else "")
    from .session import PRESENCE
    bericht = _json.dumps({"t": "bericht", **b})
    bereikt, mensen = 0, set()
    for p in list(PRESENCE.peers.values()):
        try:
            await p.ws.send_text(bericht)
            bereikt += 1
            mensen.add(p.user_id)
        except Exception:
            pass
    return {"toestellen": bereikt, "mensen": len(mensen), "bericht": b}


@router.delete("/admin/bericht")
def wis_bericht(user: User = Depends(require_admin)):
    from ..instellingen import zet_meta
    zet_meta(BERICHT_SLEUTEL, "")
    return {"ok": True}


# ── Automatisch scannen ─────────────────────────────────────────────────────
#
# Elke zoveel uur alle bibliotheken scannen (Beheer → Onderhoud). 0 = uit,
# zoals het altijd was. Eén draad die elke minuut kijkt of het tijd is.
import threading as _threading  # noqa: E402

_AUTO = {"gestart": False, "laatst": time.time()}


def _auto_scan_lus():
    from .admin_users import getal
    from ..scanner import scan_all, scan_state
    while True:
        time.sleep(60)
        try:
            db = SessionLocal()
            try:
                uur = getal(db, "auto-scan-uur")
            finally:
                db.close()
            if uur <= 0 or time.time() - _AUTO["laatst"] < uur * 3600:
                continue
            if (scan_state() or {}).get("scanning"):
                continue
            _AUTO["laatst"] = time.time()
            scan_all(full=False)
        except Exception:
            pass


if not _AUTO["gestart"]:
    _AUTO["gestart"] = True
    _threading.Thread(target=_auto_scan_lus, daemon=True, name="auto-scan").start()


# ── Logboek ─────────────────────────────────────────────────────────────────
@router.get("/admin/logboek")
def logboek(limit: int = Query(100), user: User = Depends(require_admin)):
    from ..models import BeheerLog
    limit = max(1, min(int(limit or 100), 500))
    db = SessionLocal()
    try:
        namen = _namen(db)
        rijen = db.execute(select(BeheerLog).order_by(BeheerLog.id.desc()).limit(limit)).scalars().all()
        import re as _re

        def doel(pad):
            m = _re.match(r"/admin/users/(\d+)", pad)
            return namen.get(int(m.group(1)), f"#{m.group(1)}") if m else ""
        return {"items": [{"id": r.id, "at": r.at, "naam": namen.get(r.user_id, ""), "methode": r.methode,
                           "pad": r.pad, "status": r.status, "inhoud": r.inhoud, "doel": doel(r.pad)} for r in rijen]}
    finally:
        db.close()


# ── Back-ups ────────────────────────────────────────────────────────────────
#
# Een kopie van de database met SQLite's eigen back-up (veilig terwijl de
# server draait) in /data/backups. Terugzetten gaat bewust niet vanuit de app:
# dat vervangt alles en vraagt een herstart — dat doe je op de Studio.
import sqlite3 as _sqlite3  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

BACKUP_MAP = _Path(config.DATA_DIR) / "backups"


def maak_backup(soort: str = "hand") -> dict:
    BACKUP_MAP.mkdir(parents=True, exist_ok=True)
    naam = time.strftime(f"library-%Y%m%d-%H%M%S-{soort}.db")
    doel = BACKUP_MAP / naam
    bron = _sqlite3.connect(str(config.DB_PATH))
    try:
        uit = _sqlite3.connect(str(doel))
        try:
            bron.backup(uit)
        finally:
            uit.close()
    finally:
        bron.close()
    return {"naam": naam, "grootte": doel.stat().st_size, "at": int(doel.stat().st_mtime)}


def _backups() -> list[dict]:
    if not BACKUP_MAP.exists():
        return []
    return sorted(({"naam": p.name, "grootte": p.stat().st_size, "at": int(p.stat().st_mtime),
                    "auto": p.stem.endswith("-auto")} for p in BACKUP_MAP.glob("library-*.db")),
                  key=lambda x: x["at"], reverse=True)


def _opruimen_auto(bewaren: int):
    for b in [b for b in _backups() if b["auto"]][bewaren:]:
        try:
            (BACKUP_MAP / b["naam"]).unlink()
        except Exception:
            pass


@router.get("/admin/backups")
def backups(user: User = Depends(require_admin)):
    import shutil
    try:
        vrij = shutil.disk_usage(str(config.DATA_DIR)).free
    except Exception:
        vrij = 0
    return {"items": _backups(), "vrij": vrij, "map": str(BACKUP_MAP)}


@router.post("/admin/backups")
def backup_nu(user: User = Depends(require_admin)):
    return maak_backup("hand")


@router.delete("/admin/backups/{naam}")
def backup_weg(naam: str, user: User = Depends(require_admin)):
    p = BACKUP_MAP / naam
    if "/" in naam or not naam.startswith("library-") or not p.exists():
        raise HTTPException(404, "Niet gevonden")
    p.unlink()
    return {"ok": True}


# ── Gezondheid van de bibliotheek ───────────────────────────────────────────
@router.get("/admin/gezondheid")
def gezondheid(user: User = Depends(require_admin)):
    from ..models import Album
    db = SessionLocal()
    try:
        # Dubbel: dezelfde artiest en titel in dezelfde bibliotheek, en ongeveer
        # even lang (anders is het een andere versie).
        dub_q = (select(Track.library_id, func.lower(Track.artist_name), func.lower(Track.title),
                        func.round(Track.duration_ms / 5000.0), func.count(Track.id), func.min(Track.album_id))
                 .where(Track.title != "")
                 .group_by(Track.library_id, func.lower(Track.artist_name), func.lower(Track.title),
                           func.round(Track.duration_ms / 5000.0))
                 .having(func.count(Track.id) > 1))
        dub = db.execute(dub_q.order_by(func.count(Track.id).desc()).limit(40)).all()
        dub_totaal = db.execute(select(func.count()).select_from(dub_q.subquery())).scalar() or 0

        zonder_hoes_q = select(Album).where(Album.art_path.is_(None), Album.online_art_path.is_(None))
        zonder_hoes = db.execute(zonder_hoes_q.order_by(Album.added_at.desc()).limit(40)).scalars().all()
        zonder_hoes_n = db.execute(select(func.count()).select_from(zonder_hoes_q.subquery())).scalar() or 0

        # Geen titel, of geen artiest — ook niet bij het nummer zelf. Een
        # verzamelalbum ("Various Artists") met een artiest per nummer is gewoon goed.
        onbekend = ("", "unknown artist", "onbekend", "unknown", "[unknown]")
        zonder_tags_q = select(Track).where(
            (Track.title == "")
            | (func.lower(Track.artist_name).in_(onbekend + ("various artists",))
               & func.lower(func.coalesce(Track.orig_artist, "")).in_(onbekend)))
        zonder_tags = db.execute(zonder_tags_q.limit(40)).scalars().all()
        zonder_tags_n = db.execute(select(func.count()).select_from(zonder_tags_q.subquery())).scalar() or 0

        leeg_q = select(Track).where(Track.duration_ms <= 0)
        leeg = db.execute(leeg_q.limit(40)).scalars().all()
        leeg_n = db.execute(select(func.count()).select_from(leeg_q.subquery())).scalar() or 0

        laag_q = select(Track).where(Track.bitrate > 0, Track.bitrate < 128)
        laag_n = db.execute(select(func.count()).select_from(laag_q.subquery())).scalar() or 0
        laag = db.execute(laag_q.limit(40)).scalars().all()

        def tr(t):
            return {"id": t.id, "albumId": t.album_id, "titel": t.title or t.path.rsplit("/", 1)[-1],
                    "artiest": t.artist_name, "album": t.album_title, "pad": t.path, "bitrate": t.bitrate}
        return {
            "dubbel": {"totaal": dub_totaal, "items": [
                {"artiest": a, "titel": ti, "keer": n, "albumId": alb} for _l, a, ti, _d, n, alb in dub]},
            "zonderHoes": {"totaal": zonder_hoes_n, "items": [
                {"albumId": a.id, "titel": a.title, "artiest": a.artist_name, "jaar": a.year} for a in zonder_hoes]},
            "zonderTags": {"totaal": zonder_tags_n, "items": [tr(t) for t in zonder_tags]},
            "zonderDuur": {"totaal": leeg_n, "items": [tr(t) for t in leeg]},
            "lageKwaliteit": {"totaal": laag_n, "items": [tr(t) for t in laag]},
        }
    finally:
        db.close()


# ── Onderhoudsmodus ─────────────────────────────────────────────────────────
@router.post("/admin/onderhoudsmodus")
async def onderhoudsmodus(aan: bool = Body(..., embed=True), tekst: str = Body("", embed=True),
                          user: User = Depends(require_admin)):
    from .admin_users import zet_schakelaar
    from ..instellingen import zet_meta
    db = SessionLocal()
    try:
        zet_schakelaar(db, "onderhoud", bool(aan))
        db.commit()
    finally:
        db.close()
    zet_meta("onderhoud_tekst", (tekst or "").strip()[:300])
    # Meteen naar iedereen die de app open heeft, niet pas bij het volgende ophalen.
    from .session import PRESENCE
    db = SessionLocal()
    try:
        admins = {u.id for u in db.execute(select(User).where(User.is_admin.is_(True))).scalars()}
    finally:
        db.close()
    for p in list(PRESENCE.peers.values()):
        try:
            await p.ws.send_text(_json.dumps({"t": "onderhoud", "aan": bool(aan), "blok": bool(aan) and p.user_id not in admins,
                                              "tekst": (tekst or "").strip()[:300]}))
        except Exception:
            pass
    return {"aan": bool(aan), "tekst": (tekst or "").strip()[:300]}


# ── Welkomstbericht voor nieuwe gebruikers ──────────────────────────────────
def huidig_welkom() -> dict | None:
    from ..instellingen import _meta
    try:
        w = _json.loads(_meta("welkom") or "null")
    except Exception:
        return None
    return w if w and w.get("aan") and w.get("tekst") else None


def welkom_voor(user) -> dict | None:
    """Het welkomstbericht als deze gebruiker hem nog niet zag én nieuw is."""
    w = huidig_welkom()
    if not w or (user.created_at or 0) < (w.get("sinds") or 0) - 60:
        return None
    from ..instellingen import _meta, zet_meta
    sleutel = f"welkom_gezien:{user.id}"
    if _meta(sleutel) == w["id"]:
        return None
    zet_meta(sleutel, w["id"])
    return {"id": f"welkom-{w['id']}-{user.id}", "titel": w.get("titel") or "", "tekst": w["tekst"],
            "van": "", "at": int(time.time()), "tot": 0}


@router.get("/admin/welkom")
def lees_welkom(user: User = Depends(require_admin)):
    from ..instellingen import _meta
    try:
        return {"welkom": _json.loads(_meta("welkom") or "null")}
    except Exception:
        return {"welkom": None}


@router.post("/admin/welkom")
def zet_welkom(titel: str = Body("", embed=True), tekst: str = Body("", embed=True),
               aan: bool = Body(True, embed=True), user: User = Depends(require_admin)):
    from ..instellingen import _meta, zet_meta
    try:
        oud = _json.loads(_meta("welkom") or "null") or {}
    except Exception:
        oud = {}
    tekst = (tekst or "").strip()[:800]
    nieuw = {"id": oud.get("id") if oud.get("tekst") == tekst else _secrets.token_hex(5),
             "titel": (titel or "").strip()[:80], "tekst": tekst, "aan": bool(aan and tekst),
             "sinds": oud.get("sinds") if oud.get("aan") and oud.get("tekst") == tekst else int(time.time())}
    zet_meta("welkom", _json.dumps(nieuw, ensure_ascii=False))
    return {"welkom": nieuw}


# ── Tegels op het beginscherm ───────────────────────────────────────────────
@router.get("/admin/blok")
def lees_blok_beheer(user: User = Depends(require_admin)):
    from .charts import EXTERNE_LIJSTEN, HUISLIJSTEN_BRON, META_LOOKS
    from ..models import Meta
    db = SessionLocal()
    try:
        try:
            verborgen = _json.loads((db.get(Meta, "blok_verborgen") or Meta(value="[]")).value or "[]")
        except Exception:
            verborgen = []
        try:
            looks = _json.loads((db.get(Meta, META_LOOKS) or Meta(value="{}")).value or "{}")
        except Exception:
            looks = {}
    finally:
        db.close()
    feeds = [{"feed": f, "titel": t[0]} for f, t in EXTERNE_LIJSTEN.items()]
    for f, (titel, _b) in HUISLIJSTEN_BRON.items():
        feeds.append({"feed": f, "titel": titel if isinstance(titel, str) else "Seizoen"})
    return {"feeds": feeds, "verborgen": verborgen, "volgorde": looks.get("volgorde") or []}


@router.post("/admin/blok")
def zet_blok_beheer(verborgen: list = Body([], embed=True), volgorde: list = Body([], embed=True),
                    user: User = Depends(require_admin)):
    from .charts import META_LOOKS
    from ..models import Meta
    db = SessionLocal()
    try:
        rij = db.get(Meta, "blok_verborgen") or Meta(key="blok_verborgen")
        rij.value = _json.dumps([str(f) for f in verborgen])
        db.merge(rij)
        lrij = db.get(Meta, META_LOOKS)
        try:
            looks = _json.loads(lrij.value) if lrij and lrij.value else {}
        except Exception:
            looks = {}
        looks["volgorde"] = [str(f) for f in volgorde]
        if not lrij:
            lrij = Meta(key=META_LOOKS)
            db.add(lrij)
        lrij.value = _json.dumps(looks, ensure_ascii=False)
        db.commit()
    finally:
        db.close()
    return {"ok": True}


# ── Bekijk als ──────────────────────────────────────────────────────────────
@router.post("/admin/users/{uid}/bekijk")
def bekijk_als(uid: int, admin: User = Depends(require_admin)):
    """Een tijdelijke sleutel om de app te zien zoals deze gebruiker hem ziet."""
    from ..deps import create_session
    db = SessionLocal()
    try:
        u = db.get(User, uid)
        if not u or not u.is_active:
            raise HTTPException(404, "Gebruiker niet gevonden of uit")
        token = create_session(db, u, label="bekijk-als")
        return {"token": token, "naam": u.display_name or u.email.split("@")[0], "userId": u.id}
    finally:
        db.close()


@router.post("/admin/bekijk/klaar")
def bekijk_klaar(token: str = Body(..., embed=True), admin: User = Depends(require_admin)):
    """Terug naar jezelf: de tijdelijke sleutel ongeldig maken."""
    db = SessionLocal()
    try:
        s = db.get(AuthSession, token)
        if s and s.label == "bekijk-als":
            db.delete(s)
            db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── Waarschuwingen voor de beheerder ────────────────────────────────────────
#
# Elke vijf minuten: draaien de diensten nog, en komen er ineens veel
# foutmeldingen binnen? Zo ja, dan een waarschuwing in Overzicht en meteen een
# venster bij elke beheerder die de app open heeft. Eén keer per probleem, niet
# elke vijf minuten opnieuw.
_WAARSCHUW = {"diensten": {}, "piek": 0.0}


def _waarschuwingen() -> list:
    from ..instellingen import _meta
    try:
        return _json.loads(_meta("waarschuwingen") or "[]")
    except Exception:
        return []


def _voeg_waarschuwing(soort: str, tekst: str):
    from ..instellingen import zet_meta
    lijst = [{"id": _secrets.token_hex(4), "at": int(time.time()), "soort": soort, "tekst": tekst}] + _waarschuwingen()
    zet_meta("waarschuwingen", _json.dumps(lijst[:30], ensure_ascii=False))
    _stuur_beheerders(tekst, lijst[0]["id"])


def _stuur_beheerders(tekst: str, wid: str):
    import asyncio
    try:
        from .session import PRESENCE
        db = SessionLocal()
        try:
            admins = {u.id for u in db.execute(select(User).where(User.is_admin.is_(True))).scalars()}
        finally:
            db.close()
        msg = _json.dumps({"t": "bericht", "id": f"w-{wid}", "titel": "⚠️ JLTamp-server", "tekst": tekst,
                           "van": "", "at": int(time.time()), "tot": 0})
        lus = getattr(_stuur_beheerders, "lus", None)
        for p in list(PRESENCE.peers.values()):
            if p.user_id in admins and lus:
                asyncio.run_coroutine_threadsafe(p.ws.send_text(msg), lus)
    except Exception:
        pass


@router.get("/admin/waarschuwingen")
def waarschuwingen(user: User = Depends(require_admin)):
    return {"items": _waarschuwingen()}


@router.delete("/admin/waarschuwingen")
def waarschuwingen_wis(user: User = Depends(require_admin)):
    from ..instellingen import zet_meta
    zet_meta("waarschuwingen", "[]")
    return {"ok": True}


def _controleer():
    from ..models import DiagReport
    from .admin_status import _ai
    stand = {}
    try:
        from .karaoke import beschikbaarheid
        stand["scheiden"] = bool(beschikbaarheid().get("beschikbaar"))
    except Exception:
        pass
    try:
        from .presentator import _stem_dienst_klaar
        stand["stem"] = bool(_stem_dienst_klaar())
    except Exception:
        pass
    stand["ai"] = _ai()
    namen = {"scheiden": "Scheiden (karaoke, bandmixer, spellen)", "stem": "Presentatorstem", "ai": "AI"}
    for k, ok in stand.items():
        vorig = _WAARSCHUW["diensten"].get(k)
        if vorig is True and not ok:
            _voeg_waarschuwing("dienst", f"{namen[k]} reageert niet meer.")
        elif vorig is False and ok:
            _voeg_waarschuwing("hersteld", f"{namen[k]} draait weer.")
        _WAARSCHUW["diensten"][k] = ok
    db = SessionLocal()
    try:
        n = db.query(func.count(DiagReport.id)).filter(DiagReport.at >= time.time() - 3600,
                                                       DiagReport.kind.in_(("fout", "vastloper"))).scalar() or 0
    finally:
        db.close()
    if n >= 5 and time.time() - _WAARSCHUW["piek"] > 6 * 3600:
        _WAARSCHUW["piek"] = time.time()
        _voeg_waarschuwing("meldingen", f"{n} fout- en vastlopermeldingen in het afgelopen uur.")


def _nachtwerk():
    """Elke vijf minuten de controle; tussen 4 en 5 uur 's nachts de back-up."""
    from .admin_users import schakelaar, getal
    laatste_backup_dag = ""
    while True:
        time.sleep(300)
        try:
            _controleer()
        except Exception:
            pass
        try:
            dag = time.strftime("%Y-%m-%d")
            if time.localtime().tm_hour == 4 and dag != laatste_backup_dag:
                db = SessionLocal()
                try:
                    aan, bewaren = schakelaar(db, "auto-backup"), getal(db, "backups-bewaren")
                finally:
                    db.close()
                laatste_backup_dag = dag
                if aan:
                    maak_backup("auto")
                    _opruimen_auto(bewaren)
        except Exception:
            pass


if not _AUTO.get("nacht"):
    _AUTO["nacht"] = True
    _threading.Thread(target=_nachtwerk, daemon=True, name="beheer-nacht").start()
