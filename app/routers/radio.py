"""JLTamp Radio: de routes. De zender zelf staat in app/radio.

    GET /radio/live.m3u8   de stream (HLS). Token mag in de query
                           (?X-Plex-Token=…): Chromecast, Home Assistant en
                           een <audio>-element kunnen geen header meesturen.
                           De playlist wordt per verzoek herschreven, zodat
                           elk stuk dezelfde token in zijn adres draagt.
    GET /radio/seg/{naam}  één stuk van zes seconden.
    GET /radio/nu          wat er speelt, wat er komt, de laatste tien en de
                           tekst van de laatste break.
    GET /radio/status      staat hij aan, draait hij, hoeveel luisteraars.

Aan/uit gaat met de serverschakelaar `radio` (POST /admin/server/schakelaar).
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .. import config
from ..db import SessionLocal
from ..deps import require_admin, require_user, token_from_request
from ..models import User
from ..radio import zender as radio
from ..radio.encoder import PLAYLIST
from .admin_users import schakelaar

router = APIRouter(prefix="/radio", tags=["radio"])

_SEGMENT = re.compile(r"^seg\d{1,20}\.ts$")

# Wie er de laatste halve minuut de playlist ophaalde (een speler doet dat
# elke paar seconden). Alleen een hash van de token, en alleen in het geheugen.
_luisteraars: dict[str, float] = {}
_luister_slot = threading.Lock()
_LUISTER_VENSTER = 30.0


def _tel(token: str | None) -> None:
    if not token:
        return
    nu = time.time()
    h = hashlib.sha256(token.encode()).hexdigest()[:16]
    with _luister_slot:
        _luisteraars[h] = nu
        for k in [k for k, t in _luisteraars.items() if nu - t > _LUISTER_VENSTER]:
            _luisteraars.pop(k, None)


# En wie dat zijn, voor "wie luistert er" (history.presence/sessions) en
# Beheer → Nu online: (user_id, toestel) → (naam, laatst gezien).
_gebruikers: dict[tuple[int, str], tuple[str, float]] = {}


def _tel_gebruiker(user: User, toestel: str) -> None:
    nu = time.time()
    with _luister_slot:
        _gebruikers[(user.id, toestel)] = (user.display_name or user.email.split("@")[0], nu)
        for k in [k for k, (_, t) in _gebruikers.items() if nu - t > _LUISTER_VENSTER]:
            _gebruikers.pop(k, None)


def radio_gebruikers() -> list[dict]:
    """Wie de radio nu aan heeft: [{user_id, naam, toestel}]."""
    nu = time.time()
    with _luister_slot:
        return [{"user_id": uid, "naam": naam, "toestel": toestel}
                for (uid, toestel), (naam, t) in _gebruikers.items() if nu - t <= _LUISTER_VENSTER]


def luisteraars() -> int:
    nu = time.time()
    with _luister_slot:
        return sum(1 for t in _luisteraars.values() if nu - t <= _LUISTER_VENSTER)


def _aan() -> bool:
    db = SessionLocal()
    try:
        return schakelaar(db, "radio")
    finally:
        db.close()


def _review(user: User | None) -> bool:
    """Een reviewaccount (App Store / Play) ziet de radio niet: voor hem staat
    hij gewoon "uit". Zie User.is_review."""
    return bool(getattr(user, "is_review", False))


def herschrijf(playlist: str, token: str | None) -> str:
    """Elk stuk in de playlist krijgt het adres /radio/seg/<naam>?X-Plex-Token=…"""
    staart = f"?X-Plex-Token={quote(token, safe='')}" if token else ""
    regels = []
    for regel in playlist.splitlines():
        r = regel.strip()
        if r and not r.startswith("#"):
            naam = r.rsplit("/", 1)[-1]
            if _SEGMENT.match(naam):
                regel = f"seg/{naam}{staart}"
        regels.append(regel)
    return "\n".join(regels) + "\n"


@router.get("/live.m3u8")
def live(request: Request, user: User = Depends(require_user)):
    z = radio.huidige()
    if z is None or _review(user):
        raise HTTPException(503, "De radio staat uit")
    pad = z.hls / PLAYLIST
    try:
        tekst = pad.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(503, "De radio start op, probeer het zo weer")
    token = token_from_request(request)
    _tel(token)
    _tel_gebruiker(user, request.headers.get("X-Plex-Device-Name")
                   or request.headers.get("X-Plex-Platform") or "")
    return Response(herschrijf(tekst, token), media_type="application/vnd.apple.mpegurl",
                    headers={"Cache-Control": "no-cache, no-store"})


# ── Dezelfde zender als één doorlopende AAC-stream, voor de webversie ──
#
# Chrome en Firefox spelen HLS niet af in een gewoon <audio>-element (Safari
# wel). De web-app vraagt daarom dit adres: ffmpeg leest de HLS-playlist van
# schijf en geeft de AAC-stukken door zonder ze opnieuw te coderen (-c copy),
# dus het kost bijna niets. Wel één proces per webluisteraar — vandaar de grens.
_WEB_MAX = 4
_web_bezet = threading.Semaphore(_WEB_MAX)


@router.get("/live.aac")
def live_aac(request: Request, user: User = Depends(require_user)):
    import shutil
    import subprocess
    from fastapi.responses import StreamingResponse
    z = radio.huidige()
    if z is None or _review(user):
        raise HTTPException(503, "De radio staat uit")
    pad = z.hls / PLAYLIST
    if not pad.exists():
        raise HTTPException(503, "De radio start op, probeer het zo weer")
    if not _web_bezet.acquire(blocking=False):
        raise HTTPException(503, "Te veel webluisteraars tegelijk, probeer het zo weer")
    token = token_from_request(request)
    toestel = request.headers.get("X-Plex-Device-Name") or "Web"
    proc = subprocess.Popen(
        [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error",
         "-live_start_index", "-2", "-i", str(pad), "-vn", "-c:a", "copy", "-f", "adts", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def gen():
        laatst = 0.0
        try:
            while True:
                stuk = proc.stdout.read(16 * 1024)
                if not stuk:
                    break
                # Telt mee als luisteraar zolang hij leest, net als een
                # HLS-speler die de playlist ophaalt.
                if time.time() - laatst > 10:
                    laatst = time.time()
                    _tel(token)
                    _tel_gebruiker(user, toestel)
                yield stuk
        finally:
            try:
                proc.stdout.close()
                proc.terminate()
            except Exception:
                pass
            _web_bezet.release()

    return StreamingResponse(gen(), media_type="audio/aac",
                             headers={"Cache-Control": "no-cache, no-store"})


@router.get("/seg/{naam}")
def segment(naam: str, user: User = Depends(require_user)):
    if not _SEGMENT.match(naam) or _review(user):
        raise HTTPException(404)
    z = radio.huidige()
    if z is None:
        raise HTTPException(404)
    p = z.hls / naam
    if not p.exists():
        raise HTTPException(404)
    # Een stuk verandert nooit meer; een speler mag het bewaren zolang het in
    # de lijst staat.
    return FileResponse(p, media_type="video/mp2t",
                        headers={"Cache-Control": "public, max-age=120"})


@router.get("/nu")
def nu(user: User = Depends(require_user)):
    """Wat er speelt. Tijden in epoch-seconden, gerekend naar wanneer een
    luisteraar het hoort (zie config.RADIO_VERTRAGING_S)."""
    z = radio.huidige()
    aan = _aan() and not _review(user)
    if z is None or _review(user):
        return {"aan": aan, "draait": False, "stream": "/radio/live.m3u8",
                "nu": None, "straks": None, "laatste": [], "break": None}
    from ..radio import programma
    nu_ts = time.time()
    blok = programma.blok_op(nu_ts, z._week)
    daarna = programma.blok_op(blok.eind + 1, z._week)
    met_programma = z.instellingen()["programmering"]
    return {
        "aan": aan,
        "draait": z.draait(),
        "stream": "/radio/live.m3u8",
        "nu": radio.openbaar(z.nu),
        "straks": radio.openbaar(z.straks()),
        "laatste": z.laatste(10),
        "break": z.laatste_break,
        "programma": programma.voor_app(blok) if met_programma else None,
        "daarna": programma.voor_app(daarna) if met_programma else None,
        "luisteraars": luisteraars(),
    }


# ── Nummers die nooit meer op de radio komen (zie radio/blokkeer.py) ──

@router.get("/geblokkeerd")
def geblokkeerd(_: User = Depends(require_admin)):
    from ..radio import blokkeer
    return {"nummers": blokkeer.lees(config.RADIO_DIR)}


@router.post("/blokkeer")
def blokkeer_nummer(body: dict, _: User = Depends(require_admin)):
    """{"ratingKey": "t123"} → dat nummer nooit meer op de radio;
    {"albumKey": "al387"} → dat hele album niet."""
    from ..ids import parse_key
    from ..models import Album, Track
    from ..radio import blokkeer, rotatie
    if body.get("albumKey"):
        pa = parse_key(str(body["albumKey"]))
        if not pa or pa[0] != "al":
            raise HTTPException(400, "albumKey van een album verwacht")
        db = SessionLocal()
        try:
            al = db.get(Album, pa[1])
            if not al:
                raise HTTPException(404, "album niet gevonden")
            lijst = blokkeer.voeg_album_toe(config.RADIO_DIR, al.id, al.title or "")
        finally:
            db.close()
        z = radio.huidige()
        weg = z.blokkeer(album_id=pa[1]) if z is not None else 0
        return {"ok": True, "geblokkeerd": len(lijst), "uitDeWachtrij": weg}
    p = parse_key(str(body.get("ratingKey") or ""))
    if not p or p[0] != "t":
        raise HTTPException(400, "ratingKey van een nummer verwacht")
    db = SessionLocal()
    try:
        t = db.get(Track, p[1])
        if not t:
            raise HTTPException(404, "nummer niet gevonden")
        artiest = (t.orig_artist or t.artist_name or "").strip()
        sleutel = rotatie.nummer_sleutel(artiest, t.title or "")
        lijst = blokkeer.voeg_toe(config.RADIO_DIR, t.id, sleutel, t.title or "", artiest)
    finally:
        db.close()
    z = radio.huidige()
    weg = z.blokkeer(id_=t.id, sleutel=sleutel) if z is not None else 0
    return {"ok": True, "geblokkeerd": len(lijst), "uitDeWachtrij": weg}


@router.delete("/blokkeer/{rk}")
def deblokkeer(rk: str, _: User = Depends(require_admin)):
    from ..ids import parse_key
    from ..radio import blokkeer
    p = parse_key(rk)
    if not p or p[0] not in ("t", "al"):
        raise HTTPException(400, "ratingKey van een nummer of album verwacht")
    lijst = (blokkeer.haal_weg(config.RADIO_DIR, id_=p[1]) if p[0] == "t"
             else blokkeer.haal_weg(config.RADIO_DIR, album_id=p[1]))
    return {"ok": True, "geblokkeerd": len(lijst)}


# ── Het weekschema bewerken (Beheer → Radio) ──

def _programma_naar_json(uur: int, p) -> dict:
    d = {"uur": uur, "naam": p.naam, "sfeer": p.sfeer, "hits": p.hits}
    if p.jaren:
        d["jaren"] = list(p.jaren)
    if p.genres:
        d["genres"] = list(p.genres)
    if p.geen_genres:
        d["geen_genres"] = list(p.geen_genres)
    return d


def _bekende_programmas(week) -> dict:
    """Alle programma's die je kunt kiezen: de vaste uit programma.py plus wat
    er in het huidige schema staat, op naam."""
    from ..radio import programma as pg
    uit = {}
    for dag in list(pg.STANDAARD_WEEK.values()) + list(week.values()):
        for _, p in dag:
            uit.setdefault(p.naam, p)
    return uit


@router.get("/week")
def week_lezen(_: User = Depends(require_admin)):
    from ..radio import programma as pg
    z = radio.huidige()
    week = pg.laad_week(config.RADIO_DIR) if z is None else z._week
    bekend = _bekende_programmas(week)
    return {
        "week": {str(d): [{"uur": u, "naam": p.naam} for u, p in week[d]] for d in range(7)},
        "programmas": [{"naam": p.naam, "sfeer": p.sfeer} for p in bekend.values()],
    }


@router.put("/week")
def week_opslaan(body: dict, _: User = Depends(require_admin)):
    """{"week": {"0": [{"uur": 6, "naam": "Goedemorgen JLTamp"}, …], …}} — elke
    naam moet een bekend programma zijn. Schrijft programma.json; de zender
    neemt het meteen over. {"standaard": true} zet de standaardweek terug."""
    import json as _json
    from ..radio import programma as pg
    pad = config.RADIO_DIR / "programma.json"
    z = radio.huidige()
    if body.get("standaard"):
        pad.unlink(missing_ok=True)
    else:
        huidig = pg.laad_week(config.RADIO_DIR) if z is None else z._week
        bekend = _bekende_programmas(huidig)
        uit: dict[str, list] = {}
        for dag, blokken in (body.get("week") or {}).items():
            d = int(dag)
            if not 0 <= d <= 6:
                raise HTTPException(400, "weekdag 0..6")
            rij = []
            for b in blokken:
                u, naam = int(b["uur"]), str(b["naam"])
                if not 0 <= u <= 23 or naam not in bekend:
                    raise HTTPException(400, f"onbekend blok: {u} {naam}")
                rij.append(_programma_naar_json(u, bekend[naam]))
            if not rij or min(x["uur"] for x in rij) != 0:
                raise HTTPException(400, "elke dag begint om 0 uur")
            uit[str(d)] = sorted(rij, key=lambda x: x["uur"])
        config.RADIO_DIR.mkdir(parents=True, exist_ok=True)
        pad.write_text(_json.dumps(uit, ensure_ascii=False, indent=1))
    if z is not None:
        z._week = pg.laad_week(config.RADIO_DIR)
        z._pool_ts = 0.0          # de pool opnieuw vullen voor de nieuwe programma's
    return week_lezen(_)


@router.get("/programma")
def programma_vandaag(user: User = Depends(require_user)):
    """Het programma van vandaag, blok voor blok (voor de app)."""
    from ..radio import programma
    z = radio.huidige()
    if _review(user) or (z is not None and not z.instellingen()["programmering"]):
        return {"blokken": []}
    week = z._week if z is not None else programma.STANDAARD_WEEK
    nu_ts = time.time()
    blokken = []
    lt = time.localtime(nu_ts)
    middernacht = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    t = middernacht
    while t < middernacht + 86400 - 1:
        b = programma.blok_op(t + 1, week)
        blokken.append({**programma.voor_app(b), "nu": b.start <= nu_ts < b.eind})
        t = b.eind
    return {"blokken": blokken}


@router.get("/status")
def status(_: User = Depends(require_user)):
    z = radio.huidige()
    uit = {"aan": _aan(), "draait": False, "encoder": False, "luisteraars": luisteraars(),
           "presentator": config.RADIO_PRESENTATOR}
    if z is not None:
        uit.update(z.toestand())
    return uit
