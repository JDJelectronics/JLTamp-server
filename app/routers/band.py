"""De bandmixer: elk instrument van een nummer apart harder, zachter of uit.

Karaoke haalt alleen de zang eruit. Hier gaat het om de hele band: zang,
gitaar, bas, drums, piano en de rest, elk een eigen schuif. Gitaar uit en zelf
meespelen, alleen de drums om de groove te horen, de bas harder om hem te leren.

Het scheiden gebeurt in de karaoke-dienst op de host (`karaoke_demucs.py`,
`POST /taken?model=band`): Demucs `htdemucs_6s` levert zes stereosporen die
opgeteld weer het origineel zijn. Die komen hier als m4a in
`DATA_DIR/band/<track_id>/`. Een menging maakt ffmpeg op aanvraag uit die zes
(een paar seconden) en bewaart hem naast de sporen: wie dezelfde stand nog eens
kiest, of het volgende nummer met dezelfde stand speelt, hoeft niet te wachten.

Standen per spoor in stappen van 25 %, van 0 tot 150. Boven de 100 komt er een
limiter achter, anders vervormt een opgeschroefde bas in een luid nummer.

Geen eigen schakelaar: het hoort bij karaoke (`karaoke_aan`). Alles naar
`/data`, nooit naar `/music`.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from .. import config
from ..db import SessionLocal
from ..deps import require_user
from ..models import Track, User
from . import karaoke

router = APIRouter(tags=["band"])
log = logging.getLogger("jltamp.band")

BAND_DIR = config.DATA_DIR / "band"
WERK_DIR = BAND_DIR / "_werk"
# Wat hier bij het opstarten nog staat is van een taak die een herstart afbrak.
shutil.rmtree(WERK_DIR, ignore_errors=True)

VERSIE = 1
SPOREN = ("zang", "gitaar", "bas", "drums", "piano", "rest")
STANDEN = (0, 25, 50, 75, 100, 125, 150)
MAX_RIJ = 20
MISLUKT_S = 600
# Zes sporen van een nummer van vier minuten zijn ~35 MB; met wat mengingen
# erbij ~60 MB. Hierboven gaan de langst niet gebruikte nummers weg.
MAX_MB = int(os.environ.get("JLTAMP_BAND_MAX_MB", "8000"))
MAX_MENGINGEN = 40          # per nummer


# ── de opslag ───────────────────────────────────────────────────────────────

def _map(track_id: int) -> Path:
    return BAND_DIR / str(track_id)


def _info(t: Track) -> dict | None:
    """info.json als de sporen er allemaal zijn en bij dit bestand horen."""
    m = _map(t.id)
    try:
        d = json.loads((m / "info.json").read_text())
    except Exception:
        return None
    if d.get("versie") != VERSIE or d.get("bron") != karaoke._bron_kenmerk(t):
        return None
    if not all((m / f"{s}.m4a").exists() for s in SPOREN):
        return None
    return d


def _beschikbaar() -> bool:
    b = karaoke.beschikbaarheid()
    return bool(b.get("beschikbaar") and b.get("route") == "dienst" and b.get("band"))


def _opruimen() -> None:
    try:
        mappen = [p for p in BAND_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
        grootte = {p: sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in mappen}
        totaal = sum(grootte.values())
        if totaal <= MAX_MB * 1024 * 1024:
            return
        # info.json wordt bij elk gebruik aangeraakt: de oudste mtime is het
        # langst niet gebruikt.
        def gebruikt(p: Path) -> float:
            try:
                return (p / "info.json").stat().st_mtime
            except OSError:
                return 0.0
        for p in sorted(mappen, key=gebruikt):
            if totaal <= MAX_MB * 1024 * 1024 * 0.9:
                break
            totaal -= grootte[p]
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        log.debug("[band] opruimen: %s", e)


# ── de rij ──────────────────────────────────────────────────────────────────

_rij: collections.deque[int] = collections.deque()
_staat: dict[int, dict] = {}
_slot = threading.Lock()
_wekker = threading.Event()
_werker: threading.Thread | None = None


def _zet(track_id: int, **velden) -> None:
    with _slot:
        s = _staat.setdefault(track_id, {"status": "wacht", "voortgang": 0.0, "fout": None})
        s.update(velden, wanneer=time.time())


def _start_werker() -> None:
    global _werker
    with _slot:
        if _werker is not None and _werker.is_alive():
            return
        _werker = threading.Thread(target=_werk, name="band", daemon=True)
        _werker.start()


def _werk() -> None:
    while True:
        with _slot:
            track_id = _rij.popleft() if _rij else None
        if track_id is None:
            _wekker.wait(60)
            _wekker.clear()
            continue
        _zet(track_id, status="bezig", voortgang=0.0)
        try:
            _verwerk(track_id)
            with _slot:
                _staat.pop(track_id, None)
        except Exception as e:
            log.warning("[band] nummer %s mislukt: %s", track_id, e)
            _zet(track_id, status="mislukt", fout=str(e)[:200])


def _verwerk(track_id: int) -> None:
    db = SessionLocal()
    try:
        t = db.get(Track, track_id)
        if not t:
            raise RuntimeError("nummer bestaat niet meer")
        db.expunge(t)
    finally:
        db.close()
    if t.duration_ms and t.duration_ms > karaoke.MAX_MINUTEN * 60 * 1000:
        raise RuntimeError(f"langer dan {karaoke.MAX_MINUTEN} minuten")
    if not _beschikbaar():
        raise RuntimeError("de dienst kan geen band scheiden")

    WERK_DIR.mkdir(parents=True, exist_ok=True)
    werk = WERK_DIR / str(track_id)
    shutil.rmtree(werk, ignore_errors=True)
    werk.mkdir(parents=True)
    begin = time.time()
    try:
        bron = werk / "bron.flac"
        karaoke._ffmpeg("-i", t.path, "-vn", "-map", "0:a:0", "-ac", "2", "-ar", "44100",
                        "-t", str(karaoke.MAX_MINUTEN * 60), "-c:a", "flac", str(bron))
        _zet(track_id, voortgang=0.05)
        info = _via_dienst(bron, werk, lambda f: _zet(
            track_id, voortgang=round(0.05 + 0.8 * max(0.0, min(1.0, f)), 3)))
        bron.unlink(missing_ok=True)
        aac = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
        for i, s in enumerate(SPOREN):
            flac = werk / f"band_{s}.flac"
            karaoke._ffmpeg("-i", str(flac), *aac, str(werk / f"{s}.m4a"))
            flac.unlink(missing_ok=True)
            _zet(track_id, voortgang=round(0.85 + 0.14 * (i + 1) / len(SPOREN), 3))
        (werk / "info.json").write_text(json.dumps({
            "versie": VERSIE, "bron": karaoke._bron_kenmerk(t), "gemaakt": int(time.time()),
            "seconden": round(time.time() - begin, 1),
            "niveaus": info.get("niveaus") or {},
        }))
        doel = _map(t.id)
        shutil.rmtree(doel, ignore_errors=True)
        os.replace(werk, doel)
    finally:
        shutil.rmtree(werk, ignore_errors=True)
    log.info("[band] nummer %s klaar in %.0fs", track_id, time.time() - begin)
    _opruimen()


def _via_dienst(bron: Path, werk: Path, melden) -> dict:
    with karaoke._dienst_verzoek("/taken?model=band", "POST", bron.read_bytes(), timeout=120) as r:
        tid = json.loads(r.read())["id"]
    try:
        einde = time.time() + 30 * 60
        while True:
            if time.time() > einde:
                raise RuntimeError("dienst duurde te lang")
            time.sleep(1.5)
            with karaoke._dienst_verzoek(f"/taken/{tid}") as r:
                s = json.loads(r.read())
            if s.get("status") == "klaar":
                break
            if s.get("status") == "mislukt":
                raise RuntimeError(s.get("fout") or "dienst: mislukt")
            melden(float(s.get("voortgang") or 0.0))
        for spoor in SPOREN:
            with karaoke._dienst_verzoek(f"/taken/{tid}/band_{spoor}.flac", timeout=120) as r, \
                    open(werk / f"band_{spoor}.flac", "wb") as f:
                shutil.copyfileobj(r, f, 256 * 1024)
        return s.get("info") or {}
    finally:
        try:
            karaoke._dienst_verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass


# ── het mengen ──────────────────────────────────────────────────────────────

_meng_sloten: dict[str, threading.Lock] = {}
_meng_slot = threading.Lock()


def _standen(n: str) -> tuple[int, ...]:
    """'100,0,100,100,100,100' → zes standen, elk uit STANDEN."""
    try:
        st = tuple(int(x) for x in n.split(","))
    except ValueError:
        raise HTTPException(400, "standen onleesbaar")
    if len(st) != len(SPOREN) or any(s not in STANDEN for s in st):
        raise HTTPException(400, f"zes standen uit {STANDEN}")
    if not any(st):
        raise HTTPException(400, "alles uit")
    return st


def _meng(t: Track, st: tuple[int, ...]) -> Path:
    m = _map(t.id)
    naam = "mix_" + "-".join(str(s) for s in st) + ".m4a"
    doel = m / naam
    if doel.exists():
        os.utime(doel)
        return doel
    sleutel = f"{t.id}:{naam}"
    with _meng_slot:
        slot = _meng_sloten.setdefault(sleutel, threading.Lock())
    with slot:
        if doel.exists():
            return doel
        invoer, delen = [], []
        for spoor, stand in zip(SPOREN, st):
            if stand == 0:
                continue
            i = len(invoer) // 2
            invoer += ["-i", str(m / f"{spoor}.m4a")]
            delen.append((i, stand))
        ketens = [f"[{i}:a]volume={stand / 100:.2f}[s{i}]" for i, stand in delen]
        samen = "".join(f"[s{i}]" for i, _ in delen)
        na = ",alimiter=limit=0.97:attack=5:release=50" if max(st) > 100 else ""
        filter_ = ";".join(ketens) + f";{samen}amix=inputs={len(delen)}:duration=longest:normalize=0{na}"
        tijdelijk = m / f".{naam}"
        karaoke._ffmpeg(*invoer, "-filter_complex", filter_, "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(tijdelijk))
        os.replace(tijdelijk, doel)
    # Niet onbeperkt mengingen per nummer bewaren: de oudste gaan weg.
    oud = sorted(m.glob("mix_*.m4a"), key=lambda p: p.stat().st_mtime)
    for p in oud[:-MAX_MENGINGEN]:
        p.unlink(missing_ok=True)
    with _meng_slot:
        _meng_sloten.pop(sleutel, None)
    return doel


# ── de adressen ─────────────────────────────────────────────────────────────

def _status(t: Track, aan: bool) -> dict:
    uit = {"aan": aan, "beschikbaar": False, "status": "geen", "voortgang": 0.0,
           "fout": None, "plekInRij": None, "sporen": list(SPOREN), "standen": list(STANDEN),
           "niveaus": {}, "versie": None}
    if not aan:
        return uit
    d = _info(t)
    if d is not None:
        uit.update(beschikbaar=True, status="klaar", voortgang=1.0,
                   niveaus=d.get("niveaus") or {}, versie=d.get("gemaakt"))
        return uit
    uit["beschikbaar"] = _beschikbaar()
    with _slot:
        s = dict(_staat.get(t.id) or {})
        plek = list(_rij).index(t.id) + 1 if t.id in _rij else None
    if s:
        if s["status"] == "mislukt" and time.time() - s.get("wanneer", 0) > MISLUKT_S:
            return uit
        uit.update(status=s["status"], voortgang=s.get("voortgang", 0.0),
                   fout=s.get("fout"), plekInRij=plek)
    return uit


class KlaarBody(BaseModel):
    ids: list[str]


@router.post("/band/klaar")
def band_klaar(body: KlaarBody, user: User = Depends(require_user)):
    """Welke van deze nummers al gesplitst zijn. De quiz begint daarmee, zodat
    er niet eerst een minuut gescheiden hoeft te worden. Alleen kijken, er
    wordt niets aangevraagd."""
    if not karaoke.karaoke_aan(user):
        return {"klaar": []}
    uit = []
    for rk in body.ids[:300]:
        try:
            t = karaoke._track(rk, user)
        except HTTPException:
            continue
        if _info(t) is not None:
            uit.append(rk)
    return {"klaar": uit}


@router.get("/band/{track_id}")
def band_status(track_id: str, user: User = Depends(require_user)):
    """Zijn de sporen van dit nummer er, of hoe ver het ermee staat."""
    t = karaoke._track(track_id, user)
    return _status(t, karaoke.karaoke_aan(user))


@router.post("/band/{track_id}")
def band_aanvragen(track_id: str, user: User = Depends(require_user)):
    """Zet dit nummer in de rij, tenzij het al klaar is of er al in staat."""
    t = karaoke._track(track_id, user)
    if not karaoke.karaoke_aan(user):
        raise HTTPException(403, "Karaoke staat uit")
    st = _status(t, True)
    if not st["beschikbaar"] or st["status"] in ("klaar", "wacht", "bezig"):
        return st
    with _slot:
        if t.id not in _rij:
            if len(_rij) >= MAX_RIJ:
                raise HTTPException(429, "Even te druk — probeer het zo weer")
            _rij.append(t.id)
            _staat[t.id] = {"status": "wacht", "voortgang": 0.0, "fout": None, "wanneer": time.time()}
    _start_werker()
    _wekker.set()
    return _status(t, True)


@router.get("/band/{track_id}/mix.m4a")
def band_mix(track_id: str, n: str = Query(...), v: int | None = None,
             user: User = Depends(require_user)):
    """De menging met deze standen (zang,gitaar,bas,drums,piano,rest in %).
    `v` zit er alleen in zodat een nieuwe scheiding een nieuw adres heeft."""
    t = karaoke._track(track_id, user)
    if not karaoke.karaoke_aan(user):
        raise HTTPException(403, "Karaoke staat uit")
    st = _standen(n)
    if _info(t) is None:
        raise HTTPException(404, "Nog niet gescheiden")
    try:
        os.utime(_map(t.id) / "info.json")
    except OSError:
        pass
    return FileResponse(_meng(t, st), media_type="audio/mp4")
