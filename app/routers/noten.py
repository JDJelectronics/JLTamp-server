"""Noten: wat piano, gitaar en bas in een nummer écht spelen, noot voor noot.

Voor het meespeelscherm. De akkoorden (routers/akkoorden.py) zeggen welk
akkoord er klinkt, maar een pianist speelt dat niet als één blok: een arpeggio,
een loopje in de bas, een melodie in de rechterhand. Wie meespeelt wil zien
welke toets er NU wordt aangeslagen. Daarvoor is dit.

Het rekenwerk gebeurt niet hier maar in de karaoke-dienst op de host
(`karaoke_demucs.py`, `POST /taken?model=noten`): Demucs `htdemucs_6s` maakt
piano, gitaar en bas los, Basic Pitch maakt er noten van. Dat is te zwaar
voor deze container (geen GPU, 1500 MB) — en Basic Pitch leunt op librosa en
numba, die op de Studio in Docker al eens met een segmentatiefout de hele
server meenamen (zie `_herken_los` in akkoorden.py). Is de dienst er niet,
dan zegt de status `beschikbaar: false` en blijft de app bij de akkoorden.

Uitkomst per deel een lijst `{t, d, p, v}`: begin en duur in seconden, de
midi-toonhoogte en de sterkte (0..1). Een leeg deel = dat instrument speelt
niet mee (of lekte alleen in van een ander, zie de dienst).

De bron is het origineel en niet de karaokebegeleiding: het 6s-model haalt de
zang zelf al apart, en zo hangt dit niet af van of iemand ooit karaoke heeft
aangezet.

Bewaard in `DATA_DIR/noten/<track_id>.json`, met het bronkenmerk en `VERSIE`,
net als de akkoorden. Alles naar `/data`, nooit naar `/music`.

Geen eigen schakelaar: het hoort bij de gitaarknop, dus `akkoorden_aan`.
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

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..db import SessionLocal
from ..deps import require_user
from ..models import Track, User
from . import karaoke
from .akkoorden import akkoorden_aan

router = APIRouter(tags=["noten"])
log = logging.getLogger("jltamp.noten")

NOTEN_DIR = config.DATA_DIR / "noten"
WERK_DIR = NOTEN_DIR / "_werk"
# Wat hier bij het opstarten nog staat, is van een taak die een herstart van de
# server afbrak (de `finally` die opruimt kwam nooit): weg ermee. Er draait één
# serverproces, dus er is niemand die er nu nog mee bezig is.
shutil.rmtree(WERK_DIR, ignore_errors=True)

# Verhoog dit als de herkenning in de dienst verandert (drempels, model): dan
# wordt wat er ligt bij de volgende vraag opnieuw berekend.
VERSIE = 1

MAX_MINUTEN = karaoke.MAX_MINUTEN
MAX_RIJ = 20
MISLUKT_S = 600
DELEN = ("piano", "gitaar", "bas")


# ── de cache ────────────────────────────────────────────────────────────────

def _bestand(track_id: int) -> Path:
    return NOTEN_DIR / f"{track_id}.json"


def _lees(t: Track) -> dict | None:
    try:
        d = json.loads(_bestand(t.id).read_text())
    except Exception:
        return None
    if d.get("versie") != VERSIE or d.get("bron") != karaoke._bron_kenmerk(t):
        return None
    return d


def _beschikbaar() -> bool:
    b = karaoke.beschikbaarheid()
    return bool(b.get("beschikbaar") and b.get("route") == "dienst" and b.get("noten"))


# ── de rij ──────────────────────────────────────────────────────────────────
#
# Eén werker, één nummer tegelijk — de dienst doet er ook maar één tegelijk,
# en twee keer zoveel sturen maakt het daar alleen voller.

_rij: collections.deque[int] = collections.deque()
_staat: dict[int, dict] = {}       # track_id → {"status", "voortgang", "fout", "wanneer"}
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
        _werker = threading.Thread(target=_werk, name="noten", daemon=True)
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
            log.warning("[noten] nummer %s mislukt: %s", track_id, e)
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
    if t.duration_ms and t.duration_ms > MAX_MINUTEN * 60 * 1000:
        raise RuntimeError(f"langer dan {MAX_MINUTEN} minuten")
    if not _beschikbaar():
        raise RuntimeError("de notendienst is er niet")

    WERK_DIR.mkdir(parents=True, exist_ok=True)
    werk = WERK_DIR / str(track_id)
    shutil.rmtree(werk, ignore_errors=True)
    werk.mkdir(parents=True)
    begin = time.time()
    try:
        # De bron zoals de dienst hem wil: 44,1 kHz stereo FLAC. Van /music
        # gelezen, en alleen gelezen.
        bron = werk / "bron.flac"
        karaoke._ffmpeg("-i", t.path, "-vn", "-map", "0:a:0", "-ac", "2", "-ar", "44100",
                        "-t", str(MAX_MINUTEN * 60), "-c:a", "flac", str(bron))
        _zet(track_id, voortgang=0.05)
        uitkomst, info = _via_dienst(bron, lambda f: _zet(
            track_id, voortgang=round(0.05 + 0.93 * max(0.0, min(1.0, f)), 3)))
    finally:
        shutil.rmtree(werk, ignore_errors=True)

    NOTEN_DIR.mkdir(parents=True, exist_ok=True)
    doel = _bestand(t.id)
    tijdelijk = doel.with_suffix(".tmp")
    tijdelijk.write_text(json.dumps({
        "versie": VERSIE, "bron": karaoke._bron_kenmerk(t), "gemaakt": int(time.time()),
        "seconden": round(time.time() - begin, 1), "info": info,
        **{d: uitkomst.get(d) or [] for d in DELEN},
    }, separators=(",", ":")))
    # Pas als het hele bestand er staat: een half bestand mag nooit "klaar" lijken.
    os.replace(tijdelijk, doel)
    log.info("[noten] nummer %s klaar in %.0fs (%s)", track_id, time.time() - begin,
             ", ".join(f"{d} {len(uitkomst.get(d) or [])}" for d in DELEN))


def _via_dienst(bron: Path, melden) -> tuple[dict, dict]:
    with karaoke._dienst_verzoek("/taken?model=noten", "POST", bron.read_bytes(), timeout=120) as r:
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
        with karaoke._dienst_verzoek(f"/taken/{tid}/noten.json", timeout=60) as r:
            return json.loads(r.read()), s.get("info") or {}
    finally:
        try:
            karaoke._dienst_verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass


# ── de adressen ─────────────────────────────────────────────────────────────

def _status(t: Track, aan: bool) -> dict:
    uit = {"aan": aan, "beschikbaar": False, "status": "geen", "voortgang": 0.0,
           "fout": None, "plekInRij": None, **{d: [] for d in DELEN}}
    if not aan:
        return uit
    d = _lees(t)
    if d is not None:
        # Wat er al ligt is bruikbaar, ook als de dienst nu even weg is.
        uit.update(beschikbaar=True, status="klaar", voortgang=1.0,
                   **{k: d.get(k) or [] for k in DELEN})
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


@router.get("/noten/{track_id}")
def noten_status(track_id: str, user: User = Depends(require_user)):
    """De noten van dit nummer, of hoe ver het ermee staat. Verandert niets."""
    t = karaoke._track(track_id, user)
    return _status(t, akkoorden_aan(user))


@router.post("/noten/{track_id}")
def noten_aanvragen(track_id: str, user: User = Depends(require_user)):
    """Zet dit nummer in de rij, tenzij het al klaar is, er al in staat, of er
    nergens een dienst is om het te doen (dan blijft het bij de akkoorden)."""
    t = karaoke._track(track_id, user)
    if not akkoorden_aan(user):
        raise HTTPException(403, "Akkoorden staan uit")
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
