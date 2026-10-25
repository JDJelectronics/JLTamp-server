"""Songteksten zonder tijdcodes uitlijnen op de zang.

Veel teksten (LRCLIB zonder synced, tags, .txt) zijn platte tekst: de app kan
dan alleen schatten waar je bent, en die schatting loopt achter of voor. Hier
krijgt zo'n tekst per regel een echte begintijd: de karaoke-dienst op de Mac
laat Whisper naar het zangspoor luisteren en legt de tekst daarop
(karaoke_demucs.py → karaoke_tekst.py).

  - Heeft de karaoke de zang van dit nummer al, dan gaat alleen die (mono,
    16 kHz — wat Whisper wil, en klein). Anders het hele nummer; de dienst
    haalt de zang er dan eerst zelf uit.
  - Eén nummer tegelijk, in een eigen rij. Het resultaat staat in
    DATA_DIR/uitgelijnd/<id>.json, met de bron en de tekst erbij: verandert
    het bestand of de tekst, dan telt het niet meer.
  - Vindt Whisper te weinig van de tekst terug (een andere versie, een
    instrumentaal nummer met een verkeerde tekst), dan geen tijden: liever de
    schatting in de app dan zelfverzekerd fout. Ook dat wordt onthouden.

lyrics.get_lyrics vraagt dit aan en geeft de tijden mee zodra ze er zijn.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import json
import logging
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from .. import config
from ..db import SessionLocal
from ..models import Track
from . import karaoke

log = logging.getLogger("uitlijnen")

VERSIE = 1
DIR = config.DATA_DIR / "uitgelijnd"
WERK_DIR = DIR / "_werk"
# Wat hier bij het opstarten nog staat, is van een taak die een herstart van de
# server afbrak (de `finally` die opruimt kwam nooit): weg ermee. Er draait één
# serverproces, dus er is niemand die er nu nog mee bezig is.
shutil.rmtree(WERK_DIR, ignore_errors=True)
# Minder dan dit deel van de tekstwoorden teruggevonden: geen tijden.
MIN_GEVONDEN = 0.3
# Na een mislukte poging (dienst weg, fout) pas na zo lang opnieuw.
OPNIEUW_NA_S = 6 * 3600
MAX_MINUTEN = 15
# Meer in de rij is geen wachten meer maar een opstopping (iemand bladert door
# een lijst vol platte teksten): dan later nog eens, bij de volgende keer openen.
MAX_RIJ = 20


def _kenmerk(tekst: str) -> str:
    return hashlib.sha1(tekst.strip().encode("utf-8")).hexdigest()[:16]


def _bestand(track_id: int) -> Path:
    return DIR / f"{track_id}.json"


def gelijnd(t: Track, tekst: str) -> list[dict] | None:
    """De uitgelijnde regels, als ze er zijn en nog bij dit bestand en deze tekst horen."""
    try:
        d = json.loads(_bestand(t.id).read_text(encoding="utf-8"))
    except Exception:
        return None
    if (d.get("versie") != VERSIE or d.get("bron") != karaoke._bron_kenmerk(t)
            or d.get("tekst") != _kenmerk(tekst)):
        return None
    return d.get("regels") or None


def _afgewezen(t: Track, tekst: str) -> bool:
    """Al geprobeerd en te weinig gevonden (of kort geleden mislukt)?"""
    try:
        d = json.loads(_bestand(t.id).read_text(encoding="utf-8"))
    except Exception:
        return False
    if d.get("versie") != VERSIE or d.get("tekst") != _kenmerk(tekst):
        return False
    if d.get("regels"):
        return False
    return d.get("definitief") or time.time() - float(d.get("wanneer") or 0) < OPNIEUW_NA_S


def _dienst_kan() -> bool:
    b = karaoke.beschikbaarheid()
    return bool(b.get("beschikbaar") and b.get("route") == "dienst" and b.get("tekst"))


# ── de rij ──────────────────────────────────────────────────────────────────

_rij: collections.deque[tuple[int, str]] = collections.deque()
_bezig: set[int] = set()
_slot = threading.Lock()
_wekker = threading.Event()
_werker: threading.Thread | None = None


def vraag(t: Track, tekst: str) -> str | None:
    """Zet het nummer in de rij als dat zin heeft. "bezig" als er tijden onderweg zijn."""
    if not tekst.strip() or _afgewezen(t, tekst) or not _dienst_kan():
        return None
    if t.duration_ms and t.duration_ms > MAX_MINUTEN * 60 * 1000:
        return None
    global _werker
    with _slot:
        if t.id not in _bezig:
            if len(_rij) >= MAX_RIJ:
                return None
            _bezig.add(t.id)
            _rij.append((t.id, tekst))
        if _werker is None or not _werker.is_alive():
            _werker = threading.Thread(target=_werk, name="uitlijnen", daemon=True)
            _werker.start()
    _wekker.set()
    return "bezig"


def _werk() -> None:
    while True:
        with _slot:
            klus = _rij.popleft() if _rij else None
        if klus is None:
            _wekker.wait(60)
            _wekker.clear()
            continue
        track_id, tekst = klus
        try:
            _verwerk(track_id, tekst)
        except Exception as e:
            log.warning("[uitlijnen] nummer %s mislukt: %s", track_id, e)
            _bewaar(track_id, tekst, None, definitief=False, fout=str(e)[:200])
        finally:
            with _slot:
                _bezig.discard(track_id)


def _bewaar(track_id: int, tekst: str, regels: list[dict] | None, definitief: bool,
            fout: str | None = None, gevonden: float | None = None, bron: dict | None = None) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    d = {"versie": VERSIE, "tekst": _kenmerk(tekst), "bron": bron, "regels": regels,
         "definitief": definitief, "fout": fout, "gevonden": gevonden, "wanneer": time.time()}
    tmp = _bestand(track_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_bestand(track_id))


def _verzoek(pad: str, methode: str = "GET", data: bytes | None = None, kop: dict | None = None,
             timeout: float = 10.0):
    req = urllib.request.Request(f"{karaoke.DIENST_URL}{pad}", data=data, method=methode)
    if karaoke.DIENST_SLEUTEL:
        req.add_header("X-Karaoke-Sleutel", karaoke.DIENST_SLEUTEL)
    for k, v in (kop or {}).items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=timeout)


def _verwerk(track_id: int, tekst: str) -> None:
    db = SessionLocal()
    try:
        t = db.get(Track, track_id)
        if not t:
            return
        db.expunge(t)
    finally:
        db.close()
    bron_kenmerk = karaoke._bron_kenmerk(t)

    # De zang van de karaoke als die er is (en bij dit bestand hoort), anders het nummer.
    zang = karaoke._map(t.id) / "zang.m4a"
    met_zang = zang.exists() and karaoke._is_klaar(t)
    WERK_DIR.mkdir(parents=True, exist_ok=True)
    flac = WERK_DIR / f"{track_id}.flac"
    try:
        if met_zang:
            karaoke._ffmpeg("-i", str(zang), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(flac))
        else:
            karaoke._ffmpeg("-i", t.path, "-vn", "-map", "0:a:0", "-ac", "2", "-ar", "44100",
                            "-t", str(MAX_MINUTEN * 60), "-c:a", "flac", str(flac))
        data = flac.read_bytes()
    finally:
        flac.unlink(missing_ok=True)

    kop = {"Content-Type": "audio/flac", "X-Tekst": base64.b64encode(tekst.encode("utf-8")).decode("ascii")}
    pad = "/taken?model=tekst" + ("&zang=1" if met_zang else "")
    with _verzoek(pad, "POST", data, kop, timeout=120) as r:
        tid = json.loads(r.read())["id"]
    try:
        einde = time.time() + 15 * 60
        while True:
            if time.time() > einde:
                raise RuntimeError("dienst duurde te lang")
            time.sleep(1.5)
            with _verzoek(f"/taken/{tid}") as r:
                st = json.loads(r.read())
            if st.get("status") == "klaar":
                break
            if st.get("status") == "mislukt":
                raise RuntimeError(st.get("fout") or "dienst: mislukt")
        with _verzoek(f"/taken/{tid}/tekst.json", timeout=60) as r:
            uit = json.loads(r.read())
    finally:
        try:
            _verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass

    gevonden = float(uit.get("gevonden") or 0)
    regels = uit.get("regels") or []
    goed = gevonden >= MIN_GEVONDEN and len(regels) >= 2
    log.info("[uitlijnen] nummer %s: %d regels, %.0f%% gevonden%s", track_id, len(regels),
             gevonden * 100, "" if goed else " — te weinig, geen tijden")
    _bewaar(track_id, tekst, regels if goed else None, definitief=True,
            gevonden=gevonden, bron=bron_kenmerk)
