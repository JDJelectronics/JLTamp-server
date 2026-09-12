"""Welk nummer van een album het bekendste is.

Niet uit de eigen luistergeschiedenis — dat zou alleen zeggen wat JIJ vaak
draait — maar van buiten: Deezer geeft per nummer een populariteitscijfer over
de hele wereld. Zo zie je bij een album dat je net hebt gevonden meteen welk
nummer "het" nummer is.

Deezer is hier de bron omdat de server hem al gebruikt voor hoezen en
artiestfoto's: geen sleutel nodig, en geen nieuwe partij die er iets bij komt
weten. Naar buiten gaat alleen de artiest- en albumnaam.

Leeg antwoord is een geldig antwoord en gebeurt vaak: eigen opnames,
verzamelaars en obscure uitgaven staan niet in Deezer. Dan hoort er in de app
gewoon niets te verschijnen.

Uit staat het met `JLTAMP_ALBUM_POPULAIR=false`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..deps import require_user
from ..beats import tellen_van
from ..db import SessionLocal
from ..metadata import deezer_album_track_ranks, deezer_track_bpm
from ..models import Track
from ..models import User

router = APIRouter(tags=["album"])
log = logging.getLogger("populair")

CACHE = config.DATA_DIR / "albumpopulair"
CACHE.mkdir(parents=True, exist_ok=True)

_TREFFER_GELDIG = 30 * 24 * 3600
_MISSER_GELDIG = 3 * 24 * 3600


def _aan() -> bool:
    return os.environ.get("JLTAMP_ALBUM_POPULAIR", "true").strip().lower() in ("1", "true", "yes", "on")


def _sleutel(artiest: str, album: str) -> str:
    plat = re.sub(r"[^a-z0-9]+", "-", f"{artiest}-{album}".lower()).strip("-") or "leeg"
    return f"{plat[:120]}.json"


@router.get("/album/popular")
def album_popular(artist: str, album: str, user: User = Depends(require_user)):
    """Nummers van dit album met hun populariteit, aflopend gesorteerd.

    De app zoekt daar zelf zijn eigen nummers in op. Dat is met opzet: de
    schrijfwijze verschilt vaak ("Ring Ring (English Version)" tegenover
    "Ring Ring - English Version"), en de app weet welke titels hij écht heeft.
    """
    artiest = (artist or "").strip()
    alb = (album or "").strip()
    if not artiest or not alb:
        raise HTTPException(status_code=400, detail="artist en album zijn verplicht")

    leeg = {"nummers": [], "bron": ""}
    if not _aan():
        return leeg

    bestand = CACHE / _sleutel(artiest, alb)
    if bestand.exists():
        try:
            gecacht = json.loads(bestand.read_text(encoding="utf-8"))
            ouderdom = time.time() - float(gecacht.get("opgehaald") or 0)
            geldig = _TREFFER_GELDIG if gecacht.get("nummers") else _MISSER_GELDIG
            if ouderdom < geldig:
                return {k: v for k, v in gecacht.items() if k != "opgehaald"}
        except Exception:
            pass

    try:
        nummers = deezer_album_track_ranks(artiest, alb)
    except Exception as e:
        # Deezer onbereikbaar: NIETS cachen, anders staat dit album dagenlang
        # als "niets gevonden" genoteerd door één hapering.
        log.info("deezer-opvraging mislukt: %s", e)
        return leeg

    uit = {"nummers": nummers, "bron": "deezer" if nummers else ""}
    try:
        bestand.write_text(json.dumps({**uit, "opgehaald": time.time()}), encoding="utf-8")
    except Exception as e:
        log.debug("cache schrijven mislukt: %s", e)
    return uit


TEMPO_CACHE = config.DATA_DIR / "tempo"
TEMPO_CACHE.mkdir(parents=True, exist_ok=True)


@router.get("/track/tempo")
def track_tempo(artist: str, title: str, user: User = Depends(require_user)):
    """Het tempo van een nummer, voor een gloed die met de muziek meeklopt.

    `bpm` is 0 als het onbekend is, en dat gebeurt regelmatig. De app hoort dan
    gewoon rustig te ademen in plaats van iets te verzinnen.

    Let op wat dit WEL en NIET is: dit is het tempo, niet de maat. Waar de tel
    precies valt weten we niet, dus een gloed hierop loopt in de juiste snelheid
    mee maar staat niet gegarandeerd op de tel. Echt meeklokken zou het geluid
    zelf moeten analyseren, en dat gebeurt hier niet.
    """
    art = (artist or "").strip()
    tit = (title or "").strip()
    if not art or not tit:
        raise HTTPException(status_code=400, detail="artist en title zijn verplicht")

    leeg = {"bpm": 0.0}
    if not _aan():
        return leeg

    bestand = TEMPO_CACHE / _sleutel(art, tit)
    if bestand.exists():
        try:
            gecacht = json.loads(bestand.read_text(encoding="utf-8"))
            ouderdom = time.time() - float(gecacht.get("opgehaald") or 0)
            geldig = _TREFFER_GELDIG if gecacht.get("bpm") else _MISSER_GELDIG
            if ouderdom < geldig:
                return {"bpm": gecacht.get("bpm") or 0.0}
        except Exception:
            pass

    try:
        bpm = deezer_track_bpm(art, tit)
    except Exception as e:
        # Niets cachen bij een storing; anders blijft dit nummer dagenlang
        # "geen tempo" door één hapering.
        log.info("tempo-opvraging mislukt: %s", e)
        return leeg

    try:
        bestand.write_text(json.dumps({"bpm": bpm, "opgehaald": time.time()}), encoding="utf-8")
    except Exception as e:
        log.debug("tempo cache schrijven mislukt: %s", e)
    return {"bpm": bpm}


BEAT_CACHE = config.DATA_DIR / "beats"
BEAT_CACHE.mkdir(parents=True, exist_ok=True)


@router.get("/track/beats")
def track_beats(id: str, user: User = Depends(require_user)):
    """Waar de tellen van dit nummer vallen, in milliseconden vanaf het begin.

    Uit het GELUID, niet uit een catalogus — zie app/beats.py. Daarmee klopt een
    gloed in de app echt mee in plaats van alleen op de goede snelheid te lopen,
    en werkt het ook voor nummers die nergens in een catalogus staan.

    Een lege lijst is een geldig antwoord: te kort, onleesbaar, of de analyse
    kwam er niet uit. De app hoort dan rustig te ademen.
    """
    # De app kent een nummer als "t54721": de ratingKey draagt een letter voor
    # het soort. Hier is alleen het getal bruikbaar. Als `int` binnenhalen gaf
    # een 422 en dus nooit tellen, terwijl er verder niets mis was.
    cijfers = "".join(c for c in str(id) if c.isdigit())
    if not cijfers:
        raise HTTPException(status_code=400, detail="id moet een nummer-id zijn")
    nummer_id = int(cijfers)

    db = SessionLocal()
    try:
        t = db.get(Track, nummer_id)
        if not t or not t.path:
            raise HTTPException(status_code=404, detail="nummer niet gevonden")
        pad = t.path
        artiest = (t.orig_artist or t.artist_name or "").strip()
        titel = (t.title or "").strip()
    finally:
        db.close()

    leeg = {"bpm": 0.0, "tellen": [], "energie": [], "stap": 250}
    if not _aan():
        return leeg

    bestand = BEAT_CACHE / f"{nummer_id}.json"
    if bestand.exists():
        try:
            gecacht = json.loads(bestand.read_text(encoding="utf-8"))
            ouderdom = time.time() - float(gecacht.get("opgehaald") or 0)
            geldig = _TREFFER_GELDIG if gecacht.get("tellen") else _MISSER_GELDIG
            if ouderdom < geldig:
                return {
                    "bpm": gecacht.get("bpm") or 0.0,
                    "tellen": gecacht.get("tellen") or [],
                    "energie": gecacht.get("energie") or [],
                    "stap": gecacht.get("stap") or 250,
                }
        except Exception:
            pass

    # Deezers tempo als startpunt, als hij het kent. Zie de kop van beats.py:
    # dat haalt de half/dubbel-missers er grotendeels uit. Kent hij het niet,
    # dan gaat de analyse zonder startpunt door — geen reden om af te haken.
    start = 0.0
    try:
        if artiest and titel:
            start = deezer_track_bpm(artiest, titel)
    except Exception:
        start = 0.0

    uit = tellen_van(pad, start)
    try:
        bestand.write_text(json.dumps({**uit, "opgehaald": time.time()}), encoding="utf-8")
    except Exception as e:
        log.debug("beat cache schrijven mislukt: %s", e)
    return uit
