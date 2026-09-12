"""Online metadata enrichment (Plex-style) — fetches nice artist photos and
album covers so the app looks rich even when local files have no embedded art.

Primary source: **Deezer** public API (no API key, good artist pictures + album
covers). All fetched images are cached under the writable data dir
(`/data/artwork`) — NEVER written back to the read-only NAS.

Stdlib only (urllib) to keep the server dependency-light. Best-effort and
rate-limited; failures are logged and skipped, never fatal to a scan.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import urllib.parse
import urllib.request

from sqlalchemy import select

from . import config
from .db import SessionLocal
from .models import Artist, Album

log = logging.getLogger("metadata")

_UA = "JLTamp/1.0 (+self-hosted music server)"
_DEEZER = "https://api.deezer.com"
_RATE_SLEEP = 0.25  # be polite to Deezer


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        log.debug("deezer GET failed %s: %s", url, e)
        return None


def _download(url: str, dest) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        log.debug("art download failed %s: %s", url, e)
        return False


def _deezer_artist_image(name: str) -> str | None:
    q = urllib.parse.quote(name)
    data = _get_json(f"{_DEEZER}/search/artist?q={q}&limit=1")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return None
    a = items[0]
    # prefer the largest; Deezer returns picture_xl (1000px) down to picture_small
    for k in ("picture_xl", "picture_big", "picture_medium", "picture"):
        url = a.get(k)
        if url and "/artist//" not in url:  # skip empty placeholder
            return url
    return None


def _deezer_album_cover(artist: str, album: str) -> str | None:
    q = urllib.parse.quote(f"{artist} {album}".strip())
    data = _get_json(f"{_DEEZER}/search/album?q={q}&limit=1")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return None
    a = items[0]
    for k in ("cover_xl", "cover_big", "cover_medium", "cover"):
        url = a.get(k)
        if url and "/cover//" not in url:
            return url
    return None


def zonder_artiest_ruw(artist: str, album: str) -> str:
    """De albumtitel zonder de artiest die er soms voor geplakt staat."""
    return re.sub(rf"^{re.escape(artist)}\s*[-–—:]\s*", "", album or "", flags=re.IGNORECASE)


def _zonder_haakjes(t: str) -> str:
    """De titel zonder toevoegingen tussen haakjes OF blokhaken.

    Blokhaken hoorden er eerst niet bij, en juist die staan in deze bibliotheek:
    "Loud [Hi-Res]", "hickey [Hi-Res]". Deezer noemt dat gewoon "Loud".
    """
    return _plat(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t or ""))


def deezer_album_track_ranks(artist: str, album: str) -> list[dict]:
    """Hoe populair elk nummer van een album is, volgens Deezer.

    Deezer geeft per nummer een `rank`. Dat is een cijfer over de hele wereld,
    niet iets uit de eigen luistergeschiedenis — precies wat je wilt weten als je
    een album net gevonden hebt en zelf nog niet kent.

    Leeg als het album niet gevonden wordt, en dat gebeurt: eigen opnames,
    verzamelaars en obscure uitgaven staan er niet in.

    **Gewone zoekopdracht, geen veldvorm.** Hier stond `artist:"X" album:"Y"`.
    Dat ziet er precies uit maar levert regelmatig NUL treffers op terwijl het
    album er gewoon is: bij Taylor Swift gaf de veldvorm niets voor evermore,
    folklore én 1989, en een gewone zoekopdracht vond ze alle drie meteen. Bij
    het opzoeken van het tempo speelde exact hetzelfde.

    Wat de veldvorm wél gaf — zekerheid dat je de juiste hebt — wordt nu
    nagelopen: artiest én albumtitel moeten kloppen. Zoeken op albumtitel alleen
    levert vrolijk een gelijknamig album van iemand anders op, en "folklore"
    matcht anders net zo goed met "folklore: the long pond studio sessions".
    """
    if not (artist or "").strip() or not (album or "").strip():
        return []
    # Waarmee er GEZOCHT wordt is iets anders dan waarmee er wordt gecontroleerd.
    # De titel zoals hij in de bibliotheek staat is vaak geen bruikbare
    # zoekterm: "Rihanna - Loud [Hi-Res]" wordt met de artiest ervoor
    # "Rihanna Rihanna - Loud [Hi-Res]", en daar vindt Deezer niets op — nul
    # treffers, dus de controle hieronder kwam niet eens aan bod. Zoeken doen we
    # daarom met een opgeschoonde titel; controleren met alle vormen.
    zoekterm = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", zonder_artiest_ruw(artist, album))
    zoekterm = re.sub(r"\s+", " ", zoekterm).strip() or album
    q = urllib.parse.quote(f"{artist} {zoekterm}")
    data = _get_json(f"{_DEEZER}/search/album?q={q}&limit=5")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return []

    wil_art = _plat(artist)
    wil_alb = _plat(album)
    wil_kaal = _zonder_haakjes(album)
    # Sommige albums dragen de artiest in hun eigen titel ("Kelly Clarkson -
    # Breakaway"). Deezer noemt dat gewoon "Breakaway", dus zonder deze variant
    # vindt hij het album wel maar keurt de controle het af.
    zonder_artiest = zonder_artiest_ruw(artist, album)
    wil_zonder = _plat(zonder_artiest)

    gevonden = None
    tweede_keus = None
    for kandidaat in items:
        hun_art = _plat((kandidaat.get("artist") or {}).get("name") or "")
        if hun_art != wil_art:
            continue
        hun_alb = _plat(kandidaat.get("title") or "")
        if hun_alb == wil_alb:
            gevonden = kandidaat
            break
        # Zonder de haakjes erachter mag het ook — "folklore (deluxe version)"
        # tegenover "folklore". Maar alleen als tweede keus, want dan kan het
        # een andere uitgave zijn met een andere nummerlijst.
        if hun_alb == wil_zonder:
            gevonden = kandidaat
            break
        # Alle vormen waarin deze bibliotheek een albumtitel schrijft: met de
        # artiest ervoor, met een toevoeging tussen haakjes of blokhaken, of
        # allebei tegelijk — "Rihanna - Loud [Hi-Res]" tegenover Deezers "Loud".
        if tweede_keus is None and _zonder_haakjes(kandidaat.get("title") or "") in (
            wil_kaal, _plat(zonder_artiest), _zonder_haakjes(zonder_artiest),
        ):
            tweede_keus = kandidaat

    gevonden = gevonden or tweede_keus
    if not gevonden:
        log.debug("deezer: geen passend album voor %s - %s", artist, album)
        return []

    alb = _get_json(f"{_DEEZER}/album/{gevonden.get('id')}")
    time.sleep(_RATE_SLEEP)
    nummers = ((alb or {}).get("tracks") or {}).get("data") or []
    uit = [
        {"titel": t.get("title") or "", "rang": int(t.get("rank") or 0)}
        for t in nummers if t.get("title")
    ]
    uit.sort(key=lambda x: -x["rang"])
    return uit


# Namen die met leestekens zijn geschreven. Zonder deze omzetting valt "Ke$ha"
# terug op "ke ha" en "P!nk" op "p nk", en dan matcht geen enkele catalogus.
_STIJL = str.maketrans({"$": "s", "!": "i", "€": "e", "@": "a", "0": "o"})


def _plat(tekst: str) -> str:
    plat = unicodedata.normalize("NFKD", tekst or "")
    plat = "".join(c for c in plat if not unicodedata.combining(c))
    plat = plat.lower().translate(_STIJL)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", plat)).strip()


def deezer_track_bpm(artist: str, title: str) -> float:
    """Het tempo van een nummer in slagen per minuut, volgens Deezer. 0 = onbekend.

    Niet elk nummer heeft er een — gemeten op een steekproef uit de eigen
    bibliotheek: zes van de acht. En de tags in de bestanden zelf helpen niet,
    daar had maar één op de acht een tempoveld.

    De gevonden titel wordt nagekeken. Zoeken op "a-ha Take on Me" gaf een
    akoestische heropname uit 2017, en dat is een ander tempo dan het origineel.
    Liever niets dan een verkeerde maat: een gloed die naast de muziek klopt is
    erger dan een gloed die rustig ademt.
    """
    if not (artist or "").strip() or not (title or "").strip():
        return 0.0
    q = urllib.parse.quote(f"{artist} {title}")
    data = _get_json(f"{_DEEZER}/search?q={q}&limit=1")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return 0.0
    gevonden = items[0]
    eigen, hun = _plat(title), _plat(gevonden.get("title") or "")
    if not (eigen == hun or (len(min(eigen, hun, key=len)) >= 4 and (eigen in hun or hun in eigen))):
        log.debug("deezer: titel wijkt af (%s vs %s)", hun, eigen)
        return 0.0
    track = _get_json(f"{_DEEZER}/track/{gevonden.get('id')}")
    time.sleep(_RATE_SLEEP)
    try:
        bpm = float((track or {}).get("bpm") or 0)
    except (TypeError, ValueError):
        return 0.0
    # Deezer geeft af en toe onzin; buiten dit bereik is het geen muziektempo.
    return bpm if 40 <= bpm <= 220 else 0.0


def enrich_library(library_id: int) -> None:
    """Fetch artist photos + album covers for anything not yet enriched.
    Idempotent: only touches rows with enriched=False."""
    db = SessionLocal()
    try:
        artists = list(db.execute(select(Artist).where(
            Artist.library_id == library_id, Artist.enriched == False)).scalars())  # noqa: E712
        albums = list(db.execute(select(Album).where(
            Album.library_id == library_id, Album.enriched == False)).scalars())  # noqa: E712
    finally:
        db.close()

    if artists or albums:
        log.info("enriching library %d: %d artists, %d albums",
                 library_id, len(artists), len(albums))

    # ── artists: online photo (local files rarely have one) ──
    for ar in artists:
        img = _deezer_artist_image(ar.name)
        online_path = None
        if img:
            dest = config.ARTWORK_DIR / f"artist_online_{ar.id}.jpg"
            if _download(img, dest):
                online_path = str(dest)
        db = SessionLocal()
        try:
            row = db.get(Artist, ar.id)
            if row:
                if online_path:
                    row.online_art_path = online_path
                row.enriched = True
                db.commit()
        finally:
            db.close()

    # ── albums: only fetch when there is no local cover ──
    for al in albums:
        online_path = None
        if not al.art_path:
            img = _deezer_album_cover(al.artist_name, al.title)
            if img:
                dest = config.ARTWORK_DIR / f"album_online_{al.id}.jpg"
                if _download(img, dest):
                    online_path = str(dest)
        db = SessionLocal()
        try:
            row = db.get(Album, al.id)
            if row:
                if online_path:
                    row.online_art_path = online_path
                row.enriched = True
                db.commit()
        finally:
            db.close()

    if artists or albums:
        log.info("enrichment done for library %d", library_id)
