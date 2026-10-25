"""Playlist importer — turn a Spotify playlist, album or track link into a
JLTamp playlist by matching each track (title + artist) against the user's OWN
library.

**No media is fetched, decrypted or downloaded.** All this reads is a track
LISTING — names and artists — through Spotify's official Web API, with
credentials the operator registers themselves (see `app/spotify.py` and
/admin/spotify/koppel). Without those, this import path says so and stops.

Wat er niet in de bibliotheek staat wordt onthouden mét zijn plek in de lijst
(`open_playlist_items`). Komt het nummer later alsnog in de bibliotheek, dan
schuift het na de volgende scan vanzelf op die plek de lijst in.

Het matchen zelf staat in `matching.py`, samen met de Top 40 — dat is dezelfde
vraag, en twee kopieën van die regels lopen uit elkaar.
"""
from __future__ import annotations

import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select

from .. import spotify
from ..deps import require_admin, require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import playlist_key
from ..matching import Bibliotheek, match_lijst
from ..models import OpenPlaylistItem, Playlist, PlaylistItem, Track, User

router = APIRouter()


# ── matchen ──────────────────────────────────────────────────────────────────
def _match(db, user: User, items: list[dict]) -> tuple[list[int], list[dict]]:
    """[{title, artist}] → (track-ids in volgorde, wat er niet was).

    De regels staan in matching.py: accenten, "feat"/"x"/"&", editie-achtervoegsels
    en de afspraak dat een remix nooit doorgaat voor het origineel.
    """
    allowed = accessible_library_ids(db, user)
    q = db.query(Track)
    if allowed is not None:
        q = q.filter(Track.library_id.in_(allowed or [-1]))
    gevonden, missers = match_lijst(q.all(), items)
    return [t.id for t in gevonden], missers


def _match_met_plek(db, user: User, items: list[dict]) -> tuple[list[tuple[int, int]], list[dict]]:
    """Hetzelfde, maar dan met de PLEK die elk nummer in de lijst had.

    Nodig omdat een geïmporteerde lijst gaten kan hebben: wat er niet was komt
    later alsnog binnen en moet dan op zijn eigen plek terug. Daarvoor moeten de
    nummers die er wél zijn óók op hun oorspronkelijke plek staan, en niet
    netjes achter elkaar geschoven — anders past er verderop niets meer tussen.

    Geeft [(plek, track-id)] en de missers, elk mét hun plek.
    """
    allowed = accessible_library_ids(db, user)
    q = db.query(Track)
    if allowed is not None:
        q = q.filter(Track.library_id.in_(allowed or [-1]))
    tracks = q.all()

    # ⚠️ De index ÉÉN keer bouwen, niet per nummer.
    #
    # `match_lijst` bouwt hem zelf, dus hem per regel aanroepen om de plek mee
    # te krijgen zou voor een lijst van vijftig nummers vijftig keer honderdduizend
    # titels normaliseren. Daarom hier `Bibliotheek` rechtstreeks — precies wat
    # `match_lijst` binnenin ook doet, inclusief de regel dat een nummer maar
    # één keer in de uitkomst komt.
    bib = Bibliotheek(tracks)
    uit: list[tuple[int, int]] = []
    missers: list[dict] = []
    gezien: set[int] = set()
    for volgnummer, it in enumerate(items):
        # De bron mag de plek zelf aanwijzen. Dat is nodig zodra er iets
        # tussenuit valt: een nummer dat Spotify niet vrijgeeft laat een gat
        # achter, en alles wat erna komt hoort gewoon te blijven staan waar het
        # stond. Zonder dit schoof de rest van de lijst een plaats op.
        plek = int(it.get("position", volgnummer))
        titel = (it.get("title") or "").strip()
        if not titel:
            continue
        artiest = (it.get("artist") or "").strip()
        t = bib.zoek(artiest, titel)
        if t is None:
            missers.append({"title": titel, "artist": artiest, "position": plek})
            continue
        if t.id in gezien:
            continue
        gezien.add(t.id)
        uit.append((plek, t.id))
    return uit, missers


# ── Spotify ──────────────────────────────────────────────────────────────────
def _fetch_spotify(url: str) -> tuple[str, list[dict]]:
    """Blijft bestaan voor de viraal-lijst in charts.py, die alleen naam en
    nummers wil. Wie ook wil weten of er iets is afgekapt: gebruik
    `spotify.lees` rechtstreeks."""
    db = SessionLocal()
    try:
        uit = spotify.lees(db, url)
    except spotify.SpotifyFout as e:
        raise HTTPException(502, str(e))
    finally:
        db.close()
    if not uit["items"]:
        raise HTTPException(404, "Geen nummers gevonden in die lijst")
    return uit["titel"], uit["items"]


# ── endpoint ─────────────────────────────────────────────────────────────────
def _detect(url: str) -> str:
    u = (url or "").lower()
    if "spotify.com" in u or u.startswith("spotify:"):
        return "spotify"
    return ""


@router.post("/import/playlist")
def import_playlist(url: str = "", title: str = "", user: User = Depends(require_user)):
    """Fetch a Spotify playlist, match it to the library, and create a
    JLTamp playlist from the matches. Returns a report (matched / unmatched)."""
    # De schakelaar in het beheerscherm gaat boven de omgeving van de container:
    # zo is dit om te zetten zonder herstart. Staat er nooit aan gedraaid, dan
    # geldt JLTAMP_IMPORTERS als standaard.
    from .admin_users import schakelaar as _schakelaar
    from ..db import SessionLocal as _S
    _db = _S()
    try:
        _aan = _schakelaar(_db, "importeren")
    finally:
        _db.close()
    if not _aan:
        raise HTTPException(403, "The playlist importer is disabled on this server")
    # Een account voor een winkelbeoordeling importeert niets. De kaart is voor
    # zo'n account al weg (zie /charts/looks), maar wie het verzoek zelf stuurt
    # hoort ook nul op het rekest te krijgen.
    if bool(getattr(user, "is_review", False)):
        raise HTTPException(403, "The playlist importer is disabled on this server")
    url = (url or "").strip()
    source = _detect(url)
    if not source:
        raise HTTPException(400, "Paste a Spotify playlist link")

    _db2 = _S()
    try:
        uit = spotify.lees(_db2, url)
    except spotify.SpotifyFout as e:
        raise HTTPException(502, str(e))
    finally:
        _db2.close()
    name, items = uit["titel"], uit["items"]
    if not items:
        raise HTTPException(404, "No tracks found in that playlist")

    db = SessionLocal()
    try:
        met_plek, unmatched = _match_met_plek(db, user, items)
        now = int(time.time())
        # ⚠️ De lijst wordt ALTIJD gemaakt, ook als er nu nog niets van in de
        # bibliotheek staat.
        #
        # Komen de ontbrekende nummers later in de bibliotheek, dan loopt de
        # lijst vanzelf vol (zie `los_openstaande`). Zonder lijst was er niets
        # om ze in terug te zetten.
        pl = Playlist(user_id=user.id, title=(title.strip() or name),
                      created_at=now, updated_at=now)
        db.add(pl)
        db.flush()
        for plek, tid in met_plek:
            db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=plek,
                                added_by=user.id, added_at=int(time.time())))
        # En van wat er niet was: onthouden wáár het hoorde. Zie
        # OpenPlaylistItem en `los_openstaande` hieronder.
        for mis in unmatched:
            db.add(OpenPlaylistItem(
                playlist_id=pl.id, position=int(mis.get("position") or 0),
                artist=(mis.get("artist") or "")[:300],
                title=(mis.get("title") or "")[:300],
                user_id=user.id, created_at=now,
            ))
        db.commit()
        return {
            "source": source, "title": pl.title, "ratingKey": playlist_key(pl.id),
            "total": len(items), "matched": len(met_plek),
            "unmatched": unmatched[:100],
            # Hoeveel er nog openstaan, met hun plek bewaard.
            "pending": len(unmatched),
            # De API leest een lijst helemaal; afkappen gebeurt alleen boven de
            # 5000 (zie spotify._playlist_via_api).
            "truncated": bool(uit.get("afgekapt")),
            "limit": None,
        }
    finally:
        db.close()


# ── Wat er later alsnog binnenkomt ─────────────────────────────────────────
#
# Draait na elke scan (zie scanner.py). Kijkt of er inmiddels een treffer is
# voor wat er bij een import ontbrak, en schuift die op zijn oorspronkelijke
# plek de lijst in.
#
# Waarom na de scan: pas de scan zet nieuwe bestanden in de database. Eerder
# kijken heeft geen zin.

# Zo lang blijft een openstaand nummer staan. Daarna komt het er kennelijk niet
# bij, en is stil opruimen beter dan een lijst die een jaar later alsnog iets invoegt.
BEWAAR_DAGEN = 31


def los_openstaande() -> int:
    """Zet binnengekomen nummers alsnog in de lijst waar ze bij horen."""
    db = SessionLocal()
    try:
        open_items = list(db.execute(select(OpenPlaylistItem)).scalars())
        if not open_items:
            return 0

        nu = int(time.time())
        # Per gebruiker één keer de bibliotheek opbouwen: twee mensen zien niet
        # dezelfde bibliotheken, dus het kan niet één index voor iedereen zijn.
        per_gebruiker: dict[int, Bibliotheek] = {}
        gelukt = 0
        for item in open_items:
            if nu - int(item.created_at or 0) > BEWAAR_DAGEN * 86400:
                db.delete(item)
                continue
            pl = db.get(Playlist, item.playlist_id)
            if not pl:
                db.delete(item)          # lijst is weg, dan dit ook
                continue

            bib = per_gebruiker.get(item.user_id)
            if bib is None:
                gebruiker = db.get(User, item.user_id)
                allowed = accessible_library_ids(db, gebruiker) if gebruiker else []
                q = db.query(Track)
                if allowed is not None:
                    q = q.filter(Track.library_id.in_(allowed or [-1]))
                bib = Bibliotheek(q.all())
                per_gebruiker[item.user_id] = bib

            t = bib.zoek(item.artist, item.title)
            if t is None:
                continue
            # Staat hij er al (via een andere weg), dan alleen opruimen.
            al = db.execute(
                select(PlaylistItem).where(PlaylistItem.playlist_id == pl.id,
                                           PlaylistItem.track_id == t.id)
            ).scalars().first()
            if not al:
                db.add(PlaylistItem(playlist_id=pl.id, track_id=t.id,
                                    position=item.position, added_by=item.user_id, added_at=int(time.time())))
                pl.updated_at = nu
                gelukt += 1
            db.delete(item)
        db.commit()
        return gelukt
    finally:
        db.close()


# ── De koppeling met Spotify ─────────────────────────────────────────────────
#
# Eenmalig, door een beheerder. Albums en losse nummers lukken met alleen de
# sleutel van de app; een playlist leest Spotify pas voor een gekoppeld account,
# en dan de hele lijst, hoe lang hij ook is. De sleutel van de app (client id/secret)
# en het refresh-token staan in de meta-tabel — in /data dus, en daarmee in de
# backup, zodat een nieuwe container niet opnieuw gekoppeld hoeft te worden.
class SpotifySleutelBody(BaseModel):
    clientId: str
    clientSecret: str
    redirectUri: str | None = None


@router.get("/admin/spotify/status")
def spotify_status(_: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        cid, cs = spotify.app_sleutel(db)
        return {
            "sleutel": bool(cid and cs),
            "gekoppeld": bool(spotify.meta_lees(db, spotify.META_REFRESH)),
            "redirectUri": spotify.redirect_uri(db),
            # Zonder koppeling leest deze server geen playlists (geen grens, maar nul).
            "grensZonderKoppeling": 0,
        }
    finally:
        db.close()


@router.post("/admin/spotify/sleutel")
def spotify_sleutel(body: SpotifySleutelBody, _: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        spotify.meta_zet(db, spotify.META_CLIENT_ID, body.clientId.strip())
        spotify.meta_zet(db, spotify.META_CLIENT_SECRET, body.clientSecret.strip())
        if body.redirectUri:
            spotify.meta_zet(db, spotify.META_REDIRECT, body.redirectUri.strip())
        db.commit()
        return {"ok": True, "redirectUri": spotify.redirect_uri(db)}
    finally:
        db.close()


@router.get("/admin/spotify/koppel")
def spotify_koppel(_: User = Depends(require_admin)):
    """Geeft de link waar de beheerder op moet klikken. Spotify stuurt hem
    daarna terug naar /spotify/callback."""
    db = SessionLocal()
    try:
        # Een eigen kenmerk per poging, zodat de callback weet dat het antwoord
        # bij ónze vraag hoort en niet bij een link die iemand anders stuurde.
        staat = secrets.token_urlsafe(16)
        spotify.meta_zet(db, "spotify_staat", staat)
        db.commit()
        return {"url": spotify.koppel_url(db, staat)}
    except spotify.SpotifyFout as e:
        raise HTTPException(400, str(e))
    finally:
        db.close()


@router.get("/spotify/callback")
def spotify_callback(code: str = "", state: str = "", error: str = ""):
    """Waar Spotify de browser naartoe stuurt. Publiek bereikbaar — dat moet,
    Spotify stuurt geen inlog mee — maar zonder het kenmerk uit /koppel gebeurt
    er niets."""
    if error:
        return PlainTextResponse("Spotify zegt: %s" % error, status_code=400)
    db = SessionLocal()
    try:
        verwacht = spotify.meta_lees(db, "spotify_staat")
        if not verwacht or state != verwacht:
            return PlainTextResponse(
                "Deze koppelpoging hoort niet bij deze server.", status_code=400)
        spotify.meta_zet(db, "spotify_staat", "")
        spotify.wissel_code(db, code)
        return PlainTextResponse(
            "Gelukt. JLTamp kan nu Spotify-lijsten van elke lengte lezen.\n"
            "Dit tabblad mag dicht.")
    except spotify.SpotifyFout as e:
        return PlainTextResponse("Koppelen mislukt: %s" % e, status_code=502)
    finally:
        db.close()
