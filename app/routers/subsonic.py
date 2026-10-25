"""De Subsonic-API: bestaande Subsonic-apps praten met deze server.

Subsonic is de API die Navidrome, Airsonic en Gonic ook spreken, en waar een
hele rij apps voor bestaat (Symfonium, Substreamer, play:Sub, Feishin, DSub,
Tempo, Sonixd). Wie zo'n app gebruikt, kan daarmee nu ook een JLTamp-server
kiezen, zonder van app te wisselen. Alles onder `/rest/`, zoals de spec het
wil, met de OpenSubsonic-uitbreidingen die apps het meest gebruiken.

**Inloggen.** Een Subsonic-app stuurt bij ELK verzoek `u` (gebruiker) mee, plus
een van:

- `t` + `s`: md5(wachtwoord + zout). Wat bijna elke app standaard doet. Dat kan
  alleen met het app-wachtwoord (SubsonicKey): het gewone wachtwoord staat
  alleen als hash in de database.
- `p`: het wachtwoord zelf, kaal of als `enc:<hex>`. Werkt met het
  app-wachtwoord én met het gewone wachtwoord.
- `apiKey` (OpenSubsonic): het app-wachtwoord, zonder `u`.

De gebruikersnaam is het e-mailadres, of de weergavenaam als die uniek is.

**Id's** zijn dezelfde als in de rest van de API: `ar12`, `al34`, `t56`,
`pl7`. Daardoor is een hoes-id gewoon het id van wat hij hoort, en kunnen
artiest, album en nummer nooit met elkaar verward worden.

**Toegang** gaat langs precies dezelfde regels als de app: alleen de
bibliotheken die iemand mag zien, alleen zijn eigen likes en lijsten (plus
gedeelde en openbare lijsten), en een beheerder ziet alles.

Aan/uit met de schakelaar `subsonic` in Beheer (standaard aan).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import and_, func, or_, select

from .. import config
from ..db import SessionLocal
from ..deps import accessible_library_ids, require_user
from ..ids import parse_key
from ..models import (
    Album, Artist, Library, Meta, Playlist, PlaylistItem, PlaylistMember,
    SubsonicKey, SubsonicStar, Track, User, UserTrackState,
)
from ..security import verify_password

router = APIRouter(tags=["subsonic"])
log = logging.getLogger("jltamp.subsonic")

API_VERSION = "1.16.1"
XMLNS = "http://subsonic.org/restapi"
# Wat er voor een artiestnaam weg mag bij het sorteren in de index.
IGNORED_ARTICLES = "The El La Los Las Le Les De Het Die"
MAX_LIJST = 500


# ── fouten ───────────────────────────────────────────────────────────────────
class SubsonicFout(Exception):
    """Een fout volgens de spec. Komt als HTTP 200 terug met status "failed":
    zo verwachten de apps het, een HTTP-fout snappen veel van hen niet."""

    def __init__(self, code: int, bericht: str):
        super().__init__(bericht)
        self.code = code
        self.bericht = bericht


def _ontbreekt(naam: str) -> SubsonicFout:
    return SubsonicFout(10, f"Required parameter is missing: {naam}")


NIET_GEVONDEN = SubsonicFout(70, "The requested data was not found")


# ── parameters ───────────────────────────────────────────────────────────────
class Params:
    """Query én formulier (OpenSubsonic `formPost`), met herhaalde sleutels."""

    def __init__(self, paren: list[tuple[str, str]]):
        self._paren = paren

    def get(self, naam: str, standaard: str | None = None) -> str | None:
        for k, v in self._paren:
            if k == naam:
                return v
        return standaard

    def lijst(self, naam: str) -> list[str]:
        return [v for k, v in self._paren if k == naam and v != ""]

    def nodig(self, naam: str) -> str:
        v = self.get(naam)
        if v is None or v == "":
            raise _ontbreekt(naam)
        return v

    def getal(self, naam: str, standaard: int = 0, laag: int | None = None,
              hoog: int | None = None) -> int:
        try:
            n = int(float(self.get(naam) or standaard))
        except ValueError:
            n = standaard
        if laag is not None:
            n = max(laag, n)
        if hoog is not None:
            n = min(hoog, n)
        return n

    def waar(self, naam: str, standaard: bool = False) -> bool:
        v = self.get(naam)
        if v is None:
            return standaard
        return v.lower() in ("1", "true", "yes")


# ── antwoorden ───────────────────────────────────────────────────────────────
def _omhulsel(status: str) -> dict:
    from ..main import VERSION
    return {"status": status, "version": API_VERSION, "type": "jltamp",
            "serverVersion": VERSION, "openSubsonic": True}


def _xml_waarde(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _naar_xml(el: ET.Element, data: dict) -> None:
    """dict → XML zoals Subsonic het doet: losse waarden worden attributen,
    een dict een kind, een lijst herhaalde kinderen, en `value` de tekst."""
    for k, v in data.items():
        if v is None:
            continue
        if k == "value" and not isinstance(v, (dict, list)):
            el.text = _xml_waarde(v)
        elif isinstance(v, dict):
            _naar_xml(ET.SubElement(el, k), v)
        elif isinstance(v, list):
            for item in v:
                kind = ET.SubElement(el, k)
                if isinstance(item, dict):
                    _naar_xml(kind, item)
                else:
                    kind.text = _xml_waarde(item)
        else:
            el.set(k, _xml_waarde(v))


def _antwoord(p: Params, inhoud: dict | None = None, status: str = "ok") -> Response:
    body = {**_omhulsel(status), **(inhoud or {})}
    f = (p.get("f") or "xml").lower()
    if f in ("json", "jsonp"):
        tekst = json.dumps({"subsonic-response": body}, ensure_ascii=False, separators=(",", ":"))
        if f == "jsonp":
            cb = p.get("callback") or "callback"
            if not re.fullmatch(r"[A-Za-z_$][\w$.]*", cb):
                cb = "callback"
            return Response(f"{cb}({tekst});", media_type="application/javascript")
        return Response(tekst, media_type="application/json")
    root = ET.Element("subsonic-response", {"xmlns": XMLNS})
    _naar_xml(root, body)
    return Response(b'<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(root, encoding="utf-8"),
                    media_type="application/xml")


def _fout(p: Params, fout: SubsonicFout) -> Response:
    return _antwoord(p, {"error": {"code": fout.code, "message": fout.bericht}}, status="failed")


def _iso(ts: int | float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── inloggen ─────────────────────────────────────────────────────────────────
# Een geslaagde inlog met het GEWONE wachtwoord, even onthouden. Een app stuurt
# de inlog bij elk verzoek mee, en pbkdf2 kost met opzet een tiende seconde:
# zonder dit kost een album openen met twintig hoezen twee seconden rekenen.
_ONTHOUDEN_S = 600
_onthouden: dict[str, tuple[int, float]] = {}
_onthouden_slot = threading.Lock()


def _gebruiker_op_naam(db, naam: str) -> User | None:
    naam = (naam or "").strip()
    if not naam:
        return None
    u = db.execute(select(User).where(func.lower(User.email) == naam.lower())).scalars().first()
    if u:
        return u
    # De weergavenaam alleen als hij uniek is: anders kan "Anna" de ene of de
    # andere Anna zijn, en dat mag nooit een gok worden.
    rijen = db.execute(select(User).where(func.lower(User.display_name) == naam.lower())).scalars().all()
    return rijen[0] if len(rijen) == 1 else None


def _ontsleutel(pw: str) -> str:
    if pw.startswith("enc:"):
        try:
            return bytes.fromhex(pw[4:]).decode("utf-8")
        except ValueError:
            return ""
    return pw


def _sleutel_van(db, user_id: int) -> SubsonicKey | None:
    return db.get(SubsonicKey, user_id)


def _markeer_gebruik(db, k: SubsonicKey) -> None:
    nu = int(time.time())
    if nu - (k.last_used_at or 0) > 300:
        try:
            k.last_used_at = nu
            db.commit()
        except Exception:
            db.rollback()


def _inloggen(db, p: Params) -> User:
    from .admin_users import schakelaar
    if not schakelaar(db, "subsonic"):
        raise SubsonicFout(50, "The Subsonic API is switched off on this server")

    api_key = p.get("apiKey")
    if api_key:
        if p.get("u"):
            raise SubsonicFout(43, "Multiple conflicting authentication mechanisms provided")
        k = db.execute(select(SubsonicKey).where(SubsonicKey.secret == api_key)).scalars().first()
        u = db.get(User, k.user_id) if k else None
        if not u or not u.is_active:
            raise SubsonicFout(44, "Invalid API key")
        _markeer_gebruik(db, k)
        return u

    naam = p.get("u")
    if not naam:
        raise _ontbreekt("u")
    u = _gebruiker_op_naam(db, naam)
    t, s, pw = p.get("t"), p.get("s"), p.get("p")
    if not (t and s) and not pw:
        raise _ontbreekt("p")
    if not u or not u.is_active:
        raise SubsonicFout(40, "Wrong username or password")
    k = _sleutel_van(db, u.id)

    if t and s:
        if k and hmac.compare_digest(hashlib.md5((k.secret + s).encode()).hexdigest(), t.lower()):
            _markeer_gebruik(db, k)
            return u
        raise SubsonicFout(40, "Wrong username or password. Token login needs the "
                               "Subsonic app password from your JLTamp settings")

    pw = _ontsleutel(pw or "")
    if k and hmac.compare_digest(k.secret, pw):
        _markeer_gebruik(db, k)
        return u
    sleutel = hashlib.sha256(f"{u.id}\0{pw}\0{u.password_hash}".encode()).hexdigest()
    with _onthouden_slot:
        bewaard = _onthouden.get(sleutel)
        if bewaard and bewaard[1] > time.time():
            return u
    if verify_password(pw, u.password_hash):
        with _onthouden_slot:
            for oud in [x for x, (_, tot) in _onthouden.items() if tot < time.time()]:
                _onthouden.pop(oud, None)
            _onthouden[sleutel] = (u.id, time.time() + _ONTHOUDEN_S)
        return u
    raise SubsonicFout(40, "Wrong username or password")


# ── toegang ──────────────────────────────────────────────────────────────────
class Ctx:
    """Alles wat een verzoek nodig heeft: de database, wie het is, wat hij mag zien."""

    def __init__(self, db, user: User, p: Params, request: Request):
        self.db = db
        self.user = user
        self.p = p
        self.request = request
        self.allowed = accessible_library_ids(db, user)   # None = alles (beheerder)
        folder = p.get("musicFolderId")
        if folder and folder.isdigit():
            fid = int(folder)
            self.allowed = [fid] if (self.allowed is None or fid in self.allowed) else []

    def scope(self, stmt, model):
        if self.allowed is not None:
            stmt = stmt.where(model.library_id.in_(self.allowed or [-1]))
        return stmt

    def mag(self, library_id: int) -> bool:
        return self.allowed is None or library_id in self.allowed


def _id(ruw: str | None, soort: str) -> int:
    """`al12` → 12 als het een album is; anders: niet gevonden."""
    pk = parse_key(ruw or "")
    if not pk or pk[0] != soort:
        raise NIET_GEVONDEN
    return pk[1]


def _track(c: Ctx, ruw: str | None) -> Track:
    t = c.db.get(Track, _id(ruw, "track"))
    if not t or not c.mag(t.library_id):
        raise NIET_GEVONDEN
    return t


def _album(c: Ctx, ruw: str | None) -> Album:
    a = c.db.get(Album, _id(ruw, "album"))
    if not a or not c.mag(a.library_id):
        raise NIET_GEVONDEN
    return a


def _artiest(c: Ctx, ruw: str | None) -> Artist:
    a = c.db.get(Artist, _id(ruw, "artist"))
    if not a or not c.mag(a.library_id):
        raise NIET_GEVONDEN
    return a


# ── likes en sterren ─────────────────────────────────────────────────────────
def _liked(c: Ctx) -> dict[int, int]:
    """{track_id: wanneer geliked} uit Liked Songs — één keer per verzoek."""
    if not hasattr(c, "_liked_cache"):
        from .likes import liked_playlist
        pl = liked_playlist(c.db, c.user.id)
        c._liked_cache = {} if not pl else {
            tid: (at or pl.created_at or 0) for tid, at in c.db.execute(
                select(PlaylistItem.track_id, PlaylistItem.added_at)
                .where(PlaylistItem.playlist_id == pl.id)).all()
        }
    return c._liked_cache


def _sterren(c: Ctx, soort: str) -> dict[int, int]:
    return {i: at for i, at in c.db.execute(
        select(SubsonicStar.item_id, SubsonicStar.created_at)
        .where(SubsonicStar.user_id == c.user.id, SubsonicStar.kind == soort)).all()}


# ── vertalen naar Subsonic ───────────────────────────────────────────────────
_SOORTEN = {
    "mp3": "audio/mpeg", "flac": "audio/flac", "aac": "audio/aac", "m4a": "audio/mp4",
    "mp4": "audio/mp4", "alac": "audio/mp4", "ogg": "audio/ogg", "oga": "audio/ogg",
    "opus": "audio/opus", "wav": "audio/wav", "aiff": "audio/aiff", "aif": "audio/aiff",
    "wma": "audio/x-ms-wma", "dsf": "audio/dsd", "wv": "audio/x-wavpack", "ape": "audio/ape",
}


def _relatief(pad: str) -> str:
    """Het pad zoals de app het mag zien: binnen de muziekmap, nooit het
    volledige pad op de server."""
    try:
        return str(Path(pad).relative_to(config.MUSIC_DIR))
    except ValueError:
        return Path(pad).name


def song(c: Ctx, t: Track, staat: UserTrackState | None = None) -> dict:
    ext = (t.ext or "").lstrip(".").lower()
    artiest = t.orig_artist or t.artist_name
    d = {
        "id": f"t{t.id}", "parent": f"al{t.album_id}", "isDir": False,
        "title": t.title, "album": t.album_title, "artist": artiest,
        "track": t.track_no, "discNumber": t.disc_no, "year": t.year,
        "genre": t.genre or None,
        "coverArt": f"t{t.id}" if t.art_path else f"al{t.album_id}",
        "size": t.size, "contentType": _SOORTEN.get(ext, "application/octet-stream"),
        "suffix": ext, "duration": int((t.duration_ms or 0) / 1000),
        "bitRate": t.bitrate or None, "path": _relatief(t.path),
        "albumId": f"al{t.album_id}", "artistId": f"ar{t.artist_id}",
        "type": "music", "mediaType": "song", "isVideo": False,
        "created": _iso(t.added_at),
        "samplingRate": t.samplerate or None, "channelCount": t.channels or None,
        "bitDepth": t.bits or None,
        "displayArtist": artiest,
        "displayAlbumArtist": t.album_artist or t.artist_name,
        "genres": [{"name": t.genre}] if t.genre else [],
    }
    if t.gain_db is not None:
        d["replayGain"] = {"trackGain": round(t.gain_db, 2)}
    if staat:
        d["playCount"] = staat.play_count or 0
        if staat.last_played_at:
            d["played"] = _iso(staat.last_played_at)
        if staat.rating:
            d["userRating"] = int(round(staat.rating / 2))
    geliked = _liked(c).get(t.id)
    if geliked is not None:
        d["starred"] = _iso(geliked or time.time())
    return {k: v for k, v in d.items() if v is not None}


def songs(c: Ctx, tracks: list[Track]) -> list[dict]:
    from ..serializers import track_states
    staten = track_states(c.db, c.user.id, tracks)
    return [song(c, t, staten.get(t.id)) for t in tracks]


def album(c: Ctx, a: Album, sterren: dict[int, int] | None = None) -> dict:
    d = {
        "id": f"al{a.id}", "parent": f"ar{a.artist_id}", "isDir": True,
        "name": a.title, "title": a.title, "album": a.title,
        "artist": a.artist_name, "artistId": f"ar{a.artist_id}",
        "coverArt": f"al{a.id}", "songCount": a.track_count or 0,
        "duration": int((a.duration_ms or 0) / 1000),
        "created": _iso(a.added_at) or _iso(time.time()),
        "year": a.year, "genre": a.genre or None,
        "genres": [{"name": a.genre}] if a.genre else [],
        "displayArtist": a.artist_name, "mediaType": "album",
    }
    ster = (sterren if sterren is not None else _sterren(c, "album")).get(a.id)
    if ster is not None:
        d["starred"] = _iso(ster)
    return {k: v for k, v in d.items() if v is not None}


def artiest(c: Ctx, a: Artist, aantal: int | None = None, sterren: dict[int, int] | None = None) -> dict:
    d = {"id": f"ar{a.id}", "name": a.name, "coverArt": f"ar{a.id}",
         "albumCount": aantal if aantal is not None else 0,
         "sortName": a.sort_name or None, "mediaType": "artist"}
    ster = (sterren if sterren is not None else _sterren(c, "artist")).get(a.id)
    if ster is not None:
        d["starred"] = _iso(ster)
    return {k: v for k, v in d.items() if v is not None}


def _album_aantallen(c: Ctx, artiest_ids: list[int] | None = None) -> dict[int, int]:
    q = c.scope(select(Album.artist_id, func.count(Album.id)), Album)
    if artiest_ids is not None:
        q = q.where(Album.artist_id.in_(artiest_ids or [-1]))
    return dict(c.db.execute(q.group_by(Album.artist_id)).all())


def _playlist_dict(c: Ctx, pl: Playlist, met_nummers: bool = False) -> dict:
    eigenaar = c.db.get(User, pl.user_id)
    rijen = c.db.execute(
        select(Track).join(PlaylistItem, PlaylistItem.track_id == Track.id)
        .where(PlaylistItem.playlist_id == pl.id)
        .order_by(PlaylistItem.position, PlaylistItem.id)).scalars().all()
    zichtbaar = [t for t in rijen if c.mag(t.library_id)]
    d = {
        "id": f"pl{pl.id}", "name": pl.title, "comment": "",
        "owner": (eigenaar.display_name or eigenaar.email) if eigenaar else "",
        "public": bool(pl.is_public or pl.share_code),
        "songCount": len(zichtbaar),
        "duration": int(sum((t.duration_ms or 0) for t in zichtbaar) / 1000),
        "created": _iso(pl.created_at) or _iso(time.time()),
        "changed": _iso(pl.updated_at or pl.created_at) or _iso(time.time()),
        "coverArt": f"pl{pl.id}",
        "readonly": pl.user_id != c.user.id,
    }
    if met_nummers:
        d["entry"] = songs(c, zichtbaar)
    return d


def _zichtbare_lijsten(c: Ctx) -> list[Playlist]:
    leden = select(PlaylistMember.playlist_id).where(PlaylistMember.user_id == c.user.id)
    return c.db.execute(
        select(Playlist).where(or_(Playlist.user_id == c.user.id,
                                   Playlist.is_public.is_(True),
                                   Playlist.id.in_(leden)))
        .order_by(Playlist.title)).scalars().all()


def _lijst(c: Ctx, ruw: str | None, schrijven: bool = False) -> Playlist:
    pl = c.db.get(Playlist, _id(ruw, "playlist"))
    if not pl:
        raise NIET_GEVONDEN
    eigen = pl.user_id == c.user.id
    lid = bool(c.db.execute(select(PlaylistMember.id).where(
        PlaylistMember.playlist_id == pl.id, PlaylistMember.user_id == c.user.id)).first())
    if not (eigen or lid or pl.is_public):
        raise NIET_GEVONDEN
    if schrijven and not (eigen or lid):
        raise SubsonicFout(50, "This playlist is kept by the server")
    return pl


# ── de methodes ──────────────────────────────────────────────────────────────
METHODES: dict[str, callable] = {}


def methode(*namen):
    def deco(fn):
        for n in namen:
            METHODES[n.lower()] = fn
        return fn
    return deco


@methode("ping")
def _ping(c: Ctx):
    return {}


@methode("getLicense")
def _licentie(c: Ctx):
    return {"license": {"valid": True, "email": "", "licenseExpires": "2999-12-31T00:00:00Z"}}


@methode("getOpenSubsonicExtensions")
def _uitbreidingen(c: Ctx):
    return {"openSubsonicExtensions": [
        {"name": "formPost", "versions": [1]},
        {"name": "songLyrics", "versions": [1]},
        {"name": "apiKeyAuthentication", "versions": [1]},
    ]}


@methode("getMusicFolders")
def _mappen(c: Ctx):
    q = select(Library).order_by(Library.name)
    if c.allowed is not None:
        q = q.where(Library.id.in_(c.allowed or [-1]))
    return {"musicFolders": {"musicFolder": [
        {"id": lib.id, "name": lib.name} for lib in c.db.execute(q).scalars()]}}


def _gebruiker_dict(c: Ctx, u: User) -> dict:
    mappen = accessible_library_ids(c.db, u)
    if mappen is None:
        mappen = [i for (i,) in c.db.execute(select(Library.id)).all()]
    return {
        "username": u.email, "email": u.email, "scrobblingEnabled": True,
        "adminRole": bool(u.is_admin), "settingsRole": True, "downloadRole": True,
        "uploadRole": False, "playlistRole": True, "coverArtRole": True,
        "commentRole": False, "podcastRole": False, "streamRole": True,
        "jukeboxRole": False, "shareRole": False, "videoConversionRole": False,
        "folder": mappen,
    }


@methode("getUser")
def _gebruiker(c: Ctx):
    naam = c.p.get("username") or c.user.email
    u = _gebruiker_op_naam(c.db, naam)
    if not u:
        raise NIET_GEVONDEN
    if u.id != c.user.id and not c.user.is_admin:
        raise SubsonicFout(50, "User is not authorized for the given operation")
    return {"user": _gebruiker_dict(c, u)}


@methode("getUsers")
def _gebruikers(c: Ctx):
    if not c.user.is_admin:
        raise SubsonicFout(50, "User is not authorized for the given operation")
    return {"users": {"user": [_gebruiker_dict(c, u)
                               for u in c.db.execute(select(User).order_by(User.email)).scalars()]}}


def _letter(a: Artist) -> str:
    naam = (a.sort_name or a.name or "").strip()
    for lidwoord in IGNORED_ARTICLES.split():
        if naam.lower().startswith(lidwoord.lower() + " "):
            naam = naam[len(lidwoord) + 1:]
            break
    eerste = naam[:1].upper()
    return eerste if eerste.isalpha() else "#"


def _index(c: Ctx) -> list[dict]:
    aantallen = _album_aantallen(c)
    sterren = _sterren(c, "artist")
    groepen: dict[str, list[dict]] = {}
    rijen = c.db.execute(c.scope(select(Artist), Artist).order_by(func.lower(Artist.sort_name),
                                                                   func.lower(Artist.name))).scalars()
    for a in rijen:
        n = aantallen.get(a.id, 0)
        if not n:
            continue      # een artiest zonder eigen album staat alleen als gast op nummers
        groepen.setdefault(_letter(a), []).append(artiest(c, a, n, sterren))
    return [{"name": k, "artist": groepen[k]} for k in sorted(groepen, key=lambda x: (x == "#", x))]


@methode("getArtists")
def _artiesten(c: Ctx):
    return {"artists": {"ignoredArticles": IGNORED_ARTICLES, "index": _index(c)}}


@methode("getIndexes")
def _indexen(c: Ctx):
    laatst = c.db.execute(c.scope(select(func.max(Track.added_at)), Track)).scalar() or 0
    return {"indexes": {"ignoredArticles": IGNORED_ARTICLES,
                        "lastModified": int(laatst) * 1000, "index": _index(c)}}


def _albums_van(c: Ctx, artist_id: int) -> list[Album]:
    return c.db.execute(c.scope(select(Album), Album).where(Album.artist_id == artist_id)
                        .order_by(Album.year.is_(None), Album.year, func.lower(Album.sort_title))).scalars().all()


def _nummers_van(c: Ctx, album_id: int) -> list[Track]:
    return c.db.execute(c.scope(select(Track), Track).where(Track.album_id == album_id)
                        .order_by(func.coalesce(Track.disc_no, 1), func.coalesce(Track.track_no, 0),
                                  func.lower(Track.sort_title))).scalars().all()


@methode("getArtist")
def _artiest_m(c: Ctx):
    a = _artiest(c, c.p.nodig("id"))
    albums = _albums_van(c, a.id)
    sterren = _sterren(c, "album")
    d = artiest(c, a, len(albums))
    d["album"] = [album(c, al, sterren) for al in albums]
    return {"artist": d}


@methode("getAlbum")
def _album_m(c: Ctx):
    a = _album(c, c.p.nodig("id"))
    d = album(c, a)
    d["song"] = songs(c, _nummers_van(c, a.id))
    d["songCount"] = len(d["song"])
    return {"album": d}


@methode("getSong")
def _song_m(c: Ctx):
    return {"song": songs(c, [_track(c, c.p.nodig("id"))])[0]}


@methode("getMusicDirectory")
def _map(c: Ctx):
    ruw = c.p.nodig("id")
    pk = parse_key(ruw)
    if pk and pk[0] == "artist":
        a = _artiest(c, ruw)
        sterren = _sterren(c, "album")
        kinderen = [album(c, al, sterren) for al in _albums_van(c, a.id)]
        return {"directory": {"id": f"ar{a.id}", "name": a.name, "child": kinderen}}
    if pk and pk[0] == "album":
        a = _album(c, ruw)
        return {"directory": {"id": f"al{a.id}", "parent": f"ar{a.artist_id}", "name": a.title,
                              "child": songs(c, _nummers_van(c, a.id))}}
    if ruw.isdigit():
        # Een muziekmap (bibliotheek) als bovenste map: de artiesten erin.
        fid = int(ruw)
        if not c.mag(fid):
            raise NIET_GEVONDEN
        lib = c.db.get(Library, fid)
        aantallen = _album_aantallen(c)
        rijen = c.db.execute(select(Artist).where(Artist.library_id == fid)
                             .order_by(func.lower(Artist.sort_name))).scalars()
        return {"directory": {"id": ruw, "name": lib.name if lib else ruw, "child": [
            {"id": f"ar{a.id}", "parent": ruw, "isDir": True, "title": a.name,
             "name": a.name, "coverArt": f"ar{a.id}"}
            for a in rijen if aantallen.get(a.id)]}}
    raise NIET_GEVONDEN


@methode("getGenres")
def _genres(c: Ctx):
    rijen = c.db.execute(
        c.scope(select(Track.genre, func.count(Track.id), func.count(func.distinct(Track.album_id))), Track)
        .where(Track.genre != "").group_by(func.lower(Track.genre))
        .order_by(func.lower(Track.genre))).all()
    return {"genres": {"genre": [{"value": g, "songCount": n, "albumCount": na} for g, n, na in rijen]}}


def _album_query(c: Ctx, soort: str):
    q = c.scope(select(Album), Album)
    if soort == "random":
        return q.order_by(func.random())
    if soort == "newest":
        return q.order_by(Album.added_at.desc(), Album.id.desc())
    if soort == "alphabeticalByName":
        return q.order_by(func.lower(Album.sort_title), func.lower(Album.title))
    if soort == "alphabeticalByArtist":
        return q.order_by(func.lower(Album.artist_name), Album.year, func.lower(Album.sort_title))
    if soort in ("frequent", "recent", "highest"):
        maat = {"frequent": func.sum(UserTrackState.play_count),
                "recent": func.max(UserTrackState.last_played_at),
                "highest": func.max(UserTrackState.rating)}[soort]
        veld = {"frequent": UserTrackState.play_count, "recent": UserTrackState.last_played_at,
                "highest": UserTrackState.rating}[soort]
        sub = (select(Track.album_id.label("aid"), maat.label("m"))
               .join(UserTrackState, and_(UserTrackState.track_id == Track.id,
                                          UserTrackState.user_id == c.user.id))
               .where(veld > 0).group_by(Track.album_id).subquery())
        return q.join(sub, sub.c.aid == Album.id).order_by(sub.c.m.desc())
    if soort == "starred":
        ids = list(_sterren(c, "album"))
        return q.where(Album.id.in_(ids or [-1])).order_by(func.lower(Album.sort_title))
    if soort == "byYear":
        van = c.p.getal("fromYear", 0)
        tot = c.p.getal("toYear", 9999)
        laag, hoog = min(van, tot), max(van, tot)
        q = q.where(Album.year >= laag, Album.year <= hoog)
        return q.order_by(Album.year.desc() if van > tot else Album.year, func.lower(Album.sort_title))
    if soort == "byGenre":
        g = c.p.nodig("genre")
        return q.where(func.lower(Album.genre) == g.lower()).order_by(func.lower(Album.sort_title))
    raise SubsonicFout(0, f"Unknown list type: {soort}")


def _album_lijst(c: Ctx) -> list[dict]:
    soort = c.p.nodig("type")
    grootte = c.p.getal("size", 10, 1, MAX_LIJST)
    vanaf = c.p.getal("offset", 0, 0)
    albums = c.db.execute(_album_query(c, soort).offset(vanaf).limit(grootte)).scalars().all()
    sterren = _sterren(c, "album")
    return [album(c, a, sterren) for a in albums]


@methode("getAlbumList2")
def _albumlijst2(c: Ctx):
    return {"albumList2": {"album": _album_lijst(c)}}


@methode("getAlbumList")
def _albumlijst(c: Ctx):
    return {"albumList": {"album": _album_lijst(c)}}


@methode("getRandomSongs")
def _willekeurig(c: Ctx):
    q = c.scope(select(Track), Track)
    if c.p.get("genre"):
        q = q.where(func.lower(Track.genre) == c.p.get("genre").lower())
    if c.p.get("fromYear"):
        q = q.where(Track.year >= c.p.getal("fromYear"))
    if c.p.get("toYear"):
        q = q.where(Track.year <= c.p.getal("toYear"))
    rijen = c.db.execute(q.order_by(func.random()).limit(c.p.getal("size", 10, 1, MAX_LIJST))).scalars().all()
    return {"randomSongs": {"song": songs(c, rijen)}}


@methode("getSongsByGenre")
def _per_genre(c: Ctx):
    g = c.p.nodig("genre")
    rijen = c.db.execute(
        c.scope(select(Track), Track).where(func.lower(Track.genre) == g.lower())
        .order_by(func.lower(Track.artist_name), Track.album_id, Track.disc_no, Track.track_no)
        .offset(c.p.getal("offset", 0, 0)).limit(c.p.getal("count", 10, 1, MAX_LIJST))).scalars().all()
    return {"songsByGenre": {"song": songs(c, rijen)}}


def _gelijkaardig(c: Ctx) -> list[Track]:
    """Een simpele radio: nummers uit hetzelfde genre, het eigen nummer en de
    eigen artiest gemengd. De AI-dienst kan het beter, maar die is optioneel —
    dit werkt altijd."""
    ruw = c.p.nodig("id")
    aantal = c.p.getal("count", 50, 1, MAX_LIJST)
    pk = parse_key(ruw)
    if pk and pk[0] == "artist":
        a = _artiest(c, ruw)
        genres = [g for (g,) in c.db.execute(select(Track.genre).where(
            Track.artist_id == a.id, Track.genre != "").group_by(Track.genre).limit(5)).all()]
        uitsluiten = None
    elif pk and pk[0] == "album":
        al = _album(c, ruw)
        genres = [al.genre] if al.genre else []
        uitsluiten = None
    else:
        t = _track(c, ruw)
        genres = [t.genre] if t.genre else []
        uitsluiten = t.id
    q = c.scope(select(Track), Track)
    if genres:
        q = q.where(func.lower(Track.genre).in_([g.lower() for g in genres]))
    if uitsluiten:
        q = q.where(Track.id != uitsluiten)
    return c.db.execute(q.order_by(func.random()).limit(aantal)).scalars().all()


@methode("getSimilarSongs2")
def _gelijk2(c: Ctx):
    return {"similarSongs2": {"song": songs(c, _gelijkaardig(c))}}


@methode("getSimilarSongs")
def _gelijk(c: Ctx):
    return {"similarSongs": {"song": songs(c, _gelijkaardig(c))}}


@methode("getTopSongs")
def _top(c: Ctx):
    naam = c.p.nodig("artist")
    rijen = c.db.execute(
        c.scope(select(Track), Track)
        .where(or_(func.lower(Track.artist_name) == naam.lower(), func.lower(Track.orig_artist) == naam.lower()))
        .order_by(Track.play_count.desc(), Track.year, Track.track_no)
        .limit(c.p.getal("count", 50, 1, MAX_LIJST))).scalars().all()
    return {"topSongs": {"song": songs(c, rijen)}}


def _artiest_info(c: Ctx) -> dict:
    a = _artiest(c, c.p.nodig("id"))
    return {"biography": a.summary or "", "similarArtist": []}


@methode("getArtistInfo2")
def _info2(c: Ctx):
    return {"artistInfo2": _artiest_info(c)}


@methode("getArtistInfo")
def _info(c: Ctx):
    return {"artistInfo": _artiest_info(c)}


@methode("getAlbumInfo2", "getAlbumInfo")
def _albuminfo(c: Ctx):
    _album(c, c.p.nodig("id"))
    return {"albumInfo": {"notes": ""}}


# ── zoeken ───────────────────────────────────────────────────────────────────
def _woorden(q: str) -> list[str]:
    q = (q or "").strip().strip('"').strip()
    return [w for w in re.split(r"\s+", q) if w]


def _zoek(c: Ctx, q: str):
    """Zoeken zoals de apps het gebruiken.

    Een LEGE zoekvraag hoort alles terug te geven: Symfonium en een paar andere
    halen zo de hele bibliotheek binnen, blok voor blok, om offline te kunnen
    bladeren. Wie dat afvangt als "niets gevonden", ziet in die apps een lege
    bibliotheek."""
    woorden = _woorden(q)

    def filter_(stmt, velden):
        for w in woorden:
            patroon = f"%{w.lower()}%"
            stmt = stmt.where(or_(*[func.lower(v).like(patroon) for v in velden]))
        return stmt

    p = c.p
    artiesten = c.db.execute(
        filter_(c.scope(select(Artist), Artist), [Artist.name])
        .order_by(func.lower(Artist.sort_name), Artist.id)
        .offset(p.getal("artistOffset", 0, 0)).limit(p.getal("artistCount", 20, 0, MAX_LIJST))).scalars().all()
    albums = c.db.execute(
        filter_(c.scope(select(Album), Album), [Album.title, Album.artist_name])
        .order_by(func.lower(Album.sort_title), Album.id)
        .offset(p.getal("albumOffset", 0, 0)).limit(p.getal("albumCount", 20, 0, MAX_LIJST))).scalars().all()
    nummers = c.db.execute(
        filter_(c.scope(select(Track), Track), [Track.title, Track.artist_name, Track.orig_artist, Track.album_title])
        .order_by(*( [Track.id] if not woorden else [func.lower(Track.sort_title), Track.id]))
        .offset(p.getal("songOffset", 0, 0)).limit(p.getal("songCount", 20, 0, MAX_LIJST))).scalars().all()
    aantallen = _album_aantallen(c, [a.id for a in artiesten])
    sa, sal = _sterren(c, "artist"), _sterren(c, "album")
    return ([artiest(c, a, aantallen.get(a.id, 0), sa) for a in artiesten],
            [album(c, a, sal) for a in albums], songs(c, nummers))


@methode("search3")
def _zoek3(c: Ctx):
    ar, al, so = _zoek(c, c.p.get("query") or "")
    return {"searchResult3": {"artist": ar, "album": al, "song": so}}


@methode("search2")
def _zoek2(c: Ctx):
    ar, al, so = _zoek(c, c.p.get("query") or "")
    return {"searchResult2": {"artist": ar, "album": al, "song": so}}


# ── sterren, cijfers, gespeeld ───────────────────────────────────────────────
def _gesterd(c: Ctx) -> tuple[list, list, list]:
    sa, sal = _sterren(c, "artist"), _sterren(c, "album")
    artiesten = c.db.execute(c.scope(select(Artist), Artist).where(Artist.id.in_(list(sa) or [-1]))).scalars().all()
    albums = c.db.execute(c.scope(select(Album), Album).where(Album.id.in_(list(sal) or [-1]))).scalars().all()
    liked = _liked(c)
    nummers = c.db.execute(c.scope(select(Track), Track).where(Track.id.in_(list(liked) or [-1]))).scalars().all()
    nummers.sort(key=lambda t: -(liked.get(t.id) or 0))
    aantallen = _album_aantallen(c, [a.id for a in artiesten])
    return ([artiest(c, a, aantallen.get(a.id, 0), sa) for a in artiesten],
            [album(c, a, sal) for a in albums], songs(c, nummers))


@methode("getStarred2")
def _gesterd2(c: Ctx):
    ar, al, so = _gesterd(c)
    return {"starred2": {"artist": ar, "album": al, "song": so}}


@methode("getStarred")
def _gesterd1(c: Ctx):
    ar, al, so = _gesterd(c)
    return {"starred": {"artist": ar, "album": al, "song": so}}


def _sterren_zetten(c: Ctx, aan: bool):
    from . import likes
    for ruw in c.p.lijst("id"):
        pk = parse_key(ruw)
        if pk and pk[0] == "track":
            _track(c, ruw)
            (likes.like if aan else likes.unlike)(f"t{pk[1]}", c.user)
        elif pk and pk[0] in ("album", "artist"):
            (_album if pk[0] == "album" else _artiest)(c, ruw)
            _ster(c, pk[0], pk[1], aan)
    for ruw in c.p.lijst("albumId"):
        _album(c, ruw)
        _ster(c, "album", _id(ruw, "album"), aan)
    for ruw in c.p.lijst("artistId"):
        _artiest(c, ruw)
        _ster(c, "artist", _id(ruw, "artist"), aan)
    c.db.commit()
    return {}


def _ster(c: Ctx, soort: str, item_id: int, aan: bool) -> None:
    rij = c.db.execute(select(SubsonicStar).where(
        SubsonicStar.user_id == c.user.id, SubsonicStar.kind == soort,
        SubsonicStar.item_id == item_id)).scalars().first()
    if aan and not rij:
        c.db.add(SubsonicStar(user_id=c.user.id, kind=soort, item_id=item_id, created_at=int(time.time())))
    elif rij and not aan:
        c.db.delete(rij)


@methode("star")
def _star(c: Ctx):
    return _sterren_zetten(c, True)


@methode("unstar")
def _unstar(c: Ctx):
    return _sterren_zetten(c, False)


@methode("setRating")
def _cijfer(c: Ctx):
    from .history import _state
    t = _track(c, c.p.nodig("id"))
    st = _state(c.db, c.user.id, t.id)
    st.rating = float(c.p.getal("rating", 0, 0, 5) * 2)   # Subsonic 0-5, JLTamp 0-10
    c.db.commit()
    return {}


@methode("scrobble")
def _scrobble(c: Ctx):
    """`submission=false` is "speelt nu": dan staat iemand als luisteraar in
    het overzicht. Anders telt het als gespeeld, met een regel in de
    geschiedenis — precies wat de app doet als een nummer de helft voorbij is."""
    from .history import _state, _touch_session
    from ..models import PlayEvent
    ids = c.p.lijst("id")
    if not ids:
        raise _ontbreekt("id")
    tijden = c.p.lijst("time")
    indienen = c.p.waar("submission", True)
    toestel = (c.p.get("c") or "Subsonic")[:60]
    for n, ruw in enumerate(ids):
        t = _track(c, ruw)
        if not indienen:
            _touch_session(c.user, f"subsonic:{toestel}", t.id, "playing", 0, toestel)
            continue
        try:
            wanneer = int(int(tijden[n]) / 1000) if n < len(tijden) else int(time.time())
        except ValueError:
            wanneer = int(time.time())
        st = _state(c.db, c.user.id, t.id)
        st.play_count = (st.play_count or 0) + 1
        st.last_played_at = max(st.last_played_at or 0, wanneer)
        st.view_offset_ms = 0
        c.db.add(PlayEvent(
            user_id=c.user.id, track_id=t.id,
            artist_name=(t.orig_artist or t.artist_name), album_title=t.album_title,
            track_title=t.title, started_at=wanneer - int((t.duration_ms or 0) / 1000),
            ended_at=wanneer, listened_ms=t.duration_ms or 0, duration_ms=t.duration_ms or 0,
            completed=True, device=toestel, session_id=f"subsonic:{toestel}"))
    c.db.commit()
    return {}


@methode("getNowPlaying")
def _nu(c: Ctx):
    from .history import _live_sessions
    mensen = {u.id: u for u in c.db.execute(select(User)).scalars()}
    uit = []
    for s in _live_sessions():
        u = mensen.get(s["user_id"])
        if not u:
            continue
        eigen = u.id == c.user.id
        # Wat iemand draait is van hem (zie history.presence): alleen wie dat
        # deelt, en jijzelf.
        if not eigen and (u.share_activity or "presence") != "track":
            continue
        t = c.db.get(Track, s["track_id"])
        if not t or not c.mag(t.library_id):
            continue
        d = songs(c, [t])[0]
        d.update({"username": u.display_name or u.email,
                  "minutesAgo": int((time.time() - s["updated_at"]) / 60),
                  "playerId": 0, "playerName": s.get("device") or ""})
        uit.append(d)
    return {"nowPlaying": {"entry": uit}}


# ── afspeellijsten ───────────────────────────────────────────────────────────
@methode("getPlaylists")
def _lijsten(c: Ctx):
    return {"playlists": {"playlist": [_playlist_dict(c, pl) for pl in _zichtbare_lijsten(c)]}}


@methode("getPlaylist")
def _lijst_m(c: Ctx):
    return {"playlist": _playlist_dict(c, _lijst(c, c.p.nodig("id")), met_nummers=True)}


def _toevoegen(c: Ctx, pl: Playlist, ruwe_ids: list[str]) -> None:
    pos = (c.db.execute(select(func.max(PlaylistItem.position))
                        .where(PlaylistItem.playlist_id == pl.id)).scalar())
    pos = -1 if pos is None else pos
    for ruw in ruwe_ids:
        t = _track(c, ruw)
        pos += 1
        c.db.add(PlaylistItem(playlist_id=pl.id, track_id=t.id, position=pos,
                              added_by=c.user.id, added_at=int(time.time())))


@methode("createPlaylist")
def _maak_lijst(c: Ctx):
    nu = int(time.time())
    if c.p.get("playlistId"):
        # Bestaande lijst: de inhoud wordt vervangen door wat er meekomt.
        pl = _lijst(c, c.p.get("playlistId"), schrijven=True)
        c.db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pl.id).delete()
        if c.p.get("name") and pl.user_id == c.user.id:
            pl.title = c.p.get("name")
    else:
        pl = Playlist(user_id=c.user.id, title=c.p.nodig("name"), created_at=nu, updated_at=nu)
        c.db.add(pl)
        c.db.flush()
    _toevoegen(c, pl, c.p.lijst("songId"))
    pl.updated_at = nu
    c.db.commit()
    return {"playlist": _playlist_dict(c, pl, met_nummers=True)}


@methode("updatePlaylist")
def _wijzig_lijst(c: Ctx):
    pl = _lijst(c, c.p.nodig("playlistId"), schrijven=True)
    eigen = pl.user_id == c.user.id
    if c.p.get("name") and eigen:
        pl.title = c.p.get("name")
    weg = sorted({int(i) for i in c.p.lijst("songIndexToRemove") if i.isdigit()}, reverse=True)
    if weg:
        items = c.db.execute(select(PlaylistItem).where(PlaylistItem.playlist_id == pl.id)
                             .order_by(PlaylistItem.position, PlaylistItem.id)).scalars().all()
        for i in weg:
            if i < len(items):
                c.db.delete(items[i])
    _toevoegen(c, pl, c.p.lijst("songIdToAdd"))
    pl.updated_at = int(time.time())
    c.db.commit()
    from .playlists import ROOMS
    if pl.share_code:
        ROOMS.notify(pl.id, {"t": "changed", "by": c.user.id})
    return {}


@methode("deletePlaylist")
def _weg_lijst(c: Ctx):
    from .playlists import delete_playlist
    pl = _lijst(c, c.p.nodig("id"))
    if pl.user_id != c.user.id and not c.db.execute(select(PlaylistMember.id).where(
            PlaylistMember.playlist_id == pl.id, PlaylistMember.user_id == c.user.id)).first():
        raise SubsonicFout(50, "Only the owner can delete this playlist")
    delete_playlist(f"pl{pl.id}", c.user)     # eigenaar: weg; lid: verlaat de lijst
    return {}


# ── wachtrij en bladwijzers ──────────────────────────────────────────────────
def _rij_sleutel(c: Ctx) -> str:
    return f"subsonic_queue:{c.user.id}"


@methode("getPlayQueue")
def _rij(c: Ctx):
    rij = c.db.get(Meta, _rij_sleutel(c))
    if not rij or not rij.value:
        return {}
    try:
        stand = json.loads(rij.value)
    except ValueError:
        return {}
    tracks = c.db.execute(select(Track).where(Track.id.in_(stand.get("ids") or [-1]))).scalars().all()
    per_id = {t.id: t for t in tracks if c.mag(t.library_id)}
    volgorde = [per_id[i] for i in stand.get("ids", []) if i in per_id]
    d = {"entry": songs(c, volgorde), "username": c.user.email,
         "changed": _iso(stand.get("changed")), "changedBy": stand.get("by") or ""}
    if stand.get("current") in per_id:
        d["current"] = f"t{stand['current']}"
        d["position"] = int(stand.get("position") or 0)
    return {"playQueue": d}


@methode("savePlayQueue")
def _bewaar_rij(c: Ctx):
    ids = []
    for ruw in c.p.lijst("id"):
        pk = parse_key(ruw)
        if pk and pk[0] == "track":
            ids.append(pk[1])
    huidig = parse_key(c.p.get("current") or "")
    stand = {"ids": ids[:1000], "current": huidig[1] if huidig and huidig[0] == "track" else None,
             "position": c.p.getal("position", 0, 0), "changed": int(time.time()),
             "by": (c.p.get("c") or "")[:60]}
    rij = c.db.get(Meta, _rij_sleutel(c))
    if not rij:
        rij = Meta(key=_rij_sleutel(c))
        c.db.add(rij)
    rij.value = json.dumps(stand)
    c.db.commit()
    return {}


@methode("getBookmarks")
def _bladwijzers(c: Ctx):
    return {"bookmarks": {"bookmark": []}}


@methode("createBookmark", "deleteBookmark")
def _bladwijzer(c: Ctx):
    return {}


# ── wat JLTamp niet heeft: leeg, niet kapot ──────────────────────────────────
@methode("getPodcasts")
def _podcasts(c: Ctx):
    return {"podcasts": {"channel": []}}


@methode("getNewestPodcasts")
def _nieuwste_podcasts(c: Ctx):
    return {"newestPodcasts": {"episode": []}}


@methode("getInternetRadioStations")
def _radio(c: Ctx):
    return {"internetRadioStations": {"internetRadioStation": []}}


@methode("getShares")
def _delen(c: Ctx):
    return {"shares": {"share": []}}


@methode("getChatMessages")
def _chat(c: Ctx):
    return {"chatMessages": {"chatMessage": []}}


# ── scannen ──────────────────────────────────────────────────────────────────
def _scanstand(c: Ctx) -> dict:
    from ..scanner import scan_state
    s = scan_state()
    totaal = c.db.execute(c.scope(select(func.count(Track.id)), Track)).scalar() or 0
    return {"scanStatus": {"scanning": bool(s.get("scanning")), "count": int(s.get("tracks") or totaal)}}


@methode("getScanStatus")
def _scan(c: Ctx):
    return _scanstand(c)


@methode("startScan")
def _start_scan(c: Ctx):
    if not c.user.is_admin:
        raise SubsonicFout(50, "User is not authorized for the given operation")
    from ..scanner import scan_all, scan_state
    if not scan_state().get("scanning"):
        threading.Thread(target=scan_all, daemon=True).start()
    return _scanstand(c)


# ── songteksten ─────────────────────────────────────────────────────────────
def _tekst_van(c: Ctx, t: Track) -> dict | None:
    from .lyrics import get_lyrics
    try:
        return get_lyrics(f"t{t.id}", c.user)
    except HTTPException:
        return None


@methode("getLyrics")
def _tekst(c: Ctx):
    artiest_ = (c.p.get("artist") or "").strip()
    titel = (c.p.get("title") or "").strip()
    if not titel:
        return {"lyrics": {}}
    q = c.scope(select(Track), Track).where(func.lower(Track.title) == titel.lower())
    if artiest_:
        q = q.where(or_(func.lower(Track.orig_artist) == artiest_.lower(),
                        func.lower(Track.artist_name) == artiest_.lower()))
    t = c.db.execute(q.limit(1)).scalars().first()
    tekst = _tekst_van(c, t) if t else None
    if not tekst or not (tekst.get("plain") or "").strip():
        return {"lyrics": {}}
    return {"lyrics": {"artist": t.orig_artist or t.artist_name, "title": t.title,
                       "value": tekst["plain"]}}


@methode("getLyricsBySongId")
def _tekst_per_nummer(c: Ctx):
    t = _track(c, c.p.nodig("id"))
    tekst = _tekst_van(c, t) or {}
    regels = tekst.get("synced") or []
    platte = (tekst.get("plain") or "").strip()
    if regels:
        lijn = [{"start": int(r.get("timeMs") or 0), "value": r.get("line") or ""} for r in regels]
        gesynct = True
    elif platte:
        lijn = [{"value": r} for r in platte.splitlines()]
        gesynct = False
    else:
        return {"lyricsList": {"structuredLyrics": []}}
    return {"lyricsList": {"structuredLyrics": [{
        "displayArtist": t.orig_artist or t.artist_name, "displayTitle": t.title,
        "lang": "xxx", "synced": gesynct, "line": lijn}]}}


# ── binaire antwoorden: muziek, hoezen, avatars ──────────────────────────────
BINAIR = {"stream", "download", "getcoverart", "getavatar"}


def _stream(c: Ctx, downloaden: bool) -> Response:
    from .stream import omgezet
    t = _track(c, c.p.nodig("id"))
    ext = (t.ext or "").lstrip(".").lower()
    if not downloaden:
        formaat = (c.p.get("format") or "").lower()
        max_kbps = c.p.getal("maxBitRate", 0, 0)
        # Omzetten alleen als de app erom vraagt: een ander formaat, of een
        # lagere bitrate dan het bestand heeft. Anders gaat het bestand zelf,
        # met Range — dan kan de app ook gewoon doorspoelen.
        wil_mp3 = formaat == "mp3" and ext != "mp3"
        te_zwaar = max_kbps and (t.bitrate or 0) > max_kbps
        if formaat != "raw" and (wil_mp3 or te_zwaar):
            return omgezet(t, max_kbps or 320)
    return FileResponse(t.path, media_type=_SOORTEN.get(ext, "application/octet-stream"),
                        filename=Path(t.path).name if downloaden else None)


def _hoes(c: Ctx) -> Response:
    from .artwork import photo_transcode, resolve_art_path
    ruw = c.p.nodig("id")
    if ruw.startswith("al-") or ruw.startswith("ar-"):   # andere servers gebruiken zo'n vorm
        ruw = ruw.replace("-", "", 1)
    pad = resolve_art_path(ruw, c.user)
    if not pad or not os.path.exists(pad):
        raise HTTPException(404, "No cover art")
    grootte = c.p.getal("size", 0, 0, 2000)
    if grootte:
        return photo_transcode(url=f"/art/{ruw}", width=grootte, height=grootte, user=c.user)
    return FileResponse(pad)


def _avatar(c: Ctx) -> Response:
    naam = c.p.get("username") or c.user.email
    u = _gebruiker_op_naam(c.db, naam)
    if not u or not u.thumb_path or not os.path.exists(u.thumb_path):
        raise HTTPException(404, "No avatar")
    return FileResponse(u.thumb_path, media_type="image/jpeg")


# ── de ingang ────────────────────────────────────────────────────────────────
async def _params(request: Request) -> Params:
    paren = list(request.query_params.multi_items())
    if request.method == "POST":
        try:
            formulier = await request.form()
            paren += [(k, v) for k, v in formulier.multi_items() if isinstance(v, str)]
        except Exception:
            pass
    return Params(paren)


@router.api_route("/rest/{naam}", methods=["GET", "POST"])
async def rest(naam: str, request: Request):
    p = await _params(request)
    sleutel = naam[:-5] if naam.endswith(".view") else naam
    sleutel = sleutel.lower()
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_verwerk, sleutel, p, request)


def _verwerk(sleutel: str, p: Params, request: Request) -> Response:
    db = SessionLocal()
    try:
        try:
            user = _inloggen(db, p)
        except SubsonicFout as e:
            return _fout(p, e)
        c = Ctx(db, user, p, request)
        try:
            if sleutel in BINAIR:
                if sleutel == "stream":
                    return _stream(c, False)
                if sleutel == "download":
                    return _stream(c, True)
                if sleutel == "getcoverart":
                    return _hoes(c)
                return _avatar(c)
            fn = METHODES.get(sleutel)
            if not fn:
                return _fout(p, SubsonicFout(0, f"Unknown method: {sleutel}"))
            return _antwoord(p, fn(c))
        except SubsonicFout as e:
            db.rollback()
            if sleutel in BINAIR and e.code == 70:
                raise HTTPException(404, e.bericht)
            return _fout(p, e)
        except HTTPException:
            raise
        except Exception:
            db.rollback()
            log.exception("subsonic %s mislukt", sleutel)
            return _fout(p, SubsonicFout(0, "Something went wrong on the server"))
    finally:
        db.close()


# ── het app-wachtwoord beheren (vanuit de JLTamp-app) ────────────────────────
def _sleutel_antwoord(request: Request, u: User, k: SubsonicKey | None) -> dict:
    from ..mailer import PUBLIC_URL
    basis = PUBLIC_URL or str(request.base_url).rstrip("/")
    return {"server": basis, "username": u.email, "password": k.secret if k else None,
            "createdAt": k.created_at if k else 0, "lastUsedAt": k.last_used_at if k else 0}


@router.get("/users/me/subsonic")
def mijn_subsonic(request: Request, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        from .admin_users import schakelaar
        uit = _sleutel_antwoord(request, user, _sleutel_van(db, user.id))
        uit["enabled"] = schakelaar(db, "subsonic")
        return uit
    finally:
        db.close()


@router.post("/users/me/subsonic")
def nieuw_subsonic(request: Request, user: User = Depends(require_user)):
    """Een (nieuw) app-wachtwoord. Een oud wachtwoord werkt daarna niet meer."""
    db = SessionLocal()
    try:
        k = _sleutel_van(db, user.id)
        if not k:
            k = SubsonicKey(user_id=user.id)
            db.add(k)
        # Zonder tekens die je in een app makkelijk verkeerd overtypt.
        abc = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        k.secret = "".join(secrets.choice(abc) for _ in range(20))
        k.created_at = int(time.time())
        k.last_used_at = 0
        db.commit()
        return _sleutel_antwoord(request, user, k)
    finally:
        db.close()


@router.delete("/users/me/subsonic")
def weg_subsonic(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        k = _sleutel_van(db, user.id)
        if k:
            db.delete(k)
            db.commit()
        return {"ok": True}
    finally:
        db.close()
