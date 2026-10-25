"""Een Spotify-lijst lezen — via de officiële Web API, met de sleutel van wie
deze server draait.

Twee routes:

1. **De API met een gebruikerskoppeling.** Nodig voor PLAYLISTS:
   /playlists/{id}/tracks pagineert netjes door met offset, dus een lijst van
   elke lengte. Vereist een eenmalige koppeling (zie `koppel_url` /
   `wissel_code`); daarna vernieuwt de server zijn eigen toegangsbewijs met het
   refresh-token.

2. **De API met alleen de app-sleutel** (client-credentials). Genoeg voor
   ALBUMS en losse NUMMERS. Playlist-tracks geven met alleen deze sleutel 403.

De sleutel maak je zelf aan op developer.spotify.com. Hij staat in de
meta-tabel (Beheer → Spotify) of in SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET.
Zonder sleutel leest deze server niets van Spotify.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config
from .models import Meta

_UA = "JLTamp/1.0"
# playlist, album of track — de drie soorten links die mensen plakken.
SPOTIFY_LINK = re.compile(r"\b(playlist|album|track)[/:]([A-Za-z0-9]{10,})")

# Waar de sleutels staan. Los van de schakelaars, want dit zijn geen aan/uit-
# knoppen maar geheimen; ze horen niet in een antwoord van het beheerscherm.
META_CLIENT_ID = "spotify_client_id"
META_CLIENT_SECRET = "spotify_client_secret"
META_REFRESH = "spotify_refresh_token"
META_REDIRECT = "spotify_redirect_uri"

# Zet dit op het adres van je eigen server (Beheer → Spotify), en registreer
# precies hetzelfde adres als Redirect URI bij je Spotify-app.
STANDAARD_REDIRECT = "https://your-domain.example/spotify/callback"


class SpotifyFout(Exception):
    """Wat er misging, in gewoon Nederlands — de router maakt er een HTTP-fout van."""


# ── instellingen ─────────────────────────────────────────────────────────────
def meta_lees(db, sleutel: str, standaard: str = "") -> str:
    rij = db.get(Meta, sleutel)
    return (rij.value if rij and rij.value else standaard)


def meta_zet(db, sleutel: str, waarde: str) -> None:
    rij = db.get(Meta, sleutel)
    if not rij:
        rij = Meta(key=sleutel)
        db.add(rij)
    rij.value = waarde


def app_sleutel(db) -> tuple[str, str]:
    """Uit Beheer, en anders uit de omgeving van de container."""
    return (meta_lees(db, META_CLIENT_ID) or config.SPOTIFY_CLIENT_ID,
            meta_lees(db, META_CLIENT_SECRET) or config.SPOTIFY_CLIENT_SECRET)


def redirect_uri(db) -> str:
    """Uit Beheer; anders PUBLIC_URL + /spotify/callback; anders het voorbeeld."""
    from .mailer import PUBLIC_URL
    standaard = (PUBLIC_URL.rstrip("/") + "/spotify/callback") if PUBLIC_URL else STANDAARD_REDIRECT
    return meta_lees(db, META_REDIRECT, standaard)


# ── toegangsbewijzen ─────────────────────────────────────────────────────────
#
# Een toegangsbewijs is een uur geldig. Het wordt hier vastgehouden zodat niet
# elke import opnieuw om er een vraagt; een minuut marge zodat hij nooit precies
# op de grens verloopt.
_bewaard: dict[str, tuple[str, float]] = {}


def _vraag_token(cid: str, cs: str, data: dict) -> dict:
    auth = base64.b64encode(f"{cid}:{cs}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=urllib.parse.urlencode(data).encode(),
        headers={"Authorization": "Basic " + auth,
                 "Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": _UA},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=20).read())
    except urllib.error.HTTPError as e:
        raise SpotifyFout("Spotify weigert de sleutel (%s)" % e.code)
    except Exception:
        raise SpotifyFout("Spotify is even niet bereikbaar")


def app_token(db) -> str | None:
    """Het bewijs dat alleen bij de app hoort. Genoeg voor albums."""
    cid, cs = app_sleutel(db)
    if not cid or not cs:
        return None
    bewaard = _bewaard.get("app")
    if bewaard and bewaard[1] > time.time():
        return bewaard[0]
    d = _vraag_token(cid, cs, {"grant_type": "client_credentials"})
    tok = d.get("access_token")
    if tok:
        _bewaard["app"] = (tok, time.time() + int(d.get("expires_in", 3600)) - 60)
    return tok


def gebruiker_token(db) -> str | None:
    """Het bewijs dat aan de koppeling hangt. Nodig voor playlists."""
    cid, cs = app_sleutel(db)
    refresh = meta_lees(db, META_REFRESH)
    if not cid or not cs or not refresh:
        return None
    bewaard = _bewaard.get("gebruiker")
    if bewaard and bewaard[1] > time.time():
        return bewaard[0]
    d = _vraag_token(cid, cs, {"grant_type": "refresh_token", "refresh_token": refresh})
    tok = d.get("access_token")
    if tok:
        _bewaard["gebruiker"] = (tok, time.time() + int(d.get("expires_in", 3600)) - 60)
    # Spotify mag een nieuw refresh-token meesturen; doet hij dat, dan is het
    # oude vanaf nu waardeloos en moet het nieuwe bewaard worden.
    if d.get("refresh_token") and d["refresh_token"] != refresh:
        meta_zet(db, META_REFRESH, d["refresh_token"])
        db.commit()
    return tok


def _api(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token,
                                               "User-Agent": _UA})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=25).read())
    except urllib.error.HTTPError as e:
        raise SpotifyFout("Spotify antwoordt met %s" % e.code)
    except Exception:
        raise SpotifyFout("Spotify is even niet bereikbaar")


# ── de routes zelf ───────────────────────────────────────────────────────────
def _spaties(tekst: str) -> str:
    """Spotify zet een harde spatie (U+00A0) tussen samenwerkende artiesten.

    Je ziet het verschil niet, maar het matchen tegen de bibliotheek struikelde
    er wél over, bij ongeveer één op de vier.
    """
    return re.sub(r"\s+", " ", (tekst or "").replace(" ", " ")).strip()


def _uit_api_items(items: list[dict], vanaf: int) -> list[dict]:
    """Wat de API teruggeeft → [{title, artist, position}].

    `track` mag None zijn: een nummer dat uit Spotify is gehaald of een lokaal
    bestand van de eigenaar. Die plek blijft leeg in plaats van dat alles wat
    erna komt een plaatsje opschuift — de volgorde van de lijst is het punt.
    """
    uit = []
    for n, rij in enumerate(items):
        t = rij.get("track") if "track" in rij else rij
        if not t or not t.get("name"):
            continue
        uit.append({
            "title": _spaties(t["name"]),
            "artist": _spaties(", ".join(a.get("name", "") for a in t.get("artists") or [])),
            "position": vanaf + n,
        })
    return uit


def _playlist_via_api(db, pid: str) -> tuple[str, list[dict]] | None:
    """De hele lijst, hoe lang hij ook is. None als er geen koppeling is."""
    token = gebruiker_token(db)
    if not token:
        return None
    kop = _api(f"https://api.spotify.com/v1/playlists/{pid}"
               f"?fields=name,tracks(total)", token)
    naam = (kop.get("name") or "Spotify-playlist").strip()
    items: list[dict] = []
    offset = 0
    while True:
        blok = _api(
            f"https://api.spotify.com/v1/playlists/{pid}/tracks"
            f"?limit=100&offset={offset}"
            f"&fields=total,items(track(name,artists(name)))", token)
        rijen = blok.get("items") or []
        if not rijen:
            break
        items.extend(_uit_api_items(rijen, offset))
        offset += len(rijen)
        if offset >= int(blok.get("total") or 0):
            break
        # Een lijst van tienduizend is geen import maar een ongeluk.
        if offset >= 5000:
            break
    return naam, items


def _album_via_api(db, pid: str) -> tuple[str, list[dict]] | None:
    """Albums lukken met de app-sleutel alleen — daar zegt Spotify geen nee."""
    token = app_token(db)
    if not token:
        return None
    kop = _api(f"https://api.spotify.com/v1/albums/{pid}", token)
    naam = (kop.get("name") or "Spotify-album").strip()
    artiest = ", ".join(a.get("name", "") for a in kop.get("artists") or [])
    items: list[dict] = []
    offset = 0
    while True:
        blok = _api(f"https://api.spotify.com/v1/albums/{pid}/tracks"
                    f"?limit=50&offset={offset}", token)
        rijen = blok.get("items") or []
        if not rijen:
            break
        for it in _uit_api_items(rijen, offset):
            # Op een album staat de artiest per nummer soms alleen bij de gast;
            # de albumartiest is dan wat je zoekt.
            it["artist"] = it["artist"] or _spaties(artiest)
            items.append(it)
        offset += len(rijen)
        if offset >= int(blok.get("total") or 0):
            break
    return naam, items


def _track_via_api(db, pid: str) -> tuple[str, list[dict]] | None:
    """Eén nummer is ook een lijst, van één. De app-sleutel is genoeg."""
    token = app_token(db)
    if not token:
        return None
    t = _api(f"https://api.spotify.com/v1/tracks/{pid}", token)
    naam = _spaties(t.get("name") or "")
    artiest = _spaties(", ".join(a.get("name", "") for a in t.get("artists") or []))
    return naam, ([{"title": naam, "artist": artiest, "position": 0}] if naam else [])


GEEN_SLEUTEL = ("Er staat nog geen Spotify-sleutel op deze server "
                "(Beheer → Spotify, of SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET)")


def lees(db, url: str) -> dict:
    """Een Spotify-link → {titel, items, bron, afgekapt}.

    `afgekapt` is alleen waar boven de 5000 nummers (zie _playlist_via_api).
    """
    m = SPOTIFY_LINK.search(url or "")
    if not m:
        raise SpotifyFout("Dat lijkt geen Spotify-link")
    soort, pid = m.group(1), m.group(2)

    if soort == "track":
        uit = _track_via_api(db, pid)
        if not uit:
            raise SpotifyFout(GEEN_SLEUTEL)
        return {"titel": uit[0], "items": uit[1], "bron": "api", "afgekapt": False}

    if soort == "album":
        uit = _album_via_api(db, pid)
        if not uit:
            raise SpotifyFout(GEEN_SLEUTEL)
        return {"titel": uit[0], "items": uit[1], "bron": "api", "afgekapt": False}

    uit = _playlist_via_api(db, pid)
    if not uit:
        cid, cs = app_sleutel(db)
        if not (cid and cs):
            raise SpotifyFout(GEEN_SLEUTEL)
        raise SpotifyFout("Een playlist lezen kan pas na een koppeling met een "
                          "Spotify-account (Beheer → Spotify → Koppelen)")
    return {"titel": uit[0], "items": uit[1], "bron": "api",
            "afgekapt": len(uit[1]) >= 5000}


# ── de koppeling ─────────────────────────────────────────────────────────────
#
# Eenmalig: de beheerder klikt op de link die `koppel_url` teruggeeft, logt bij
# Spotify in en komt terug op /spotify/callback. Vanaf dat moment kan de server
# zelf lijsten van elke lengte lezen. Het refresh-token staat in de meta-tabel
# en gaat dus gewoon mee in de backup van /data.
KOPPEL_RECHTEN = "playlist-read-private playlist-read-collaborative"


def koppel_url(db, staat: str) -> str:
    cid, _ = app_sleutel(db)
    if not cid:
        raise SpotifyFout("Er staat nog geen Spotify-sleutel in de instellingen")
    vraag = urllib.parse.urlencode({
        "client_id": cid, "response_type": "code",
        "redirect_uri": redirect_uri(db), "scope": KOPPEL_RECHTEN, "state": staat,
    })
    return "https://accounts.spotify.com/authorize?" + vraag


def wissel_code(db, code: str) -> None:
    cid, cs = app_sleutel(db)
    if not cid or not cs:
        raise SpotifyFout("Er staat nog geen Spotify-sleutel in de instellingen")
    d = _vraag_token(cid, cs, {"grant_type": "authorization_code", "code": code,
                               "redirect_uri": redirect_uri(db)})
    if not d.get("refresh_token"):
        raise SpotifyFout("Spotify gaf geen refresh-token terug")
    meta_zet(db, META_REFRESH, d["refresh_token"])
    db.commit()
    _bewaard.pop("gebruiker", None)
