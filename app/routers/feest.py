"""Feestmodus — gasten vragen nummers aan zonder app of account.

De gastheer start in de app een feest. Hij krijgt een link (en de app maakt er
een QR van); wie die scant komt op een gewone webpagina die deze server zelf
serveert (`static/feest.html`). Daar zie je wat er speelt en wat er aankomt, kun
je zoeken in de bibliotheek, een nummer aanvragen en op aanvragen van anderen
stemmen. De aanvraag met de meeste stemmen gaat als volgende in de wachtrij van
de gastheer.

Een gast heeft GEEN account. Hij kiest een bijnaam en krijgt een willekeurig
geheimpje terug dat zijn browser bewaart; daarmee herkennen we zijn stemmen en
houden we zijn limieten bij. Dat is alles wat we van hem weten.

⚠️ Wat een gast nooit krijgt, en waarom dat hier zo hard staat:

- **Geen muziek.** Er is geen stream- of downloadroute achter de feestlink. Een
  gast ziet titel, artiest en hoes — dat is het. Afspelen doet alleen de
  telefoon van de gastheer.
- **Geen gegevens van de gastheer.** Geen e-mail, geen id, geen token. Op de
  pagina staat de naam van het feest en verder niets over wie het geeft.
- **Niets buiten de bibliotheken van de gastheer.** Zoeken en hoezen gaan langs
  dezelfde toegangscontrole als bij de gastheer zelf.

Een feest leeft in het geheugen, net als een sessie van samen luisteren (zie
session.py): na een herstart van de server is het voorbij en start de gastheer
een nieuw. Dat is met opzet — er komt niets van gasten in de database terecht.

Een wens (iets dat niet in de bibliotheek staat) komt op het verlanglijstje van
de gastheer (`noteer_misser`), net als een zoekopdracht die niets opleverde. Met
de schakelaar `wensen` en een eigen dak per gast.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key, track_key
from ..models import Track, User
from ..serializers import track_dict

router = APIRouter(tags=["feest"])

# Hoe lang een feestlink geldig is. Een avond, en dan nog wat; daarna moet de
# gastheer een nieuw feest starten. Een link die op een groepsapp rondgaat hoort
# niet over een week nog te werken.
FEEST_DUUR = 12 * 3600
# Hoeveel wensen (nummers die er niet zijn) één gast per feest mag doen.
MAX_WENSEN_PER_GAST = 3
# En hoeveel wensen een heel feest bij elkaar: het verlanglijstje van de
# gastheer is van hem, een feest mag het niet volschrijven.
MAX_WENSEN_PER_FEEST = 15
# Hoeveel aanvragen één gast tegelijk open mag hebben staan.
MAX_OPEN_PER_GAST = 5
# Hoeveel aanvragen een feest in totaal bijhoudt (alles, ook afgehandeld).
MAX_AANVRAGEN = 300
MAX_GASTEN = 300
BIJNAAM_MAX = 24
# Een nummer dat is ingepland maar na zoveel seconden niet meer in de wachtrij
# van de gastheer staat (en ook niet speelt), is voorbij of overgeslagen.
INPLAN_GRATIE = 20
# Hoe lang een afgelopen feest nog als "afgelopen" herkend wordt, voor het weg
# mag. Daarna is het gewoon een onbekende link.
OPRUIMEN_NA = 6 * 3600

_PAGINA = Path(__file__).resolve().parent.parent / "static" / "feest.html"

_SLOT = threading.RLock()


# ── Het feest zelf ──────────────────────────────────────────────────────────
@dataclass
class Gast:
    geheim: str
    bijnaam: str
    sinds: float
    wensen: int = 0


@dataclass
class Aanvraag:
    id: int
    track_id: int
    titel: str
    artiest: str
    door: str                   # geheim van de gast die hem aanvroeg
    bijnaam: str
    tijd: float
    stemmen: set[str] = field(default_factory=set)
    # open → ingepland → gespeeld; of geweigerd door de gastheer.
    status: str = "open"
    ingepland_op: float = 0.0


@dataclass
class Feest:
    token: str
    host_user_id: int
    naam: str
    gestart: float
    verloopt: float
    sessie: str | None = None
    gestopt: float = 0.0
    gasten: dict[str, Gast] = field(default_factory=dict)        # geheim → Gast
    aanvragen: dict[int, Aanvraag] = field(default_factory=dict)
    wensen: list[dict] = field(default_factory=list)
    # Wat de app van de gastheer laatst doorgaf.
    nu: dict | None = None
    wachtrij: list[dict] = field(default_factory=list)
    volgnummer: int = 0

    def voorbij(self) -> bool:
        return bool(self.gestopt) or time.time() >= self.verloopt


FEESTEN: dict[str, Feest] = {}


def _opruimen() -> None:
    nu = time.time()
    for k in [k for k, f in FEESTEN.items()
              if f.voorbij() and nu - max(f.gestopt or 0, min(f.verloopt, nu)) > OPRUIMEN_NA]:
        FEESTEN.pop(k, None)


def _mijn_feest(user_id: int) -> Feest | None:
    for f in FEESTEN.values():
        if f.host_user_id == user_id and not f.voorbij():
            return f
    return None


# ── Rustig aan ──────────────────────────────────────────────────────────────
#
# Een feestlink staat open op internet, dus alles wat een gast kan, kan een
# script ook — en dan duizend keer per seconde. Per handeling een venster met
# een maximum; daarboven een 429. Bewust eenvoudig: een rij tijdstempels per
# sleutel, in het geheugen, net als het feest zelf.
_TELLERS: dict[str, deque] = {}


def _beperk(sleutel: str, maximum: int, venster: float) -> None:
    nu = time.monotonic()
    with _SLOT:
        rij = _TELLERS.setdefault(sleutel, deque())
        while rij and nu - rij[0] > venster:
            rij.popleft()
        if len(rij) >= maximum:
            wacht = int(venster - (nu - rij[0])) + 1
            raise HTTPException(429, "Even rustig aan", headers={"Retry-After": str(wacht)})
        rij.append(nu)
        # Niet laten uitdijen: lege of oude rijen eruit.
        if len(_TELLERS) > 5000:
            for k in [k for k, r in _TELLERS.items() if not r or nu - r[-1] > 3600]:
                _TELLERS.pop(k, None)


# De grenzen per IP zijn ruim met opzet: op een feest zitten alle gasten op
# dezelfde wifi en komen ze hier dus van één adres binnen. De echte rem is die
# per gast; die per IP vangt alleen een script dat steeds nieuwe gasten maakt.
def _ip(request: Request) -> str:
    # Achter Caddy is de eerste in X-Forwarded-For de echte bezoeker.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


# ── Hulpjes ─────────────────────────────────────────────────────────────────
def feest_mag(db, user: User | None) -> bool:
    """Mag deze gebruiker een feest geven? Schakelaar aan, actief account en
    geen account voor een winkelbeoordeling — die ziet de feestmodus niet."""
    from .admin_users import schakelaar
    if not user or not getattr(user, "is_active", True):
        return False
    if getattr(user, "is_review", False):
        return False
    return schakelaar(db, "feest")


def _wensen_mogelijk(db, host: User) -> bool:
    from .admin_users import schakelaar
    if getattr(host, "is_review", False):
        return False
    return schakelaar(db, "wensen") and not schakelaar(db, "winkelstand")


def _publieke_basis(request: Request) -> str:
    """Het adres waar een gast de pagina kan openen.

    Zelfde regel als de uitnodigingsmail (mailer.send_invite): eerst PUBLIC_URL,
    want het LAN-adres waarmee de app van de gastheer toevallig verbonden is
    werkt op de telefoon van een gast niet. Pas als die niet gezet is het adres
    waarlangs dit verzoek binnenkwam.
    """
    from ..mailer import PUBLIC_URL
    if PUBLIC_URL:
        return PUBLIC_URL.rstrip("/")
    schema = request.headers.get("x-forwarded-proto") or request.url.scheme or "http"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    if host:
        return f"{schema}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _voornaam(user: User) -> str:
    # Alleen het eerste woord van de weergavenaam. Nooit het e-mailadres: de
    # pagina staat open voor iedereen met de link.
    naam = (getattr(user, "display_name", "") or "").strip().split(" ")[0]
    return naam[:BIJNAAM_MAX]


def _schoon(tekst, lengte: int) -> str:
    return " ".join(str(tekst or "").split())[:lengte]


def _hoes(token: str, rk: str | None) -> str | None:
    if not rk or not str(rk).startswith("t") or not str(rk)[1:].isdigit():
        return None
    return f"/feest/{token}/hoes/{rk}"


def _gesorteerd(f: Feest, statussen=("ingepland", "open")) -> list[Aanvraag]:
    rij = [a for a in f.aanvragen.values() if a.status in statussen]
    # Ingepland bovenaan (die komt echt als volgende), dan op stemmen, en bij
    # gelijke stand wie het eerst vroeg.
    rij.sort(key=lambda a: (0 if a.status == "ingepland" else 1, -len(a.stemmen), a.tijd))
    return rij


def _track_van(db, host: User, rk: str) -> Track | None:
    """Een nummer uit de bibliotheken van de gastheer, of niets."""
    p = parse_key(rk)
    if not p or p[0] != "track":
        return None
    t = db.get(Track, p[1])
    if not t:
        return None
    allowed = accessible_library_ids(db, host)
    if allowed is not None and t.library_id not in allowed:
        return None
    return t


def _host(db, f: Feest) -> User | None:
    u = db.get(User, f.host_user_id)
    if u is not None:
        db.expunge(u)
    return u


def _gast_feest(token: str) -> tuple[Feest, User]:
    """Het feest achter een gastlink — of een nette weigering.

    404 als de link nooit bestond; 410 als het feest voorbij is, gestopt, of als
    de gastheer het niet (meer) mag geven: schakelaar uit, review-account,
    account uitgezet. Voor een gast is dat allemaal hetzelfde: afgelopen.
    """
    f = FEESTEN.get(token or "")
    if f is None:
        raise HTTPException(404, "Dit feest bestaat niet")
    if f.voorbij():
        raise HTTPException(410, "Dit feest is afgelopen")
    db = SessionLocal()
    try:
        host = _host(db, f)
        if not feest_mag(db, host):
            raise HTTPException(410, "Dit feest is afgelopen")
        return f, host
    finally:
        db.close()


def _gast(f: Feest, geheim: str | None) -> Gast:
    g = f.gasten.get(geheim or "")
    if g is None:
        raise HTTPException(401, "Kies eerst een bijnaam")
    return g


# ── Wat de gastheer ziet ────────────────────────────────────────────────────
def _host_stand(f: Feest, request: Request) -> dict:
    return {
        "actief": True,
        "token": f.token,
        "url": f"{_publieke_basis(request)}/feest/{f.token}",
        "naam": f.naam,
        "gestart": int(f.gestart),
        "verloopt": int(f.verloopt),
        "gasten": len(f.gasten),
        "aanvragen": [
            {"id": a.id, "ratingKey": track_key(a.track_id), "titel": a.titel,
             "artiest": a.artiest, "bijnaam": a.bijnaam, "stemmen": len(a.stemmen),
             "status": a.status}
            for a in _gesorteerd(f)
        ],
        "wensen": list(f.wensen[-30:]),
    }


class StartBody(BaseModel):
    naam: str | None = None
    sessie: str | None = None


class NuItem(BaseModel):
    ratingKey: str | None = None
    title: str | None = None
    artist: str | None = None


class StandBody(BaseModel):
    nu: NuItem | None = None
    wachtrij: list[NuItem] = []


@router.get("/feest/beschikbaar")
def beschikbaar(user: User = Depends(require_user)):
    """Of de app de knop "Feest starten" laat zien."""
    db = SessionLocal()
    try:
        return {"aan": feest_mag(db, user)}
    finally:
        db.close()


@router.post("/feest/start")
def start(request: Request, body: StartBody | None = None,
          user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        if not feest_mag(db, user):
            raise HTTPException(403, "Feestmodus staat uit")
    finally:
        db.close()
    body = body or StartBody()
    with _SLOT:
        _opruimen()
        # Eén feest tegelijk per gastheer. Opnieuw starten is ook de manier om
        # een link die te ver rondging onbruikbaar te maken: de oude stopt.
        oud = _mijn_feest(user.id)
        if oud is not None:
            oud.gestopt = time.time()
        nu = time.time()
        naam = _schoon(body.naam, 40) or (f"Feest van {_voornaam(user)}" if _voornaam(user) else "Feest")
        f = Feest(token=secrets.token_urlsafe(16), host_user_id=user.id, naam=naam,
                  gestart=nu, verloopt=nu + FEEST_DUUR,
                  sessie=_schoon(body.sessie, 12).upper() or None)
        FEESTEN[f.token] = f
        return _host_stand(f, request)


@router.get("/feest/mijn")
def mijn(request: Request, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        if not feest_mag(db, user):
            return {"actief": False, "aan": False}
    finally:
        db.close()
    with _SLOT:
        f = _mijn_feest(user.id)
        if f is None:
            return {"actief": False, "aan": True}
        return _host_stand(f, request)


@router.post("/feest/mijn/stand")
def stand_gastheer(request: Request, body: StandBody, user: User = Depends(require_user)):
    """De app van de gastheer meldt wat er speelt en wat er aankomt, en krijgt
    terug wat er nu in zijn wachtrij moet.

    Eén aanvraag tegelijk staat "ingepland": die heeft de app als volgende in de
    wachtrij gezet. Speelt hij, of staat hij niet meer in de wachtrij (gespeeld,
    overgeslagen, weggehaald), dan is hij klaar en gaat de volgende met de meeste
    stemmen erin. Zo blijft stemmen tot het laatste moment zin hebben: alleen
    het nummer dat écht als volgende komt ligt vast.
    """
    db = SessionLocal()
    try:
        if not feest_mag(db, user):
            return {"actief": False, "aan": False}
        with _SLOT:
            f = _mijn_feest(user.id)
            if f is None:
                return {"actief": False, "aan": True}

            def item(n: NuItem | None) -> dict | None:
                if n is None or not (n.title or n.ratingKey):
                    return None
                rk = n.ratingKey if (n.ratingKey or "").startswith("t") and (n.ratingKey or "")[1:].isdigit() else None
                return {"titel": _schoon(n.title, 120), "artiest": _schoon(n.artist, 120), "rk": rk}

            f.nu = item(body.nu)
            f.wachtrij = [x for x in (item(n) for n in body.wachtrij[:10]) if x]

            nu_rk = (f.nu or {}).get("rk")
            in_rij = {x["rk"] for x in f.wachtrij if x.get("rk")}
            tijd = time.time()
            for a in f.aanvragen.values():
                rk = track_key(a.track_id)
                if a.status in ("open", "ingepland") and rk == nu_rk:
                    a.status = "gespeeld"
                elif a.status == "ingepland" and rk not in in_rij \
                        and tijd - a.ingepland_op > INPLAN_GRATIE:
                    a.status = "gespeeld"

            inplannen = None
            if not any(a.status == "ingepland" for a in f.aanvragen.values()):
                for a in _gesorteerd(f, ("open",)):
                    t = _track_van(db, user, track_key(a.track_id))
                    if t is None:
                        a.status = "geweigerd"
                        continue
                    a.status = "ingepland"
                    a.ingepland_op = tijd
                    inplannen = {"aanvraag": a.id, "track": track_dict(t)}
                    break

            uit = _host_stand(f, request)
            uit["inplannen"] = inplannen
            return uit
    finally:
        db.close()


@router.post("/feest/mijn/aanvraag/{aid}/weiger")
def weiger(aid: int, request: Request, user: User = Depends(require_user)):
    with _SLOT:
        f = _mijn_feest(user.id)
        if f is None:
            raise HTTPException(404, "Geen feest")
        a = f.aanvragen.get(aid)
        if a is None:
            raise HTTPException(404, "Onbekende aanvraag")
        a.status = "geweigerd"
        return _host_stand(f, request)


@router.post("/feest/stop")
def stop(user: User = Depends(require_user)):
    with _SLOT:
        f = _mijn_feest(user.id)
        if f is not None:
            f.gestopt = time.time()
        return {"actief": False}


# ── Wat de gast ziet ────────────────────────────────────────────────────────
class GastBody(BaseModel):
    bijnaam: str


class AanvraagBody(BaseModel):
    id: str


class WensBody(BaseModel):
    tekst: str


def _gast_stand(f: Feest, host: User, g: Gast | None) -> dict:
    db = SessionLocal()
    try:
        wens = _wensen_mogelijk(db, host)
    finally:
        db.close()
    nu = dict(f.nu) if f.nu else None
    if nu:
        nu["hoes"] = _hoes(f.token, nu.pop("rk", None))
    rij = []
    for x in f.wachtrij:
        y = dict(x)
        y["hoes"] = _hoes(f.token, y.pop("rk", None))
        rij.append(y)
    return {
        "naam": f.naam,
        "verloopt": int(f.verloopt),
        "nu": nu,
        "wachtrij": rij,
        "aanvragen": [
            {"id": a.id, "titel": a.titel, "artiest": a.artiest,
             "hoes": _hoes(f.token, track_key(a.track_id)),
             "bijnaam": a.bijnaam, "stemmen": len(a.stemmen),
             "gestemd": bool(g and g.geheim in a.stemmen),
             "volgende": a.status == "ingepland"}
            for a in _gesorteerd(f)
        ],
        "ik": ({"bijnaam": g.bijnaam,
                "wensenOver": max(0, MAX_WENSEN_PER_GAST - g.wensen)} if g else None),
        "wensen": wens,
    }


@router.get("/feest/{token}")
def pagina(token: str):
    """De pagina zelf. Altijd dezelfde statische pagina; die vraagt daarna de
    stand op en laat zelf zien als het feest niet (meer) bestaat."""
    return FileResponse(_PAGINA, media_type="text/html", headers={
        "Cache-Control": "no-store",
        # De link IS de toegang: niet doorgeven aan wie er vanaf hier gelinkt wordt.
        "Referrer-Policy": "no-referrer",
        "X-Robots-Tag": "noindex, nofollow",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/feest/{token}/stand")
def stand_gast(token: str, request: Request,
               x_feest_gast: str | None = Header(None)):
    _beperk(f"{_ip(request)}|stand", 600, 60)
    f, host = _gast_feest(token)
    with _SLOT:
        return _gast_stand(f, host, f.gasten.get(x_feest_gast or ""))


@router.post("/feest/{token}/gast")
def word_gast(token: str, body: GastBody, request: Request):
    _beperk(f"{_ip(request)}|gast", 40, 600)
    f, host = _gast_feest(token)
    bijnaam = _schoon(body.bijnaam, BIJNAAM_MAX)
    if len(bijnaam) < 2:
        raise HTTPException(400, "Kies een bijnaam van minstens twee tekens")
    with _SLOT:
        if len(f.gasten) >= MAX_GASTEN:
            raise HTTPException(429, "Het is vol op dit feest")
        g = Gast(geheim=secrets.token_urlsafe(18), bijnaam=bijnaam, sinds=time.time())
        f.gasten[g.geheim] = g
        return {"gast": g.geheim, **_gast_stand(f, host, g)}


@router.get("/feest/{token}/zoek")
def zoek(token: str, request: Request, q: str = Query("", max_length=100),
         x_feest_gast: str | None = Header(None)):
    """Zoeken in de bibliotheek van de gastheer. Alleen nummers, en alleen wat
    een gast mag zien: titel, artiest, album en een hoes."""
    f, host = _gast_feest(token)
    g = _gast(f, x_feest_gast)
    _beperk(f"{token}|{g.geheim}|zoek", 30, 60)
    _beperk(f"{_ip(request)}|zoek", 300, 60)

    # Dezelfde woordzoeker en rangschikking als de zoekbalk van de app, maar
    # zonder het verlanglijstje: wat een gast niet vindt hoort niet tussen de
    # missers van de gastheer.
    from .media import _woorden, _elk_woord, _scope, _score, _vouw
    woorden = _woorden(q)
    if not woorden:
        return {"items": []}
    hele = " ".join(woorden)
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, host)
        stmt = _scope(select(Track), Track, allowed)
        for v in _elk_woord(woorden, [Track.title, Track.artist_name, Track.album_artist,
                                      Track.orig_artist, Track.album_title]):
            stmt = stmt.where(v)
        paren, gezien = [], set()
        for t in db.execute(stmt.limit(300)).scalars():
            omg = _vouw(f"{t.artist_name} {t.album_artist} {t.orig_artist} {t.album_title}")
            hooi = f"{_vouw(t.title)} {omg}"
            if not all(w in hooi for w in woorden):
                continue
            paren.append((_score(t.title, omg, woorden, hele) + min(60, (t.play_count or 0) * 3), t))
        paren.sort(key=lambda p: -p[0])
        items = []
        for _, t in paren:
            artiest = (t.orig_artist or "").strip() or t.artist_name or t.album_artist or ""
            k = f"{_vouw(t.title)}|{_vouw(artiest)}"
            if k in gezien:
                continue
            gezien.add(k)
            rk = track_key(t.id)
            items.append({"id": rk, "titel": t.title, "artiest": artiest,
                          "album": t.album_title or "", "hoes": _hoes(token, rk)})
            if len(items) >= 20:
                break
        return {"items": items}
    finally:
        db.close()


@router.post("/feest/{token}/aanvraag")
def vraag_aan(token: str, body: AanvraagBody, request: Request,
              x_feest_gast: str | None = Header(None)):
    f, host = _gast_feest(token)
    g = _gast(f, x_feest_gast)
    _beperk(f"{token}|{g.geheim}|aanvraag", 8, 600)
    db = SessionLocal()
    try:
        t = _track_van(db, host, body.id)
        if t is None:
            raise HTTPException(404, "Dat nummer staat er niet")
        with _SLOT:
            # Staat hij er al? Dan is aanvragen hetzelfde als stemmen.
            for a in f.aanvragen.values():
                if a.track_id == t.id and a.status in ("open", "ingepland"):
                    a.stemmen.add(g.geheim)
                    return {"status": "gestemd", **_gast_stand(f, host, g)}
            eigen = sum(1 for a in f.aanvragen.values()
                        if a.door == g.geheim and a.status == "open")
            if eigen >= MAX_OPEN_PER_GAST:
                raise HTTPException(429, "Je hebt al genoeg aanvragen open staan")
            if len(f.aanvragen) >= MAX_AANVRAGEN:
                raise HTTPException(429, "De lijst zit vol")
            f.volgnummer += 1
            artiest = (t.orig_artist or "").strip() or t.artist_name or t.album_artist or ""
            f.aanvragen[f.volgnummer] = Aanvraag(
                id=f.volgnummer, track_id=t.id, titel=t.title, artiest=artiest,
                door=g.geheim, bijnaam=g.bijnaam, tijd=time.time(), stemmen={g.geheim})
            return {"status": "aangevraagd", **_gast_stand(f, host, g)}
    finally:
        db.close()


@router.post("/feest/{token}/stem/{aid}")
def stem(token: str, aid: int, x_feest_gast: str | None = Header(None)):
    """Duim omhoog — of weer terug. Eén stem per gast per aanvraag."""
    f, host = _gast_feest(token)
    g = _gast(f, x_feest_gast)
    _beperk(f"{token}|{g.geheim}|stem", 40, 60)
    with _SLOT:
        a = f.aanvragen.get(aid)
        if a is None or a.status not in ("open", "ingepland"):
            raise HTTPException(404, "Die aanvraag staat er niet meer")
        if g.geheim in a.stemmen:
            a.stemmen.discard(g.geheim)
        else:
            a.stemmen.add(g.geheim)
        return _gast_stand(f, host, g)


@router.post("/feest/{token}/wens")
def wens(token: str, body: WensBody, request: Request,
         x_feest_gast: str | None = Header(None)):
    """Staat het er niet? Dan mag een gast het wensen — op naam van de gastheer,
    op het verlanglijstje van de gastheer (`noteer_misser`).

    Alleen als de gastheer zelf wensen mag doen (schakelaar `wensen` aan, geen
    winkelstand, geen review-account). Geen links: een geplakte link is een
    import, en dat is aan de gastheer, niet aan een gast.
    """
    f, host = _gast_feest(token)
    g = _gast(f, x_feest_gast)
    _beperk(f"{token}|{g.geheim}|wens", 5, 600)
    tekst = _schoon(body.tekst, 120)
    laag = tekst.lower()
    if "http" in laag or "://" in laag or "spotify" in laag or "www." in laag:
        return {"status": "geen_link"}
    db = SessionLocal()
    try:
        if not _wensen_mogelijk(db, host):
            return {"status": "uit"}
    finally:
        db.close()
    with _SLOT:
        if g.wensen >= MAX_WENSEN_PER_GAST:
            return {"status": "genoeg", "wensenOver": 0}
        if sum(1 for w in f.wensen if w.get("geteld")) >= MAX_WENSEN_PER_FEEST:
            return {"status": "genoeg", "wensenOver": max(0, MAX_WENSEN_PER_GAST - g.wensen)}

    # Op het verlanglijstje van de gastheer, net als een zoekopdracht die
    # niets opleverde.
    from .wishlist import noteer_misser
    db = SessionLocal()
    try:
        noteer_misser(db, host.id, tekst)
    finally:
        db.close()
    uit: dict = {}
    status = "aangevraagd"
    with _SLOT:
        geteld = status in ("aangevraagd", "opnieuw")
        if geteld:
            g.wensen += 1
        f.wensen.append({"tekst": tekst, "bijnaam": g.bijnaam, "status": status,
                         "titel": uit.get("title") or "", "artiest": uit.get("artist") or "",
                         "geteld": geteld, "tijd": int(time.time())})
        del f.wensen[:-100]
        # Alleen wat een gast hoeft te weten; geen albumnummers of dakgetallen.
        return {"status": status, "titel": uit.get("title") or "",
                "artiest": uit.get("artist") or "",
                "wensenOver": max(0, MAX_WENSEN_PER_GAST - g.wensen)}


@router.get("/feest/{token}/hoes/{rk}")
def hoes(token: str, rk: str, request: Request):
    """De hoes van een nummer, verkleind. Alleen nummers, alleen uit de
    bibliotheken van de gastheer — dezelfde controle als bij hem zelf."""
    _beperk(f"{_ip(request)}|hoes", 1500, 60)
    f, host = _gast_feest(token)
    if not rk.startswith("t") or not rk[1:].isdigit():
        raise HTTPException(404, "Geen hoes")
    from .artwork import photo_transcode
    antwoord = photo_transcode(url=f"/art/{rk}", width=160, height=160, quality=75, user=host)
    antwoord.headers["Cache-Control"] = "private, max-age=3600"
    return antwoord
