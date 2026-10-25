"""Samen spelen — de quiz "Raad het nummer" met meerdere telefoons.

Zoals bij een kwisavond: de telefoon van de gastheer speelt de muziek (op de
speaker) en laat de stand zien. Wie meedoet scant de QR, kiest een bijnaam op
een gewone webpagina die deze server serveert (`static/spel.html`) en krijgt
bij elke ronde de vier keuzes op zijn eigen scherm. Snel én goed geeft de
meeste punten.

Gebouwd naar het voorbeeld van de feestmodus (feest.py), met dezelfde regels:

- **Geen account, geen muziek.** Een speler krijgt een willekeurig geheimpje
  terug dat zijn browser bewaart. Er is geen stream achter de spellink; een
  speler ziet alleen de vier titels en, na het antwoord, of hij goed zat.
  Afspelen doet alleen de telefoon van de gastheer.
- **Geen gegevens van de gastheer.** Alleen zijn voornaam als kamernaam.
- **Alleen in het geheugen.** Na een herstart van de server is het spel weg.
- Dezelfde schakelaar als het feest (`feest`): staat die uit, dan geen spel.

De gastheer stuurt per ronde de keuzes (titel en artiest; welke goed is
houdt hij voor zich tot het onthullen), en zegt daarna welke het was. Pas dan
rekent de server de punten uit: zo kan een speler het goede antwoord niet uit
de stand halen terwijl de ronde loopt.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..db import SessionLocal
from ..deps import require_user
from ..models import User
from .feest import _beperk, _ip, _publieke_basis, _schoon, _voornaam, feest_mag

router = APIRouter(tags=["spel"])

SPEL_DUUR = 4 * 3600
OPRUIMEN_NA = 2 * 3600
MAX_SPELERS = 40
BIJNAAM_MAX = 18
MAX_OPTIES = 6
# Punten voor een goed antwoord: van MAX_PUNTEN (meteen) tot MIN_PUNTEN (op de bel).
MAX_PUNTEN = 1000
MIN_PUNTEN = 200

_PAGINA = Path(__file__).resolve().parent.parent / "static" / "spel.html"
_SLOT = threading.RLock()


@dataclass
class Speler:
    geheim: str
    bijnaam: str
    sinds: float
    punten: int = 0
    reeks: int = 0
    gezien: float = 0.0
    # Wie vanuit de app meedoet is ingelogd: dan herkennen we hem aan zijn
    # account en kiest hij geen bijnaam (zie /lid).
    user_id: int | None = None
    # En aan zijn toestel: met één account op twee telefoons ben je twee spelers.
    toestel: str = ""
    # Het adres van de profielfoto (/users/<id>/thumb), als hij er een heeft.
    foto: str | None = None
    # Zelf weggegaan (terugknop), of al gemeld als "verbinding kwijt".
    vertrokken: bool = False
    weg_gemeld: bool = False


@dataclass
class Antwoord:
    keuze: int
    na: float           # seconden na het begin van de ronde


@dataclass
class Spel:
    token: str
    host_user_id: int
    naam: str
    gestart: float
    verloopt: float
    # Welk spel: 'quiz' (raad het nummer), 'zing' (zing verder), 'artiest'.
    soort: str = "quiz"
    gestopt: float = 0.0
    spelers: dict[str, Speler] = field(default_factory=dict)
    # 'wacht' (lobby), 'ronde' (er kan geantwoord worden), 'uitslag', 'einde'
    fase: str = "wacht"
    ronde: int = 0
    rondes: int = 0
    opties: list[dict] = field(default_factory=list)
    begin: float = 0.0
    duur: float = 30.0
    goed: int = -1
    antwoorden: dict[str, Antwoord] = field(default_factory=dict)
    # Wat elke speler in de laatste ronde verdiende (geheim → punten).
    verdiend: dict[str, int] = field(default_factory=dict)
    # Wat er gebeurde met de spelers (erbij, weg, terug, verlaten), voor de
    # meldingen in de app. Oplopend id, alleen de laatste paar.
    gebeurtenissen: list[dict] = field(default_factory=list)
    volg_id: int = 0

    def voorbij(self) -> bool:
        return bool(self.gestopt) or time.time() >= self.verloopt


SPELLEN: dict[str, Spel] = {}


def _opruimen() -> None:
    nu = time.time()
    for k in [k for k, s in SPELLEN.items()
              if s.voorbij() and nu - max(s.gestopt or 0, min(s.verloopt, nu)) > OPRUIMEN_NA]:
        SPELLEN.pop(k, None)


def _mijn_spel(user_id: int) -> Spel | None:
    for s in SPELLEN.values():
        if s.host_user_id == user_id and not s.voorbij():
            return s
    return None


def _mag(user: User) -> None:
    db = SessionLocal()
    try:
        if not feest_mag(db, user):
            raise HTTPException(403, "Samen spelen staat uit")
    finally:
        db.close()


def _spel(token: str) -> Spel:
    s = SPELLEN.get(token or "")
    if s is None:
        raise HTTPException(404, "Dit spel bestaat niet")
    if s.voorbij():
        raise HTTPException(410, "Dit spel is afgelopen")
    db = SessionLocal()
    try:
        host = db.get(User, s.host_user_id)
        if not feest_mag(db, host):
            raise HTTPException(410, "Dit spel is afgelopen")
    finally:
        db.close()
    return s


WEG_NA_S = 20


def _meld(s: Spel, soort: str, p: Speler) -> None:
    s.volg_id += 1
    s.gebeurtenissen.append({"id": s.volg_id, "soort": soort, "naam": p.bijnaam, "foto": p.foto})
    del s.gebeurtenissen[:-20]


def _is_weg(p: Speler, nu: float) -> bool:
    return p.vertrokken or nu - p.gezien > WEG_NA_S


def _peil(s: Spel) -> None:
    """Wie de verbinding kwijt is of terug is: een melding, één keer."""
    nu = time.time()
    for p in s.spelers.values():
        if p.vertrokken:
            continue
        weg = nu - p.gezien > WEG_NA_S
        if weg and not p.weg_gemeld:
            p.weg_gemeld = True
            _meld(s, "weg", p)
        elif not weg and p.weg_gemeld:
            p.weg_gemeld = False
            _meld(s, "terug", p)


def _spelers_lijst(s: Spel) -> list[dict]:
    nu = time.time()
    return [{"bijnaam": p.bijnaam, "foto": p.foto, "punten": p.punten,
             "beantwoord": p.geheim in s.antwoorden, "weg": _is_weg(p, nu),
             "vertrokken": p.vertrokken}
            for p in sorted(s.spelers.values(), key=lambda p: (-p.punten, p.sinds))]


def _ranglijst(s: Spel) -> list[dict]:
    rij = sorted(s.spelers.values(), key=lambda p: (-p.punten, p.sinds))
    return [{"bijnaam": p.bijnaam, "punten": p.punten, "reeks": p.reeks, "foto": p.foto,
             "erbij": s.verdiend.get(p.geheim, 0) if s.fase in ("uitslag", "einde") else 0}
            for p in rij]


def _over(s: Spel) -> float:
    return max(0.0, s.duur - (time.time() - s.begin)) if s.fase == "ronde" else 0.0


# ── De gastheer ─────────────────────────────────────────────────────────────
def _host_stand(s: Spel, request: Request) -> dict:
    _peil(s)
    return {
        "actief": True,
        "token": s.token,
        "url": f"{_publieke_basis(request)}/spel/{s.token}",
        "naam": s.naam,
        "soort": s.soort,
        "fase": s.fase,
        "ronde": s.ronde,
        "spelers": _spelers_lijst(s),
        "gebeurtenissen": s.gebeurtenissen[-8:],
        "antwoorden": len(s.antwoorden),
        "ranglijst": _ranglijst(s),
    }


class RondeOptie(BaseModel):
    titel: str
    artiest: str = ""


class RondeBody(BaseModel):
    ronde: int
    rondes: int = 0
    opties: list[RondeOptie]
    duur: float = 30.0


class LidBody(BaseModel):
    toestel: str = ""


class StartBody(BaseModel):
    soort: str = "quiz"


class OnthulBody(BaseModel):
    goed: int


@router.post("/spel/start")
def start(request: Request, body: StartBody | None = None, user: User = Depends(require_user)):
    """Een nieuwe kamer. Een lopend spel van dezelfde gastheer stopt dan."""
    _mag(user)
    with _SLOT:
        _opruimen()
        oud = _mijn_spel(user.id)
        if oud:
            oud.gestopt = time.time()
        nu = time.time()
        naam = _voornaam(user)
        soort = (body.soort if body else "quiz")
        s = Spel(token=secrets.token_urlsafe(9), host_user_id=user.id,
                 naam=naam, gestart=nu, verloopt=nu + SPEL_DUUR,
                 soort=soort if soort in ("quiz", "zing", "artiest") else "quiz")
        SPELLEN[s.token] = s
        return _host_stand(s, request)


@router.get("/spel/mijn")
def mijn(request: Request, user: User = Depends(require_user),
         x_spel_speler: str | None = Header(None)):
    with _SLOT:
        s = _mijn_spel(user.id)
        # De kwismaster speelt zelf ook mee. Hij vraagt de stand hier op en niet
        # via /stand, dus hier ook bijhouden dat hij er nog is. Anders telde hij
        # na 20 seconden als "weg", en eindigde een ronde zodra één ander koos.
        if s and x_spel_speler and x_spel_speler in s.spelers:
            s.spelers[x_spel_speler].gezien = time.time()
        return _host_stand(s, request) if s else {"actief": False}


@router.post("/spel/mijn/ronde")
def ronde(body: RondeBody, request: Request, user: User = Depends(require_user)):
    with _SLOT:
        s = _mijn_spel(user.id)
        if not s:
            raise HTTPException(404, "Geen spel")
        opties = [{"titel": _schoon(o.titel, 80), "artiest": _schoon(o.artiest, 60)}
                  for o in body.opties[:MAX_OPTIES]]
        if len(opties) < 2:
            raise HTTPException(400, "Te weinig keuzes")
        s.fase = "ronde"
        s.ronde = int(body.ronde)
        s.rondes = max(0, int(body.rondes))
        s.opties = opties
        s.begin = time.time()
        s.duur = max(5.0, min(120.0, float(body.duur)))
        s.goed = -1
        s.antwoorden = {}
        s.verdiend = {}
        return _host_stand(s, request)


@router.post("/spel/mijn/onthul")
def onthul(body: OnthulBody, request: Request, user: User = Depends(require_user)):
    """Het goede antwoord, en dus de punten van deze ronde."""
    with _SLOT:
        s = _mijn_spel(user.id)
        if not s:
            raise HTTPException(404, "Geen spel")
        if s.fase != "ronde":
            return _host_stand(s, request)
        s.goed = int(body.goed)
        s.verdiend = {}
        for geheim, p in s.spelers.items():
            a = s.antwoorden.get(geheim)
            if a and a.keuze == s.goed:
                deel = min(1.0, a.na / s.duur)
                pt = round(MAX_PUNTEN - (MAX_PUNTEN - MIN_PUNTEN) * deel)
                p.reeks += 1
                # Drie of meer op rij: een kleine bonus, zoals bij een echte kwis.
                if p.reeks >= 3:
                    pt += 100
                p.punten += pt
                s.verdiend[geheim] = pt
            else:
                p.reeks = 0
        s.fase = "uitslag"
        return _host_stand(s, request)


@router.post("/spel/mijn/einde")
def einde(request: Request, user: User = Depends(require_user)):
    with _SLOT:
        s = _mijn_spel(user.id)
        if not s:
            raise HTTPException(404, "Geen spel")
        s.fase = "einde"
        return _host_stand(s, request)


@router.post("/spel/stop")
def stop(user: User = Depends(require_user)):
    with _SLOT:
        s = _mijn_spel(user.id)
        if s:
            s.gestopt = time.time()
        return {"actief": False}


# ── De spelers ──────────────────────────────────────────────────────────────
class SpelerBody(BaseModel):
    bijnaam: str


class AntwoordBody(BaseModel):
    ronde: int
    keuze: int


def _speler_stand(s: Spel, p: Speler | None) -> dict:
    _peil(s)
    a = s.antwoorden.get(p.geheim) if p else None
    uit = {
        "naam": s.naam,
        "soort": s.soort,
        "fase": s.fase,
        "ronde": s.ronde,
        "rondes": s.rondes,
        "spelers": len([q for q in s.spelers.values() if not q.vertrokken]),
        "spelersLijst": _spelers_lijst(s),
        "gebeurtenissen": s.gebeurtenissen[-8:],
        # Hoeveel er deze ronde al gekozen hebben (niet wie of wat).
        "gekozenAantal": len(s.antwoorden) if s.fase == "ronde" else 0,
        "over": round(_over(s), 1),
        "duur": s.duur,
        "opties": s.opties if s.fase in ("ronde", "uitslag") else [],
        "gekozen": a.keuze if a else None,
        "goed": s.goed if s.fase in ("uitslag", "einde") else None,
        "ranglijst": _ranglijst(s)[:10] if s.fase in ("uitslag", "einde", "wacht") else [],
        "ik": None,
    }
    if p:
        plek = 1 + sum(1 for q in s.spelers.values() if q.punten > p.punten)
        uit["ik"] = {"bijnaam": p.bijnaam, "punten": p.punten, "reeks": p.reeks, "plek": plek,
                     "erbij": s.verdiend.get(p.geheim, 0) if s.fase in ("uitslag", "einde") else 0}
    return uit


@router.get("/spel/{token}")
def pagina(token: str):
    return FileResponse(_PAGINA, media_type="text/html", headers={
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Robots-Tag": "noindex, nofollow",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/spel/{token}/stand")
def stand(token: str, request: Request, x_spel_speler: str | None = Header(None)):
    # Ruim per IP: aan een tafel zit iedereen op dezelfde wifi.
    _beperk(f"{_ip(request)}|spelstand", 2400, 60)
    s = _spel(token)
    with _SLOT:
        p = s.spelers.get(x_spel_speler or "")
        if p:
            p.gezien = time.time()
        return _speler_stand(s, p)


@router.post("/spel/{token}/speler")
def word_speler(token: str, body: SpelerBody, request: Request):
    _beperk(f"{_ip(request)}|spelspeler", 60, 600)
    s = _spel(token)
    bijnaam = _schoon(body.bijnaam, BIJNAAM_MAX)
    if len(bijnaam) < 2:
        raise HTTPException(400, "Kies een bijnaam van minstens twee tekens")
    with _SLOT:
        if len(s.spelers) >= MAX_SPELERS:
            raise HTTPException(429, "Het spel zit vol")
        if any(p.bijnaam.lower() == bijnaam.lower() for p in s.spelers.values()):
            raise HTTPException(409, "Die bijnaam is al gekozen")
        nu = time.time()
        p = Speler(geheim=secrets.token_urlsafe(18), bijnaam=bijnaam, sinds=nu, gezien=nu)
        s.spelers[p.geheim] = p
        _meld(s, "erbij", p)
        return {"speler": p.geheim, **_speler_stand(s, p)}


@router.post("/spel/{token}/lid")
def word_lid(token: str, request: Request, body: LidBody | None = None,
             user: User = Depends(require_user)):
    """Meedoen vanuit de app: als ingelogde gebruiker, met je eigen naam.

    Nog eens aanroepen (de app opnieuw geopend) geeft dezelfde speler terug, met
    zijn punten. Heet er al iemand zo (een gast via de webpagina), dan komt er
    een cijfer achter.
    """
    # Per gebruiker, niet per IP: op het LAN komt alles via hetzelfde adres binnen.
    _beperk(f"u{user.id}|spellid", 30, 60)
    s = _spel(token)
    toestel = ((body.toestel if body else "") or "")[:80]
    with _SLOT:
        for p in s.spelers.values():
            if p.user_id == user.id and p.toestel == toestel:
                p.gezien = time.time()
                if p.vertrokken:
                    p.vertrokken = False
                    _meld(s, "terug", p)
                return {"speler": p.geheim, **_speler_stand(s, p)}
        if len(s.spelers) >= MAX_SPELERS:
            raise HTTPException(429, "Het spel zit vol")
        basis = (_voornaam(user) or "Speler")[:BIJNAAM_MAX]
        naam, n = basis, 2
        bezet = {p.bijnaam.lower() for p in s.spelers.values()}
        while naam.lower() in bezet:
            naam = f"{basis[:BIJNAAM_MAX - 3]} {n}"
            n += 1
        nu = time.time()
        heeft = bool(getattr(user, "thumb_path", None))
        p = Speler(geheim=secrets.token_urlsafe(18), bijnaam=naam, sinds=nu, gezien=nu,
                   user_id=user.id, toestel=toestel, foto=f"/users/{user.id}/thumb" if heeft else None)
        s.spelers[p.geheim] = p
        _meld(s, "erbij", p)
        return {"speler": p.geheim, **_speler_stand(s, p)}


@router.post("/spel/{token}/verlaat")
def verlaat(token: str, request: Request, x_spel_speler: str | None = Header(None)):
    """Een speler gaat weg (terugknop). De anderen zien dat, en er wordt niet
    meer op hem gewacht."""
    _beperk(f"{_ip(request)}|spelverlaat", 600, 60)
    s = _spel(token)
    with _SLOT:
        p = s.spelers.get(x_spel_speler or "")
        if p and not p.vertrokken:
            p.vertrokken = True
            _meld(s, "verlaten", p)
        return {"ok": True}


@router.post("/spel/{token}/antwoord")
def antwoord(token: str, body: AntwoordBody, request: Request,
             x_spel_speler: str | None = Header(None)):
    _beperk(f"{_ip(request)}|spelantwoord", 600, 60)
    s = _spel(token)
    with _SLOT:
        p = s.spelers.get(x_spel_speler or "")
        if p is None:
            raise HTTPException(401, "Kies eerst een bijnaam")
        if s.fase != "ronde" or body.ronde != s.ronde:
            raise HTTPException(409, "Deze ronde is voorbij")
        if p.geheim in s.antwoorden:
            raise HTTPException(409, "Je hebt al gekozen")
        if not 0 <= body.keuze < len(s.opties):
            raise HTTPException(400, "Onbekende keuze")
        na = time.time() - s.begin
        if na > s.duur + 1:
            raise HTTPException(409, "Te laat")
        s.antwoorden[p.geheim] = Antwoord(keuze=body.keuze, na=na)
        return _speler_stand(s, p)
