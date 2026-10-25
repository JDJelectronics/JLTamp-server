"""Welk nummer de radio hierna draait.

Twee delen, met opzet los van elkaar:

  bouw_pool(db)   leest de bibliotheek en zet er een gewicht op: wat in de
                  Top 40 staat, wat hier veel gedraaid en veel geliefd wordt,
                  krijgt voorrang; daarnaast een ruime greep uit de rest, zodat
                  de zender niet na een dag rondjes draait. Draait af en toe
                  (elk half uur), niet per nummer.
  kies(...)       de rotatieregels, zonder database: geen artiest die binnen
                  de laatste acht nummers al langskwam, geen nummer dat de
                  laatste zes uur al speelde. Puur, dus te testen.

Een nummer staat er vaak twee keer in: als mp3 in Muzyk en als flac in High-Res.
Voor de regels is dat hetzelfde nummer — daarom telt de `sleutel` (artiest +
titel in vergelijkvorm, zie matching.py) en niet het id.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field

from sqlalchemy import func, or_, select

from ..matching import artiest_delen, sleutel as vergelijkvorm
from ..models import Track

log = logging.getLogger("radio")

ARTIEST_AFSTAND = 8           # zoveel nummers geen zelfde artiest
NUMMER_AFSTAND_S = 6 * 3600   # zolang geen zelfde nummer

# Als er met de strenge regels niets overblijft (een kleine bibliotheek, of een
# pool die net leeg is): stap voor stap losser, en pas helemaal aan het eind
# maakt het niet meer uit. Stilte is erger dan een herhaling.
_LOSSER = [(ARTIEST_AFSTAND, NUMMER_AFSTAND_S), (4, 3 * 3600), (2, 3600), (1, 0), (0, 0)]


@dataclass(frozen=True)
class Kandidaat:
    id: int
    sleutel: str                       # artiest|titel in vergelijkvorm
    artiesten: frozenset[str]          # alle delen van de artiestnaam
    gewicht: float = 1.0
    dagdelen: frozenset[str] = field(default_factory=frozenset)   # extra kans hier
    jaar: int | None = None            # voor de programma's (programma.py)
    album_id: int | None = None        # voor geblokkeerde albums (blokkeer.py)
    genre: str = ""                    # in kleine letters


@dataclass(frozen=True)
class Gespeeld:
    """Iets dat gespeeld heeft of al ingepland staat (ts = wanneer)."""
    id: int
    sleutel: str
    artiesten: frozenset[str]
    ts: float


def dagdeel(uur: int) -> str:
    if uur < 6:
        return "nacht"
    if uur < 9:
        return "ochtend"
    if uur < 18:
        return "dag"
    return "avond"


# Waar een artiestnaam in delen uiteenvalt. Ruimer dan matching.GAST (daar
# splitst "&" tussen spaties niet, met opzet of niet): voor de rotatie is te
# streng geen probleem — "Simon & Garfunkel" houdt Paul Simon dan ook even weg.
_DELER = re.compile(r"\s*(?:&|,|\+|/|;|\bx\b|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b|"
                    r"\bmet\b|\bvs\.?|\band\b|\ben\b)\s*", re.I)


def artiest_sleutels(artiest: str) -> frozenset[str]:
    """"Queen & David Bowie" → {queendavidbowie, queen, davidbowie}. Raakt één
    deel een recente artiest, dan telt het als dezelfde artiest."""
    delen = set(artiest_delen(artiest or ""))
    for stuk in _DELER.split(artiest or ""):
        k = re.sub(r"[^a-z0-9]+", "", vergelijkvorm(stuk))
        if k:
            delen.add(k)
    return frozenset(delen) or frozenset({"?"})


def nummer_sleutel(artiest: str, titel: str) -> str:
    return f"{vergelijkvorm(artiest)}|{vergelijkvorm(titel)}"


# Zoveel nummers moet een programma minstens hebben, anders draait de zender de
# gewone mix: met twintig nummers in een uur blokkeren de rotatieregels zichzelf.
MIN_PROGRAMMA = 25


def kies(kandidaten: list[Kandidaat], recent: list[Gespeeld], nu: float,
         deel: str | None = None, rng: random.Random | None = None,
         programma=None, artiest_afstand: int = ARTIEST_AFSTAND,
         nummer_s: float = NUMMER_AFSTAND_S) -> Kandidaat | None:
    """Het volgende nummer volgens de rotatieregels, of None bij een lege pool.

    `recent` is alles wat net speelde én wat al ingepland staat, oudste eerst.
    Gewogen loting: een groter gewicht is een grotere kans, en in zijn eigen
    dagdeel telt een nummer dubbel.
    """
    if not kandidaten:
        return None
    r = rng or random
    # Het programma (programma.Programma) beperkt de keus tot wat erbij past, en
    # laat de bekende nummers zwaarder wegen naar gelang `hits`.
    hits = 1.0
    if programma is not None:
        hits = programma.hits
        if programma.heeft_filter:
            passend = [k for k in kandidaten if programma.past(k.jaar, k.genre)]
            if len(passend) >= MIN_PROGRAMMA:
                kandidaten = passend
    # De strengste trap komt uit de instellingen (Beheer → Radio); daarna
    # dezelfde versoepelingen als altijd.
    trappen = [(artiest_afstand, nummer_s)] + [x for x in _LOSSER[1:]
                                                 if x[0] < artiest_afstand]
    for artiest_afstand, nummer_afstand in trappen:
        bezet: set[str] = set()
        for g in recent[-artiest_afstand:] if artiest_afstand else []:
            bezet |= g.artiesten
        net = {g.sleutel for g in recent if nu - g.ts < nummer_afstand}
        net |= {g.sleutel for g in recent[-1:]}      # nooit twee keer direct achter elkaar
        mag = [k for k in kandidaten
               if k.sleutel not in net and not (k.artiesten & bezet)]
        if not mag and artiest_afstand == 0:
            mag = kandidaten
        if mag:
            gewichten = [k.gewicht * (2.0 if deel and deel in k.dagdelen else 1.0)
                         * (hits if k.gewicht > _BREEDTE_GEWICHT else 1.0) for k in mag]
            return r.choices(mag, weights=gewichten)[0]
    return None


# ── De pool ──────────────────────────────────────────────────────────────────

# Geen nummers maar stukjes: een intro van twaalf seconden of een skit tussen
# twee raps hoort niet op de radio.
_GEEN_NUMMER = re.compile(r"\b(intro|outro|interlude|skit|reprise|hidden track)\b", re.I)
_MIN_MS = 90_000
_MAX_MS = 8 * 60_000

# Hoe groot de greep uit de rest van de bibliotheek is, en hoe zwaar die weegt
# tegenover de bekende nummers. Met ~400 bekende nummers van gemiddeld gewicht
# drie is dit grofweg zestig procent bekend, veertig procent ontdekken.
_BREEDTE = 1500
_BREEDTE_GEWICHT = 0.5
# Zoveel nummers per komend programma extra in de pool.
_PROGRAMMA_GREEP = 600


def _kandidaat(t: Track, gewicht: float, dagdelen: frozenset[str] = frozenset()) -> Kandidaat:
    artiest = (t.orig_artist or t.artist_name or "").strip()
    return Kandidaat(id=t.id, sleutel=nummer_sleutel(artiest, t.title or ""),
                     artiesten=artiest_sleutels(artiest), gewicht=gewicht, dagdelen=dagdelen,
                     jaar=t.year, genre=(t.genre or "").lower(), album_id=t.album_id)


# Hoort niet op een algemene zender, in geen enkel programma: kinderliedjes en
# filmmuziek (samen duizenden nummers in een grote bibliotheek).
_NIET_OP_DE_RADIO = ("kids", "films/games", "film scores", "classical")


def _bruikbaar(t: Track, bibliotheken: list[int] | None) -> bool:
    if bibliotheken and t.library_id not in bibliotheken:
        return False
    if t.duration_ms and not (_MIN_MS <= t.duration_ms <= _MAX_MS):
        return False
    if any(x in (t.genre or "").lower() for x in _NIET_OP_DE_RADIO):
        return False
    return not _GEEN_NUMMER.search(t.title or "")


def bouw_pool(db, bibliotheken: list[int] | None = None, programmas=(),
              geen_albums: set[int] | frozenset[int] = frozenset()) -> list[Kandidaat]:
    """Alles waar de radio uit kiest, met een gewicht per nummer.

    De lijsten komen uit charts.py, dezelfde die het beginscherm vullen: die
    weten al wat hier bekend en geliefd is. Wat ontbreekt (een lege
    luistergeschiedenis op een nieuwe server) maakt de pool alleen kleiner,
    nooit leeg: de greep uit de bibliotheek is er altijd.
    """
    from ..routers import charts
    from ..routers.admin_users import schakelaar

    gewicht: dict[int, float] = {}
    dagdelen: dict[int, set[str]] = {}

    def geef(ids, g: float, deel: str | None = None):
        for i in ids:
            gewicht[i] = max(gewicht.get(i, 0.0), g)
            if deel:
                dagdelen.setdefault(i, set()).add(deel)

    def veilig(naam, f):
        try:
            return f()
        except Exception as e:      # één lijst die stuk is, houdt de rest niet tegen
            log.info("radio: lijst %s overgeslagen: %s", naam, e)
            return []

    nu = int(time.time())
    geef(veilig("top40", lambda: charts._extern_ids(db, charts.FEED_TOP40)[1]), 6.0)
    if veilig("viraal-aan", lambda: [schakelaar(db, "viraal")]) == [True]:
        geef(veilig("viraal", lambda: charts._extern_ids(db, charts.FEED_VIRAAL)[1]), 3.0)
    for tid, stemmen in veilig("stemmen", lambda: charts._stemmen(db, nu - 365 * 86400)):
        geef([tid], 2.0 + min(stemmen, 20) / 5.0)
    geef([t.id for t in veilig("geliefd", lambda: charts._meest_geliefd(db))], 3.0)
    geef([t.id for t in veilig("nacht", lambda: charts._bouw_nacht(db))], 2.0, "nacht")
    geef([t.id for t in veilig("ochtend", lambda: charts._bouw_ochtend(db))], 2.0, "ochtend")
    # Wat het vaakst gedraaid is over de hele geschiedenis (de teller op het
    # nummer zelf), voor een server waar play_events nog jong is.
    geef(veilig("teller", lambda: db.execute(
        select(Track.id).where(Track.play_count > 0)
        .order_by(Track.play_count.desc()).limit(300)).scalars().all()), 1.5)

    pool: dict[str, Kandidaat] = {}
    if gewicht:
        for t in db.query(Track).filter(Track.id.in_(list(gewicht))).all():
            if _bruikbaar(t, bibliotheken):
                k = _kandidaat(t, gewicht[t.id], frozenset(dagdelen.get(t.id, ())))
                if k.sleutel not in pool or pool[k.sleutel].gewicht < k.gewicht:
                    pool[k.sleutel] = k
    bekend = len(pool)

    # De greep uit de rest. ORDER BY random() over honderdduizend rijen is in SQLite
    # een fractie van een seconde, en dit gebeurt maar elk half uur.
    q = select(Track).order_by(func.random()).limit(_BREEDTE * 2)
    if geen_albums:
        q = q.where(Track.album_id.notin_(list(geen_albums)))
    if bibliotheken:
        q = q.where(Track.library_id.in_(bibliotheken))
    for t in db.execute(q).scalars():
        if len(pool) >= bekend + _BREEDTE:
            break
        if _bruikbaar(t, bibliotheken):
            k = _kandidaat(t, _BREEDTE_GEWICHT)
            pool.setdefault(k.sleutel, k)
    # En per komend programma een eigen greep die erbij past: in de algemene
    # greep zitten maar een stuk of dertig nummers uit de jaren tachtig.
    for p in programmas:
        if not p.heeft_filter:
            continue
        q = select(Track)
        if p.jaren:
            q = q.where(Track.year >= p.jaren[0], Track.year <= p.jaren[1])
        if p.genres:
            q = q.where(or_(*[func.lower(Track.genre).contains(g) for g in p.genres]))
        if bibliotheken:
            q = q.where(Track.library_id.in_(bibliotheken))
        if geen_albums:
            q = q.where(Track.album_id.notin_(list(geen_albums)))
        erbij = 0
        for t in db.execute(q.order_by(func.random()).limit(_PROGRAMMA_GREEP * 2)).scalars():
            if erbij >= _PROGRAMMA_GREEP:
                break
            if _bruikbaar(t, bibliotheken):
                k = _kandidaat(t, _BREEDTE_GEWICHT)
                if p.past(k.jaar, k.genre) and k.sleutel not in pool:
                    pool[k.sleutel] = k
                    erbij += 1
        log.info("radio: %s: %d nummers erbij", p.naam, erbij)
    uit = [k for k in pool.values() if k.album_id not in geen_albums] if geen_albums else list(pool.values())
    log.info("radio: pool van %d nummers (%d bekend)", len(uit), bekend)
    return uit
