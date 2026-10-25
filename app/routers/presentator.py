"""De presentator: een stem die tussen twee nummers iets zegt, zoals op de radio.

De app vraagt vlak voordat een nummer afloopt om een aankondiging voor de
overgang naar het volgende. Hier wordt die tekst geschreven (door het model op
de eigen machine, Ollama) en uitgesproken (door Piper, een kleine TTS die op de
processor draait). Terug gaat een kort audiobestand dat de app over het einde
van het ene en het begin van het volgende nummer heen afspeelt.

**Geen verzonnen feiten.** Dit gaat over echte artiesten. Een taalmodel schrijft
zonder blikken of blozen dat een nummer "in 1987 drie weken op één stond", en
dan zegt een stem dat met overtuiging in je huiskamer. Daarom krijgt het model
ALLEEN wat er in de bibliotheek staat (titel, artiest, album, jaar, genre) en
de inleiding van het Wikipedia-artikel dat artistinfo al ophaalt — en daarna
wordt nagekeken: een getal in de tekst dat niet in de feiten staat, en de tekst
gaat niet door. Liever een keer geen aankondiging dan een verkeerde.

**Alles is optioneel.** Staat de schakelaar uit, is het model niet bereikbaar,
of is er geen stem: dan zegt dit endpoint netjes "niet beschikbaar" en speelt de
app gewoon door. Er is geen pad waarop het ontbreken van de presentator de
muziek ophoudt.

**Niet voor de zender.** Dit is voor de app van de luisteraar zelf, en voor
JLTamp Radio (app/radio), de 24/7-stream van deze server die dezelfde functies
gebruikt (feiten_voor, maak_aankondiging). De live zender van JLTamp is een
aparte dienst en heeft hier niets mee te maken; de stem komt daar nooit in een
uitzending terecht.

Stem: zie docs/PRESENTATOR.md voor hoe Piper en een Nederlandse stem op de
Studio komen. Omgeving:

    JLTAMP_AI_URL          Ollama (gedeeld met artistinfo)
    JLTAMP_AI_MODEL        idem; vrij te wisselen, de opdrachten zijn gewone
                           instructies zonder trucs voor één bepaald model
    JLTAMP_PIPER_URL       een Piper-HTTP-dienst (python -m piper.http_server),
                           bv. http://jltamp-piper:5000
    JLTAMP_PIPER_MODEL     óf: pad naar een .onnx-stem, dan draait Piper hier
    JLTAMP_PIPER_TEMPO     length_scale, < 1 is sneller praten (standaard 0.95)
    JLTAMP_STEM_URL        de stemdienst op de Mac zelf (stem_dienst.py, Chatterbox):
                           een natuurlijke man- en vrouwenstem die elkaar afwisselen.
                           Piper blijft de achtervang als die er niet is of te
                           lang nadenkt.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import config
from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key
from ..models import Track, User
from . import artistinfo
from .. import presentator_talen as talen
from ..presentator_talen import taal_van
from .admin_users import schakelaar

router = APIRouter(prefix="/presentator", tags=["presentator"])
log = logging.getLogger("presentator")

CACHE = config.DATA_DIR / "presentator"
CACHE.mkdir(parents=True, exist_ok=True)

_AI = os.environ.get("JLTAMP_AI_URL", "http://host.docker.internal:11434").rstrip("/")
_AI_MODEL = os.environ.get("JLTAMP_AI_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
_PIPER_URL = os.environ.get("JLTAMP_PIPER_URL", "").strip().rstrip("/")
_PIPER_MODEL = os.environ.get("JLTAMP_PIPER_MODEL", "").strip()
_STEM_URL = os.environ.get("JLTAMP_STEM_URL", "").strip().rstrip("/")

# Zo lang mag de stemdienst over één aankondiging doen. Daarna wordt het Piper:
# de app vraagt 40 s voor het einde en wil het geluid er ruim voor de overgang
# hebben, en het model heeft dan ook al een paar seconden geschreven.
_STEM_GEDULD_S = 24.0


def _tempo() -> float:
    try:
        return max(0.6, min(1.5, float(os.environ.get("JLTAMP_PIPER_TEMPO", "0.95"))))
    except ValueError:
        return 0.95


# Een aankondiging is kort. Twaalf seconden is ruim dertig woorden in een
# rustig spreektempo; alles daarboven is geen overgang meer maar een praatje.
MAX_WOORDEN = 32
MAX_SECONDEN = 13.0

# Hoeveel aankondigingen er tegelijk gemaakt mogen worden. Het model is één
# machine voor het hele huis (en voor de AI-weeklijsten); twee tegelijk is
# genoeg, de rest krijgt "even niet" en de muziek loopt gewoon door.
_BEZIG = threading.BoundedSemaphore(2)

# Per gebruiker niet vaker dan dit. De app vraagt één keer per nummer; dit vangt
# een app die door een fout in een lus belandt.
_MIN_TUSSENPOZE = 15.0
_laatste: dict[int, float] = {}
_laatste_slot = threading.Lock()

# Dezelfde overgang twee keer vragen (een herstart van de app, een tweede
# toestel) hoeft het model niet twee keer te laten schrijven.
_TEKST_GELDIG = 15 * 60
_teksten: dict[tuple, tuple[float, str, str]] = {}

# Man en vrouw om de beurt, per luisteraar, zoals een duo achter de microfoon.
_beurt: dict[int, int] = {}


def _wie_spreekt(user_id: int) -> str:
    n = _beurt.get(user_id, random.randrange(2))
    _beurt[user_id] = n + 1
    return ("man", "vrouw")[n % 2]

# Oude geluidjes opruimen: na een week hoort niemand ze nog terug.
_AUDIO_GELDIG = 7 * 24 * 3600
_MAX_BESTANDEN = 600


# ── Wat er beschikbaar is ────────────────────────────────────────────────────

_ai_stand: tuple[float, bool] = (0.0, False)


def _ai_bereikbaar() -> bool:
    """Staat het model aan? Eén minuut onthouden, de vraag kost een rondje net."""
    global _ai_stand
    nu = time.time()
    if nu - _ai_stand[0] < 60:
        return _ai_stand[1]
    ok = False
    try:
        with urllib.request.urlopen(f"{_AI}/api/tags", timeout=3) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _ai_stand = (nu, ok)
    return ok


_stem = None             # PiperVoice, als Piper hier zelf draait
_stem_slot = threading.Lock()


def _stem_in_huis():
    """Laad de Piper-stem in dit proces, één keer. None als dat niet kan."""
    global _stem
    if _stem is not None:
        return _stem
    if not _PIPER_MODEL or not Path(_PIPER_MODEL).exists():
        return None
    with _stem_slot:
        if _stem is None:
            try:
                from piper import PiperVoice  # type: ignore
                _stem = PiperVoice.load(_PIPER_MODEL)
                log.info("presentator: stem geladen uit %s", _PIPER_MODEL)
            except Exception as e:
                log.warning("presentator: stem laden mislukt: %s", e)
                return None
    return _stem


_stem_dienst_stand: tuple[float, bool, tuple] = (0.0, False, ())


def _stem_dienst_klaar(taal: str = "nl") -> bool:
    """Staat de stemdienst aan, heeft hij zijn model geladen, en spreekt hij
    deze taal? Eén minuut onthouden.

    Een stemdienst van vóór de talen meldt geen "talen"; die spreekt alleen
    Nederlands. Een Engelse tekst naar zo'n dienst sturen geeft een Engelse
    zin met een Nederlandse uitspraak, en dat is erger dan geen break."""
    global _stem_dienst_stand
    if not _STEM_URL:
        return False
    nu = time.time()
    if nu - _stem_dienst_stand[0] >= 60:
        ok, talen = False, ()
        try:
            with urllib.request.urlopen(f"{_STEM_URL}/gezond", timeout=3) as r:
                antwoord = json.loads(r.read().decode())
            ok = bool(antwoord.get("klaar"))
            talen = tuple(antwoord.get("talen") or ("nl",))
        except Exception:
            ok = False
        _stem_dienst_stand = (nu, ok, talen)
    return _stem_dienst_stand[1] and taal in _stem_dienst_stand[2]


def _stem_beschikbaar(taal: str = "nl") -> bool:
    if _stem_dienst_klaar(taal):
        return True
    # Piper heeft alleen een Nederlandse stem: geen terugval voor de rest.
    if taal != "nl":
        return False
    if _PIPER_URL:
        return True
    if not _PIPER_MODEL or not Path(_PIPER_MODEL).exists():
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


@router.get("/status")
def status(user: User = Depends(require_user), taal: str = "nl"):
    """Kan de presentator hier iets, in deze taal? De app laat de instelling dan zien."""
    db = SessionLocal()
    try:
        aan = schakelaar(db, "presentator")
    finally:
        db.close()
    stem = _stem_beschikbaar(taal_van(taal))
    ai = _ai_bereikbaar() if aan else False
    return {"aan": aan, "ai": ai, "stem": stem, "beschikbaar": aan and ai and stem}


# ── De feiten ────────────────────────────────────────────────────────────────

class Weer(BaseModel):
    temp: float


class AankondigingBody(BaseModel):
    vorig: str | None = None          # ratingKey van het nummer dat afloopt
    volgend: str                      # ratingKey van het nummer dat komt
    weer: Weer | None = None          # alleen als de luisteraar het weer aan heeft
    uur: int | None = None            # de klok van het toestel, niet van de server
    minuut: int | None = None
    weekdag: int | None = None        # 0 = maandag
    sessie: str | None = None         # samen-luisteren-code, voor de meeluisteraars
    # De taal van de app: nl, en, de, es of pt. Oude apps sturen niets: nl.
    # Iets onbekends wordt ook nl (zie presentator_talen.taal_van).
    taal: str = "nl"


def _track(db, user: User, rk: str | None) -> Track | None:
    if not rk:
        return None
    p = parse_key(rk)
    if not p or p[0] != "track":
        return None
    t = db.get(Track, p[1])
    if not t:
        return None
    libs = accessible_library_ids(db, user)
    if libs is not None and t.library_id not in libs:
        return None
    return t


def _artiest(t: Track) -> str:
    return (t.orig_artist or t.artist_name or "").strip()


_VERZAMEL = {"various artists", "various", "diverse artiesten", "va", "verschillende artiesten"}
_TOEVOEGSEL = re.compile(
    r"\s*[(\[][^)\]]*\b(featuring|feat|ft|with|remix|mix|edit|version|versie|acoustic|akoestisch|live|"
    r"remaster(ed)?|mono|stereo|demo|instrumental|radio|single|extended|bonus|deluxe|edition|"
    r"expanded|anniversary|originele|from|sessions?|re-?recording|re-?recorded|hi-res)\b[^)\]]*[)\]]"
    r"|\s*\(\d{4}\)|\s*\[[^\]]*\]",
    re.I)
_STAART = re.compile(
    r"\s+-\s+.*\b(featuring|feat|ft|remix|mix|edit|version|versie|acoustic|live|remaster(ed)?|mono|"
    r"stereo|demo|instrumental|radio|single|bonus)\b.*$", re.I)


def _beschaafd(s: str) -> str:
    """HOOFDLETTERS → Gewone Hoofdletters; Piper spelt anders 'OP' als O-P."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) > 3 and all(c.isupper() for c in letters):
        # Afkortingen met tekens erin (AC/DC, R&B) blijven zoals ze zijn.
        def woord(i: int, w: str) -> str:
            kern = w.strip("-'.,!?")
            if not kern.isalpha():
                return w
            if len(kern) <= 2 and i > 0:
                return w.lower()
            return w.replace(kern, kern.capitalize())
        return " ".join(woord(i, w) for i, w in enumerate(s.split()))
    return s


# Een verzamelalbum zegt niets over het nummer: "uit het album Top 100 over
# 2005" klopt niet, en het jaar is dat van de verzamelaar, niet van het nummer.
_VERZAMELALBUM = re.compile(
    r"\b(top ?\d+|hits|best of|greatest|collection|verzamel|compilation|anthology|"
    r"the essential|piepschuim|club|now that|radio \d|jaren \d|\d0s|\d0's)\b", re.I)


def _verzamelalbum(t: Track) -> bool:
    return ((t.album_artist or "").strip().lower() in _VERZAMEL
            or bool(_VERZAMELALBUM.search(t.album_title or "")))


def spreek_artiest(t: Track) -> str:
    """De artiest zoals je hem uitspreekt; leeg bij een verzamelalbum."""
    a = _beschaafd(_artiest(t))
    return "" if a.lower() in _VERZAMEL else a


def spreek_titel(t: Track) -> str:
    """De titel zoals een dj hem uitspreekt: zonder tracknummer, artiest ervoor,
    'feat.', '(Remastered)', '(Film Version)' en dergelijke."""
    titel = (t.title or "").strip()
    titel = re.sub(r"^\d{1,3}\s*[-.]\s+", "", titel)
    a = _artiest(t)
    if a and titel.lower().startswith(a.lower()):
        rest = re.sub(r"^\s*[-–:]\s*", "", titel[len(a):])
        if rest:
            titel = rest
    kaal = _STAART.sub("", _TOEVOEGSEL.sub("", titel)).strip(" -–")
    return _beschaafd(kaal or titel)


_UREN = ["twaalf", "een", "twee", "drie", "vier", "vijf", "zes", "zeven",
         "acht", "negen", "tien", "elf"]


def tijd_in_woorden(uur: int, minuut: int) -> str:
    """'kwart over negen' — zoals je het zegt, niet zoals het op de klok staat.

    Afgerond op vijf minuten; een presentator zegt niet "zeventien over negen".
    """
    m = int(round(minuut / 5.0) * 5)
    if m == 60:
        uur, m = uur + 1, 0
    h = _UREN[uur % 12]
    volgend = _UREN[(uur + 1) % 12]
    return {
        0: f"{h} uur",
        5: f"vijf over {h}", 10: f"tien over {h}", 15: f"kwart over {h}",
        20: f"tien voor half {volgend}", 25: f"vijf voor half {volgend}",
        30: f"half {volgend}",
        35: f"vijf over half {volgend}", 40: f"tien over half {volgend}",
        45: f"kwart voor {volgend}", 50: f"tien voor {volgend}", 55: f"vijf voor {volgend}",
    }[m]


def _bio_kort(naam: str, taal: str = "nl") -> str:
    """De eerste twee zinnen uit Wikipedia, of niets.

    Eerst wat artistinfo al in zijn cache heeft (Nederlands, dan Engels). Is er
    nog niets, dan één opvraging in het Engels — níet in het Nederlands, want
    dan gaat artistinfo bij een ontbrekend artikel vertalen, en dat is nog een
    rondje over hetzelfde model dat deze tekst moet schrijven.
    """
    if not naam:
        return ""
    bio = ""
    # Voor een andere taal dan nl alleen het Engelse artikel: een Nederlandse
    # inleiding in de feiten trekt het model naar het Nederlands.
    for bron_taal in (("nl", "en") if taal == "nl" else ("en",)):
        f = artistinfo.CACHE / artistinfo._sleutel(naam, bron_taal)
        if f.exists():
            try:
                bio = json.loads(f.read_text(encoding="utf-8")).get("bio") or ""
            except Exception:
                bio = ""
            if bio:
                break
    if not bio:
        try:
            # De gebruiker doet er voor artist_info niet toe; hij vraagt hem
            # alleen omdat het een route is.
            bio = (artistinfo.artist_info(name=naam, lang="en", user=None) or {}).get("bio") or ""
        except Exception as e:
            log.debug("presentator: geen biografie voor %s: %s", naam, e)
            bio = ""
    # artistinfo zoekt op naam en pakt soms een artikel over een andere
    # artiest met een langere naam ("Black Box" → "The Black Box Revelation").
    # De naam moet erin staan, en niet als begin van een langere naam.
    m = re.search(rf"(?<!\w){re.escape(naam)}(?!\w)", bio, re.I)
    # En het moet over muziek gaan, niet over de stad Boston of een spookstad.
    # Zonder \b vooraan: 'popzangeres', 'rockband' en 'volkszanger' tellen mee.
    muziek = re.search(r"(band|zanger|zangeres|singer|rapper|musician|muzikant|artiest|artist|"
                       r"groep|group|duo|componist|composer|songwriter|dj|producer|"
                       r"album|orkest|orchestra|koor|choir)\b", bio, re.I)
    # Een langere naam is bij een persoon gewoon de volledige naam (Anouk
    # Teeuwe); bij een band is het een andere band.
    langer = bool(re.match(r"\s+[A-Z][a-z]", bio[m.end():])) if m else False
    groep = re.search(r"(band|groep|group|duo|trio)\b", bio[:200], re.I)
    if not m or not muziek or (langer and groep):
        if bio:
            log.info("presentator: biografie lijkt niet over %s te gaan", naam)
        return ""
    zinnen = re.split(r"(?<=[.!?])\s+", bio.strip())
    return " ".join(zinnen[:2])[:420]


def _meeluisteraars(code: str | None, user: User) -> list[str]:
    """Wie er via samen luisteren meedoet, zonder jezelf. Alleen als jij er ook in zit."""
    if not code:
        return []
    try:
        from .session import MANAGER
        s = MANAGER.get(code)
        if not s or not any(p.user_id == user.id for p in s.participants.values()):
            return []
        namen: dict[int, str] = {}
        for p in s.participants.values():
            if p.user_id != user.id and p.name:
                namen.setdefault(p.user_id, p.name.split()[0])
        return list(namen.values())[:4]
    except Exception:
        return []


def _feiten(db, user: User, body: AankondigingBody) -> tuple[dict, list[str]] | None:
    """Wat de presentator mag weten, en welke invalshoeken daarmee kunnen."""
    volgend = _track(db, user, body.volgend)
    if not volgend:
        return None
    vorig = _track(db, user, body.vorig)
    if body.uur is not None and 0 <= body.uur <= 23:
        mn = body.minuut if body.minuut is not None and 0 <= body.minuut <= 59 else 0
        uur = body.uur
    else:
        lt = time.localtime()
        uur, mn = lt.tm_hour, lt.tm_min
    weekdag = body.weekdag if body.weekdag is not None and 0 <= body.weekdag <= 6 else None
    temp = body.weer.temp if body.weer else None
    return feiten_voor(volgend, vorig, uur, mn, weekdag, temp,
                       _meeluisteraars(body.sessie, user), taal_van(body.taal))


def feiten_voor(volgend: Track, vorig: Track | None, uur: int, mn: int,
                weekdag: int | None = None, temp: float | None = None,
                mee: list[str] | None = None, taal: str = "nl") -> tuple[dict, list[str]]:
    """De feiten bij een overgang, los van wie erom vraagt.

    Gedeeld door de route hierboven (de app van de luisteraar) en JLTamp Radio
    (app/radio), die zelf weet welk nummer er komt en hoe laat het dan is.
    Tijd, dagdeel, dag en temperatuur staan er in `taal` in: het model neemt
    ze letterlijk over, en een Nederlandse tijd in een Engelse zin is precies
    wat niet mag.
    """
    def nummer(t: Track) -> dict:
        d = {"titel": spreek_titel(t), "artiest": spreek_artiest(t)}
        if not _verzamelalbum(t):
            album = _STAART.sub("", _TOEVOEGSEL.sub("", t.album_title or "")).strip(" -–")
            if album and album.lower() != d["titel"].lower():
                d["album"] = _beschaafd(album)
            if t.year:
                d["jaar"] = t.year
        if t.genre:
            d["genre"] = t.genre.split(";")[0].split("/")[0].strip()
        return d

    f: dict = {"volgend": nummer(volgend)}
    hoeken = ["volgend"]
    if vorig and vorig.id != volgend.id:
        # Alleen titel en artiest: met het album en jaar erbij ging het model
        # die door elkaar halen met die van het volgende nummer.
        f["vorig"] = {"titel": spreek_titel(vorig), "artiest": spreek_artiest(vorig)}

    nu = time.time()
    if volgend.added_at and nu - volgend.added_at < 14 * 86400:
        f["volgend"]["netBinnen"] = True
        hoeken += ["nieuw", "nieuw"]      # dubbel: dit is nieuws, dat mag vaker

    bio = _bio_kort(spreek_artiest(volgend), taal)
    if bio:
        f["overArtiest"] = bio
        hoeken += ["artiest", "artiest"]

    f["tijd"] = talen.tijd(uur, mn, taal)
    f["dagdeel"] = talen.dagdeel(uur, taal)
    if weekdag is not None and 0 <= weekdag <= 6:
        f["dag"] = talen.dag(weekdag, taal)
    hoeken.append("tijd")

    if temp is not None and -50 <= temp <= 60:
        f["temperatuurBuiten"] = talen.graden(temp, taal)
        hoeken.append("weer")

    if mee:
        f["luistertMee"] = mee
        hoeken += ["meeluisteraars", "meeluisteraars"]

    return f, hoeken


# ── De tekst ─────────────────────────────────────────────────────────────────
#
# Een dj zegt niet elke keer hetzelfde. Eerst stond hier één sjabloon ("Dat was
# X. Nu Y."), en na een uur luisteren hoorde je alleen nog dat sjabloon. Daarom
# nu twee keuzes per aankondiging:
#
#   de VORM   hoe de break in elkaar zit: terugblik én vooruitblik, alleen
#             vooruit, alleen terug, of (op de radio, met twee stemmen) een
#             kort gesprekje tussen de man- en de vrouwenstem;
#   de HOEK   waar het over gaat: het nummer, de artiest (een feitje uit de
#             Wikipedia-inleiding), de tijd, het weer, wie er meeluistert.
#
# En een geheugen (Geheugen hieronder) van de laatste openingszinnen en vormen,
# zodat dezelfde opening niet twee keer kort na elkaar valt. De feitencheck
# (_waarom_niet) blijft voor elke vorm dezelfde: geen getal en geen lofwoord
# dat niet in de feiten staat.

_HOEK_UITLEG = {
    "volgend": "vertel iets over het volgende nummer uit de feiten (album, jaar of genre)",
    "nieuw": "vertel dat het volgende nummer net nieuw binnen is in de verzameling",
    "artiest": "vertel één ding over de artiest van het volgende nummer, uit 'overArtiest'",
    "tijd": "noem terloops hoe laat het is of welk dagdeel het is",
    "weer": "noem terloops de temperatuur buiten",
    "meeluisteraars": "groet wie er meeluistert (bij voornaam)",
    # Alleen de radio kiest deze, rond het hele uur (zie app/radio/planning.py).
    "uur": "zeg aan het begin hoe laat het is, precies zoals in 'tijd' staat, zoals op "
           "de radio rond het hele uur",
    # Alleen de radio: het begin van een nieuw programma (app/radio/programma.py).
    "programma": "zeg dat nu het programma uit 'programma' begint, met de naam letterlijk "
                 "zoals die er staat, en in een paar woorden wat er te horen is (uit 'wat'). "
                 "Noem eerst hoe laat het is als 'tijd' er staat",
}

# Hoeken die niet over het volgende nummer gaan. Alleen die passen bij een
# break die het volgende nummer niet noemt (de vorm "terug").
_HOEK_ZONDER_VOLGEND = {"tijd", "weer", "meeluisteraars"}

VORMEN = ("terug_vooruit", "vooruit", "terug", "dialoog")


class Geheugen:
    """Wat er onlangs gezegd is: openingen en vormen.

    De app-presentator heeft er één voor alle luisteraars samen (_GEHEUGEN), de
    radio heeft een eigen. Klein en begrensd: alleen de laatste paar."""

    def __init__(self, openingen: int = 8, vormen: int = 3):
        self._openingen: deque[str] = deque(maxlen=openingen)
        self._vormen: deque[str] = deque(maxlen=vormen)
        self._slot = threading.Lock()

    @staticmethod
    def opening(tekst: str) -> str:
        """De eerste twee woorden, zonder hoofdletters en leestekens."""
        return " ".join(_kaal(tekst).split()[:2])

    def herhaalt(self, tekst: str) -> bool:
        o = self.opening(tekst)
        with self._slot:
            return bool(o) and o in self._openingen

    def recente_openingen(self) -> list[str]:
        with self._slot:
            return list(self._openingen)

    def laatste_vorm(self) -> str | None:
        with self._slot:
            return self._vormen[-1] if self._vormen else None

    def onthoud(self, tekst: str, vorm: str) -> None:
        o = self.opening(tekst)
        with self._slot:
            if o:
                self._openingen.append(o)
            self._vormen.append(vorm)


_GEHEUGEN = Geheugen()


def kies_vorm_en_hoek(feiten: dict, hoeken: list[str], geheugen: Geheugen,
                      dialoog: bool = False, hoek: str | None = None,
                      rng: random.Random | None = None) -> tuple[str, str]:
    """Welke vorm en welke hoek deze keer. Nooit twee keer achter elkaar
    dezelfde vorm, als er iets anders kan.

    `hoek` dwingt een hoek af (de radio rond het hele uur: "uur"). `dialoog`
    mag alleen als er twee echte stemmen zijn — de radio vraagt erom, de app
    niet."""
    r = rng or random
    hoek = hoek or r.choice(hoeken)
    gewicht = {"vooruit": 3.0}
    if "vorig" in feiten:
        gewicht["terug_vooruit"] = 4.5
        if hoek in _HOEK_ZONDER_VOLGEND:
            gewicht["terug"] = 2.0
    if dialoog and hoek != "uur":
        gewicht["dialoog"] = 2.5
    vorig = geheugen.laatste_vorm()
    if vorig in gewicht and len(gewicht) > 1:
        del gewicht[vorig]
    vormen = list(gewicht)
    return r.choices(vormen, weights=[gewicht[v] for v in vormen])[0], hoek


def _wie(n: dict, taal: str = "nl") -> str:
    return talen.wie(n, taal)


# Hoe de terugblik begint staat per taal in presentator_talen.TERUG_BEGIN. Door
# de code gekozen (het model zelf koos negen van de tien keer "Je hoorde net"),
# en niet een opening die net nog viel.


def _vers(opties: list[str], geheugen: Geheugen, rng=None) -> str:
    """Een van de opties waarvan de opening niet net nog gezegd is."""
    r = rng or random
    vrij = [o for o in opties if not geheugen.herhaalt(o)]
    return r.choice(vrij or opties)


_SPREEKTAAL = (
    "- Praat zoals een ontspannen Nederlandse radio-dj: kort, spreektaal, 'je' en "
    "'we', gewone woorden. Geen schrijftaal, geen opsommingen, geen superlatieven, "
    "geen uitroeptekens, geen emoji, geen hashtags.\n"
)

_FEITREGELS = (
    "- Haal 'vorig' en 'volgend' niet door elkaar: het album, jaar en genre "
    "horen bij het volgende nummer. Staat er geen jaar of album, noem er dan geen.\n"
    "- Gebruik ALLEEN de feiten hieronder. Verzin niets: geen jaartallen, "
    "hitnoteringen, albums, prijzen of weetjes die er niet staan, en geen eigen "
    "beschrijving van hoe de muziek klinkt of waar ze naar verwijst. Noem iets "
    "geen hit, klassieker of succes als dat niet in de feiten staat.\n"
    "- Noem titels en namen precies zoals ze er staan.\n"
    "- Over het weer weet je alleen de temperatuur: niets over zon, regen of wind.\n"
    "- Je hoeft niet alle feiten te gebruiken; één of twee is genoeg.\n"
)


def _opdracht(feiten: dict, hoek: str, kort: bool = False, vorm: str = "terug_vooruit",
              geheugen: Geheugen | None = None, begin: str | None = None,
              taal: str = "nl") -> str:
    """De opdracht voor het model. Gewone instructies, niets dat alleen bij één
    model werkt: het model op de Studio kan gewisseld worden (JLTAMP_AI_MODEL).
    Nederlands in het Nederlands; de andere talen via _opdracht_anders."""
    if taal != "nl":
        return _opdracht_anders(feiten, hoek, kort, vorm, geheugen, begin, taal)
    v = feiten["volgend"]
    woorden = 16 if kort else MAX_WOORDEN - 8
    begin = begin or "Dat was"
    if vorm == "terug_vooruit":
        opbouw = (
            f"- Begin met \"{begin} {_wie(feiten['vorig'])}\" (zo, of een woordje anders als "
            "de zin dat vraagt). Zeg over dat vorige nummer verder NIETS: geen sfeer, "
            "geen beschrijving, geen jaar. "
            f"Kondig daarna {_wie(v)} aan, met één ding volgens de invalshoek.\n"
            "- Noem de titel van het volgende nummer precies ÉÉN keer, in één lopende zin.\n"
        )
    elif vorm == "terug":
        opbouw = (
            f"- Begin met \"{begin} {_wie(feiten['vorig'])}\" en zeg daarna één ding volgens "
            "de invalshoek. Zeg over dat vorige nummer verder niets. Het volgende "
            "nummer noem je NIET; dat begint vanzelf.\n"
        )
    elif vorm == "dialoog":
        terug = (f"A blikt kort terug op {_wie(feiten['vorig'])}, of zegt iets volgens de "
                 "invalshoek. " if "vorig" in feiten else "A zegt iets volgens de invalshoek. ")
        opbouw = (
            "- Je schrijft voor TWEE presentatoren, A en B, die even met elkaar praten. "
            "Precies twee regels: de eerste begint met 'A:', de tweede met 'B:'. "
            f"Elke regel hooguit {9 if kort else 14} woorden. "
            + terug +
            f"B reageert daar kort op en kondigt {_wie(v)} aan.\n"
            "- De titel van het volgende nummer staat in totaal precies ÉÉN keer in de "
            "twee regels.\n"
        )
        woorden = MAX_WOORDEN
    else:  # vooruit
        opbouw = (
            f"- Zeg één ding volgens de invalshoek en kondig {_wie(v)} aan. Noem geen "
            "vorig nummer.\n"
            "- Noem de titel van het volgende nummer precies ÉÉN keer, in één lopende zin. "
            "Niet eerst erover vertellen en dan nog eens aankondigen.\n"
        )
    recent = (geheugen or _GEHEUGEN).recente_openingen()
    vermijd = ""
    if recent and vorm not in ("terug_vooruit", "terug"):
        vermijd = ("- Begin NIET met een van deze openingen, die zijn net gebruikt: "
                   + ", ".join(f"'{o}'" for o in recent) + ".\n")
    return (
        "Je bent de presentator van JLTamp, een kleine Nederlandse thuisradio. "
        "Schrijf wat je zegt tussen twee nummers in.\n\n"
        "Regels:\n"
        f"- Hooguit {woorden} woorden, één of twee korte zinnen. Het wordt hardop "
        "voorgelezen en moet binnen tien seconden klaar zijn.\n"
        + opbouw + _SPREEKTAAL + vermijd +
        "- Schrijf hele, lopende zinnen. Nooit een titel of naam als losse zin, en geen "
        "'Volgend nummer:' als opsomming.\n"
        + _FEITREGELS +
        "- Begin niet met 'Hallo' of 'Goedemorgen' tenzij je iemand groet.\n"
        "- Geef ALLEEN de gesproken tekst terug, zonder aanhalingstekens of uitleg.\n\n"
        f"Invalshoek: {_HOEK_UITLEG.get(hoek, _HOEK_UITLEG['volgend'])}.\n\n"
        f"Feiten (JSON):\n{json.dumps(feiten, ensure_ascii=False)}\n"
    )


def _opdracht_anders(feiten: dict, hoek: str, kort: bool, vorm: str,
                     geheugen: Geheugen | None, begin: str | None, taal: str) -> str:
    """Dezelfde opdracht voor en/de/es/pt. De regels in het Engels (dat volgt
    elk model het best), de toon in de taal zelf (talen.SPREEKTAAL), en twee
    keer met nadruk in welke taal het antwoord moet."""
    v = feiten["volgend"]
    naam = talen.TAALNAAM[taal]
    woorden = 16 if kort else MAX_WOORDEN - 8
    begin = begin or talen.TERUG_BEGIN[taal][0]
    if vorm == "terug_vooruit":
        opbouw = (
            f"- Start with \"{begin} {_wie(feiten['vorig'], taal)}\" (like that, or a word "
            "different if the sentence needs it). Say NOTHING else about that previous song. "
            f"Then announce {_wie(v, taal)}, with one thing from the angle below.\n"
            "- Mention the title of the next song exactly ONCE, in one flowing sentence.\n"
        )
    elif vorm == "terug":
        opbouw = (
            f"- Start with \"{begin} {_wie(feiten['vorig'], taal)}\" and then say one thing "
            "from the angle below. Do NOT mention the next song.\n"
        )
    elif vorm == "dialoog":
        opbouw = (
            "- Write for TWO presenters, A and B, chatting briefly. Exactly two lines: the "
            f"first starts with 'A:', the second with 'B:'. Each line at most {9 if kort else 14} "
            f"words. B announces {_wie(v, taal)}. The title of the next song appears exactly "
            "ONCE in total.\n"
        )
        woorden = MAX_WOORDEN
    else:
        opbouw = (
            f"- Say one thing from the angle below and announce {_wie(v, taal)}. Do not "
            "mention a previous song.\n"
            "- Mention the title of the next song exactly ONCE, in one flowing sentence.\n"
        )
    recent = (geheugen or _GEHEUGEN).recente_openingen()
    vermijd = ""
    if recent and vorm not in ("terug_vooruit", "terug"):
        vermijd = ("- Do NOT start with any of these openings, they were just used: "
                   + ", ".join(f"'{o}'" for o in recent) + ".\n")
    return (
        f"You are the presenter of JLTamp, a small home radio. Write what you say between "
        f"two songs. Write it in {naam}, and ONLY in {naam}.\n\n"
        "Rules:\n"
        f"- At most {woorden} words, one or two short sentences. It is read aloud and must "
        "be done within ten seconds.\n"
        + opbouw +
        f"- {talen.SPREEKTAAL[taal]}\n" + vermijd +
        "- Whole, flowing sentences. Never a title or name as a sentence on its own.\n"
        "- The keys in the facts are Dutch; 'vorig' is the previous song, 'volgend' the "
        "next. Album, year and genre belong to the next song. No year or album in the "
        "facts: mention none.\n"
        "- Use ONLY the facts below. Invent nothing: no years, chart positions, albums, "
        "awards or trivia that are not there, and no description of how the music sounds. "
        "Do not call anything a hit, classic or success unless the facts say so.\n"
        "- Use titles and names exactly as written. Use the time ('tijd'), part of day "
        "('dagdeel') and day ('dag') exactly as written; they are already in "
        f"{naam}.\n"
        "- About the weather you only know the temperature.\n"
        "- No exclamation marks, no emoji, no hashtags, no superlatives.\n"
        f"- Return ONLY the spoken text in {naam}, without quotes or explanation.\n\n"
        f"Angle: {talen.HOEK_UITLEG_EN.get(hoek, talen.HOEK_UITLEG_EN['volgend'])}.\n\n"
        f"Facts (JSON):\n{json.dumps(feiten, ensure_ascii=False)}\n"
    )


def _schoon(tekst: str) -> str:
    tekst = re.sub(r"<think>.*?</think>", "", tekst, flags=re.S)
    tekst = tekst.strip().strip('"“”„\'').strip()
    tekst = re.sub(r"[*_#`~]|[\U0001F000-\U0001FAFF☀-➿]", "", tekst)   # emoji
    tekst = re.sub(r"[\"“”„«»]|(?<!\w)'|'(?!\w)", "", tekst)               # aanhalingstekens
    tekst = re.sub(r"\s+", " ", tekst)
    tekst = re.sub(r"!+", ".", tekst)
    tekst = re.sub(r"\s*(\.\.\.|…)\s*", ", ", tekst)
    tekst = re.sub(r"([.,?]),\s*", r"\1 ", tekst).strip(" ,")
    return tekst.strip()


def _delen(ruw: str, vorm: str) -> list[str]:
    """Het antwoord van het model als lijst gesproken stukken: één, of bij een
    dialoog twee (A en B). Een dialoog die niet netjes in twee regels A/B
    terugkomt is leeg — dan gaat hij niet door."""
    ruw = re.sub(r"<think>.*?</think>", "", ruw or "", flags=re.S)
    if vorm != "dialoog":
        return [_schoon(ruw)]
    stukken = re.findall(r"(?:^|\n)[ \t*]*([AB])[ \t*]*:[ \t]*(.+)", ruw.strip())
    if [s[0] for s in stukken] != ["A", "B"]:
        return []
    return [_schoon(s[1]) for s in stukken]


def _kaal(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _telt(tekst: str, naam: str) -> int:
    """Hoe vaak `naam` als geheel in `tekst` voorkomt, zonder op leestekens te letten."""
    n = _kaal(naam)
    if not n:
        return 0
    return len(re.findall(rf"(?<!\w){re.escape(n)}(?!\w)", _kaal(tekst)))


# Woorden waarmee het model graag iets aandikt dat niet in de feiten staat.
# Daarnaast de sfeerwoorden: het model beschrijft graag hoe iets klinkt, en dat
# weet het niet.
# Dezelfde lijst voor alle vijf talen staat in presentator_talen.LOF.
_LOF = talen.LOF


def _waarom_niet(delen: list[str] | str, feiten: dict, vorm: str = "vooruit",
                 geheugen: Geheugen | None = None, taal: str = "nl") -> str | None:
    """Wat er mis is met deze tekst, in een zin die het model bij de volgende
    poging te lezen krijgt; None als hij goed is. Geen verzonnen getallen of
    lofwoorden, niet te lang, niet leeg, het volgende nummer precies één keer
    (bij "terug" hooguit één keer), het vorige erin als daarom gevraagd is, en
    geen opening die net nog gebruikt is."""
    if isinstance(delen, str):
        delen = [delen]
    if not delen or any(not d or len(d) < 8 for d in delen):
        return ("er kwamen geen twee regels A: en B:" if vorm == "dialoog"
                else "er kwam geen tekst")
    tekst = " ".join(delen)
    if vorm == "dialoog":
        if any(len(d.split()) > 18 for d in delen):
            return "een van de regels was te lang"
    elif len(tekst.split()) > MAX_WOORDEN:
        return "hij was te lang"
    v = feiten["volgend"]
    titel = v.get("titel") or ""
    # Een titel die ook gewoon een woord is ("Hello", "Home") of in de
    # artiestnaam zit, telt te makkelijk dubbel; dan alleen "niet weg".
    gewoon = len(_kaal(titel)) < 6 or _kaal(titel) in _kaal(v.get("artiest") or "")
    keer = _telt(tekst, titel)
    if vorm == "terug":
        if keer > 1 and not gewoon:
            return f"de titel {titel} stond er {keer} keer in; noem het volgende nummer niet"
    elif keer == 0 or (keer > 1 and not gewoon):
        log.info("presentator: titel %d keer in %r", keer, tekst)
        return (f"de titel {titel} stond er niet in" if keer == 0
                else f"de titel {titel} stond er {keer} keer in, dat moet één keer zijn")
    if vorm in ("terug_vooruit", "terug"):
        w = feiten.get("vorig") or {}
        if not (_telt(tekst, w.get("titel") or "") or _telt(tekst, w.get("artiest") or "")):
            log.info("presentator: vorig nummer ontbreekt in %r", tekst)
            return f"het vorige nummer ({_wie(w, taal)}) werd niet genoemd"
    # Bij de terugblik kiest de code de opening zelf (al vers); daar niet nog
    # eens op afkeuren.
    if vorm not in ("terug_vooruit", "terug") and geheugen and geheugen.herhaalt(delen[0]):
        return (f"de opening '{geheugen.opening(delen[0])}' is net nog gebruikt; "
                "begin anders")
    if taal != "nl":
        namen = [x for n in (v, feiten.get("vorig") or {}) for x in (n.get("titel"), n.get("artiest"))]
        if talen.klinkt_nederlands(tekst, namen):
            log.info("presentator: Nederlands in een %s-tekst: %r", taal, tekst)
            return f"the text contains Dutch words; write it in {talen.TAALNAAM[taal]} only"
    bron = json.dumps(feiten, ensure_ascii=False)
    for woord in _LOF.findall(tekst):
        if not re.search(rf"\b{re.escape(woord)}", bron, re.I):
            log.info("presentator: %r staat niet in de feiten: %r", woord, tekst)
            return (f"het woord '{woord}' staat niet in de feiten; beschrijf niet hoe "
                    "het nummer klinkt of hoe goed het is")
    for getal in re.findall(r"\d+", tekst):
        if getal not in bron:
            log.info("presentator: verzonnen getal %s in %r", getal, tekst)
            return f"het getal {getal} staat niet in de feiten"
    return None


def _vast(feiten: dict, vorm: str = "terug_vooruit", geheugen: Geheugen | None = None,
          rng=None, taal: str = "nl") -> str:
    """Als het model twee keer iets verzint: een gewone zin uit de feiten zelf.
    Saai, maar hij klopt altijd. Wel een opening die niet net nog viel, en in
    de taal van de app (presentator_talen.vaste_zinnen)."""
    g = geheugen or _GEHEUGEN
    v = _wie(feiten["volgend"], taal)
    w = _wie(feiten["vorig"], taal) if "vorig" in feiten else None
    return _vers(talen.vaste_zinnen(vorm, v, w, taal), g, rng)


# Ollama kent "think": false om het nadenken van een denkend model over te
# slaan. Een model dat dat niet kent, geeft daar bij sommige versies een fout
# op; dan vanaf dat moment zonder. Het antwoord gaat altijd nog door _schoon,
# die een <think>-blok weghaalt — zo werkt het met elk model.
_zonder_think = False


def _vraag_model(prompt: str, temperatuur: float) -> str | None:
    """Eén antwoord van het model, of None. Het model komt uit JLTAMP_AI_MODEL."""
    global _zonder_think
    for poging in range(2):
        verzoek_data = {
            "model": _AI_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperatuur, "num_predict": 160},
        }
        if not _zonder_think:
            verzoek_data["think"] = False
        try:
            verzoek = urllib.request.Request(
                f"{_AI}/api/generate", data=json.dumps(verzoek_data).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(verzoek, timeout=45) as r:
                return json.loads(r.read().decode()).get("response") or ""
        except urllib.error.HTTPError as e:
            if e.code == 400 and not _zonder_think and poging == 0:
                log.info("presentator: model kent 'think' niet, verder zonder")
                _zonder_think = True
                continue
            log.info("presentator: model gaf %s", e.code)
            return None
        except Exception as e:
            log.info("presentator: model gaf geen antwoord: %s", e)
            return None
    return None


def _schrijf(feiten: dict, hoek: str, kort: bool = False, vorm: str = "vooruit",
             geheugen: Geheugen | None = None, begin: str | None = None,
             taal: str = "nl") -> list[str] | None:
    """De gesproken stukken (één, of twee bij een dialoog), of None."""
    opdracht = _opdracht(feiten, hoek, kort, vorm, geheugen, begin, taal)
    uit, reden = "", ""
    for poging in range(2):
        # De tweede keer met de reden erbij: zonder die zin schreef het model
        # vier keer achter elkaar hetzelfde "met een zachte sfeer".
        ruw = _vraag_model(
            opdracht if poging == 0 else
            f"{opdracht}\nJe vorige versie was: {uit}\nDie is afgekeurd: {reden}. "
            "Schrijf een nieuwe versie zonder die fout.\n",
            0.75 if poging == 0 else 0.4)
        if ruw is None:
            return None
        delen = _delen(ruw, vorm)
        uit = " / ".join(delen) if delen else _schoon(ruw)
        reden = _waarom_niet(delen, feiten, vorm, geheugen, taal)
        if reden is None:
            return delen
    return None


# ── De stem ──────────────────────────────────────────────────────────────────

_EENHEDEN = ["nul", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen",
             "tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien",
             "zeventien", "achttien", "negentien"]
_TIENTALLEN = ["", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig",
               "tachtig", "negentig"]


def _onder_honderd(n: int) -> str:
    if n < 20:
        return _EENHEDEN[n]
    t, e = divmod(n, 10)
    if not e:
        return _TIENTALLEN[t]
    en = "ën" if _EENHEDEN[e].endswith("e") else "en"     # tweeëntwintig, drieëntwintig
    return f"{_EENHEDEN[e]}{en}{_TIENTALLEN[t]}"


def getal_in_woorden(n: int) -> str:
    """Hoe je een getal op de radio zegt. Jaartallen zoals een mens ze zegt:
    1975 → negentien vijfenzeventig, 2013 → tweeduizend dertien."""
    if n < 100:
        return _onder_honderd(n)
    if 1100 <= n <= 1999:
        if n % 100 == 0:
            return f"{_onder_honderd(n // 100)}honderd"            # negentienhonderd
        return f"{_onder_honderd(n // 100)} {_onder_honderd(n % 100)}"
    if n < 1000:
        h, r = divmod(n, 100)
        kop = "honderd" if h == 1 else f"{_EENHEDEN[h]}honderd"
        return kop + (_onder_honderd(r) if r else "")
    if n < 10000:
        d, r = divmod(n, 1000)
        kop = "duizend" if d == 1 else f"{_EENHEDEN[d]}duizend"
        return kop + (f" {getal_in_woorden(r)}" if r else "")
    return str(n)


def _uitspreekbaar(tekst: str, taal: str = "nl") -> str:
    """Getallen voluit voor de stemdienst: Chatterbox maakte van '1975' hoorbaar
    'negentien zevenenvijftig'. Piper leest ze zelf goed voor, die krijgt de tekst
    zoals hij is. Alleen in het Nederlands: in een andere taal blijven de cijfers
    staan (een Nederlands telwoord in een Engelse zin is erger)."""
    if taal != "nl":
        return tekst
    return re.sub(r"\b\d{1,4}\b", lambda m: getal_in_woorden(int(m.group())), tekst)


def _spreek_echt(tekst: str, stem: str, taal: str = "nl") -> bytes | None:
    """Tekst → WAV via de stemdienst, of None (dan wordt het Piper, alleen in nl)."""
    try:
        verzoek = urllib.request.Request(
            f"{_STEM_URL}/spreek",
            data=json.dumps({"tekst": _uitspreekbaar(tekst, taal), "stem": stem,
                             "taal": taal}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(verzoek, timeout=_STEM_GEDULD_S) as r:
            data = r.read()
        if data[:4] == b"RIFF":
            return data
    except Exception as e:
        log.info("presentator: stemdienst gaf geen geluid (%s), dan Piper", e)
    return None


def _spreek_wav(tekst: str) -> bytes | None:
    """Tekst → WAV, via de Piper-dienst of via Piper in dit proces."""
    if _PIPER_URL:
        # piper.http_server luistert sinds 1.3 op /synthesize; de oudere op /.
        for pad in ("/synthesize", "/"):
            try:
                verzoek = urllib.request.Request(
                    _PIPER_URL + pad,
                    data=json.dumps({"text": tekst, "length_scale": _tempo()}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(verzoek, timeout=30) as r:
                    data = r.read()
                if data[:4] == b"RIFF":
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (404, 405):
                    continue
                log.info("presentator: Piper-dienst gaf %s", e.code)
                return None
            except Exception as e:
                log.info("presentator: Piper-dienst gaf geen geluid: %s", e)
                return None
        return None
    stem = _stem_in_huis()
    if stem is None:
        return None
    try:
        from piper import SynthesisConfig  # type: ignore
        cfg = SynthesisConfig(length_scale=_tempo())
    except Exception:
        cfg = None
    buf = io.BytesIO()
    try:
        with _stem_slot, wave.open(buf, "wb") as w:
            if cfg is not None:
                stem.synthesize_wav(tekst, w, syn_config=cfg)
            else:
                stem.synthesize_wav(tekst, w)
        return buf.getvalue()
    except Exception as e:
        log.info("presentator: Piper mislukte: %s", e)
        return None


def _duur(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


def _bewaar(sleutel: str, wav: bytes) -> Path:
    """Als mp3 als ffmpeg er is (een tiende van de grootte, telt op 4G), anders wav.

    Meteen gelijkgetrokken op ongeveer het niveau van de muziek (-16 LUFS), zodat
    de stem niet fluistert onder een hard nummer of schreeuwt na een zacht.
    """
    ff = shutil.which("ffmpeg")
    if ff:
        doel = CACHE / f"{sleutel}.mp3"
        try:
            subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "wav", "-i", "pipe:0",
                 "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ac", "1", "-ar", "44100",
                 "-c:a", "libmp3lame", "-b:a", "80k", str(doel)],
                input=wav, check=True, timeout=30,
            )
            return doel
        except Exception as e:
            log.info("presentator: mp3 maken mislukt, dan wav: %s", e)
            doel.unlink(missing_ok=True)
    doel = CACHE / f"{sleutel}.wav"
    doel.write_bytes(wav)
    return doel


def _opruimen() -> None:
    try:
        nu = time.time()
        bestanden = sorted(CACHE.glob("*.*"), key=lambda p: p.stat().st_mtime)
        for i, p in enumerate(bestanden):
            if nu - p.stat().st_mtime > _AUDIO_GELDIG or i < len(bestanden) - _MAX_BESTANDEN:
                p.unlink(missing_ok=True)
    except Exception:
        pass


def _uit_cache(sleutel: str) -> tuple[str, float] | None:
    for ext in ("mp3", "wav"):
        p = CACHE / f"{sleutel}.{ext}"
        if p.exists():
            meta = CACHE / f"{sleutel}.json"
            try:
                sec = float(json.loads(meta.read_text()).get("sec") or 0)
            except Exception:
                sec = 0.0
            return p.name, sec
    return None


def _audio_voor(tekst: str, stem: str = "man", taal: str = "nl") -> tuple[str, float] | None:
    """(bestandsnaam, seconden) voor deze tekst; uit de cache of nieuw gemaakt.

    Eerst de natuurlijke stem (`stem` = man of vrouw) in `taal`; lukt dat niet,
    Piper — maar alleen in het Nederlands, de enige taal die Piper hier kent.
    De taal zit in de cachesleutel."""
    wav = None
    if _stem_dienst_klaar(taal):
        sleutel = hashlib.sha256(f"chatterbox|{taal}|{stem}|{tekst}".encode()).hexdigest()
        oud = _uit_cache(sleutel)
        if oud:
            return oud
        wav = _spreek_echt(tekst, stem, taal)
    if not wav and taal != "nl":
        return None
    if not wav:
        stemnaam = _PIPER_URL or _PIPER_MODEL
        sleutel = hashlib.sha256(f"{stemnaam}|nl|{_tempo()}|{tekst}".encode()).hexdigest()
        oud = _uit_cache(sleutel)
        if oud:
            return oud
        wav = _spreek_wav(tekst)
    if not wav:
        return None
    sec = _duur(wav)
    if sec <= 0 or sec > MAX_SECONDEN:
        log.info("presentator: %.1f s is te lang, overgeslagen", sec)
        return None
    p = _bewaar(sleutel, wav)
    try:
        (CACHE / f"{sleutel}.json").write_text(json.dumps({"sec": sec, "tekst": tekst}))
    except Exception:
        pass
    _opruimen()
    return p.name, sec


# ── Tekst én geluid in één keer (gedeeld met de radio) ─────────────────────

def maak_aankondiging(feiten: dict, hoeken: list[str], stem: str, model: bool = True,
                      geheugen: Geheugen | None = None, hoek: str | None = None,
                      dialoog: bool = False, taal: str = "nl") -> list[dict] | None:
    """De aankondiging als lijst gesproken stukken, of None.

    Elk stuk is {"stem", "tekst", "naam" (bestand in CACHE), "sec"}. Meestal
    één; bij een dialoog (alleen als `dialoog` mag én de stemdienst met twee
    stemmen er is) twee, om de beurt `stem` en de andere.

    Twee pogingen bij het model. De tweede met een kortere opdracht: een tekst
    die uitgesproken langer duurt dan MAX_SECONDEN valt af, en dan is een
    kortere van dezelfde invalshoek beter dan niets. Lukt ook dat niet, dan de
    vaste zin. Met `model=False` (het model is er niet) meteen de vaste zin.

    De route hieronder en JLTamp Radio gebruiken allebei deze functie; de
    feitencheck in `_schrijf` geldt dus voor allebei.

    `taal` is die van de app (de radio: altijd nl). De feiten moeten in
    dezelfde taal gemaakt zijn (feiten_voor(..., taal=)).
    """
    g = geheugen or _GEHEUGEN
    dialoog = dialoog and _stem_dienst_klaar(taal)
    vorm, hoek = kies_vorm_en_hoek(feiten, hoeken, g, dialoog=dialoog, hoek=hoek)
    begin = _vers(talen.TERUG_BEGIN[taal], g)
    ander = "vrouw" if stem == "man" else "man"
    for kort in ((False, True, None) if model else (None,)):
        if kort is None:
            # De vaste zin is altijd één stem; een dialoog wordt dan een
            # gewone terug- en vooruitblik.
            vast_vorm = "terug_vooruit" if vorm == "dialoog" else vorm
            tekst = _vast(feiten, vast_vorm, g, taal=taal)
            if hoek == "uur" and feiten.get("tijd"):
                tekst = f"{talen.het_is(feiten['tijd'], taal)} {tekst}"
            elif hoek == "programma" and feiten.get("programma"):
                # Alleen op de radio, en die praat Nederlands.
                tekst = f"Dit is {feiten['programma']['naam']}. {tekst}"
            delen, vorm = [tekst], vast_vorm
        else:
            delen = _schrijf(feiten, hoek, kort, vorm, g, begin, taal)
        if not delen:
            continue
        stukken = []
        for i, tekst in enumerate(delen):
            wie = stem if i % 2 == 0 else ander
            audio = _audio_voor(tekst, wie, taal)
            if not audio:
                break
            stukken.append({"stem": wie, "tekst": tekst, "naam": audio[0], "sec": audio[1]})
        if len(stukken) == len(delen):
            g.onthoud(delen[0], vorm)
            return stukken
    return None


@contextmanager
def een_plek(wacht: float):
    """Wacht hooguit `wacht` seconden op een plek bij het model (zie _BEZIG).

    Voor de radio, die een aankondiging minuten vooruit maakt en dus best even
    kan wachten. De route van de app wacht juist niet (die zegt "bezig")."""
    ok = _BEZIG.acquire(timeout=wacht)
    try:
        yield ok
    finally:
        if ok:
            _BEZIG.release()


def model_bereikbaar() -> bool:
    return _ai_bereikbaar()


def stem_beschikbaar(taal: str = "nl") -> bool:
    return _stem_beschikbaar(taal)


# ── De routes ────────────────────────────────────────────────────────────────

def _niet(reden: str) -> dict:
    return {"beschikbaar": False, "reden": reden}


@router.post("/aankondiging")
def aankondiging(body: AankondigingBody, user: User = Depends(require_user)):
    """Een aankondiging voor de overgang naar `volgend`. Nooit een fout: bij elk
    probleem komt er `beschikbaar: false` terug en speelt de app gewoon door."""
    db = SessionLocal()
    try:
        if not schakelaar(db, "presentator"):
            return _niet("uit")
        taal = taal_van(body.taal)
        if not _stem_beschikbaar(taal):
            return _niet("geen stem")
        if not _ai_bereikbaar():
            return _niet("geen model")
        uitkomst = _feiten(db, user, body)
    finally:
        db.close()
    if not uitkomst:
        return _niet("onbekend nummer")
    feiten, hoeken = uitkomst

    nu = time.time()
    # Dezelfde overgang binnen het kwartier: dezelfde tekst, geen nieuw werk
    # voor het model.
    sleutel = (user.id, body.vorig, body.volgend, taal)
    oud = _teksten.get(sleutel)
    tekst = oud[1] if oud and nu - oud[0] < _TEKST_GELDIG else None
    stem = oud[2] if tekst else _wie_spreekt(user.id)
    if not tekst:
        with _laatste_slot:
            if nu - _laatste.get(user.id, 0) < _MIN_TUSSENPOZE:
                return _niet("te vaak")
            _laatste[user.id] = nu

    if not _BEZIG.acquire(blocking=False):
        return _niet("bezig")
    try:
        audio = _audio_voor(tekst, stem, taal) if tekst else None
        if not audio:
            uit = maak_aankondiging(feiten, hoeken, stem, taal=taal)
            if not uit:
                return _niet("geen geluid")
            # Zonder dialoog (die is alleen voor de radio) is het één stuk.
            tekst, audio = uit[0]["tekst"], (uit[0]["naam"], uit[0]["sec"])
            _teksten[sleutel] = (nu, tekst, stem)
            for k in [k for k, (t, *_) in _teksten.items() if nu - t > _TEKST_GELDIG]:
                _teksten.pop(k, None)
        naam, sec = audio
        return {
            "beschikbaar": True,
            "tekst": tekst,
            "seconden": round(sec, 2),
            "stem": stem,
            "taal": taal,
            "url": f"/presentator/audio/{naam}",
        }
    finally:
        _BEZIG.release()


_NAAM = re.compile(r"^[0-9a-f]{64}\.(mp3|wav)$")


@router.get("/audio/{naam}")
def audio(naam: str, _: User = Depends(require_user)):
    if not _NAAM.match(naam):
        raise HTTPException(404)
    p = CACHE / naam
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(
        p, media_type="audio/mpeg" if naam.endswith(".mp3") else "audio/wav",
        headers={"Cache-Control": "private, max-age=604800"},
    )
