"""Het huis doet mee met de muziek — via Home Assistant.

Twee dingen:

1. **Lampen in de kleur van de hoes.** Begint er een nummer, dan krijgen de
   gekozen lampen de hoofdkleur van de albumhoes. Rustig: een overgang van twee
   seconden, een gematigde helderheid, en alleen lampen die al AAN staan — de
   muziek doet geen lampen aan in een kamer waar niemand zit. Bij pauze gebeurt
   er niets; de app meldt alleen een nieuw nummer.

2. **Afspelen op een speaker in huis.** Een lijst van de media_players uit
   Home Assistant (de Google-speakers, de tv's) en "speel dit hier af". De
   speaker haalt het nummer zelf bij deze server op, met een ondertekend adres
   dat na een paar uur verloopt — het token van de gebruiker gaat nooit naar
   Home Assistant of de speaker. Verhuizen naar een andere kamer is een
   handmatige keuze ("verplaats naar…"): automatisch volgen zonder echte
   aanwezigheidsdetectie zou raden zijn.

   De rij speelt de SERVER af, niet de telefoon. Een telefoon in je zak die in
   slaap valt kan geen volgend nummer meer sturen; deze server is altijd wakker.
   Een draadje kijkt elke paar seconden hoe de speaker ervoor staat en zet het
   volgende nummer op als het vorige uit is.

Wie mag dit: alleen de eigenaar van de server. Andere gebruikers — ook een
tweede beheerder — en zeker een reviewaccount zien het huis niet en kunnen het
niet bedienen. Het is het huis van de eigenaar, niet dat van iedereen met een account.

Alles staat standaard UIT (`meta.huis_aan`), en zonder HA_URL/HA_TOKEN in de
omgeving antwoordt alles met "niet ingesteld".

JLTamp praat alleen met de REST-API van Home Assistant; Home Assistant doet het
LAN-werk (Cast, Tuya). Zelfde opzet als in JLTwatch.
"""
from __future__ import annotations

import colorsys
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from .. import config
from ..db import SessionLocal
from ..deps import require_user
from ..ids import parse_key, track_key
from ..models import Album, Meta, Track, User
from ..security import norm_email
from .admin_users import schakelaar, zet_schakelaar
from .stream import _CONTENT_TYPES, omgezet

router = APIRouter(prefix="/huis", tags=["huis"])
log = logging.getLogger("jltamp.huis")

# ── instellingen in `meta` ───────────────────────────────────────────────────
#
# `huis_aan` is de hoofdschakelaar (standaard UIT). Hij staat bewust niet in
# SCHAKELAARS van admin_users: die lijst ziet elke beheerder, en dit is alleen
# voor de eigenaar.
META_AAN = "huis_aan"
META_LAMPEN = "huis_lampen"          # JSON-lijst van light.* die meedoen
META_HELDERHEID = "huis_helderheid"  # procent
META_ALLEEN_THUIS = "huis_alleen_thuis"

HELDERHEID_STANDAARD = 45
OVERGANG_S = 2
# Wacht even na een nieuw nummer voor de lampen omgaan. Wie snel door de rij
# bladert, ziet anders de kamer van kleur naar kleur springen; zo telt alleen
# het nummer waar je op blijft staan.
LAMP_WACHT_S = 1.5

# Kleurmodi waarmee een lamp een kleur kan aannemen (en niet alleen wit).
KLEUR_MODI = {"hs", "rgb", "rgbw", "rgbww", "xy"}
# media_player.supported_features: PLAY_MEDIA.
KAN_AFSPELEN = 512


class HuisFout(RuntimeError):
    pass


# ── wie mag het huis zien ────────────────────────────────────────────────────
def is_eigenaar(user: User) -> bool:
    return (bool(user.is_admin) and not bool(getattr(user, "is_review", False))
            and norm_email(user.email) == norm_email(config.ADMIN_EMAIL))


def require_eigenaar(request: Request) -> User:
    user = require_user(request)
    if not is_eigenaar(user):
        raise HTTPException(403, "Het huis is alleen voor de eigenaar van de server")
    return user


def ingesteld() -> bool:
    return bool(config.HA_URL and config.HA_TOKEN)


def _meta(db, sleutel: str, standaard: str = "") -> str:
    rij = db.get(Meta, sleutel)
    return rij.value if rij and rij.value is not None else standaard


def _zet_meta(db, sleutel: str, waarde: str) -> None:
    rij = db.get(Meta, sleutel)
    if not rij:
        rij = Meta(key=sleutel)
        db.add(rij)
    rij.value = waarde


def _instellingen(db) -> dict:
    try:
        lampen = [x for x in json.loads(_meta(db, META_LAMPEN, "[]")) if isinstance(x, str)]
    except ValueError:
        lampen = []
    try:
        helderheid = int(_meta(db, META_HELDERHEID, str(HELDERHEID_STANDAARD)))
    except ValueError:
        helderheid = HELDERHEID_STANDAARD
    return {
        "aan": schakelaar(db, META_AAN),
        "lampen": lampen,
        "helderheid": helderheid,
        "alleenThuis": _meta(db, META_ALLEEN_THUIS, "1") != "0",
    }


def huis_aan() -> bool:
    if not ingesteld():
        return False
    db = SessionLocal()
    try:
        return schakelaar(db, META_AAN)
    finally:
        db.close()


# ── Home Assistant ───────────────────────────────────────────────────────────
def _ha(methode: str, pad: str, data: dict | None = None, timeout: float = 8):
    """Eén verzoek aan de REST-API van Home Assistant."""
    if not ingesteld():
        raise HuisFout("Home Assistant is niet ingesteld (HA_URL en HA_TOKEN)")
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        config.HA_URL + pad, data=body, method=methode,
        headers={"Authorization": f"Bearer {config.HA_TOKEN}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ruw = r.read()
    except urllib.error.HTTPError as e:
        raise HuisFout(f"Home Assistant weigerde: {e.code}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise HuisFout(f"Home Assistant antwoordt niet: {e}") from None
    try:
        return json.loads(ruw) if ruw else None
    except ValueError:
        return None


def _dienst(domein: str, dienst: str, data: dict) -> None:
    _ha("POST", f"/api/services/{domein}/{dienst}", data)


def _staat(entity_id: str) -> dict:
    s = _ha("GET", f"/api/states/{quote(entity_id)}")
    if not isinstance(s, dict):
        raise HuisFout(f"Onbekend in Home Assistant: {entity_id}")
    return s


def _alle_staten() -> list[dict]:
    s = _ha("GET", "/api/states")
    return s if isinstance(s, list) else []


def _geldig_id(entity_id: str, domein: str) -> bool:
    if not entity_id.startswith(domein + "."):
        return False
    return entity_id.replace(".", "").replace("_", "").isalnum()


def _naam(s: dict) -> str:
    return (s.get("attributes") or {}).get("friendly_name") or s.get("entity_id", "")


def kan_kleur(s: dict) -> bool:
    modi = (s.get("attributes") or {}).get("supported_color_modes") or []
    return bool(KLEUR_MODI.intersection(modi))


def kan_afspelen(s: dict) -> bool:
    try:
        f = int((s.get("attributes") or {}).get("supported_features") or 0)
    except (TypeError, ValueError):
        f = 0
    return bool(f & KAN_AFSPELEN)


# ── de kleur van de hoes ─────────────────────────────────────────────────────
#
# Op de server en niet in de app: dan geldt dezelfde kleur op elk toestel, en
# kan de speaker-rij (die hier draait) de lampen ook meenemen zonder telefoon.
_kleur_cache: dict[str, tuple[float, float] | None] = {}
# Een grijze of zwart-witte hoes heeft geen kleur om over te nemen. Dan een
# zacht warm licht in plaats van koud wit of — erger — een toevallige tint.
WARM = (32.0, 38.0)


def _hoes_pad(t: Track) -> str | None:
    if t.art_path and Path(t.art_path).exists():
        return t.art_path
    db = SessionLocal()
    try:
        al = db.get(Album, t.album_id)
        for p in ((al.art_path, al.online_art_path) if al else ()):
            if p and Path(p).exists():
                return p
    finally:
        db.close()
    return None


def hoofdkleur(pad: str) -> tuple[float, float] | None:
    """(tint 0-360, verzadiging 0-100) van de kleur die de hoes draagt.

    Niet het gemiddelde — dat is bij bijna elke hoes bruin. De hoes gaat terug
    naar acht kleuren; daarvan wint de kleur die veel ruimte inneemt ÉN echt
    kleur is. Bijna zwart en bijna wit tellen niet mee: daar is de tint ruis.
    None als er niets kleurigs in zit.
    """
    if pad in _kleur_cache:
        return _kleur_cache[pad]
    uitkomst: tuple[float, float] | None = None
    try:
        with Image.open(pad) as im:
            klein = im.convert("RGB")
            klein.thumbnail((64, 64))
        q = klein.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
        palet = q.getpalette() or []
        tellingen = q.getcolors() or []
        totaal = sum(n for n, _ in tellingen) or 1
        beste = 0.0
        for n, idx in tellingen:
            r, g, b = palet[idx * 3: idx * 3 + 3]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if v < 0.18 or s < 0.22:
                continue
            score = (n / totaal) * s * (0.4 + v)
            if score > beste:
                beste = score
                uitkomst = (round(h * 360, 1), round(s * 100, 1))
        # Een snippertje kleur op een verder grijze hoes is niet "de kleur".
        if beste < 0.02:
            uitkomst = None
    except Exception as e:      # een kapotte hoes mag de muziek niet storen
        log.debug("hoeskleur %s: %s", pad, e)
        uitkomst = None
    if len(_kleur_cache) > 2000:
        _kleur_cache.clear()
    _kleur_cache[pad] = uitkomst
    return uitkomst


def lampkleur(t: Track) -> tuple[float, float]:
    """De kleur voor de lampen. Wat verzadigder dan de hoes: een lamp in
    pastel ziet eruit als vies wit."""
    pad = _hoes_pad(t)
    k = hoofdkleur(pad) if pad else None
    if k is None:
        return WARM
    return (k[0], round(max(45.0, min(100.0, k[1] * 1.15)), 1))


def _hex(hs: tuple[float, float]) -> str:
    r, g, b = colorsys.hsv_to_rgb(hs[0] / 360, hs[1] / 100, 1)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


# ── de lampen omzetten ───────────────────────────────────────────────────────
_lamp_slot = threading.Lock()
_lamp_timer: threading.Timer | None = None
_laatste_kleur: tuple[float, float] | None = None


def _bijna_gelijk(a: tuple[float, float] | None, b: tuple[float, float]) -> bool:
    if a is None:
        return False
    dh = abs(a[0] - b[0]) % 360
    return min(dh, 360 - dh) < 8 and abs(a[1] - b[1]) < 10


def _zet_lampen(hs: tuple[float, float]) -> None:
    """Zet de gekozen lampen die AAN staan in deze kleur."""
    global _laatste_kleur
    db = SessionLocal()
    try:
        inst = _instellingen(db)
    finally:
        db.close()
    if not inst["aan"] or not inst["lampen"]:
        return
    if _bijna_gelijk(_laatste_kleur, hs):
        return          # zelfde album, zelfde kleur: niets aanraken
    try:
        staten = {s.get("entity_id"): s for s in _alle_staten()}
        doen = [e for e in inst["lampen"]
                if e in staten and staten[e].get("state") == "on" and kan_kleur(staten[e])]
        if not doen:
            return
        _dienst("light", "turn_on", {
            "entity_id": doen,
            "hs_color": [hs[0], hs[1]],
            "brightness_pct": max(5, min(80, inst["helderheid"])),
            "transition": OVERGANG_S,
        })
        _laatste_kleur = hs
    except HuisFout as e:
        log.info("[huis] lampen niet gezet: %s", e)


def lampen_volgen(t: Track) -> tuple[float, float]:
    """Plan de lampen voor dit nummer in. Een nieuw nummer binnen de wachttijd
    vervangt het vorige — zo knippert er niets als je doorbladert."""
    global _lamp_timer
    hs = lampkleur(t)
    with _lamp_slot:
        if _lamp_timer is not None:
            _lamp_timer.cancel()
        _lamp_timer = threading.Timer(LAMP_WACHT_S, _zet_lampen, args=(hs,))
        _lamp_timer.daemon = True
        _lamp_timer.start()
    return hs


# ── ondertekende adressen voor de speaker ────────────────────────────────────
#
# Een speaker kan geen X-Plex-Token sturen, en het token van de eigenaar hoort
# niet in de geschiedenis van Home Assistant of in een Cast-log. Dus een adres
# dat precies één nummer opent en na een paar uur niets meer waard is.
def _sleutel() -> bytes:
    pad = config.DATA_DIR / "huis_sleutel"
    try:
        if pad.exists():
            return bytes.fromhex(pad.read_text().strip())
    except (OSError, ValueError):
        pass
    nieuw = secrets.token_hex(32)
    pad.write_text(nieuw)
    try:
        os.chmod(pad, 0o600)
    except OSError:
        pass
    return bytes.fromhex(nieuw)


def teken(soort: str, track_id: int, geldig_s: int) -> str:
    verloopt = int(time.time()) + int(geldig_s)
    inhoud = f"{soort}.{track_id}.{verloopt}"
    sig = hmac.new(_sleutel(), inhoud.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{inhoud}.{sig}"


def controleer(tok: str, soort: str) -> int:
    try:
        s, tid, verloopt, sig = tok.split(".")
        tid_i, verloopt_i = int(tid), int(verloopt)
    except ValueError:
        raise HTTPException(403, "Ongeldig adres") from None
    goed = hmac.new(_sleutel(), f"{s}.{tid}.{verloopt}".encode(), hashlib.sha256).hexdigest()[:32]
    if s != soort or not hmac.compare_digest(sig, goed):
        raise HTTPException(403, "Ongeldig adres")
    if verloopt_i < time.time():
        raise HTTPException(403, "Verlopen")
    return tid_i


def _basis(request: Request | None) -> str:
    if config.HA_STREAM_URL:
        return config.HA_STREAM_URL
    for deel in (config.LOCAL_URL or "").split(","):
        deel = deel.strip().rstrip("/")
        if deel.startswith("http"):
            return deel
    if request is not None:
        return str(request.base_url).rstrip("/")
    raise HuisFout("Geen adres voor de speakers (HA_STREAM_URL of LOCAL_URL)")


def _hoeft_omzetten(t: Track) -> bool:
    return (t.ext or "").lstrip(".").lower() in config.HIRES_CONTAINERS


def _geldig_voor(t: Track) -> int:
    # Lang genoeg voor het nummer plus een pauze ertussen; daarna dicht.
    return int((t.duration_ms or 0) / 1000) + 3 * 3600


def stream_adres(t: Track, basis: str) -> str:
    ext = "mp3" if _hoeft_omzetten(t) else ((t.ext or ".mp3").lstrip(".").lower() or "mp3")
    return f"{basis}/huis/stream/{teken('s', t.id, _geldig_voor(t))}/nummer.{ext}"


def hoes_adres(t: Track, basis: str) -> str:
    return f"{basis}/huis/hoes/{teken('h', t.id, _geldig_voor(t))}/hoes.jpg"


@router.get("/stream/{tok}/{naam}")
def huis_stream(tok: str, naam: str):
    """Het nummer voor een speaker. Geen token: het ondertekende adres is de
    toegang, en het geldt voor dit ene nummer."""
    tid = controleer(tok, "s")
    db = SessionLocal()
    try:
        t = db.get(Track, tid)
        if not t:
            raise HTTPException(404, "Nummer niet gevonden")
        db.expunge(t)
    finally:
        db.close()
    if _hoeft_omzetten(t):
        return omgezet(t)
    ext = (t.ext or "").lstrip(".").lower()
    return FileResponse(t.path, media_type=_CONTENT_TYPES.get(ext, "audio/mpeg"))


@router.get("/hoes/{tok}/{naam}")
def huis_hoes(tok: str, naam: str):
    tid = controleer(tok, "h")
    db = SessionLocal()
    try:
        t = db.get(Track, tid)
        if t:
            db.expunge(t)
    finally:
        db.close()
    pad = _hoes_pad(t) if t else None
    if not pad:
        raise HTTPException(404, "Geen hoes")
    return FileResponse(pad)


# ── de rij op de speaker ─────────────────────────────────────────────────────
@dataclass
class Sessie:
    entity_id: str
    rij: list[int]
    index: int = 0
    basis: str = ""
    # Elke keer dat er een nummer opgezet wordt, gaat dit omhoog. Zo weet het
    # draadje dat een oude toestand niet meer over het huidige nummer gaat.
    beurt: int = 0
    gestart: float = field(default_factory=time.time)
    gezien_spelen: bool = False
    zoek_naar_s: float = 0.0
    # Positie zoals de speaker hem het laatst meldde, met het moment erbij.
    positie_s: float = 0.0
    positie_tijd: float = 0.0
    # Speelde hij bij de laatste blik? Alleen dan loopt de positie door.
    speelde: bool = False
    fouten: int = 0
    stop: threading.Event = field(default_factory=threading.Event)


_sessie: Sessie | None = None
_sessie_slot = threading.RLock()
POLL_S = 3.0
# Zo lang mag een nummer erover doen om te gaan spelen voor we opgeven.
OPSTART_S = 45.0
EIND_MARGE_S = 20.0


def _laad(track_id: int) -> Track | None:
    db = SessionLocal()
    try:
        t = db.get(Track, track_id)
        if t:
            db.expunge(t)
        return t
    finally:
        db.close()


def _speel_op(entity_id: str, t: Track, basis: str) -> None:
    artiest = t.orig_artist or t.artist_name or ""
    _dienst("media_player", "play_media", {
        "entity_id": entity_id,
        "media_content_id": stream_adres(t, basis),
        "media_content_type": "music",
        # Cast laat hiermee titel, artiest en hoes op het scherm zien; andere
        # spelers negeren het.
        "extra": {
            "title": t.title,
            "thumb": hoes_adres(t, basis),
            "metadata": {
                "metadataType": 3,
                "title": t.title,
                "artist": artiest,
                "albumName": t.album_title or "",
                "images": [{"url": hoes_adres(t, basis)}],
            },
        },
    })


def _zet_nummer(s: Sessie, positie_s: float = 0.0) -> Track:
    t = _laad(s.rij[s.index])
    if t is None:
        raise HuisFout("Nummer niet gevonden")
    _speel_op(s.entity_id, t, s.basis)
    s.beurt += 1
    s.gestart = time.time()
    s.gezien_spelen = False
    s.speelde = False
    s.zoek_naar_s = positie_s if positie_s > 3 else 0.0
    s.positie_s, s.positie_tijd = positie_s, time.time()
    if huis_aan():
        lampen_volgen(t)
    return t


def _geschatte_positie(s: Sessie, staat: dict | None = None) -> float:
    """Waar de speaker nu is. HA geeft een positie met het moment waarop die
    gold; tijdens het spelen tel je de verstreken tijd erbij op."""
    if staat:
        a = staat.get("attributes") or {}
        pos = a.get("media_position")
        if isinstance(pos, (int, float)):
            bijgewerkt = a.get("media_position_updated_at")
            extra = 0.0
            if staat.get("state") == "playing" and bijgewerkt:
                try:
                    from datetime import datetime
                    extra = max(0.0, time.time() - datetime.fromisoformat(bijgewerkt).timestamp())
                except ValueError:
                    extra = 0.0
            return float(pos) + extra
    return s.positie_s + (time.time() - s.positie_tijd if s.speelde else 0.0)


def _beeindig(s: Sessie, reden: str) -> None:
    global _sessie
    with _sessie_slot:
        s.stop.set()
        if _sessie is s:
            _sessie = None
    log.info("[huis] rij op %s gestopt: %s", s.entity_id, reden)


def _volg(s: Sessie) -> None:
    """Het draadje dat de rij gaande houdt."""
    while not s.stop.wait(POLL_S):
        try:
            staat = _staat(s.entity_id)
            s.fouten = 0
        except HuisFout as e:
            s.fouten += 1
            if s.fouten >= 10:
                _beeindig(s, f"speaker onbereikbaar ({e})")
                return
            continue
        with _sessie_slot:
            if s.stop.is_set():
                return
            toestand = staat.get("state")
            a = staat.get("attributes") or {}
            inhoud = str(a.get("media_content_id") or "")
            if inhoud and "/huis/stream/" not in inhoud:
                _beeindig(s, "iemand anders speelt nu op deze speaker")
                return
            if toestand == "playing":
                if not s.gezien_spelen and s.zoek_naar_s:
                    try:
                        _dienst("media_player", "media_seek",
                                {"entity_id": s.entity_id, "seek_position": s.zoek_naar_s})
                    except HuisFout:
                        pass
                    s.zoek_naar_s = 0.0
                s.gezien_spelen = True
                s.speelde = True
                s.positie_s = _geschatte_positie(s, staat)
                s.positie_tijd = time.time()
                continue
            if toestand in ("paused", "buffering"):
                if toestand == "paused":
                    s.speelde = False
                    s.positie_s = _geschatte_positie(s, staat)
                    s.positie_tijd = time.time()
                continue
            # idle / off / standby / unavailable
            if not s.gezien_spelen:
                if time.time() - s.gestart > OPSTART_S:
                    _beeindig(s, f"nummer kwam niet op gang ({toestand})")
                    return
                continue
            # Was hij klaar, of heeft iemand op stop gedrukt? Alleen bij het
            # einde van het nummer gaan we door.
            t = _laad(s.rij[s.index])
            lengte = (t.duration_ms or 0) / 1000 if t else 0
            pos = s.positie_s + (time.time() - s.positie_tijd if s.speelde else 0.0)
            if lengte and pos < lengte - EIND_MARGE_S:
                _beeindig(s, "gestopt op de speaker")
                return
            if s.index + 1 >= len(s.rij):
                _beeindig(s, "rij is op")
                return
            s.index += 1
            try:
                _zet_nummer(s)
            except HuisFout as e:
                _beeindig(s, str(e))
                return


def start_sessie(entity_id: str, rij: list[int], index: int, positie_s: float, basis: str) -> Sessie:
    global _sessie
    with _sessie_slot:
        oud = _sessie
        if oud is not None:
            oud.stop.set()
        s = Sessie(entity_id=entity_id, rij=rij, index=index, basis=basis)
        _zet_nummer(s, positie_s)
        _sessie = s
        if oud is not None and oud.entity_id != entity_id:
            try:
                _dienst("media_player", "media_stop", {"entity_id": oud.entity_id})
            except HuisFout:
                pass
    threading.Thread(target=_volg, args=(s,), daemon=True, name="huis-rij").start()
    return s


def _sessie_dict(s: Sessie | None) -> dict:
    if s is None:
        return {"actief": False}
    t = _laad(s.rij[s.index])
    naam = s.entity_id
    toestand = None
    try:
        st = _staat(s.entity_id)
        naam, toestand = _naam(st), st.get("state")
    except HuisFout:
        pass
    return {
        "actief": True,
        "entityId": s.entity_id,
        "naam": naam,
        "staat": toestand,
        "index": s.index,
        "lengte": len(s.rij),
        "positie": round(_geschatte_positie(s)),
        "nummer": None if t is None else {
            "ratingKey": track_key(t.id), "title": t.title,
            "artist": t.orig_artist or t.artist_name, "album": t.album_title,
            "duration": t.duration_ms,
        },
    }


# ── de API voor de app ───────────────────────────────────────────────────────
def _fout(e: HuisFout) -> HTTPException:
    return HTTPException(502, str(e))


@router.get("/status")
def status(_: User = Depends(require_eigenaar)):
    """Is het huis ingesteld, staat het aan, en wat doet er mee."""
    db = SessionLocal()
    try:
        inst = _instellingen(db)
    finally:
        db.close()
    return {"ingesteld": ingesteld(), **inst,
            **({} if ingesteld() else {"reden": "niet ingesteld"})}


class InstellingenBody(BaseModel):
    aan: bool | None = None
    lampen: list[str] | None = None
    helderheid: int | None = None
    alleenThuis: bool | None = None


@router.post("/instellingen")
def zet_instellingen(body: InstellingenBody, _: User = Depends(require_eigenaar)):
    global _sessie
    db = SessionLocal()
    try:
        if body.aan is not None:
            zet_schakelaar(db, META_AAN, body.aan)
        if body.lampen is not None:
            lampen = sorted({e for e in body.lampen if _geldig_id(e, "light")})[:50]
            _zet_meta(db, META_LAMPEN, json.dumps(lampen))
        if body.helderheid is not None:
            _zet_meta(db, META_HELDERHEID, str(max(5, min(80, int(body.helderheid)))))
        if body.alleenThuis is not None:
            _zet_meta(db, META_ALLEEN_THUIS, "1" if body.alleenThuis else "0")
        db.commit()
        inst = _instellingen(db)
    finally:
        db.close()
    if body.aan is False:
        # Uit is uit: ook een rij die op een speaker loopt houdt op met
        # doorschuiven (het nummer dat speelt, speelt nog uit).
        with _sessie_slot:
            if _sessie is not None:
                _sessie.stop.set()
                _sessie = None
    return {"ingesteld": ingesteld(), **inst}


def _vereist_aan() -> None:
    if not ingesteld():
        raise HTTPException(409, "niet ingesteld")
    if not huis_aan():
        raise HTTPException(409, "Het huis staat uit")


@router.get("/lampen")
def lampen(_: User = Depends(require_eigenaar)):
    """De lampen die een kleur kunnen aannemen. Witte lampen doen niet mee."""
    if not ingesteld():
        raise HTTPException(409, "niet ingesteld")
    db = SessionLocal()
    try:
        gekozen = set(_instellingen(db)["lampen"])
    finally:
        db.close()
    try:
        staten = _alle_staten()
    except HuisFout as e:
        raise _fout(e)
    lijst = [{
        "id": s["entity_id"], "naam": _naam(s), "staat": s.get("state"),
        "beschikbaar": s.get("state") not in ("unavailable", "unknown"),
        "gekozen": s["entity_id"] in gekozen,
    } for s in staten
        if str(s.get("entity_id", "")).startswith("light.") and kan_kleur(s)]
    lijst.sort(key=lambda x: x["naam"].lower())
    return {"lampen": lijst}


@router.get("/speakers")
def speakers(_: User = Depends(require_eigenaar)):
    """Alles in huis dat een adres kan afspelen."""
    _vereist_aan()
    try:
        staten = _alle_staten()
    except HuisFout as e:
        raise _fout(e)
    with _sessie_slot:
        actief = _sessie.entity_id if _sessie else None
    lijst = []
    for s in staten:
        eid = str(s.get("entity_id", ""))
        if not eid.startswith("media_player.") or not kan_afspelen(s):
            continue
        a = s.get("attributes") or {}
        lijst.append({
            "id": eid, "naam": _naam(s), "staat": s.get("state"),
            "beschikbaar": s.get("state") not in ("unavailable", "unknown"),
            "volume": None if a.get("volume_level") is None else round(a["volume_level"] * 100),
            "actief": eid == actief,
        })
    lijst.sort(key=lambda x: (not x["beschikbaar"], x["naam"].lower()))
    return {"speakers": lijst}


class NummerBody(BaseModel):
    ratingKey: str
    thuis: bool = True


@router.post("/nummer")
def nummer(body: NummerBody, _: User = Depends(require_eigenaar)):
    """De app meldt: er begint een nieuw nummer. Alleen bij een nieuw nummer —
    pauze en verder spelen sturen niets, dus de lampen blijven staan."""
    if not ingesteld():
        return {"ok": False, "reden": "niet ingesteld"}
    db = SessionLocal()
    try:
        inst = _instellingen(db)
    finally:
        db.close()
    if not inst["aan"] or not inst["lampen"]:
        return {"ok": False, "reden": "uit"}
    if inst["alleenThuis"] and not body.thuis:
        return {"ok": False, "reden": "niet thuis"}
    # Speelt er een rij op een speaker, dan bepaalt die de kleur.
    with _sessie_slot:
        if _sessie is not None:
            return {"ok": False, "reden": "speaker"}
    parsed = parse_key(body.ratingKey)
    if not parsed or parsed[0] != "track":
        raise HTTPException(400, "Geen nummer")
    t = _laad(parsed[1])
    if t is None:
        raise HTTPException(404, "Nummer niet gevonden")
    hs = lampen_volgen(t)
    return {"ok": True, "kleur": _hex(hs)}


class SpeelBody(BaseModel):
    entityId: str
    ratingKeys: list[str]
    index: int = 0
    positie: float = 0.0      # seconden


@router.post("/speel")
def speel(body: SpeelBody, request: Request, _: User = Depends(require_eigenaar)):
    """Speel de rij vanaf `index` af op deze speaker. De server houdt de rij
    gaande; de app mag daarna gewoon in slaap vallen."""
    _vereist_aan()
    if not _geldig_id(body.entityId, "media_player"):
        raise HTTPException(400, "Geen speaker")
    rij: list[int] = []
    for rk in body.ratingKeys[:500]:
        p = parse_key(rk)
        if p and p[0] == "track":
            rij.append(p[1])
    if not rij:
        raise HTTPException(400, "Niets om af te spelen")
    index = max(0, min(int(body.index or 0), len(rij) - 1))
    try:
        s = start_sessie(body.entityId, rij, index, max(0.0, float(body.positie or 0)),
                         _basis(request))
    except HuisFout as e:
        raise _fout(e)
    return _sessie_dict(s)


class VerplaatsBody(BaseModel):
    entityId: str


@router.post("/verplaats")
def verplaats(body: VerplaatsBody, request: Request, _: User = Depends(require_eigenaar)):
    """Neem wat er speelt mee naar een andere speaker, vanaf dezelfde plek."""
    _vereist_aan()
    if not _geldig_id(body.entityId, "media_player"):
        raise HTTPException(400, "Geen speaker")
    with _sessie_slot:
        s = _sessie
        if s is None:
            raise HTTPException(409, "Er speelt niets op een speaker")
        if s.entity_id == body.entityId:
            return _sessie_dict(s)
        try:
            positie = _geschatte_positie(s, _staat(s.entity_id))
        except HuisFout:
            positie = _geschatte_positie(s)
        try:
            nieuw = start_sessie(body.entityId, list(s.rij), s.index, positie, s.basis or _basis(request))
        except HuisFout as e:
            raise _fout(e)
    return _sessie_dict(nieuw)


@router.get("/sessie")
def sessie(_: User = Depends(require_eigenaar)):
    with _sessie_slot:
        s = _sessie
    return _sessie_dict(s)


class BedienBody(BaseModel):
    actie: str                 # pauze | verder | stop | volgende | vorige | volume
    waarde: float | None = None


@router.post("/bedien")
def bedien(body: BedienBody, _: User = Depends(require_eigenaar)):
    with _sessie_slot:
        s = _sessie
        if s is None:
            return {"actief": False}
        try:
            if body.actie == "pauze":
                _dienst("media_player", "media_pause", {"entity_id": s.entity_id})
            elif body.actie == "verder":
                _dienst("media_player", "media_play", {"entity_id": s.entity_id})
            elif body.actie == "stop":
                _dienst("media_player", "media_stop", {"entity_id": s.entity_id})
                _beeindig(s, "gestopt vanuit de app")
                return {"actief": False}
            elif body.actie in ("volgende", "vorige"):
                stap = 1 if body.actie == "volgende" else -1
                nieuw = s.index + stap
                if 0 <= nieuw < len(s.rij):
                    s.index = nieuw
                    _zet_nummer(s)
            elif body.actie == "volume" and body.waarde is not None:
                _dienst("media_player", "volume_set", {
                    "entity_id": s.entity_id,
                    "volume_level": max(0.0, min(100.0, body.waarde)) / 100})
            else:
                raise HTTPException(400, "Onbekende actie")
        except HuisFout as e:
            raise _fout(e)
    return _sessie_dict(s)
