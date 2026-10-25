"""Karaoke: een nummer uit de eigen bibliotheek zonder (of met een zachte) zang.

De app vraagt per nummer om een scheiding; die gebeurt op de achtergrond, één
tegelijk, en wat eruit komt wordt bewaard zodat het tweede keer meteen klaarstaat.

Waar het scheiden gebeurt, in volgorde:

  1. **De dienst op de host** (`JLTAMP_KARAOKE_URL`, bv.
     `http://host.docker.internal:8425`). Op de Mac Studio is dat de route: de
     container kan niet bij de GPU, de host wel. Zie docs/KARAOKE.md en
     `karaoke_demucs.py`.
  2. **Demucs in deze omgeving zelf** — als los proces. Voor een
     ontwikkelmachine, of een server waar geen Mac naast staat.
  3. **Nergens.** Dan zegt de status `beschikbaar: false` met een reden, en
     verder gebeurt er niets. Er gaat niets stuk als Demucs er niet is; de app
     laat de knop dan gewoon niet werken.

Wat er bewaard wordt, per nummer in `DATA_DIR/karaoke/<id>/`:

  - `begeleiding.m4a`  alles behalve de zang
  - `zang.m4a`         alleen de zang
  - `gids.m4a`         de begeleiding met de zang zacht erdoorheen
  - `info.json`        van welk bestand (pad, grootte, tijd) dit gemaakt is

Waarom die derde: de app speelt één bron tegelijk af. Twee spelers naast elkaar
(begeleiding + zang met een eigen volume) lopen na een paar minuten hoorbaar uit
de pas, en op het vergrendelscherm, in de auto en bij een netwerkwissel worden
het twee nummers. Dus mengt de server de "gidsstem" vooraf, en kiest de schuif
in de app tussen drie echte bestanden: weg, zacht, of het origineel.

Alles naar `/data`, nooit naar `/music` — zie server/CLAUDE.md. De cache heeft
een plafond (`JLTAMP_KARAOKE_MAX_MB`); wat het langst niet gebruikt is gaat er
het eerst uit.

En alles achter de schakelaar `karaoke` (meta `karaoke_aan`), standaard UIT. Zo
verandert er live niets tot hij in het beheerscherm wordt aangezet.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .. import config
from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..models import Track, User

router = APIRouter(tags=["karaoke"])
log = logging.getLogger("jltamp.karaoke")

KARAOKE_DIR = config.DATA_DIR / "karaoke"
WERK_DIR = KARAOKE_DIR / "_werk"
# Wat hier bij het opstarten nog staat, is van een taak die een herstart van de
# server afbrak (de `finally` die opruimt kwam nooit): weg ermee. Er draait één
# serverproces, dus er is niemand die er nu nog mee bezig is.
shutil.rmtree(WERK_DIR, ignore_errors=True)

# Eenmalig na 17-10: de eerste versie van de standen boven de 100 heette
# mix110.m4a … mix200.m4a, zonder de heldere stemketen. Die worden nooit meer
# gebruikt (nu mix110b.m4a …) en stonden alleen nog schijf vol te maken.
for _oud in KARAOKE_DIR.glob("*/mix[12][0-9]0.m4a"):
    if _oud.name not in ("mix100.m4a",) and int(_oud.name[3:-4]) > 100:
        try:
            _oud.unlink()
        except OSError:
            pass

DIENST_URL = os.environ.get("JLTAMP_KARAOKE_URL", "").strip().rstrip("/")
DIENST_SLEUTEL = os.environ.get("JLTAMP_KARAOKE_SLEUTEL", "").strip()
# De Python waarmee Demucs als los proces draait (route 2). Standaard deze zelf.
DEMUCS_PYTHON = os.environ.get("JLTAMP_KARAOKE_PYTHON", "").strip() or sys.executable
# Vier minuten muziek is zo'n 3 × 6 MB aan m4a. 4000 MB is dus een paar honderd
# nummers — ruim genoeg voor de liedjes die echt gezongen worden.
MAX_MB = int(os.environ.get("JLTAMP_KARAOKE_MAX_MB", "4000") or 4000)
# Een DJ-mix van een uur is geen karaokenummer, en houdt de rij een kwartier bezet.
MAX_MINUTEN = int(os.environ.get("JLTAMP_KARAOKE_MAX_MIN", "15") or 15)
# Hoe hard de zang in de gidsversie meeklinkt (lineair; 0,25 ≈ -12 dB). Hard
# genoeg om de melodie te volgen, zacht genoeg om er zelf overheen te zingen.
GIDS_VOLUME = 0.25
# Langer dan dit in de rij is geen wachten meer maar een opstopping.
MAX_RIJ = 20

STEMS = ("begeleiding", "zang", "gids")

# De zangschuif in de app: per tien procent een eigen mix, gemaakt uit de
# begeleiding en de zang. Eén bestand per stand, zodat er altijd één speler
# speelt en zang en muziek nooit uit de pas kunnen lopen. 0 % is de begeleiding
# zelf. Ook 100 % is een mix en niet het origineel: het origineel heeft een
# andere encoder-vertraging, en dan verspringt de overvloei een paar tienden.
MIX_STANDEN = tuple(range(10, 101, 10))
# Boven de 100: de zang blijft vol en de BEGELEIDING gaat zachter, tot 200 =
# alleen de stem (a capella). De schuif in de app loopt zo van zonder zang (0)
# via het origineel (100, het midden) naar alleen zang (200).
MIX_BOVEN = tuple(range(110, 201, 10))
_mix_sloten: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
_mix_bezig: set[int] = set()
_mix_bezig_slot = threading.Lock()


# ── de schakelaar ───────────────────────────────────────────────────────────

def karaoke_aan(user: User | None = None) -> bool:
    """Staat karaoke aan voor deze gebruiker?

    Uit voor een account voor een winkelbeoordeling, net als de wensen: een
    knop die een nummer op de achtergrond laat verwerken valt daar niet uit te
    leggen, en een beoordelaar hoeft hem niet te zien.
    """
    if user is not None and bool(getattr(user, "is_review", False)):
        return False
    from .admin_users import schakelaar, recht
    db = SessionLocal()
    try:
        return schakelaar(db, "karaoke") and recht(db, user, "karaoke")
    finally:
        db.close()


# ── is er iets om mee te scheiden? ─────────────────────────────────────────
#
# Dit wordt bij elke statusvraag gevraagd, en de app vraagt elke paar seconden
# zolang er gescheiden wordt. Dus een halve minuut onthouden — maar niet langer,
# want wie de dienst op de host start, wil niet een kwartier wachten tot de
# server dat merkt.

_beschikbaar: tuple[float, dict] = (0.0, {})
_beschikbaar_slot = threading.Lock()


def _dienst_verzoek(pad: str, methode: str = "GET", data: bytes | None = None,
                    timeout: float = 10.0):
    req = urllib.request.Request(f"{DIENST_URL}{pad}", data=data, method=methode)
    if DIENST_SLEUTEL:
        req.add_header("X-Karaoke-Sleutel", DIENST_SLEUTEL)
    if data is not None:
        req.add_header("Content-Type", "audio/flac")
    return urllib.request.urlopen(req, timeout=timeout)


def beschikbaarheid(vers: bool = False) -> dict:
    """{"beschikbaar": bool, "route": "dienst"|"lokaal"|None, "reden": str|None}."""
    global _beschikbaar
    with _beschikbaar_slot:
        wanneer, antwoord = _beschikbaar
        if not vers and antwoord and time.time() - wanneer < 30:
            return antwoord

    if not shutil.which("ffmpeg"):
        uit = {"beschikbaar": False, "route": None, "reden": "geen_ffmpeg"}
    elif DIENST_URL:
        try:
            with _dienst_verzoek("/gezond", timeout=3) as r:
                d = json.loads(r.read() or b"{}")
            uit = ({"beschikbaar": True, "route": "dienst", "reden": None,
                    "apparaat": d.get("apparaat"), "roformer": bool(d.get("roformer")),
                    "noten": bool(d.get("noten")),
                    "akkoorden": bool(d.get("akkoorden")),
                    "tekst": bool(d.get("tekst")),
                    "band": bool(d.get("band")),
                    "tempo": bool(d.get("tempo"))} if d.get("ok")
                   else {"beschikbaar": False, "route": "dienst", "reden": "dienst_zonder_demucs"})
        except Exception:
            uit = {"beschikbaar": False, "route": "dienst", "reden": "dienst_onbereikbaar"}
    elif _demucs_lokaal():
        uit = {"beschikbaar": True, "route": "lokaal", "reden": None}
    else:
        uit = {"beschikbaar": False, "route": None, "reden": "geen_demucs"}

    with _beschikbaar_slot:
        _beschikbaar = (time.time(), uit)
    return uit


def _demucs_lokaal() -> bool:
    if DEMUCS_PYTHON == sys.executable:
        return (importlib.util.find_spec("demucs") is not None
                and importlib.util.find_spec("soundfile") is not None)
    try:
        return subprocess.run([DEMUCS_PYTHON, "-c", "import demucs, soundfile"],
                              capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


# ── de cache ────────────────────────────────────────────────────────────────

def _map(track_id: int) -> Path:
    return KARAOKE_DIR / str(track_id)


def _bron_kenmerk(t: Track) -> dict:
    """Waarmee we zien dat het bestand nog hetzelfde is.

    De tagronde op de Studio hernoemt en herschrijft bestanden; een scheiding
    van de oude versie hoort dan niet meer bij dit nummer. Pad, grootte en
    wijzigingstijd samen vangen dat — de inhoud zelf hashen zou bij elke
    statusvraag een heel bestand over NFS lezen.
    """
    try:
        st = os.stat(t.path)
        return {"pad": t.path, "grootte": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"pad": t.path, "grootte": None, "mtime": None}


def _is_klaar(t: Track) -> bool:
    m = _map(t.id)
    if not all((m / f"{s}.m4a").is_file() for s in STEMS):
        return False
    try:
        info = json.loads((m / "info.json").read_text())
    except Exception:
        return False
    return info.get("bron") == _bron_kenmerk(t)


def _gebruikt(track_id: int) -> None:
    """Laatst gebruikt bijhouden, voor het opruimen.

    Op `info.json` en niet op de m4a's: een speler vraagt tijdens het afspelen
    tientallen stukjes op (Range), en een `utime` per stukje is zonde. Eens per
    minuut is precies genoeg voor "wat is het langst niet gebruikt".
    """
    p = _map(track_id) / "info.json"
    try:
        if time.time() - p.stat().st_mtime > 60:
            os.utime(p, None)
    except OSError:
        pass


def _grootte(map_: Path) -> int:
    return sum(f.stat().st_size for f in map_.glob("*") if f.is_file())


def opruimen(behoud: int | None = None) -> int:
    """Tot onder het plafond, langst niet gebruikt eerst. Geeft terug hoeveel weg.

    `behoud` is het nummer dat net klaar is: dat gaat er nooit uit, ook niet als
    het plafond lager staat dan één nummer groot is. Anders vraagt iemand om
    karaoke, wacht hij tot de balk vol is, en is het er alweer niet.
    """
    if not KARAOKE_DIR.is_dir():
        return 0
    mappen = []
    for m in KARAOKE_DIR.iterdir():
        if not m.is_dir() or not m.name.isdigit() or m.name == str(behoud):
            continue
        try:
            gebruikt = (m / "info.json").stat().st_mtime
        except OSError:
            gebruikt = 0.0       # half of kapot: als eerste weg
        mappen.append((gebruikt, m, _grootte(m)))
    totaal = sum(g for _, _, g in mappen)
    plafond = MAX_MB * 1024 * 1024
    weg = 0
    for _, m, g in sorted(mappen):
        if totaal <= plafond:
            break
        shutil.rmtree(m, ignore_errors=True)
        totaal -= g
        weg += 1
    if weg:
        log.info("[karaoke] cache opgeruimd: %d nummers weg, nu %.0f MB", weg, totaal / 1e6)
    return weg


# ── de rij ──────────────────────────────────────────────────────────────────
#
# Eén werker, één nummer tegelijk. Scheiden is zwaar (op de CPU minuten per
# nummer, op de GPU van de Studio seconden maar wel de hele GPU), en de server
# draait met 1500 MB: twee tegelijk levert niets op behalve twee keer het risico.

_rij: collections.deque[int] = collections.deque()
_staat: dict[int, dict] = {}       # track_id → {"status", "voortgang", "fout", "wanneer"}
_slot = threading.Lock()
_wekker = threading.Event()
_werker: threading.Thread | None = None

# Een mislukte poging blijft zo lang staan; daarna mag er opnieuw om gevraagd
# worden zonder dat de app eerst "mislukt" laat zien.
MISLUKT_S = 600


def _zet(track_id: int, **velden) -> None:
    with _slot:
        s = _staat.setdefault(track_id, {"status": "wacht", "voortgang": 0.0, "fout": None})
        s.update(velden, wanneer=time.time())


def _start_werker() -> None:
    global _werker
    with _slot:
        if _werker is not None and _werker.is_alive():
            return
        _werker = threading.Thread(target=_werk, name="karaoke", daemon=True)
        _werker.start()


def _werk() -> None:
    while True:
        with _slot:
            track_id = _rij.popleft() if _rij else None
        if track_id is None:
            _wekker.wait(60)
            _wekker.clear()
            continue
        _zet(track_id, status="bezig", voortgang=0.0)
        try:
            _verwerk(track_id)
            with _slot:
                _staat.pop(track_id, None)
                voor_akkoorden = track_id in _alleen_akkoorden
                _alleen_akkoorden.discard(track_id)
            # De schone RoFormer-versie (minuten GPU) alleen voor wie karaoke wil:
            # voor de akkoorden is de snelle Demucs-zang ruim genoeg. Zet iemand
            # later karaoke aan, dan vraagt _status hem alsnog aan.
            if not voor_akkoorden:
                _vraag_verbetering(track_id)
        except Exception as e:
            log.warning("[karaoke] nummer %s mislukt: %s", track_id, e)
            _zet(track_id, status="mislukt", fout=str(e)[:200])
        finally:
            try:
                opruimen(behoud=track_id)
            except Exception as e:
                log.warning("[karaoke] opruimen mislukt: %s", e)


def _lees_info(track_id: int) -> dict | None:
    try:
        return json.loads((_map(track_id) / "info.json").read_text())
    except Exception:
        return None


def _is_roformer(info: dict | None) -> bool:
    return bool(info) and "roformer" in str(info.get("model", ""))


# ── de tweede ronde: RoFormer ───────────────────────────────────────────────
# Demucs maakt in seconden een versie waarmee je meteen kunt zingen; daarna
# rekent RoFormer op de achtergrond een schonere uit (minuten). Een eigen rij en
# een eigen werker, zodat een nieuw nummer nooit achter een lange RoFormer-taak
# hoeft te wachten. Klaar → de map wordt in één keer vervangen, `versie` in de
# status verandert, en de app wisselt op dezelfde plek over.

_verbeter_rij: collections.deque[int] = collections.deque()
_verbeter_staat: dict[int, dict] = {}
_verbeter_wekker = threading.Event()
_verbeteraar: threading.Thread | None = None
# Mislukt RoFormer op een nummer, dan niet elke statusvraag opnieuw proberen.
_verbeter_mislukt: dict[int, float] = {}


def _zet_verbeter(track_id: int, **velden) -> None:
    with _slot:
        s = _verbeter_staat.setdefault(track_id, {"status": "wacht", "voortgang": 0.0})
        s.update(velden)


def _vraag_verbetering(track_id: int) -> None:
    """In de RoFormer-rij, als dat kan en nog niet gebeurd is."""
    if not beschikbaarheid().get("roformer"):
        return
    if time.time() - _verbeter_mislukt.get(track_id, 0) < 3600:
        return
    if _is_roformer(_lees_info(track_id)):
        return
    global _verbeteraar
    with _slot:
        if track_id in _verbeter_staat or len(_verbeter_rij) >= MAX_RIJ:
            return
        _verbeter_rij.append(track_id)
        _verbeter_staat[track_id] = {"status": "wacht", "voortgang": 0.0}
        if _verbeteraar is None or not _verbeteraar.is_alive():
            _verbeteraar = threading.Thread(target=_werk_verbeter, name="karaoke-roformer", daemon=True)
            _verbeteraar.start()
    _verbeter_wekker.set()


def _werk_verbeter() -> None:
    while True:
        with _slot:
            track_id = _verbeter_rij.popleft() if _verbeter_rij else None
        if track_id is None:
            _verbeter_wekker.wait(60)
            _verbeter_wekker.clear()
            continue
        _zet_verbeter(track_id, status="bezig", voortgang=0.0)
        try:
            _verwerk(track_id, "roformer")
            log.info("[karaoke] nummer %s verbeterd met RoFormer", track_id)
        except Exception as e:
            log.warning("[karaoke] RoFormer voor %s mislukt: %s", track_id, e)
            _verbeter_mislukt[track_id] = time.time()
        finally:
            with _slot:
                _verbeter_staat.pop(track_id, None)


def _ffmpeg(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr.decode(errors="replace").strip()[-200:] or "?"))


def _verwerk(track_id: int, model: str = "demucs") -> None:
    """Scheiden en klaarzetten. `model="roformer"` is de schone tweede ronde
    (zie hieronder): dezelfde stappen, maar de voortgang gaat naar zijn eigen
    staat, want voor de app is het nummer dan al klaar."""
    zet = (lambda **v: _zet(track_id, **v)) if model == "demucs" else (lambda **v: _zet_verbeter(track_id, **v))
    db = SessionLocal()
    try:
        t = db.get(Track, track_id)
        if not t:
            raise RuntimeError("nummer bestaat niet meer")
        db.expunge(t)
    finally:
        db.close()
    if t.duration_ms and t.duration_ms > MAX_MINUTEN * 60 * 1000:
        raise RuntimeError(f"langer dan {MAX_MINUTEN} minuten")

    WERK_DIR.mkdir(parents=True, exist_ok=True)
    werk = WERK_DIR / f"{track_id}-{model}"
    shutil.rmtree(werk, ignore_errors=True)
    werk.mkdir(parents=True)
    begin = time.time()
    try:
        # 1. De bron naar 44,1 kHz stereo FLAC: wat Demucs verwacht, en klein
        #    genoeg om over HTTP naar de host te sturen. Gelezen van /music, en
        #    alleen gelezen.
        bron = werk / "bron.flac"
        _ffmpeg("-i", t.path, "-vn", "-map", "0:a:0", "-ac", "2", "-ar", "44100",
                "-t", str(MAX_MINUTEN * 60), "-c:a", "flac", str(bron))
        zet(voortgang=0.05)

        # 2. Scheiden — 5 % tot 90 % van de balk.
        def melden(f: float) -> None:
            zet(voortgang=round(0.05 + 0.85 * max(0.0, min(1.0, f)), 3))
        route = beschikbaarheid().get("route")
        if model == "roformer":
            if route != "dienst" or not DIENST_URL:
                raise RuntimeError("roformer kan alleen via de dienst")
            info = _scheid_via_dienst(bron, werk, melden, "roformer")
        elif route == "dienst" and DIENST_URL:
            info = _scheid_via_dienst(bron, werk, melden)
        elif route == "lokaal":
            info = _scheid_lokaal(bron, werk, melden)
        else:
            raise RuntimeError("geen Demucs beschikbaar")
        bron.unlink(missing_ok=True)

        # 3. Naar m4a, en de gidsversie mengen.
        aac = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
        _ffmpeg("-i", str(werk / "begeleiding.flac"), *aac, str(werk / "begeleiding.m4a"))
        zet(voortgang=0.93)
        _ffmpeg("-i", str(werk / "zang.flac"), *aac, str(werk / "zang.m4a"))
        zet(voortgang=0.96)
        # normalize=0: anders deelt amix alles door twee en wordt de gidsversie
        # zachter dan de begeleiding — dan maakt de schuif een volumesprong.
        _ffmpeg("-i", str(werk / "begeleiding.flac"), "-i", str(werk / "zang.flac"),
                "-filter_complex",
                f"[1:a]volume={GIDS_VOLUME}[z];[0:a][z]amix=inputs=2:duration=longest:normalize=0",
                *aac, str(werk / "gids.m4a"))
        for f in werk.glob("*.flac"):
            f.unlink()

        (werk / "info.json").write_text(json.dumps({
            "bron": _bron_kenmerk(t), "gemaakt": int(time.time()),
            "seconden": round(time.time() - begin, 1), **(info or {}),
        }))
        doel = _map(track_id)
        if model == "roformer":
            # Intussen een ander bronbestand (opnieuw neergezet)? Dan is deze
            # schone versie van het verkeerde nummer, en blijft de oude staan.
            info_nu = _lees_info(track_id)
            if info_nu is None or info_nu.get("bron") != _bron_kenmerk(t):
                raise RuntimeError("bron veranderd tijdens het verbeteren")
        # De vorige versie niet weggooien maar opzij zetten: een speler die er nog
        # uit leest (Range-verzoeken, ?v=oud) kreeg anders ineens bytes uit een
        # ander bestand — ruis, of de speler viel uit. Na OUD_BEWAAR_S weg.
        # Hernoemen in plaats van rmtree vangt ook een mix die net in de oude
        # map schrijft: die faalt netjes, en `doel` is leeg voor os.replace.
        if doel.exists():
            oud_v = (_lees_info(track_id) or {}).get("gemaakt") or int(time.time())
            opzij = KARAOKE_DIR / f"{track_id}.v{oud_v}"
            shutil.rmtree(opzij, ignore_errors=True)
            try:
                os.replace(doel, opzij)
            except OSError:
                shutil.rmtree(doel, ignore_errors=True)
        # Pas als alles er staat op de echte plek: een halve map mag nooit
        # "klaar" lijken voor een speler die toevallig net kijkt.
        os.replace(werk, doel)
        log.info("[karaoke] nummer %s klaar in %.0fs (%s)", track_id,
                 time.time() - begin, (info or {}).get("apparaat", "?"))
    finally:
        shutil.rmtree(werk, ignore_errors=True)


def _scheid_via_dienst(bron: Path, werk: Path, melden, model: str = "demucs") -> dict:
    pad = "/taken?model=roformer" if model == "roformer" else "/taken"
    with _dienst_verzoek(pad, "POST", bron.read_bytes(), timeout=120) as r:
        tid = json.loads(r.read())["id"]
    try:
        einde = time.time() + 30 * 60
        while True:
            if time.time() > einde:
                raise RuntimeError("dienst duurde te lang")
            time.sleep(1.5)
            with _dienst_verzoek(f"/taken/{tid}") as r:
                s = json.loads(r.read())
            if s.get("status") == "klaar":
                break
            if s.get("status") == "mislukt":
                raise RuntimeError(s.get("fout") or "dienst: mislukt")
            melden(float(s.get("voortgang") or 0.0))
        for naam in ("zang", "begeleiding"):
            with _dienst_verzoek(f"/taken/{tid}/{naam}.flac", timeout=300) as r, \
                    open(werk / f"{naam}.flac", "wb") as f:
                shutil.copyfileobj(r, f, 256 * 1024)
        return s.get("info") or {}
    finally:
        try:
            _dienst_verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass


def _scheid_lokaal(bron: Path, werk: Path, melden) -> dict:
    script = Path(__file__).resolve().parent.parent / "karaoke_demucs.py"
    proc = subprocess.Popen([DEMUCS_PYTHON, str(script), "los", str(bron), str(werk)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    laatste: dict = {}
    try:
        for regel in proc.stdout:
            try:
                d = json.loads(regel)
            except ValueError:
                continue
            if "voortgang" in d:
                melden(float(d["voortgang"]))
            laatste = d
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    if laatste.get("fout") or not laatste.get("klaar"):
        raise RuntimeError(laatste.get("fout") or f"demucs stopte met code {proc.returncode}")
    return {k: v for k, v in laatste.items() if k != "klaar"}


# ── de adressen ─────────────────────────────────────────────────────────────

def _track(track_id: str, user: User) -> Track:
    # Een string, zoals bij de songteksten: de app geeft de ratingKey terug die
    # hij kreeg, en dat is `t7937`, geen 7937.
    raw = str(track_id).strip()
    if raw[:1] in ("t", "T"):
        raw = raw[1:]
    if not raw.isdigit():
        raise HTTPException(400, "Bad track id")
    db = SessionLocal()
    try:
        t = db.get(Track, int(raw))
        if not t:
            raise HTTPException(404, "Track not found")
        allowed = accessible_library_ids(db, user)
        if allowed is not None and t.library_id not in allowed:
            raise HTTPException(403, "Gjin tagong ta dit nûmer")
        db.expunge(t)
        return t
    finally:
        db.close()


def _status(t: Track, aan: bool) -> dict:
    b = beschikbaarheid() if aan else {"beschikbaar": False, "reden": "uit"}
    uit = {"aan": aan, "beschikbaar": bool(b.get("beschikbaar")), "reden": b.get("reden"),
           "status": "geen", "voortgang": 0.0, "plekInRij": None, "fout": None}
    if not aan:
        return uit
    if _is_klaar(t):
        uit.update(status="klaar", voortgang=1.0)
        # De mixen voor de zangschuif alvast maken (een paar seconden per
        # stand): wie karaoke aanzet, gaat schuiven, en dan moet de stand er
        # al zijn. Eén keer per nummer; bestaat alles al, dan gebeurt er niets.
        if aan and not all((_map(t.id) / _mix_naam(p)).exists() for p in MIX_STANDEN + MIX_BOVEN):
            _maak_alle_mixen(t.id)
        info = _lees_info(t.id) or {}
        uit.update(model=info.get("model"), versie=info.get("gemaakt"))
        if aan and not _is_roformer(info):
            _vraag_verbetering(t.id)
        with _slot:
            v = _verbeter_staat.get(t.id)
        uit["verbetering"] = dict(v) if v else None
        return uit
    with _slot:
        s = dict(_staat.get(t.id) or {})
        plek = list(_rij).index(t.id) + 1 if t.id in _rij else None
    if s:
        if s["status"] == "mislukt" and time.time() - s.get("wanneer", 0) > MISLUKT_S:
            return uit
        uit.update(status=s["status"], voortgang=s.get("voortgang", 0.0),
                   fout=s.get("fout"), plekInRij=plek)
    return uit


# Scheidingen die alleen voor de akkoorden/noten zijn aangevraagd (zie hierboven).
_alleen_akkoorden: set[int] = set()


def vraag_scheiding(track_id: int) -> None:
    """De scheiding aanvragen zonder gebruiker (voor de akkoorden).

    Zelfde rij als een aanvraag uit de app; staat hij er al of is hij klaar,
    dan gebeurt er niets. Alleen als de dienst er is.
    """
    if not beschikbaarheid().get("beschikbaar"):
        return
    with _slot:
        if track_id in _rij or (_staat.get(track_id) or {}).get("status") in ("wacht", "bezig"):
            return
        if len(_rij) >= MAX_RIJ:
            return
        _rij.append(track_id)
        _alleen_akkoorden.add(track_id)
        _staat[track_id] = {"status": "wacht", "voortgang": 0.0, "fout": None, "wanneer": time.time()}
    _start_werker()
    _wekker.set()


@router.get("/karaoke/{track_id}")
def karaoke_status(track_id: str, user: User = Depends(require_user)):
    """Hoe staat het met de karaokeversie van dit nummer? Verandert niets."""
    t = _track(track_id, user)
    return _status(t, karaoke_aan(user))


@router.post("/karaoke/{track_id}")
def karaoke_aanvragen(track_id: str, user: User = Depends(require_user)):
    """Zet dit nummer in de rij, tenzij het al klaar is of er al in staat."""
    t = _track(track_id, user)
    if not karaoke_aan(user):
        raise HTTPException(403, "Karaoke staat uit")
    st = _status(t, True)
    if not st["beschikbaar"] or st["status"] in ("klaar", "wacht", "bezig"):
        return st
    with _slot:
        if len(_rij) >= MAX_RIJ:
            raise HTTPException(429, "Even te druk — probeer het zo weer")
        _rij.append(t.id)
        _staat[t.id] = {"status": "wacht", "voortgang": 0.0, "fout": None, "wanneer": time.time()}
    _start_werker()
    _wekker.set()
    return _status(t, True)


@router.get("/karaoke/{track_id}/{stem}.m4a")
def karaoke_stream(track_id: str, stem: str, v: int | None = None, user: User = Depends(require_user)):
    """Begeleiding, zang of de gidsversie — met Range, net als /library/parts.

    FileResponse doet Range zelf, dus spoelen werkt zoals bij elk ander nummer.
    """
    if stem not in STEMS:
        raise HTTPException(404, "Onbekend deel")
    t = _track(track_id, user)
    if not karaoke_aan(user):
        raise HTTPException(403, "Karaoke staat uit")
    if not _is_klaar(t):
        raise HTTPException(404, "Nog niet gescheiden")
    _gebruikt(t.id)
    return FileResponse(_bron_map(t.id, v) / f"{stem}.m4a", media_type="audio/mp4")


OUD_BEWAAR_S = 15 * 60


def _ruim_oude_versies_op() -> None:
    """Opzijgezette versies (`<id>.v<gemaakt>`) ouder dan OUD_BEWAAR_S weg."""
    nu = time.time()
    try:
        for p in KARAOKE_DIR.glob("*.v*"):
            try:
                if p.is_dir() and nu - p.stat().st_mtime > OUD_BEWAAR_S:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


def _bron_map(track_id: int, v: int | None) -> Path:
    """De map voor versie `v`: de huidige, of de opzijgezette als de speler die nog speelt."""
    if v:
        huidig = (_lees_info(track_id) or {}).get("gemaakt")
        if huidig and int(huidig) != int(v):
            oud = KARAOKE_DIR / f"{track_id}.v{int(v)}"
            if oud.is_dir():
                return oud
    return _map(track_id)


def _mix_naam(pct: int) -> str:
    return f"mix{pct}b.m4a" if pct > 100 else f"mix{pct}.m4a"


def _maak_mix(track_id: int, pct: int) -> Path:
    """De mix voor één stand van de schuif; bestaat hij al, dan die."""
    map_ = _map(track_id)
    # Boven de 100 een eigen naam: die mixen kregen later de heldere stemketen,
    # en een oude mix zonder die keten mag dan niet blijven hangen.
    naam = _mix_naam(pct)
    doel = map_ / naam
    if doel.exists():
        return doel
    with _mix_sloten[f"{track_id}:{pct}"]:
        if doel.exists():
            return doel
        # Wordt de map tijdens het mengen vervangen (de RoFormer-ronde), dan
        # hoort deze mix bij de oude scheiding; niet in de nieuwe map laten staan.
        versie_voor = (_lees_info(track_id) or {}).get("gemaakt")
        tijdelijk = map_ / f".{naam}"
        # normalize=0, net als bij de gidsversie: anders deelt amix door twee
        # en springt het volume bij elke stand.
        zang = min(pct, 100) / 100
        muziek = (200 - pct) / 100 if pct > 100 else 1.0
        # Hoe ver richting "alleen zang" (0 bij het origineel, 1 bij 200).
        k = max(0.0, (pct - 100) / 100)
        if k > 0:
            # De stem helder maken naarmate de muziek wegvalt: de brom en het
            # lekken van bas en drums eruit (hoogdoorlaat), het gesis van de
            # scheiding eruit (ruisonderdrukking), wat meer aanwezigheid rond
            # 3 kHz, en harder — anders is een stem zonder muziek een dun
            # geluidje op de helft van het volume. Een begrenzer houdt het heel.
            stem = (f"[1:a]highpass=f={int(70 + 50 * k)},"
                    f"afftdn=nr={6 + 14 * k:.0f}:nf=-42,"
                    f"equalizer=f=3200:t=q:w=1.1:g={4 * k:.1f},"
                    f"equalizer=f=250:t=q:w=1:g={-3 * k:.1f}[z]")
            na = f",volume={1 + 0.9 * k:.2f},alimiter=limit=0.93:level=disabled"
        else:
            stem = f"[1:a]volume={zang:.2f}[z]"
            na = ""
        _ffmpeg("-i", str(map_ / "begeleiding.m4a"), "-i", str(map_ / "zang.m4a"),
                "-filter_complex",
                f"[0:a]volume={muziek:.2f}[b];{stem};"
                f"[b][z]amix=inputs=2:duration=longest:normalize=0{na}",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4",
                str(tijdelijk))
        if (_lees_info(track_id) or {}).get("gemaakt") != versie_voor:
            tijdelijk.unlink(missing_ok=True)
            raise RuntimeError("scheiding vervangen tijdens het mengen")
        tijdelijk.replace(doel)
    return doel


def _maak_alle_mixen(track_id: int) -> None:
    """In de achtergrond de andere standen alvast maken: wie schuift, wil niet wachten."""
    with _mix_bezig_slot:
        if track_id in _mix_bezig:
            return
        _mix_bezig.add(track_id)

    def werk() -> None:
        try:
            for pct in MIX_STANDEN + MIX_BOVEN:
                try:
                    _maak_mix(track_id, pct)
                except Exception:
                    log.exception("karaoke: mix %s%% van %s mislukt", pct, track_id)
                    return
        finally:
            with _mix_bezig_slot:
                _mix_bezig.discard(track_id)

    threading.Thread(target=werk, name=f"karaoke-mix-{track_id}", daemon=True).start()


@router.get("/karaoke/{track_id}/mix/{pct}.m4a")
def karaoke_mix(track_id: str, pct: int, v: int | None = None, user: User = Depends(require_user)):
    """De zangschuif als één bestand: 0–100 = zang op `pct` procent, 110–200 = muziek op 200 − `pct`."""
    if pct != 0 and pct not in MIX_STANDEN and pct not in MIX_BOVEN:
        raise HTTPException(404, "Onbekende stand")
    t = _track(track_id, user)
    if not karaoke_aan(user):
        raise HTTPException(403, "Karaoke staat uit")
    if not _is_klaar(t):
        raise HTTPException(404, "Nog niet gescheiden")
    _gebruikt(t.id)
    _ruim_oude_versies_op()
    # Speelt de app nog een vorige versie (?v=), dan uit die map — zolang hij er is.
    bron = _bron_map(t.id, v)
    if bron != _map(t.id):
        oud = bron / ("begeleiding.m4a" if pct == 0 else _mix_naam(pct))
        if oud.exists():
            return FileResponse(oud, media_type="audio/mp4")
    if pct == 0:
        return FileResponse(_map(t.id) / "begeleiding.m4a", media_type="audio/mp4")
    try:
        pad = _maak_mix(t.id, pct)
    except Exception:
        log.exception("karaoke: mix %s%% van %s mislukt", pct, t.id)
        raise HTTPException(500, "Mengen mislukt")
    _maak_alle_mixen(t.id)
    return FileResponse(pad, media_type="audio/mp4")
