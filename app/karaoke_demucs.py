"""Zang en begeleiding uit elkaar halen met Demucs — los van de rest van de server.

Dit bestand importeert met opzet NIETS uit de server (geen config, geen db), en
bij het laden alleen de standaardbibliotheek. Zo kan hetzelfde bestand drie
dingen zijn:

  1. **De losse dienst op de Mac Studio** (`python karaoke_demucs.py dienst`).
     Daar draait Demucs op de GPU (MPS); in een Docker-container kan dat niet,
     want OrbStack geeft een container geen toegang tot Metal. Op de CPU van
     een container duurt een nummer van vier minuten al gauw een paar minuten
     en vraagt het meer geheugen dan de 1500 MB die de server heeft; op MPS is
     het een kwestie van seconden. Zie docs/KARAOKE.md.
  2. **Een eenmalige opdracht** (`python karaoke_demucs.py los bron.flac map/`),
     die de server zelf start als Demucs in zijn eigen omgeving staat — handig
     op een ontwikkelmachine, of op een server zonder Mac ernaast. Een apart
     proces en geen import: torch blijft dan niet na afloop honderden MB in de
     server hangen.
  3. **Niets**, als Demucs nergens staat. Dan importeert de server dit bestand
     wel, maar roept hij er niets uit aan — `karaoke.py` meldt dan netjes dat
     karaoke er niet is.

Waarom niet gewoon `demucs --two-stems=vocals` aanroepen: dat leest en schrijft
via torchaudio, en de nieuwste torchaudio wil daarvoor torchcodec mét de
ffmpeg-bibliotheken op het systeem. Hier lezen en schrijven we FLAC met
soundfile (een wiel met libsndfile erin) en doen we wat de opdrachtregel doet
met dezelfde twee functies, `get_model` en `apply_model`. Het omzetten naar
FLAC (en straks naar m4a) doet de server met zijn eigen ffmpeg.

Wat er uitkomt is precies `--two-stems=vocals`: zang, en al het andere bij
elkaar opgeteld als begeleiding.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import subprocess
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "htdemucs"
# Demucs werkt op 44,1 kHz stereo. De bron wordt daar vooraf naartoe gezet
# (door de server, met ffmpeg); klopt het niet, dan weigeren we liever dan dat
# er op de verkeerde snelheid gezongen wordt.
SAMPLERATE = 44100

# Per modelnaam één keer geladen: htdemucs voor karaoke, htdemucs_6s voor de
# noten (zie hieronder). Allebei vasthouden kost ~0,5 GB; dat is op de Studio
# niets, en steeds wisselen kost per nummer seconden.
_modellen: dict = {}
_model_slot = threading.Lock()


def apparaat() -> str:
    """Waar Demucs op rekent. MPS op een Mac, anders CUDA, anders de CPU.

    `KARAOKE_APPARAAT=cpu` dwingt het af — voor als MPS na een macOS- of
    torch-update ineens rare dingen doet en je wilt weten of het daaraan ligt.
    """
    gekozen = os.environ.get("KARAOKE_APPARAAT", "").strip().lower()
    if gekozen:
        return gekozen
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _laad_model(naam: str = MODEL, repo: str | None = None):
    """Het model één keer laden en vasthouden.

    In de dienst scheelt dat per nummer een paar seconden (en het lezen van
    80 MB van schijf). De eerste keer haalt Demucs het model op (Hugging Face,
    anders de oude AWS-opslag) — daarna niet meer. `repo` is een map waar het
    model met de hand is neergezet (.yaml + .th); dan wordt er niets opgehaald.
    """
    with _model_slot:
        if naam not in _modellen:
            from demucs.pretrained import get_model
            from pathlib import Path
            m = get_model(naam, repo=Path(repo) if repo else None)
            m.eval()
            _modellen[naam] = m
        return _modellen[naam]


def _met_voortgang(melden):
    """Demucs' eigen voortgangsbalk ombuigen naar een functie.

    `apply_model(progress=True)` loopt met tqdm over de stukken waarin hij het
    nummer knipt. Dat is de enige plek waar je ziet hoe ver hij is, dus vangen
    we die af in plaats van zelf te gaan schatten. Lukt dat niet (een andere
    versie van Demucs), dan valt er alleen de voortgang weg — het scheiden zelf
    gaat gewoon door.
    """
    try:
        import demucs.apply as da
    except Exception:
        return lambda: None
    oud = getattr(da, "tqdm", None)

    def tqdm(iterable, **_kw):
        stukken = list(iterable)
        totaal = max(1, len(stukken))
        for i, stuk in enumerate(stukken):
            yield stuk
            try:
                melden((i + 1) / totaal)
            except Exception:
                pass

    da.tqdm = types.SimpleNamespace(tqdm=tqdm)

    def herstel():
        if oud is not None:
            da.tqdm = oud
    return herstel


def scheiden(bron: str, doelmap: str, melden=None) -> dict:
    """`bron` (FLAC/WAV, 44,1 kHz) → `doelmap/zang.flac` + `doelmap/begeleiding.flac`.

    `melden(fractie)` krijgt tussendoor 0..1. Geeft wat informatie terug voor
    in het logboek.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    begin = time.time()
    model = _laad_model()
    dev = apparaat()

    data, sr = sf.read(bron, dtype="float32", always_2d=True)    # (n, kanalen)
    if sr != SAMPLERATE:
        raise ValueError(f"bron staat op {sr} Hz, verwacht {SAMPLERATE}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))        # (2, n)

    # Zoals `demucs.separate` het doet: normaliseren op het gemiddelde van de
    # twee kanalen en het na afloop terugzetten. De epsilon is voor stilte —
    # een nummer dat begint met een minuut niets geeft anders een deling door 0.
    ref = wav.mean(0)
    gem, spreiding = ref.mean(), ref.std() + 1e-8
    wav = (wav - gem) / spreiding

    herstel = _met_voortgang(melden) if melden else (lambda: None)
    try:
        with torch.no_grad():
            bronnen = apply_model(model, wav[None], device=dev, shifts=1,
                                  split=True, overlap=0.25, progress=bool(melden))[0]
    finally:
        herstel()
    bronnen = bronnen * spreiding + gem

    i_zang = model.sources.index("vocals")
    zang = bronnen[i_zang]
    begeleiding = bronnen.sum(0) - zang

    os.makedirs(doelmap, exist_ok=True)
    # 24 bit: het gaat nog één keer door een encoder, dan is 16 bit zonde.
    sf.write(os.path.join(doelmap, "zang.flac"), zang.cpu().numpy().T, SAMPLERATE, subtype="PCM_24")
    sf.write(os.path.join(doelmap, "begeleiding.flac"), begeleiding.cpu().numpy().T, SAMPLERATE, subtype="PCM_24")
    return {"apparaat": dev, "model": MODEL, "seconden": round(time.time() - begin, 1),
            "lengte": round(data.shape[0] / SAMPLERATE, 1)}


# ── RoFormer: de schone tweede ronde ────────────────────────────────────────
#
# BS-RoFormer haalt de stem veel schoner weg dan Demucs (minder zang die in de
# begeleiding achterblijft, vooral bij galm en achtergrondkoren), maar kost op
# de Studio twee tot vier minuten per nummer in plaats van zeven seconden. Dus
# is het een TWEEDE ronde: Demucs eerst, zodat je meteen kunt zingen, en dit
# erachteraan; de app wisselt ongemerkt zodra het klaar is.
#
# Het draait als los programma uit een eigen venv (audio-separator, MIT): die
# heeft andere eisen dan Demucs, en zo kan het één het ander niet breken. Het
# model (639 MB) staat in KARAOKE_ROFORMER_MODELLEN — downloaden lukt op de
# Studio niet betrouwbaar, dus het wordt er met de hand neergezet.

_HIER = os.path.dirname(os.path.abspath(__file__))
ROFORMER_BIN = os.environ.get("KARAOKE_ROFORMER_BIN", os.path.join(_HIER, "venv-roformer", "bin", "audio-separator"))
ROFORMER_MODELLEN = os.environ.get("KARAOKE_ROFORMER_MODELLEN", os.path.join(_HIER, "modellen"))
ROFORMER_MODEL = os.environ.get("KARAOKE_ROFORMER_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
# Overlap 4: tussen de beste kwaliteit (8, ~4 min) en snel (2, ~2 min) in.
ROFORMER_OVERLAP = os.environ.get("KARAOKE_ROFORMER_OVERLAP", "4")


def roformer_beschikbaar() -> bool:
    return os.path.isfile(ROFORMER_BIN) and os.path.isfile(os.path.join(ROFORMER_MODELLEN, ROFORMER_MODEL))


def scheiden_roformer(bron: str, doelmap: str, melden=None) -> dict:
    """Zelfde belofte als scheiden(): `doelmap/zang.flac` + `doelmap/begeleiding.flac`."""
    import soundfile as sf
    begin = time.time()
    lengte = sf.info(bron).duration
    uit = os.path.join(doelmap, "roformer-uit")
    os.makedirs(uit, exist_ok=True)
    omgeving = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1",
                    PATH="/opt/homebrew/bin:" + os.environ.get("PATH", "/usr/bin:/bin"))
    # stderr naar een bestand, niet een pijp: audio-separator logt en tekent een
    # voortgangsbalk, en een volle pijp die pas na afloop gelezen wordt zet het
    # programma stil — en daarmee _roformer_slot voorgoed (zie akkoorden()).
    log = open(os.path.join(doelmap, "roformer.log"), "w+b")
    proces = subprocess.Popen(
        [ROFORMER_BIN, bron, "-m", ROFORMER_MODEL, "--model_file_dir", ROFORMER_MODELLEN,
         "--output_dir", uit, "--output_format", "FLAC", "--mdxc_overlap", ROFORMER_OVERLAP],
        stdout=subprocess.DEVNULL, stderr=log, env=omgeving)
    # Voortgang: het programma meldt die niet bruikbaar, dus schatten op de
    # tijd — ongeveer 0,7 seconde rekenen per seconde muziek bij overlap 4.
    verwacht = max(30.0, lengte * 0.7)
    grens = time.time() + max(20 * 60, verwacht * 6)
    while proces.poll() is None:
        if time.time() > grens:
            proces.kill()
            proces.wait(timeout=30)
            log.close()
            raise RuntimeError("audio-separator: duurde te lang, afgebroken")
        if melden:
            melden(min(0.95, (time.time() - begin) / verwacht))
        time.sleep(1)
    log.seek(0)
    fout = log.read().decode(errors="replace")
    log.close()
    if proces.returncode != 0:
        raise RuntimeError("audio-separator: " + (fout.strip().splitlines() or ["?"])[-1][-200:])
    stem = {}
    for naam in os.listdir(uit):
        if "(Vocals)" in naam:
            stem["zang"] = naam
        elif "(Instrumental)" in naam:
            stem["begeleiding"] = naam
    if set(stem) != {"zang", "begeleiding"}:
        raise RuntimeError(f"onverwachte uitvoer: {os.listdir(uit)}")
    for wat, naam in stem.items():
        os.replace(os.path.join(uit, naam), os.path.join(doelmap, f"{wat}.flac"))
    shutil.rmtree(uit, ignore_errors=True)
    return {"apparaat": "mps", "model": "bs_roformer", "seconden": round(time.time() - begin, 1),
            "lengte": round(lengte, 1)}


# ── Noten: wat piano, gitaar en bas écht spelen ─────────────────────────────
#
# Voor het meespeelscherm. Akkoorden zeggen welke noten er ongeveer klinken,
# maar niet wanneer: een arpeggio, een loopje of een melodie komt eruit als
# "alles tegelijk". Hier wordt het noot voor noot, in twee stappen:
#
#   1. **Demucs `htdemucs_6s`** — zes delen in plaats van twee: drums, bas,
#      overig, zang, gitaar en piano. Alleen gitaar, piano en bas gaan door.
#      Het 6s-model is wat minder schoon dan htdemucs (de piano lekt soms in
#      "overig"), maar het is het enige vrije model dat gitaar en piano los
#      kan maken, en voor noten hoeft het niet om te horen te zijn.
#   2. **Basic Pitch** (Spotify, Apache-2.0) — van geluid naar noten, ook
#      meerdere tegelijk. Een klein net (20.000 gewichten), op de CPU met
#      ONNX of CoreML, zonder TensorFlow.
#
# Basic Pitch draait als los programma uit een eigen venv (`venv-noten`), net
# als RoFormer: het wil librosa, scikit-learn en onnxruntime/coremltools, en
# dat hoort niet in de venv van Demucs waar karaoke van afhangt. Het programma
# is dít bestand (`karaoke_demucs.py noten …`): dat importeert bij het laden
# alleen de standaardbibliotheek, dus het draait in elke venv, en er hoeft
# niets extra's naar de Studio.

NOTEN_MODEL = os.environ.get("KARAOKE_NOTEN_MODEL", "htdemucs_6s")
# Een map met het model als .yaml + .th, neergezet met de hand (zie
# docs/NOTEN.md). Staat die er niet, dan haalt Demucs het zelf op — en dat lukt
# op de Studio niet betrouwbaar.
NOTEN_REPO = os.environ.get("KARAOKE_NOTEN_REPO", os.path.join(_HIER, "modellen", "demucs"))
NOTEN_PYTHON = os.environ.get("KARAOKE_NOTEN_PYTHON", os.path.join(_HIER, "venv-noten", "bin", "python"))
# Welke delen, met hun naam in Demucs. De volgorde is ook de volgorde in de uitkomst.
NOTEN_DELEN = (("piano", "piano"), ("gitaar", "guitar"), ("bas", "bass"))
# Stiller dan dit ten opzichte van de hele mix = dat instrument speelt niet mee.
# Demucs zet in een deel zonder instrument nooit echt stilte: er lekt altijd
# wat van de rest in. Zonder deze drempel "speelt" in elk rocknummer een piano
# — de gitaar, maar dan slecht herkend. -24 dB is gemeten: een echte
# pianopartij zit er ruim boven, lekkage ruim onder.
NOTEN_STIL_DB = float(os.environ.get("KARAOKE_NOTEN_STIL_DB", "-24"))

# Per deel wat Basic Pitch moet doen en wat erna overblijft:
#   bereik   midi laag..hoog; daarbuiten kan het instrument niet (een "gitaarnoot"
#            onder de lage E is de bas die erin lekt)
#   poly     hoeveel noten tegelijk maximaal; meer = de zachtste gaan weg
#   min_ms   korter is een tik of een aanzet, geen noot
#   sterkte  zachter is ruis of nagalm (Basic Pitch' "amplitude", 0..1)
NOTEN_INSTELLINGEN = {
    # 0,35: daaronder zitten vooral boventonen (een "noot" een octaaf of
    # duodecime boven een echte); gemeten op Für Elise, waar de echte noten
    # tussen 0,45 en 0,75 liggen.
    "piano":  {"bereik": (21, 108), "poly": 6, "min_ms": 70, "sterkte": 0.35},
    "gitaar": {"bereik": (40, 88), "poly": 6, "min_ms": 70, "sterkte": 0.35},
    # Een baslijn is (bijna) altijd één noot tegelijk.
    "bas":    {"bereik": (28, 67), "poly": 1, "min_ms": 90, "sterkte": 0.30},
}
# Bovengrens per deel: een nummer van een kwartier met razendsnelle loopjes
# blijft zo onder de paar honderd kB.
NOTEN_MAX = 8000
# Lekkage tussen delen: is een deel veel zachter dan een ander (zoveel dB) en
# vallen de meeste van zijn noten samen met noten van dat andere deel (zelfde
# toonklasse, vrijwel tegelijk), dan is het geen eigen partij maar wat Demucs
# van het andere instrument niet goed kon toewijzen. Voorbeeld: Für Elise solo
# piano gaf een "baspartij" van 36 noten — allemaal de linkerhand.
LEK_DB = 6.0
LEK_DEEL = 0.6


def _is_lek(zwak: list[dict], sterk: list[dict]) -> bool:
    if not zwak or not sterk:
        return False
    begins: dict[int, list[dict]] = {}
    for n in sterk:
        begins.setdefault(int(n["t"] * 10), []).append(n)
    samen = 0
    for n in zwak:
        b = int(n["t"] * 10)
        if any(abs(m["t"] - n["t"]) <= 0.1 and m["p"] % 12 == n["p"] % 12
               for k in (b - 1, b, b + 1) for m in begins.get(k, ())):
            samen += 1
    return samen >= LEK_DEEL * len(zwak)


def noten_beschikbaar() -> bool:
    return os.path.isfile(NOTEN_PYTHON)


def _noten_repo() -> str | None:
    """De handmatige map als het model daar staat, anders None (= ophalen)."""
    try:
        if any(f.endswith(".th") for f in os.listdir(NOTEN_REPO)):
            return NOTEN_REPO
    except OSError:
        pass
    return None


def noten(bron: str, doelmap: str, melden=None) -> dict:
    """`bron` (FLAC, 44,1 kHz) → `doelmap/noten.json` = {piano, gitaar, bas}.

    Elk deel een lijst {t, d, p, v}: begin en duur in seconden, midi-toonhoogte,
    sterkte 0..1. Een leeg deel = dat instrument speelt niet mee.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    melden = melden or (lambda f: None)
    begin = time.time()
    model = _laad_model(NOTEN_MODEL, _noten_repo())
    dev = apparaat()

    data, sr = sf.read(bron, dtype="float32", always_2d=True)
    if sr != SAMPLERATE:
        raise ValueError(f"bron staat op {sr} Hz, verwacht {SAMPLERATE}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    ref = wav.mean(0)
    gem, spreiding = ref.mean(), ref.std() + 1e-8
    wav = (wav - gem) / spreiding

    # Scheiden is 0–60 % van de balk, de noten de rest. Het scheiden zelf in
    # de baan van karaoke (_reken_slot): twee keer Demucs tegelijk op de GPU is
    # niet sneller, en _met_voortgang buigt een functie in demucs.apply om die
    # er maar één keer is. Een karaokenummer wacht zo hooguit dit stukje; het
    # trage deel (Basic Pitch, op de CPU) houdt de GPU niet bezet.
    with _reken_slot:
        herstel = _met_voortgang(lambda f: melden(0.6 * f))
        try:
            with torch.no_grad():
                bronnen = apply_model(model, wav[None], device=dev, shifts=1,
                                      split=True, overlap=0.25, progress=True)[0]
        finally:
            herstel()
    bronnen = (bronnen * spreiding + gem).cpu().numpy()        # (delen, 2, n)
    mix_rms = float(np.sqrt(np.mean(np.square(data)))) + 1e-9

    # Ook binnen een nummer: een piano die alleen in het refrein meedoet, is in
    # het couplet een deel vol lekkage van de gitaar, en daar maakt Basic Pitch
    # vrolijk "pianonoten" van. Dus per venster dezelfde drempel als voor het
    # hele deel, ten opzichte van de mix op dat moment, en wat eronder valt
    # stil. Zacht in en uit (lineair tussen de vensters), anders maakt de
    # knip zelf een aanslag.
    venster = SAMPLERATE // 2
    n_vensters = max(1, data.shape[0] // venster)

    def per_venster(x):
        x = x[:n_vensters * venster].reshape(n_vensters, venster)
        return np.sqrt(np.mean(np.square(x), axis=1)) + 1e-9
    mix_venster = per_venster(data.mean(1))
    midden = (np.arange(n_vensters) + 0.5) * venster

    # Per deel mono naar schijf, als het luid genoeg is. Basic Pitch leest
    # mono op 22,05 kHz en zet zelf om; 16 bit is voor noten meer dan genoeg.
    os.makedirs(doelmap, exist_ok=True)
    delen, niveaus = [], {}
    for naam, demucs_naam in NOTEN_DELEN:
        mono = bronnen[model.sources.index(demucs_naam)].mean(0)
        db = 20 * np.log10(float(np.sqrt(np.mean(np.square(mono)))) / mix_rms + 1e-9)
        niveaus[naam] = round(float(db), 1)
        if db >= NOTEN_STIL_DB:
            aan = (20 * np.log10(per_venster(mono) / mix_venster) >= NOTEN_STIL_DB).astype("float32")
            mono = mono * np.interp(np.arange(mono.size), midden, aan).astype("float32")
            sf.write(os.path.join(doelmap, f"{naam}.flac"), mono, SAMPLERATE, subtype="PCM_16")
            delen.append(naam)
    del bronnen
    melden(0.6)
    na_scheiden = time.time() - begin

    uitkomst = {naam: [] for naam, _ in NOTEN_DELEN}
    lek: dict[str, str] = {}
    if delen:
        omgeving = dict(os.environ, PYTHONUNBUFFERED="1")
        # stderr naar een bestand (zie scheiden_roformer): anders kan een volle
        # pijp het programma stilzetten terwijl wij op stdout wachten.
        with open(os.path.join(doelmap, "noten.log"), "w+") as log:
            proces = subprocess.Popen([NOTEN_PYTHON, os.path.abspath(__file__), "noten", doelmap, *delen],
                                      stdout=subprocess.PIPE, stderr=log, text=True, env=omgeving)
            laatste: dict = {}
            for regel in proces.stdout:
                try:
                    d = json.loads(regel)
                except ValueError:
                    continue
                if "voortgang" in d:
                    melden(0.6 + 0.4 * float(d["voortgang"]))
                laatste = d
            proces.wait(timeout=60)
            log.seek(0)
            fout = log.read()
        if proces.returncode != 0 or not laatste.get("klaar"):
            raise RuntimeError("basic-pitch: " + (laatste.get("fout") or
                               (fout.strip().splitlines() or ["?"])[-1][-200:]))
        with open(os.path.join(doelmap, "noten.json")) as f:
            uitkomst.update(json.load(f))
        for zwak in delen:
            for sterk in delen:
                if (zwak != sterk and uitkomst[zwak] and niveaus[zwak] <= niveaus[sterk] - LEK_DB
                        and _is_lek(uitkomst[zwak], uitkomst[sterk])):
                    uitkomst[zwak] = []
                    lek[zwak] = sterk
    for naam in delen:
        try:
            os.remove(os.path.join(doelmap, f"{naam}.flac"))
        except OSError:
            pass
    with open(os.path.join(doelmap, "noten.json"), "w") as f:
        json.dump(uitkomst, f, separators=(",", ":"))
    return {"apparaat": dev, "model": NOTEN_MODEL, "seconden": round(time.time() - begin, 1),
            "scheiden": round(na_scheiden, 1), "lengte": round(data.shape[0] / SAMPLERATE, 1),
            "niveaus": niveaus, "lek": lek, "aantal": {k: len(v) for k, v in uitkomst.items()}}


# De bandmixer: alle zes delen van htdemucs_6s, elk in stereo, zodat de
# luisteraar zelf mengt (gitaar eruit, alleen de drums, de bas harder).
BAND_DELEN = (("zang", "vocals"), ("gitaar", "guitar"), ("bas", "bass"),
              ("drums", "drums"), ("piano", "piano"), ("rest", "other"))
BAND_BESTANDEN = tuple(f"band_{naam}.flac" for naam, _ in BAND_DELEN)


def band(bron: str, doelmap: str, melden=None) -> dict:
    """`bron` (FLAC, 44,1 kHz) → zes stereo-FLACs `doelmap/band_{deel}.flac`.

    Opgeteld zijn ze weer het origineel (dat is hoe Demucs scheidt), dus de
    server kan er elke verhouding van mengen. `niveaus` zegt per deel hoe luid
    het is ten opzichte van de mix: een nummer zonder piano heeft daar -40 dB,
    en dan grijst de app die schuif uit.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    melden = melden or (lambda f: None)
    begin = time.time()
    model = _laad_model(NOTEN_MODEL, _noten_repo())
    dev = apparaat()

    data, sr = sf.read(bron, dtype="float32", always_2d=True)
    if sr != SAMPLERATE:
        raise ValueError(f"bron staat op {sr} Hz, verwacht {SAMPLERATE}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    ref = wav.mean(0)
    gem, spreiding = ref.mean(), ref.std() + 1e-8
    wav = (wav - gem) / spreiding

    with _reken_slot:
        herstel = _met_voortgang(lambda f: melden(0.9 * f))
        try:
            with torch.no_grad():
                bronnen = apply_model(model, wav[None], device=dev, shifts=1,
                                      split=True, overlap=0.25, progress=True)[0]
        finally:
            herstel()
    bronnen = (bronnen * spreiding + gem).cpu().numpy()        # (delen, 2, n)
    mix_rms = float(np.sqrt(np.mean(np.square(data)))) + 1e-9

    os.makedirs(doelmap, exist_ok=True)
    niveaus = {}
    for naam, demucs_naam in BAND_DELEN:
        deel = bronnen[model.sources.index(demucs_naam)]
        rms = float(np.sqrt(np.mean(np.square(deel))))
        niveaus[naam] = round(20 * np.log10(rms / mix_rms + 1e-9), 1)
        sf.write(os.path.join(doelmap, f"band_{naam}.flac"), np.clip(deel.T, -1, 1),
                 SAMPLERATE, subtype="PCM_16")
    del bronnen
    melden(1.0)
    return {"apparaat": dev, "model": NOTEN_MODEL, "seconden": round(time.time() - begin, 1),
            "lengte": round(data.shape[0] / SAMPLERATE, 1), "niveaus": niveaus}


# Het tempo voor de DJ-decks: librosa's beat tracker in venv-akkoorden (daar
# staat librosa al). De eigen schatting in de server zat er soms een kwart
# naast (4/3); librosa vergist zich hooguit een octaaf (2× of ½×), en dat
# vangt SYNC in de app zelf op.
def tempo(bron: str, doelmap: str, melden=None) -> dict:
    """`bron` (elk formaat dat librosa leest) → `doelmap/tempo.json` = {bpm, telStart, tellen}."""
    begin = time.time()
    with open(os.path.join(doelmap, "tempo.log"), "w+") as log:
        r = subprocess.run([AKKOORDEN_PYTHON, os.path.abspath(__file__), "tempo", bron, doelmap],
                           stdout=subprocess.PIPE, stderr=log, text=True, timeout=300,
                           env=dict(os.environ, OMP_NUM_THREADS="2"))
        log.seek(0)
        fout = log.read()
    if r.returncode != 0 or not os.path.isfile(os.path.join(doelmap, "tempo.json")):
        raise RuntimeError("tempo: " + ((fout.strip().splitlines() or ["?"])[-1][-200:]))
    with open(os.path.join(doelmap, "tempo.json")) as f:
        d = json.load(f)
    return {"apparaat": "cpu", "model": "librosa", "seconden": round(time.time() - begin, 1),
            "lengte": d.get("lengte"), "bpm": d.get("bpm")}


def _tempo_los(bron: str, doelmap: str) -> int:
    import numpy as np
    import librosa
    y, sr = librosa.load(bron, sr=22050, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    _, tellen = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, start_bpm=120, units="time")
    tellen = np.asarray(tellen, dtype=float)
    uit = {"lengte": round(len(y) / sr, 2), "bpm": None, "telStart": 0.0, "tellen": []}
    if len(tellen) >= 8:
        p = float(np.median(np.diff(tellen)))
        k = np.round((tellen - tellen[0]) / p)
        # Het raster dat het best op alle tellen past, en daarvan de eerste tel.
        fase = float(np.median(tellen - k * p))
        fase -= np.floor(fase / p) * p
        uit.update(bpm=round(60.0 / p, 2), telStart=round(fase, 3),
                   tellen=[round(float(x), 3) for x in tellen])
    with open(os.path.join(doelmap, "tempo.json"), "w") as f:
        json.dump(uit, f, separators=(",", ":"))
    return 0


def _beperk_poly(noten_: list[dict], poly: int) -> list[dict]:
    """Nooit meer dan `poly` noten tegelijk: de sterkste blijven.

    Van sterk naar zwak: een noot mag erbij als er op zijn begin nog geen
    `poly` andere klinken. Bakjes van een seconde, zodat het niet elke noot
    tegen alle andere hoeft te leggen.
    """
    bakken: dict[int, list[dict]] = {}
    gehouden = []
    for n in sorted(noten_, key=lambda n: -n["v"]):
        t = n["t"]
        tegelijk = 0
        for b in range(int(t) - 30, int(t) + 1):       # geen noot is langer dan 30 s
            for m in bakken.get(b, ()):
                if m["t"] <= t < m["t"] + m["d"]:
                    tegelijk += 1
        if tegelijk >= poly:
            continue
        gehouden.append(n)
        bakken.setdefault(int(t), []).append(n)
    return sorted(gehouden, key=lambda n: (n["t"], n["p"]))


def _noten_los(map_: str, delen: list[str]) -> int:
    """In venv-noten: elk deel in `map_` door Basic Pitch, → `map_/noten.json`.

    Voortgang als JSON-regels op stdout, net als `_los`. Basic Pitch print zelf
    ook naar stdout ("Predicting MIDI for…"); dat gaat naar stderr.
    """
    import contextlib

    def melden(f: float) -> None:
        print(json.dumps({"voortgang": round(f, 3)}), flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict
            model = Model(ICASSP_2022_MODEL_PATH)
        uit: dict[str, list] = {}
        for i, naam in enumerate(delen):
            ins = NOTEN_INSTELLINGEN[naam]
            laag, hoog = ins["bereik"]
            with contextlib.redirect_stdout(sys.stderr):
                _, _, gebeurtenissen = predict(
                    os.path.join(map_, f"{naam}.flac"), model,
                    minimum_note_length=ins["min_ms"],
                    minimum_frequency=440.0 * 2 ** ((laag - 69) / 12),
                    maximum_frequency=440.0 * 2 ** ((hoog - 69) / 12),
                )
            rij = [{"t": round(float(a), 3), "d": round(float(b) - float(a), 3),
                    "p": int(p), "v": round(float(v), 2)}
                   for a, b, p, v, _ in gebeurtenissen
                   if laag <= int(p) <= hoog and float(v) >= ins["sterkte"]
                   and (float(b) - float(a)) * 1000 >= ins["min_ms"]]
            rij = _beperk_poly(rij, ins["poly"])
            if len(rij) > NOTEN_MAX:
                # Te veel: de zachtste weg, in de tijd laten staan.
                drempel = sorted((n["v"] for n in rij), reverse=True)[NOTEN_MAX - 1]
                rij = [n for n in rij if n["v"] >= drempel][:NOTEN_MAX]
            uit[naam] = rij
            melden((i + 1) / len(delen))
        with open(os.path.join(map_, "noten.json"), "w") as f:
            json.dump(uit, f, separators=(",", ":"))
    except Exception as e:
        print(json.dumps({"fout": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"klaar": True}), flush=True)
    return 0


# ── Akkoorden: BTC, een getraind model ──────────────────────────────────────
#
# Voor de gitaarknop. De server deed dit zelf, met sjablonen op een chromagram
# (routers/akkoorden.py), maar dat mist in een volle mix te veel: een basnoot
# of een melodienoot trekt het naar een buurakkoord. BTC (zie karaoke_btc.py)
# is getraind op met de hand gelabelde nummers en ziet dat verschil wel.
#
# Draait als los programma uit een eigen venv (`venv-akkoorden`: torch +
# librosa), net als de noten: librosa hoort niet in de venv van Demucs, en
# venv-noten heeft geen torch. Het programma is dít bestand
# (`karaoke_demucs.py akkoorden …`), met karaoke_btc.py ernaast. De gewichten
# (twee keer 12 MB) staan in KARAOKE_AKKOORDEN_MODELLEN — met de hand neergezet,
# want downloaden lukt op de Studio niet betrouwbaar. Zie docs/AKKOORDEN.md.
#
# Rekenen gebeurt op de CPU: het net is klein (een nummer van vier minuten is
# alles bij elkaar een seconde of twee). De GPU blijft zo vrij voor Demucs.

AKKOORDEN_PYTHON = os.environ.get("KARAOKE_AKKOORDEN_PYTHON",
                                  os.path.join(_HIER, "venv-akkoorden", "bin", "python"))
AKKOORDEN_MODELLEN = os.environ.get("KARAOKE_AKKOORDEN_MODELLEN", os.path.join(_HIER, "modellen", "btc"))
AKKOORDEN_APPARAAT = os.environ.get("KARAOKE_AKKOORDEN_APPARAAT", "cpu")


def akkoorden_beschikbaar() -> bool:
    return (os.path.isfile(AKKOORDEN_PYTHON)
            and os.path.isfile(os.path.join(_HIER, "karaoke_btc.py"))
            and all(os.path.isfile(os.path.join(AKKOORDEN_MODELLEN, n))
                    for n in ("btc_model.pt", "btc_model_large_voca.pt")))


def akkoorden(bron: str, doelmap: str, melden=None) -> dict:
    """`bron` (FLAC, elke samplerate) → `doelmap/akkoorden.json`.

    Hetzelfde formaat als de oude herkenning in de server: {segmenten,
    segmentenExpert, toonsoort, duur, bpm, telStart}, plus methode "btc".
    """
    melden = melden or (lambda f: None)
    begin = time.time()
    omgeving = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    # stderr naar een bestand en niet naar een pijp: librosa en torch zijn
    # spraakzaam, en een volle pijp die niemand leest zet het programma stil.
    with open(os.path.join(doelmap, "akkoorden.log"), "w+") as log:
        proces = subprocess.Popen([AKKOORDEN_PYTHON, os.path.abspath(__file__), "akkoorden", bron, doelmap],
                                  stdout=subprocess.PIPE, stderr=log, text=True, env=omgeving)
        laatste: dict = {}
        for regel in proces.stdout:
            try:
                d = json.loads(regel)
            except ValueError:
                continue
            if "voortgang" in d:
                melden(float(d["voortgang"]))
            laatste = d
        proces.wait(timeout=60)
        log.seek(0)
        fout = log.read()
    if proces.returncode != 0 or not laatste.get("klaar"):
        raise RuntimeError("btc: " + (laatste.get("fout") or
                           (fout.strip().splitlines() or ["?"])[-1][-200:]))
    return {"apparaat": AKKOORDEN_APPARAAT, "model": "btc", "seconden": round(time.time() - begin, 1),
            "lengte": laatste.get("lengte"), "segmenten": laatste.get("segmenten")}


def _akkoorden_los(bron: str, map_: str) -> int:
    """In venv-akkoorden: `bron` → `map_/akkoorden.json`. Voortgang als JSON op stdout."""
    def melden(f: float) -> None:
        print(json.dumps({"voortgang": round(f, 3)}), flush=True)
    try:
        sys.path.insert(0, _HIER)
        import karaoke_btc
        y = karaoke_btc.lees_audio(bron)
        uit = karaoke_btc.herken(y, AKKOORDEN_MODELLEN, AKKOORDEN_APPARAAT, melden)
        with open(os.path.join(map_, "akkoorden.json"), "w") as f:
            json.dump(uit, f, separators=(",", ":"))
    except Exception as e:
        print(json.dumps({"fout": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"klaar": True, "lengte": uit["duur"], "segmenten": len(uit["segmenten"])}), flush=True)
    return 0


# ── Songtekst uitlijnen (Whisper) ───────────────────────────────────────────
#
# Een tekst zonder tijdcodes krijgt per regel een begintijd: Whisper luistert
# naar het zangspoor en de tekst wordt daarop gelegd (karaoke_tekst.py, in
# venv-tekst met mlx-whisper op de GPU). Heeft de server de zang al (van de
# karaoke), dan stuurt hij die mee (?zang=1); anders eerst Demucs.

TEKST_PYTHON = os.environ.get("KARAOKE_TEKST_PYTHON", os.path.join(_HIER, "venv-tekst", "bin", "python"))
TEKST_MODEL = os.environ.get("KARAOKE_TEKST_MODEL", os.path.join(_HIER, "modellen", "whisper-turbo"))


def tekst_beschikbaar() -> bool:
    return (os.path.isfile(TEKST_PYTHON)
            and os.path.isfile(os.path.join(_HIER, "karaoke_tekst.py"))
            and os.path.isfile(os.path.join(TEKST_MODEL, "weights.safetensors")))


def tekst(bron: str, doelmap: str, melden=None, zang_al: bool = False) -> dict:
    """`bron` + `doelmap/tekst.txt` → `doelmap/tekst.json` = {regels: [{timeMs, line}], gevonden}."""
    melden = melden or (lambda f: None)
    begin = time.time()
    zang = bron
    if not zang_al:
        # Eerst de zang los, in de baan van karaoke (één keer Demucs tegelijk).
        with _reken_slot:
            scheiden(bron, doelmap, lambda f: melden(0.4 * f))
        zang = os.path.join(doelmap, "zang.flac")
    omgeving = dict(os.environ, PYTHONUNBUFFERED="1", KARAOKE_TEKST_MODEL=TEKST_MODEL)
    with open(os.path.join(doelmap, "tekst.log"), "w+") as log:
        proces = subprocess.Popen([TEKST_PYTHON, os.path.join(_HIER, "karaoke_tekst.py"), zang,
                                   os.path.join(doelmap, "tekst.txt"), os.path.join(doelmap, "tekst.json")],
                                  stdout=subprocess.PIPE, stderr=log, text=True, env=omgeving)
        laatste: dict = {}
        for regel in proces.stdout:
            try:
                d = json.loads(regel)
            except ValueError:
                continue
            if "voortgang" in d:
                melden((0.0 if zang_al else 0.4) + (0.6 if not zang_al else 1.0) * float(d["voortgang"]))
            laatste = d
        proces.wait(timeout=60)
        log.seek(0)
        fout = log.read()
    if proces.returncode != 0 or not laatste.get("klaar"):
        raise RuntimeError("tekst: " + (laatste.get("fout") or
                           (fout.strip().splitlines() or ["?"])[-1][-200:]))
    return {"apparaat": "mlx", "model": "whisper-turbo", "seconden": round(time.time() - begin, 1),
            "lengte": None, "gevonden": laatste.get("gevonden"), "regels": laatste.get("regels")}


# ── 2. Eenmalig, als los proces van de server ───────────────────────────────

def _los(bron: str, doelmap: str) -> int:
    """Eén nummer, voortgang als JSON-regels op stdout.

    De server leest die regels terwijl het loopt. Alles wat Demucs of torch zelf
    te melden heeft gaat naar stderr, zodat stdout schoon blijft.
    """
    def melden(f: float) -> None:
        print(json.dumps({"voortgang": round(f, 3)}), flush=True)
    try:
        info = scheiden(bron, doelmap, melden)
    except Exception as e:
        print(json.dumps({"fout": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"klaar": True, **info}), flush=True)
    return 0


# ── 1. De dienst op de host ──────────────────────────────────────────────────
#
# Een handvol adressen, met de standaardbibliotheek — geen FastAPI hier: dit
# draait in een eigen venv op de Mac, en hoe minder daarin staat, hoe minder er
# bij een macOS-update kan omvallen.
#
#   GET    /gezond                    leeft hij, en waarop rekent hij
#   POST   /taken                     body = de bron als FLAC → {"id": ...}
#   GET    /taken/{id}                {"status": wacht|bezig|klaar|mislukt, "voortgang", "fout"}
#   GET    /taken/{id}/zang.flac      het resultaat
#   GET    /taken/{id}/begeleiding.flac
#   POST   /taken?model=noten         idem, maar piano/gitaar/bas als noten (zie boven)
#   GET    /taken/{id}/noten.json     het resultaat daarvan
#   POST   /taken?model=akkoorden     idem, de akkoorden met BTC (zie boven)
#   GET    /taken/{id}/akkoorden.json het resultaat daarvan
#   POST   /taken?model=tekst[&zang=1] idem, de songtekst (X-Tekst, base64) uitgelijnd
#   GET    /taken/{id}/tekst.json     het resultaat daarvan
#   DELETE /taken/{id}                opruimen (gebeurt anders na een uur vanzelf)
#
# Eén nummer tegelijk. Twee tegelijk op de GPU is niet sneller, wel twee keer
# zoveel geheugen; de server stuurt er ook nooit meer dan één, maar de dienst
# rekent daar niet op.

MAX_BRON = 400 * 1024 * 1024        # een FLAC van 40 minuten; meer is geen nummer
BEWAAR_S = 3600

_taken: dict[str, dict] = {}
_taken_slot = threading.Lock()
_reken_slot = threading.Lock()
# Een eigen baan voor RoFormer: die rekent minuten, en een nieuw nummer dat
# Demucs in zeven seconden kan, mag daar niet achter wachten.
_roformer_slot = threading.Lock()
# En een voor de noten: Basic Pitch rekent op de CPU, tientallen seconden per
# nummer. Alleen het Demucs-stuk ervan leent _reken_slot (zie noten()).
_noten_slot = threading.Lock()
_band_slot = threading.Lock()
_tempo_slot = threading.Lock()
# En een voor de akkoorden: BTC op de CPU, een paar seconden per nummer. Los
# van de rest, zodat de gitaarknop niet achter een RoFormer-ronde wacht.
_akkoorden_slot = threading.Lock()
# En een voor het uitlijnen van songteksten (Whisper op de GPU, ~10–20 s).
_tekst_slot = threading.Lock()
_werkmap = ""


def _ruim_oude_op() -> None:
    nu = time.time()
    with _taken_slot:
        weg = [i for i, t in _taken.items() if nu - t["gemaakt"] > BEWAAR_S]
        for i in weg:
            shutil.rmtree(_taken.pop(i)["map"], ignore_errors=True)
        # Ook mappen die niemand meer kent: van voor een herstart van de dienst,
        # of van een taak die de server nooit opruimde. Die bleven anders voor
        # altijd staan (op 17-10 stonden er 17, samen bijna een gigabyte).
        bekend = {t["map"] for t in _taken.values()}
    try:
        for naam in os.listdir(_werkmap):
            pad = os.path.join(_werkmap, naam)
            if pad in bekend or not os.path.isdir(pad):
                continue
            if nu - os.path.getmtime(pad) > BEWAAR_S:
                shutil.rmtree(pad, ignore_errors=True)
    except OSError:
        pass


def _ruim_op_klok() -> None:
    """Elk kwartier opruimen, niet alleen als er een nieuwe taak binnenkomt."""
    while True:
        try:
            _ruim_oude_op()
        except Exception:
            pass
        time.sleep(900)


def _draai(tid: str) -> None:
    taak = _taken[tid]
    soort = taak.get("model")
    # Band pakt zelf _reken_slot voor het GPU-stuk (net als noten); hier dus
    # een eigen baan, anders wacht hij op zichzelf — en alle karaoke erachter.
    baan = {"roformer": _roformer_slot, "noten": _noten_slot, "band": _band_slot, "tempo": _tempo_slot,
            "akkoorden": _akkoorden_slot, "tekst": _tekst_slot}.get(soort, _reken_slot)
    with baan:
        taak["status"] = "bezig"
        try:
            def melden(f: float) -> None:
                taak["voortgang"] = f
            doe = {"roformer": scheiden_roformer, "noten": noten, "akkoorden": akkoorden, "band": band,
                   "tempo": tempo,
                   "tekst": lambda b, m, me: tekst(b, m, me, bool(taak.get("zang")))}.get(soort, scheiden)
            info = doe(os.path.join(taak["map"], "bron.flac"), taak["map"], melden)
            taak.update(status="klaar", voortgang=1.0, info=info)
            print(f"[karaoke] {tid}: klaar in {info['seconden']}s "
                  f"({info['lengte']}s audio, {info['apparaat']})", flush=True)
        except Exception as e:
            taak.update(status="mislukt", fout=f"{type(e).__name__}: {e}")
            print(f"[karaoke] {tid}: mislukt — {taak['fout']}", flush=True)
        finally:
            try:
                os.remove(os.path.join(taak["map"], "bron.flac"))
            except OSError:
                pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "jltamp-karaoke/1"

    def log_message(self, fmt, *args):     # stil: launchd-log vol met GET /taken/…
        pass

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mag(self) -> bool:
        """Een gedeeld geheim, als er een is ingesteld.

        De dienst luistert alleen op 127.0.0.1 en is dus van buiten de Mac niet
        te zien. Het geheim is voor als iemand hem later toch op een netwerkadres
        zet: dan kan niet zomaar iedereen er de GPU mee bezighouden.
        """
        sleutel = os.environ.get("KARAOKE_SLEUTEL", "")
        if sleutel and self.headers.get("X-Karaoke-Sleutel", "") != sleutel:
            self._json(403, {"fout": "verkeerde sleutel"})
            return False
        return True

    def _taak(self) -> tuple[str, dict | None, str]:
        delen = self.path.strip("/").split("/")
        tid = delen[1] if len(delen) > 1 else ""
        rest = delen[2] if len(delen) > 2 else ""
        with _taken_slot:
            return tid, _taken.get(tid), rest

    def do_GET(self):
        if self.path == "/gezond":
            try:
                dev = apparaat()
                import demucs  # noqa: F401 — staat het er écht?
            except Exception as e:
                return self._json(503, {"ok": False, "fout": str(e)})
            return self._json(200, {"ok": True, "model": MODEL, "apparaat": dev,
                                    "bezig": _reken_slot.locked(),
                                    "roformer": roformer_beschikbaar(),
                                    "noten": noten_beschikbaar(),
                                    "akkoorden": akkoorden_beschikbaar(),
                                    "tekst": tekst_beschikbaar(),
                                    "band": True,
                                    "tempo": akkoorden_beschikbaar()})
        if not self._mag():
            return
        if not self.path.startswith("/taken/"):
            return self._json(404, {"fout": "onbekend"})
        tid, taak, rest = self._taak()
        if not taak:
            return self._json(404, {"fout": "onbekende taak"})
        if not rest:
            return self._json(200, {k: taak.get(k) for k in ("status", "voortgang", "fout", "info")})
        if (rest not in ("zang.flac", "begeleiding.flac", "noten.json", "akkoorden.json", "tekst.json",
                         "tempo.json")
                + BAND_BESTANDEN or taak["status"] != "klaar"):
            return self._json(404, {"fout": "niet klaar"})
        pad = os.path.join(taak["map"], rest)
        if not os.path.isfile(pad):
            return self._json(404, {"fout": "niet klaar"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if rest.endswith(".json") else "audio/flac")
        self.send_header("Content-Length", str(os.path.getsize(pad)))
        self.end_headers()
        with open(pad, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 256 * 1024)

    def do_POST(self):
        if not self._mag():
            return
        pad, _, vraag = self.path.partition("?")
        if pad != "/taken":
            return self._json(404, {"fout": "onbekend"})
        model = ("roformer" if "model=roformer" in vraag
                 else "noten" if "model=noten" in vraag
                 else "akkoorden" if "model=akkoorden" in vraag
                 else "tekst" if "model=tekst" in vraag
                 else "band" if "model=band" in vraag
                 else "tempo" if "model=tempo" in vraag else "demucs")
        if model == "roformer" and not roformer_beschikbaar():
            return self._json(503, {"fout": "roformer staat hier niet"})
        if model == "noten" and not noten_beschikbaar():
            return self._json(503, {"fout": "venv-noten staat hier niet"})
        if model == "akkoorden" and not akkoorden_beschikbaar():
            return self._json(503, {"fout": "venv-akkoorden of de BTC-gewichten staan hier niet"})
        if model == "tekst" and not tekst_beschikbaar():
            return self._json(503, {"fout": "venv-tekst of het Whisper-model staat hier niet"})
        tekst_b64 = self.headers.get("X-Tekst") or ""
        if model == "tekst" and not tekst_b64:
            return self._json(400, {"fout": "geen tekst (X-Tekst)"})
        lengte = int(self.headers.get("Content-Length") or 0)
        if lengte <= 0 or lengte > MAX_BRON:
            return self._json(413, {"fout": "bron leeg of te groot"})
        _ruim_oude_op()
        tid = uuid.uuid4().hex
        map_ = tempfile.mkdtemp(prefix=f"{tid}-", dir=_werkmap)
        with open(os.path.join(map_, "bron.flac"), "wb") as f:
            over = lengte
            while over > 0:
                stuk = self.rfile.read(min(over, 256 * 1024))
                if not stuk:
                    break
                f.write(stuk)
                over -= len(stuk)
        if over > 0:
            shutil.rmtree(map_, ignore_errors=True)
            return self._json(400, {"fout": "bron niet helemaal ontvangen"})
        if model == "tekst":
            import base64
            try:
                with open(os.path.join(map_, "tekst.txt"), "w", encoding="utf-8") as f:
                    f.write(base64.b64decode(tekst_b64).decode("utf-8"))
            except Exception:
                shutil.rmtree(map_, ignore_errors=True)
                return self._json(400, {"fout": "tekst onleesbaar"})
        with _taken_slot:
            _taken[tid] = {"status": "wacht", "voortgang": 0.0, "fout": None,
                           "map": map_, "gemaakt": time.time(), "model": model,
                           "zang": "zang=1" in vraag}
        threading.Thread(target=_draai, args=(tid,), daemon=True).start()
        return self._json(202, {"id": tid})

    def do_DELETE(self):
        if not self._mag():
            return
        tid, taak, _ = self._taak()
        if taak and taak["status"] in ("klaar", "mislukt"):
            with _taken_slot:
                _taken.pop(tid, None)
            shutil.rmtree(taak["map"], ignore_errors=True)
        return self._json(200, {"ok": True})


def _dienst(adres: str, poort: int, werkmap: str) -> int:
    global _werkmap
    os.makedirs(werkmap, exist_ok=True)
    _werkmap = werkmap
    # Het model meteen laden: dan merkt launchd bij het opstarten al dat er iets
    # mis is (en staat het in het logboek), in plaats van bij het eerste nummer.
    _laad_model()
    threading.Thread(target=_ruim_op_klok, name="opruimen", daemon=True).start()
    print(f"[karaoke] {MODEL} op {apparaat()}, luistert op {adres}:{poort}", flush=True)
    ThreadingHTTPServer((adres, poort), _Handler).serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Demucs-scheiding voor de JLTamp-karaoke")
    sub = p.add_subparsers(dest="wat", required=True)
    d = sub.add_parser("dienst", help="HTTP-dienst voor de server in de container")
    d.add_argument("--adres", default=os.environ.get("KARAOKE_ADRES", "127.0.0.1"))
    d.add_argument("--poort", type=int, default=int(os.environ.get("KARAOKE_POORT", "8425")))
    d.add_argument("--werkmap", default=os.environ.get(
        "KARAOKE_WERKMAP", os.path.join(tempfile.gettempdir(), "jltamp-karaoke")))
    l = sub.add_parser("los", help="één bestand scheiden (voortgang als JSON op stdout)")
    l.add_argument("bron")
    l.add_argument("doelmap")
    n = sub.add_parser("noten", help="(in venv-noten) delen in een map naar noten.json")
    n.add_argument("map")
    n.add_argument("delen", nargs="+", choices=[d for d, _ in NOTEN_DELEN])
    t = sub.add_parser("noten-proef", help="één bestand helemaal: scheiden + noten (zonder dienst)")
    t.add_argument("bron")
    t.add_argument("doelmap")
    tp = sub.add_parser("tempo", help="(in venv-akkoorden) één bestand → doelmap/tempo.json")
    tp.add_argument("bron")
    tp.add_argument("doelmap")
    k = sub.add_parser("akkoorden", help="(in venv-akkoorden) één bestand → doelmap/akkoorden.json")
    k.add_argument("bron")
    k.add_argument("doelmap")
    a = p.parse_args(argv)
    if a.wat == "dienst":
        return _dienst(a.adres, a.poort, a.werkmap)
    if a.wat == "noten":
        return _noten_los(a.map, a.delen)
    if a.wat == "akkoorden":
        return _akkoorden_los(a.bron, a.doelmap)
    if a.wat == "tempo":
        return _tempo_los(a.bron, a.doelmap)
    if a.wat == "noten-proef":
        info = noten(a.bron, a.doelmap, lambda f: print(f"\r{f:4.0%}", end="", file=sys.stderr))
        print(json.dumps(info), flush=True)
        return 0
    return _los(a.bron, a.doelmap)


if __name__ == "__main__":
    sys.exit(main())
