"""Akkoorden: welke akkoorden er in een nummer zitten, uit het geluid zelf.

Voor de gitaarknop in de songtekst: wie meespeelt, wil zien welk akkoord er nu
klinkt en welk er komt. Een akkoordensite kent maar een deel van deze
bibliotheek (veel Nederlands en obscuur materiaal), dus wordt het uit de audio
gehaald. Twee routes:

  * **BTC op de karaoke-dienst** (sinds versie 6, de gewone route). Een
    getraind model — de Bi-directional Transformer for Chord recognition —
    in `karaoke_btc.py`, dat op de Mac Studio draait (`POST
    /taken?model=akkoorden`). Gebruikt zodra de dienst `"akkoorden": true`
    meldt in /gezond. Op GuitarSet (echte opnames met bekende akkoorden) 78 %
    goed tegen 61 % voor de route hieronder, en de septiemen voor expert
    65 % tegen 24 %. Zie docs/AKKOORDEN.md.
  * **De eigen herkenning hieronder**, met wat er al in het image zit (librosa,
    numpy, scipy), als de dienst er niet is of het daar misgaat. Geen neuraal
    net, geen extra pakket van een paar honderd MB. In een los proces
    (`_herken_los`): librosa/numba kan hier segfaulten.

De eigen herkenning, in het kort:

  1. **De bron.** Is er een karaokeversie (routers/karaoke.py), dan de
     begeleiding daarvan: zang zit vol glijtonen en versieringen die in een
     chromagram als "verkeerde" noten opduiken. Anders het origineel.
  2. **Per tel.** De tellen via `librosa.beat.beat_track`. Akkoorden wisselen
     bijna altijd op een tel, dus is één meting per tel zowel rustiger als
     sneller dan één per frame. Geen bruikbare tellen (ambient, een rubato
     piano) → een vast raster van een halve seconde.
  3. **Chroma.** CQT-chroma op het harmonische deel (`hpss`), zodat drums en
     tikken niet als twaalf noten tegelijk meetellen. In blokken van een
     halve minuut: een hele STFT van een nummer van een kwartier kost
     honderden MB, en de server draait met 1500 MB.
  4. **Sjablonen.** 24 akkoorden (12 majeur, 12 mineur) als drieklank, plus
     "geen akkoord" (N) voor stiltes en lawaai.
  5. **Gladstrijken.** Een HMM met Viterbi (`librosa.sequence.viterbi`) en een
     overgangsmatrix die liever blijft dan wisselt. Zonder dat verspringt het
     akkoord bij elke doorgangsnoot in de bas.
  6. **Segmenten.** Gelijke tellen samen, en wat korter is dan
     `MIN_SEGMENT_S` gaat op in de buren: een akkoord dat je niet kunt
     pakken voor het weer weg is, helpt niemand.

Het blijft een schatting. Drieklanken herkennen gaat goed; septiemakkoorden,
sus en slash-akkoorden worden hun dichtstbijzijnde drieklank. Voor meespelen
met een gitaar is dat precies het niveau dat je wilt.

Bewaard in `DATA_DIR/akkoorden/<track_id>.json`, met het bronkenmerk (pad,
grootte, mtime) zodat een hernoemd of herschreven bestand opnieuw wordt
berekend. Alles naar `/data`, nooit naar `/music` — zie server/CLAUDE.md.

Achter de schakelaar `akkoorden` (meta `akkoorden_aan`), standaard UIT.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..db import SessionLocal
from ..deps import require_user
from ..models import Track, User
from . import karaoke

router = APIRouter(tags=["akkoorden"])
log = logging.getLogger("jltamp.akkoorden")

AKKOORDEN_DIR = config.DATA_DIR / "akkoorden"

# Verhoog dit als de herkenning verandert: dan wordt alles wat er al ligt bij
# de volgende vraag opnieuw berekend in plaats van dat oude uitkomsten blijven.
VERSIE = 6   # 6: BTC via de dienst; 5: bpm/telStart; 4: segmentenExpert; 3: vaste stap; 2: N strenger

SR = 22050
HOP = 512
# Een DJ-mix van een uur is geen nummer om op mee te spelen, en houdt de rij bezet.
MAX_MINUTEN = int(os.environ.get("JLTAMP_KARAOKE_MAX_MIN", "15") or 15)
MAX_RIJ = 20
MISLUKT_S = 600
# Korter dan dit is geen akkoord om te pakken maar een doorgang.
MIN_SEGMENT_S = 0.8
# Een uitkomst van de eigen herkenning terwijl BTC er inmiddels wél is: opnieuw,
# maar niet vaker dan dit. Zo rekent een dienst die steeds faalt niet bij elke
# vraag van de app opnieuw.
OPNIEUW_NA_S = 3600
WERK_DIR = AKKOORDEN_DIR / "_werk"
# Wat hier bij het opstarten nog staat, is van een taak die een herstart van de
# server afbrak (de `finally` die opruimt kwam nooit): weg ermee. Er draait één
# serverproces, dus er is niemand die er nu nog mee bezig is.
shutil.rmtree(WERK_DIR, ignore_errors=True)
# Kans dat het akkoord op de volgende tel hetzelfde blijft. Hoog, omdat een
# akkoord meestal een maat of langer duurt; te hoog en snelle wisselingen
# (twee tellen per akkoord) worden overgeslagen.
BLIJF_KANS = 0.93   # per stap van ~0,19 s (was 0,75 per tel)
# Frames per stap: 8 × 512 / 22050 ≈ 0,19 s.
STAP_FRAMES = 8
# Hoe zwaar "geen akkoord" telt tegen een echte drieklank (zie herken()).
N_WEGING = 0.72

GRONDTONEN = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# Mineur met de gebruikelijke spelling op de gitaar: G#m en niet Abm.
MINEUR = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")
NAMEN = (*GRONDTONEN, *MINEUR, "N")


# ── de schakelaar ───────────────────────────────────────────────────────────

def akkoorden_aan(user: User | None = None) -> bool:
    """Staat de gitaarknop aan voor deze gebruiker? Uit voor een beoordelaar,
    zoals alles wat het huis aanzet en op de achtergrond rekent."""
    if user is not None and bool(getattr(user, "is_review", False)):
        return False
    from .admin_users import schakelaar
    db = SessionLocal()
    try:
        return schakelaar(db, "akkoorden")
    finally:
        db.close()


# ── de herkenning ───────────────────────────────────────────────────────────

def _decodeer(pad: str):
    """Mono, 22,05 kHz, als float32. Via ffmpeg (zit in het image en leest
    alles, ook de m4a van de karaoke); waar dat ontbreekt, zoals op een
    ontwikkelmachine, via librosa/soundfile."""
    import numpy as np
    if shutil.which("ffmpeg"):
        uit = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", pad, "-vn", "-ac", "1", "-ar", str(SR),
             "-t", str(MAX_MINUTEN * 60), "-f", "f32le", "-"],
            capture_output=True, timeout=300,
        ).stdout
        return np.frombuffer(uit, dtype="<f4").copy()
    import librosa
    y, _ = librosa.load(pad, sr=SR, mono=True, duration=MAX_MINUTEN * 60)
    return y.astype("float32")


def _kenmerken(y, np, librosa):
    """CQT-chroma van het harmonische deel én de onset-curve voor de tellen,
    in blokken van een halve minuut.

    In blokken omdat het geheugen anders met de lengte meegroeit: op een
    nummer van vier minuten in één keer kwam er ruim een halve GB bij, en de
    server draait met 1500 MB (naast een scan en de rest). Per blok blijft het
    rond de 150 MB, hoe lang het nummer ook is.

    Een blok is een veelvoud van HOP, zodat de frames van opeenvolgende blokken
    netjes aansluiten; het laatste frame van een blok (half over de rand) valt
    weg, het eerste van het volgende neemt het over.
    """
    blok = 1280 * HOP           # ≈ 30 s
    chroma, onset = [], []
    for begin in range(0, y.size, blok):
        stuk = y[begin:begin + blok]
        if stuk.size < SR:          # een restje van minder dan een tel
            break
        harm = librosa.effects.harmonic(stuk, margin=2.0)
        c = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP)
        del harm
        o = librosa.onset.onset_strength(y=stuk, sr=SR, hop_length=HOP)
        n = min(c.shape[1], o.size)
        if begin + blok < y.size:
            n -= 1
        chroma.append(c[:, :n])
        onset.append(o[:n])
    if not chroma:
        return np.zeros((12, 0)), np.zeros(0)
    return np.concatenate(chroma, axis=1), np.concatenate(onset)


def _tel_grenzen(onset, np, librosa):
    """Frame-grenzen van de stappen: de tellen, of een vast raster als die er niet zijn."""
    n_frames = onset.size
    tellen = np.array([], dtype=int)
    try:
        # Het tempo uit hooguit drie minuten rond het midden. beat_track schat
        # het anders zelf via een tempogram van het hele nummer, en dat groeit
        # mee: bij een kwartier kwam er zo een halve GB bij. Het tempo van een
        # nummer zit in elke drie minuten ervan.
        midden, half = n_frames // 2, int(90 * SR / HOP)
        stuk = onset[max(0, midden - half):midden + half]
        bpm_schat = float(np.atleast_1d(librosa.feature.tempo(
            onset_envelope=stuk, sr=SR, hop_length=HOP))[0])
        # Niet librosa.beat.beat_track: die rekent via numba (localmax), en
        # dat crashte op de Studio (ARM, in Docker) met een segmentatiefout die
        # de hele server meenam. Dus zelf, met gewone numpy: een raster op het
        # geschatte tempo, gelegd waar de aanslagen het sterkst samenvallen.
        bpm = bpm_schat
        if 50 <= bpm <= 220:
            periode = 60.0 * SR / HOP / bpm                  # frames per tel
            fasen = np.arange(0.0, periode, 0.5)
            beste, beste_som = 0.0, -1.0
            for f in fasen:
                idx = np.round(np.arange(f, n_frames, periode)).astype(int)
                idx = idx[idx < n_frames]
                som = float(onset[idx].sum())
                if som > beste_som:
                    beste, beste_som = f, som
            tellen = np.round(np.arange(beste, n_frames, periode)).astype(int)
        else:
            tellen = np.array([], dtype=int)
    except Exception as e:
        log.info("[akkoorden] tellen mislukt: %s", e)
    duur = n_frames * HOP / SR
    # Te weinig tellen (minder dan een per anderhalve seconde) = geen ritme om
    # op te leunen. Dan een halve seconde per stap.
    if len(tellen) < max(4, duur / 1.5):
        stap = int(round(0.5 * SR / HOP))
        tellen = np.arange(stap, n_frames, stap)
    tellen = tellen[(tellen > 0) & (tellen < n_frames)]
    return np.unique(np.concatenate([[0], tellen, [n_frames]])).astype(int)


def _tempo_en_fase(onset, np, librosa) -> tuple[float | None, float]:
    """(bpm, eerste tel in seconden) voor het slagpatroon in de app.

    Zelfde aanpak als vroeger het telraster: het tempo uit drie minuten rond
    het midden, de fase waar de aanslagen het sterkst samenvallen. Alleen voor
    het tonen van op/neer-slagen; de akkoorden zelf hangen er niet meer van af.
    """
    try:
        n = onset.size
        midden, half = n // 2, int(90 * SR / HOP)
        bpm = float(np.atleast_1d(librosa.feature.tempo(
            onset_envelope=onset[max(0, midden - half):midden + half], sr=SR, hop_length=HOP))[0])
        if not 50 <= bpm <= 220:
            return None, 0.0
        periode = 60.0 * SR / HOP / bpm
        beste, som_beste = 0.0, -1.0
        for f in np.arange(0.0, periode, 0.5):
            idx = np.round(np.arange(f, n, periode)).astype(int)
            som = float(onset[idx[idx < n]].sum())
            if som > som_beste:
                beste, som_beste = f, som
        return round(bpm, 2), round(beste * HOP / SR, 3)
    except Exception:
        return None, 0.0


def _sjablonen(np):
    """24 drieklanken + N. Grondtoon iets zwaarder: die zit ook in de bas, en
    het onderscheidt bv. Am (A-C-E) van C6-achtige klanken."""
    rij = []
    for kwint_terts in ((4, 7), (3, 7)):
        for r in range(12):
            v = np.zeros(12)
            v[r] = 1.2
            v[(r + kwint_terts[0]) % 12] = 1.0
            v[(r + kwint_terts[1]) % 12] = 1.0
            rij.append(v / np.linalg.norm(v))
    return np.array(rij)       # (24, 12)


# Expert: naast de 24 drieklanken ook septiemen en sus-akkoorden.
RIJKE_SOORTEN = [
    ("", (0, 4, 7)), ("m", (0, 3, 7)),
    ("7", (0, 4, 7, 10)), ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)),
    ("sus2", (0, 2, 7)), ("sus4", (0, 5, 7)),
]
GRONDEN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _rijke_sjablonen(np):
    namen, rij = [], []
    for achter, iv in RIJKE_SOORTEN:
        for r in range(12):
            v = np.zeros(12)
            for k, x in enumerate(iv):
                v[(r + x) % 12] = 1.2 if k == 0 else (0.85 if x in (10, 11, 2, 5) else 1.0)
            namen.append(GRONDEN[r] + achter)
            rij.append(v / np.linalg.norm(v))
    return namen, np.array(rij)


def herken(y) -> dict:
    """{segmenten: [{start, eind, akkoord}], toonsoort, duur} uit monsters op SR."""
    import numpy as np
    import librosa

    duur = y.size / SR
    if y.size < SR * 2:
        return {"segmenten": [], "toonsoort": None, "duur": round(duur, 2)}

    chroma, onset = _kenmerken(y, np, librosa)
    if chroma.shape[1] < 4:
        return {"segmenten": [], "toonsoort": None, "duur": round(duur, 2)}
    # Een vaste stap van ~0,19 s in plaats van de tellen. Een telraster op één
    # geschat tempo schuift weg zodra het tempo een fractie afwijkt (live, een
    # band die versnelt): na een paar minuten zat het er seconden naast. Met
    # een vaste stap kan niets wegschuiven; het gladstrijken (Viterbi, BLIJF_KANS)
    # houdt het rustig. Gemeten op If I Ain't Got You: dezelfde akkoorden,
    # wissels binnen een paar tienden van het telraster.
    grenzen = np.unique(np.concatenate([np.arange(0, chroma.shape[1], STAP_FRAMES),
                                        [chroma.shape[1]]])).astype(int)

    # Per stap de mediaan: een uitschieter (een tik, een glijtoon) telt dan
    # niet mee.
    stappen = np.stack([np.median(chroma[:, a:b], axis=1)
                        for a, b in zip(grenzen[:-1], grenzen[1:])], axis=1)
    # Per stap de mediaan van de luidheid per frame, rechtstreeks uit de
    # monsters (librosa.feature.rms maakt eerst een kopie van het hele nummer
    # in frames van 2048 — 75 MB voor niets). De mediaan, zodat een stap die
    # net over de rand van een stilte valt niet door dat randje luid lijkt.
    nf = min(chroma.shape[1], y.size // HOP)
    frame_rms = np.sqrt(np.mean(np.square(y[:nf * HOP].reshape(nf, HOP)), axis=1))
    frame_rms = np.pad(frame_rms, (0, chroma.shape[1] - nf))
    luid = np.array([float(np.median(frame_rms[a:b])) for a, b in zip(grenzen[:-1], grenzen[1:])])
    stil = luid < max(1e-4, 0.05 * float(np.percentile(luid, 90)))

    normen = np.linalg.norm(stappen, axis=0) + 1e-9
    eenheid = stappen / normen
    score = _sjablonen(np) @ eenheid                 # (24, stappen), cosinus 0..1
    # N: hoe vlak het chromagram is. Bij lawaai of een drumbreak zijn alle
    # twaalf noten ongeveer even sterk; een drieklank scoort dan niet veel
    # hoger dan dat vlak.
    vlak = np.ones(12) / np.sqrt(12)
    # 0,72 en niet hoger: in een volle mix (zang, drums) is het chromagram van
    # nature vrij vlak, en met 0,92 won "geen akkoord" op bijna de helft van de
    # tellen — na het gladstrijken stond driekwart van het nummer op een
    # streepje. Gemeten op Rolling in the Deep: 0,92 → 48 %, 0,75 → 7 % N.
    n_score = (vlak @ eenheid) * N_WEGING
    n_score[stil] = 1.5
    alles = np.vstack([score, n_score])              # (25, stappen)

    # Naar kansen: een scherpe softmax. Het verschil tussen het beste en het
    # één-na-beste akkoord is vaak maar een paar honderdsten cosinus.
    kans = np.exp(25.0 * (alles - alles.max(axis=0, keepdims=True)))
    kans /= kans.sum(axis=0, keepdims=True)
    overgang = librosa.sequence.transition_loop(len(NAMEN), BLIJF_KANS)
    pad = librosa.sequence.viterbi(kans, overgang)

    tijden = librosa.frames_to_time(grenzen, sr=SR, hop_length=HOP)
    tijden[-1] = min(float(tijden[-1]), duur)
    segmenten = _samenvoegen([(float(tijden[i]), float(tijden[i + 1]), NAMEN[int(s)])
                              for i, s in enumerate(pad)])
    # Expert: dezelfde stappen, maar met de rijke sjablonen (7, maj7, m7,
    # sus2, sus4). Iets losser gladgestreken, zodat snelle wissels blijven.
    rnamen, rsj = _rijke_sjablonen(np)
    rscore = rsj @ eenheid
    ralles = np.vstack([rscore, n_score])
    rkans = np.exp(25.0 * (ralles - ralles.max(axis=0, keepdims=True)))
    rkans /= rkans.sum(axis=0, keepdims=True)
    rpad = librosa.sequence.viterbi(rkans, librosa.sequence.transition_loop(len(rnamen) + 1, 0.88))
    alle_namen = rnamen + ["N"]
    rijk = _samenvoegen([(float(tijden[i]), float(tijden[i + 1]), alle_namen[int(s_)])
                         for i, s_ in enumerate(rpad)], min_s=0.4)
    bpm, tel_start = _tempo_en_fase(onset, np, librosa)
    return {"segmenten": segmenten, "segmentenExpert": rijk, "toonsoort": _toonsoort(chroma, np),
            "duur": round(duur, 2), "bpm": bpm, "telStart": tel_start}


def _samenvoegen(stappen: list[tuple[float, float, str]], min_s: float = MIN_SEGMENT_S) -> list[dict]:
    """Gelijke opeenvolgende stappen samen; te korte segmenten opgaan in de buren."""
    segs: list[list] = []
    for a, b, k in stappen:
        if segs and segs[-1][2] == k:
            segs[-1][1] = b
        else:
            segs.append([a, b, k])
    # Steeds het kortste te korte segment weg, tot er geen meer is. Het gaat
    # naar de langste buur: dat is meestal het akkoord waar de doorgang bij
    # hoorde. Daarna buren met hetzelfde akkoord weer samen.
    while len(segs) > 1:
        kort = [i for i, s in enumerate(segs) if s[1] - s[0] < min_s]
        if not kort:
            break
        i = min(kort, key=lambda j: segs[j][1] - segs[j][0])
        links = segs[i - 1] if i > 0 else None
        rechts = segs[i + 1] if i + 1 < len(segs) else None
        if rechts is None or (links is not None and links[1] - links[0] >= rechts[1] - rechts[0]):
            links[1] = segs[i][1]
        else:
            rechts[0] = segs[i][0]
        del segs[i]
        samen: list[list] = []
        for s in segs:
            if samen and samen[-1][2] == s[2]:
                samen[-1][1] = s[1]
            else:
                samen.append(s)
        segs = samen
    return [{"start": round(a, 2), "eind": round(b, 2), "akkoord": k} for a, b, k in segs]


# Krumhansl-Kessler: hoe goed elke toon "past" in een majeur- of mineurtoonsoort.
_KK_MAJ = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MIN = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _toonsoort(chroma, np) -> str | None:
    """De toonsoort van het hele nummer, als akkoordnaam ("G", "Em").

    Een schatting op het gemiddelde chromagram; bij een nummer dat moduleert
    wint de toonsoort waar het het langst in staat.
    """
    if chroma.size == 0:
        return None
    gem = chroma.mean(axis=1)
    if not np.any(gem):
        return None
    best, naam = -2.0, None
    for profiel, namen in ((_KK_MAJ, GRONDTONEN), (_KK_MIN, MINEUR)):
        p = np.array(profiel)
        for r in range(12):
            c = float(np.corrcoef(gem, np.roll(p, r))[0, 1])
            if c > best:
                best, naam = c, namen[r]
    return naam


# ── de cache ────────────────────────────────────────────────────────────────

def _bestand(track_id: int) -> Path:
    return AKKOORDEN_DIR / f"{track_id}.json"


def _lees(t: Track) -> dict | None:
    """Wat er bewaard is, als het nog bij dit bestand en deze versie hoort."""
    try:
        d = json.loads(_bestand(t.id).read_text())
    except Exception:
        return None
    if d.get("versie") != VERSIE or d.get("bron") != karaoke._bron_kenmerk(t):
        return None
    return d


# ── de rij ──────────────────────────────────────────────────────────────────
#
# Eén werker, één nummer tegelijk, net als karaoke. Een nummer van vier minuten
# kost op de CPU een halve minuut tot een minuut; twee tegelijk maakt niets
# sneller en verdubbelt alleen het geheugen.

_rij: collections.deque[int] = collections.deque()
_staat: dict[int, dict] = {}       # track_id → {"status", "fout", "wanneer"}
_slot = threading.Lock()
_wekker = threading.Event()
_werker: threading.Thread | None = None


def _start_werker() -> None:
    global _werker
    with _slot:
        if _werker is not None and _werker.is_alive():
            return
        _werker = threading.Thread(target=_werk, name="akkoorden", daemon=True)
        _werker.start()


def _werk() -> None:
    while True:
        with _slot:
            track_id = _rij.popleft() if _rij else None
            if track_id is not None:
                _staat[track_id] = {"status": "bezig", "fout": None, "wanneer": time.time()}
        if track_id is None:
            _wekker.wait(60)
            _wekker.clear()
            continue
        try:
            _verwerk(track_id)
            with _slot:
                _staat.pop(track_id, None)
        except Exception as e:
            log.warning("[akkoorden] nummer %s mislukt: %s", track_id, e)
            with _slot:
                _staat[track_id] = {"status": "mislukt", "fout": str(e)[:200],
                                    "wanneer": time.time()}


def _herken_los(bron: str) -> dict:
    """De herkenning in een eigen proces.

    librosa rekent deels via numba, en dat is op de Studio (ARM, in Docker)
    al eens hard gecrasht: een segmentatiefout, geen exceptie — en in de
    server zelf nam die het hele proces mee, muziek van iedereen incluis. In
    een los proces is een crash alleen "dit nummer lukte niet".
    """
    wortel = str(Path(__file__).resolve().parents[2])
    code = ("import json, sys\n"
            "from app.routers.akkoorden import _decodeer, herken\n"
            "y = _decodeer(sys.argv[1])\n"
            "if y.size == 0:\n"
            "    raise SystemExit('bestand niet te lezen')\n"
            "print(json.dumps(herken(y)))\n")
    omgeving = dict(os.environ, PYTHONPATH=wortel, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code, bron], capture_output=True, text=True,
                       timeout=900, env=omgeving, cwd=wortel)
    if r.returncode != 0:
        fout = (r.stderr.strip().splitlines() or [f"rc={r.returncode}"])[-1][-200:]
        raise RuntimeError(f"herkenning stopte (rc={r.returncode}): {fout}")
    regels = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if not regels:
        raise RuntimeError("herkenning gaf niets terug")
    return json.loads(regels[-1])


def _dienst_akkoorden() -> bool:
    """Kan de karaoke-dienst akkoorden (BTC)? Een halve minuut onthouden door karaoke."""
    b = karaoke.beschikbaarheid()
    return bool(b.get("beschikbaar") and b.get("route") == "dienst" and b.get("akkoorden"))


def _via_dienst(track_id: int, bron: str) -> dict:
    """BTC op de karaoke-dienst. Zelfde uitkomst als herken(), plus methode "btc".

    De bron gaat als FLAC, mono op 22,05 kHz — wat BTC wil, en een kwart van
    een stereo-FLAC op 44,1 kHz. Van /music gelezen, en alleen gelezen.
    """
    WERK_DIR.mkdir(parents=True, exist_ok=True)
    flac = WERK_DIR / f"{track_id}.flac"
    try:
        karaoke._ffmpeg("-i", bron, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
                        "-t", str(MAX_MINUTEN * 60), "-c:a", "flac", str(flac))
        data = flac.read_bytes()
    finally:
        flac.unlink(missing_ok=True)
    with karaoke._dienst_verzoek("/taken?model=akkoorden", "POST", data, timeout=120) as r:
        tid = json.loads(r.read())["id"]
    try:
        einde = time.time() + 10 * 60
        while True:
            if time.time() > einde:
                raise RuntimeError("dienst duurde te lang")
            time.sleep(1.0)
            with karaoke._dienst_verzoek(f"/taken/{tid}") as r:
                st = json.loads(r.read())
            if st.get("status") == "klaar":
                break
            if st.get("status") == "mislukt":
                raise RuntimeError(st.get("fout") or "dienst: mislukt")
        with karaoke._dienst_verzoek(f"/taken/{tid}/akkoorden.json", timeout=60) as r:
            uit = json.loads(r.read())
    finally:
        try:
            karaoke._dienst_verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass
    if not isinstance(uit.get("segmenten"), list):
        raise RuntimeError("dienst gaf geen segmenten")
    return uit


def _verwerk(track_id: int) -> None:
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

    # De begeleiding als die er is: zonder zang klopt de herkenning veel vaker.
    # Die is van precies dit bestand gemaakt (karaoke._is_klaar kijkt naar het
    # bronkenmerk), dus de tijden lopen gelijk met het origineel.
    begeleiding = karaoke._map(t.id) / "begeleiding.m4a"
    uit_begeleiding = karaoke._is_klaar(t)
    bron = str(begeleiding) if uit_begeleiding else t.path

    begin = time.time()
    uitkomst = None
    if _dienst_akkoorden():
        try:
            uitkomst = _via_dienst(t.id, bron)
        except Exception as e:
            # Dan de eigen herkenning: liever iets dan niets. Na OPNIEUW_NA_S
            # wordt het (bij een volgende vraag) nog eens met BTC geprobeerd.
            log.warning("[akkoorden] nummer %s: BTC mislukt (%s), eigen herkenning", track_id, e)
    if uitkomst is None:
        uitkomst = {**_herken_los(bron), "methode": "librosa"}
    AKKOORDEN_DIR.mkdir(parents=True, exist_ok=True)
    doel = _bestand(t.id)
    tijdelijk = doel.with_suffix(".tmp")
    tijdelijk.write_text(json.dumps({
        "versie": VERSIE, "bron": karaoke._bron_kenmerk(t),
        "uitBegeleiding": uit_begeleiding, "gemaakt": int(time.time()),
        "seconden": round(time.time() - begin, 1), **uitkomst,
    }))
    # Pas als het hele bestand er staat: een half bestand mag nooit "klaar" lijken.
    os.replace(tijdelijk, doel)
    log.info("[akkoorden] nummer %s klaar in %.0fs (%s, %d segmenten, %s, %s)", track_id,
             time.time() - begin, uitkomst.get("methode"), len(uitkomst["segmenten"]),
             uitkomst.get("toonsoort"), "begeleiding" if uit_begeleiding else "origineel")


# ── de adressen ─────────────────────────────────────────────────────────────

def _status(t: Track, aan: bool) -> dict:
    uit = {"aan": aan, "status": "geen", "fout": None, "plekInRij": None,
           "segmenten": [], "toonsoort": None, "uitBegeleiding": False}
    if not aan:
        return uit
    d = _lees(t)
    if d is not None:
        uit.update(status="klaar", segmenten=d.get("segmenten") or [],
                   segmentenExpert=d.get("segmentenExpert") or [],
                   bpm=d.get("bpm"), telStart=d.get("telStart"),
                   toonsoort=d.get("toonsoort"), uitBegeleiding=bool(d.get("uitBegeleiding")),
                   methode=d.get("methode"))
        return uit
    with _slot:
        s = dict(_staat.get(t.id) or {})
        plek = list(_rij).index(t.id) + 1 if t.id in _rij else None
    if s:
        if s["status"] == "mislukt" and time.time() - s.get("wanneer", 0) > MISLUKT_S:
            return uit
        uit.update(status=s["status"], fout=s.get("fout"), plekInRij=plek)
    return uit


def _in_rij(track_id: int) -> None:
    with _slot:
        if track_id in _rij or (_staat.get(track_id) or {}).get("status") == "bezig":
            return
        if len(_rij) >= MAX_RIJ:
            raise HTTPException(429, "Even te druk — probeer het zo weer")
        _rij.append(track_id)
        _staat[track_id] = {"status": "wacht", "fout": None, "wanneer": time.time()}
    _start_werker()
    _wekker.set()


def _zonder_zang(t: Track) -> None:
    """Zorg dat er een begeleiding zonder zang komt, en reken daaruit opnieuw.

    Uit het origineel mist de herkenning veel: de zang zit boven op de
    akkoorden en trekt het chromagram scheef. De karaokescheiding kost op de
    Studio een seconde of tien, dus vragen we die zelf aan. Is hij er, en komen
    de akkoorden nog uit het origineel, dan opnieuw — de oude uitkomst blijft
    tot dan staan.

    Zelfde voor de methode: komt wat er ligt van de eigen herkenning terwijl
    BTC er nu wel is (de dienst lag eruit, of faalde), dan opnieuw — maar pas
    na OPNIEUW_NA_S, zodat een dienst die blijft falen niet elke vraag kost.
    """
    try:
        d = _lees(t)
        if (d is not None and d.get("methode") != "btc"
                and time.time() - float(d.get("gemaakt") or 0) > OPNIEUW_NA_S
                and _dienst_akkoorden()):
            _in_rij(t.id)
        if not karaoke._is_klaar(t):
            karaoke.vraag_scheiding(t.id)
            return
        if d is not None and not d.get("uitBegeleiding"):
            _in_rij(t.id)
    except Exception as e:   # druk of geen dienst: dan blijft het bij het origineel
        log.info("[akkoorden] zonder zang lukt nu niet: %s", e)


@router.get("/akkoorden/{track_id}")
def akkoorden_status(track_id: str, user: User = Depends(require_user)):
    """De akkoorden van dit nummer, of hoe ver het ermee staat. Verandert niets."""
    t = karaoke._track(track_id, user)
    aan = akkoorden_aan(user)
    if aan:
        _zonder_zang(t)
    return _status(t, aan)


@router.post("/akkoorden/{track_id}")
def akkoorden_aanvragen(track_id: str, user: User = Depends(require_user)):
    """Zet dit nummer in de rij, tenzij het al klaar is of er al in staat.

    Is het klaar maar uit het origineel berekend, en is er inmiddels een
    karaokeversie, dan wordt het op de achtergrond opnieuw gedaan — de oude
    uitkomst blijft tot dan gewoon staan.
    """
    t = karaoke._track(track_id, user)
    if not akkoorden_aan(user):
        raise HTTPException(403, "Akkoorden staan uit")
    _zonder_zang(t)
    st = _status(t, True)
    if st["status"] == "klaar":
        if not st["uitBegeleiding"] and karaoke._is_klaar(t):
            try:
                _in_rij(t.id)
            except HTTPException:
                pass        # druk: dan blijft de versie van het origineel staan
        return st
    if st["status"] in ("wacht", "bezig"):
        return st
    _in_rij(t.id)
    return _status(t, True)
