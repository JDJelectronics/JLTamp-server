"""Wat de DJ-decks over een nummer moeten weten: golfvorm, tempo en de tel.

De decks in de app laten een golfvorm lopen met de tellen erin, synchroniseren
twee nummers op tempo en tel (SYNC), en zetten cues en loops op de beat. Daar
is per nummer nodig:

- `golf`: de luidheid over het hele nummer in GOLF_N stukjes (0..1), voor de
  golfvorm. Uit het bestand zelf, met ffmpeg naar 8 kHz mono en numpy.
- `bpm` en `telStart`: het tempo en waar de eerste tel valt. Van librosa's
  beat tracker in de karaoke-dienst (`model=tempo`); die vergist zich hooguit
  een octaaf, en dat vangt SYNC op. Is de dienst er niet, dan een eigen
  schatting uit de aanslagen — die zat er soms een kwart naast (4/3).

Bewaard in `DATA_DIR/dj/<track_id>.json` met het bronkenmerk. Alles naar
`/data`, nooit naar `/music`.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Depends

from .. import config
from ..deps import require_user
from ..models import Track, User
from . import karaoke

router = APIRouter(tags=["dj"])
log = logging.getLogger("jltamp.dj")

DJ_DIR = config.DATA_DIR / "dj"
VERSIE = 2   # 2: tempo van librosa via de dienst
GOLF_N = 1500           # stukjes; bij vier minuten ~0,16 s per stukje
SR = 8000               # genoeg voor luidheid en aanslagen, en snel te lezen


def _bestand(track_id: int) -> Path:
    return DJ_DIR / f"{track_id}.json"


def _pcm(t: Track):
    import numpy as np
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", t.path, "-vn", "-map", "0:a:0",
         "-ac", "1", "-ar", str(SR), "-t", str(karaoke.MAX_MINUTEN * 60), "-f", "s16le", "-"],
        capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-200:])
    return np.frombuffer(r.stdout, dtype="<i2").astype("float32") / 32768.0


def _golf(x) -> list[float]:
    import numpy as np
    n = max(1, len(x) // GOLF_N)
    stukken = x[: n * GOLF_N].reshape(GOLF_N, n) if len(x) >= GOLF_N else x.reshape(1, -1)
    rms = np.sqrt(np.mean(np.square(stukken), axis=1))
    piek = float(np.percentile(rms, 99)) or 1.0
    return [round(min(1.0, float(v) / piek), 3) for v in rms]


def _tempo(x) -> tuple[float | None, float]:
    """(bpm, eerste tel in s) uit de aanslagen: spectrale flux, autocorrelatie
    tussen 70 en 180 BPM, en de fase waar de aanslagen het sterkst samenvallen.

    Een schatting met een vast tempo; voor popmuziek op de decks is dat genoeg.
    """
    import numpy as np
    hop, raam = 128, 512
    fps = SR / hop
    if len(x) < raam * 4:
        return None, 0.0
    n = (len(x) - raam) // hop
    idx = np.arange(raam)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(raam), axis=1))
    spec = np.log1p(spec * 10)
    flux = np.maximum(0.0, np.diff(spec, axis=0)).sum(axis=1)
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    if not flux.any():
        return None, 0.0
    # Hooguit drie minuten uit het midden, net als bij de akkoorden.
    m = len(flux) // 2
    stuk = flux[max(0, m - int(90 * fps)): m + int(90 * fps)]
    ac = np.correlate(stuk, stuk, mode="full")[len(stuk) - 1:]
    lags = np.arange(len(ac))
    beste, beste_v = None, -1.0
    for bpm in np.arange(70.0, 180.0, 0.25):
        lag = 60.0 * fps / bpm
        # De tel én twee tellen verder: zo wint de echte tel van een halve.
        v = (np.interp(lag, lags, ac) + 0.5 * np.interp(2 * lag, lags, ac)
             + 0.25 * np.interp(4 * lag, lags, ac))
        # Een zachte voorkeur voor 90–140: daar zit de meeste muziek.
        v *= 1.0 + 0.15 * np.exp(-((bpm - 118) / 30) ** 2)
        if v > beste_v:
            beste, beste_v = float(bpm), float(v)
    if beste is None:
        return None, 0.0
    periode = 60.0 * fps / beste
    fasen = np.arange(0, periode, 0.5)
    scores = [flux[np.round(np.arange(f, len(flux), periode)).astype(int).clip(0, len(flux) - 1)].sum()
              for f in fasen]
    fase = float(fasen[int(np.argmax(scores))]) / fps
    # 'hop' verschuift de flux een half raam ten opzichte van het geluid.
    return round(beste, 2), round(fase + raam / 2 / SR, 3)


def _tempo_dienst(t: Track) -> dict | None:
    """{bpm, telStart} van librosa in de karaoke-dienst, of None."""
    b = karaoke.beschikbaarheid()
    if not (b.get("beschikbaar") and b.get("route") == "dienst" and b.get("tempo")):
        return None
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dj-") as m:
        bron = Path(m) / "bron.flac"
        karaoke._ffmpeg("-i", t.path, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "22050",
                        "-t", str(karaoke.MAX_MINUTEN * 60), "-c:a", "flac", str(bron))
        with karaoke._dienst_verzoek("/taken?model=tempo", "POST", bron.read_bytes(), timeout=60) as r:
            tid = json.loads(r.read())["id"]
    try:
        einde = time.time() + 120
        while time.time() < einde:
            time.sleep(0.7)
            with karaoke._dienst_verzoek(f"/taken/{tid}") as r:
                st = json.loads(r.read())
            if st.get("status") == "klaar":
                with karaoke._dienst_verzoek(f"/taken/{tid}/tempo.json", timeout=30) as r:
                    return json.loads(r.read())
            if st.get("status") == "mislukt":
                log.info("[dj] tempo via de dienst mislukt: %s", st.get("fout"))
                return None
        return None
    finally:
        try:
            karaoke._dienst_verzoek(f"/taken/{tid}", "DELETE").close()
        except Exception:
            pass


def _maak(t: Track) -> dict:
    begin = time.time()
    x = _pcm(t)
    bpm, tel, bron = None, 0.0, "eigen"
    try:
        lt = _tempo_dienst(t)
        if lt and lt.get("bpm"):
            bpm, tel, bron = float(lt["bpm"]), float(lt.get("telStart") or 0.0), "librosa"
    except Exception as e:
        log.info("[dj] tempo via de dienst: %s", e)
    if bpm is None:
        bpm, tel = _tempo(x)
    d = {"versie": VERSIE, "bron": karaoke._bron_kenmerk(t), "duur": round(len(x) / SR, 2),
         "bpm": bpm, "telStart": tel, "golf": _golf(x), "bpmBron": bron}
    DJ_DIR.mkdir(parents=True, exist_ok=True)
    tijdelijk = _bestand(t.id).with_suffix(".tmp")
    tijdelijk.write_text(json.dumps(d, separators=(",", ":")))
    tijdelijk.replace(_bestand(t.id))
    log.info("[dj] nummer %s: %s bpm (%s) in %.1fs", t.id, bpm, bron, time.time() - begin)
    return d


@router.get("/dj/{track_id}")
def dj_info(track_id: str, user: User = Depends(require_user)):
    """Golfvorm, tempo en eerste tel voor de decks. De eerste keer een seconde
    of twee rekenen, daarna uit de bewaarde versie."""
    t = karaoke._track(track_id, user)
    d = None
    try:
        d = json.loads(_bestand(t.id).read_text())
        if d.get("versie") != VERSIE or d.get("bron") != karaoke._bron_kenmerk(t):
            d = None
    except Exception:
        d = None
    if d is None:
        d = _maak(t)
    # Liggen de akkoorden er, dan hun tempo: dat is uit de begeleiding berekend
    # (zonder zang), en dezelfde tel als waar de strumpijltjes op lopen.
    try:
        from . import akkoorden
        a = akkoorden._lees(t)
        if d.get("bpmBron") == "eigen" and a and a.get("bpm") and 50 <= float(a["bpm"]) <= 220:
            d = {**d, "bpm": round(float(a["bpm"]), 2), "telStart": float(a.get("telStart") or 0.0),
                 "bpmBron": "akkoorden"}
    except Exception:
        pass
    return {k: d.get(k) for k in ("duur", "bpm", "telStart", "golf", "bpmBron")}
