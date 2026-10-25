"""De stem van de presentator — Chatterbox op de GPU van de Mac Studio.

Piper (de stem in de container) is klein en snel, maar klinkt als een
navigatiesysteem. Chatterbox Multilingual (Resemble AI, MIT-licentie) spreekt
Nederlands met echte intonatie: adem, nadruk, een zin die ergens naartoe gaat.
Het model is te zwaar voor de CPU van een container en OrbStack geeft een
container geen Metal, dus net als de karaokedienst draait dit los op de Mac
zelf, onder launchd. Zie docs/PRESENTATOR.md.

**Twee stemmen, man en vrouw.** Chatterbox neemt de klank over van een kort
voorbeeld. Die voorbeelden zijn zelf synthetisch (de systeemstemmen van macOS,
zie `maak_voorbeelden`), dus er wordt niemands echte stem nagedaan. De server
kiest per aankondiging welke van de twee; dit bestand zegt alleen hoe.

Net als karaoke_demucs.py importeert dit bij het laden alleen de
standaardbibliotheek; torch en chatterbox komen pas als de dienst start.

    python stem_dienst.py dienst             # de dienst (launchd)
    python stem_dienst.py voorbeelden        # de twee voorbeeldstemmen maken

    GET  /gezond                       {"klaar": true, "stemmen": ["man", "vrouw"],
                                        "talen": ["nl", "en", "de", "es", "pt"]}
    POST /spreek {tekst, stem, taal}   → audio/wav

**Talen.** Chatterbox Multilingual spreekt ze alle vijf; de app-presentator
praat in de taal van de app. Zonder `taal` is het Nederlands (een server van
vóór de talen). Een taal die hier niet bij staat krijgt een 400 in plaats van
een Nederlandse uitspraak. De man- en vrouwenstem blijven dezelfde: het
voorbeeld bepaalt alleen de klank, de taal komt van het model. Dat de
voorbeelden Nederlands spreken kan een licht accent geven in de andere talen.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAP = Path(os.environ.get("STEM_MAP", Path(__file__).resolve().parent / "stemmen"))
# Het model als losse bestanden in een map. Hugging Face zelf ophalen lukt op
# de Studio niet betrouwbaar (SSL breekt halverwege grote downloads af), dus
# staat het er vooraf; zonder map probeert hij het toch via de hub.
MODEL_MAP = Path(os.environ.get("STEM_MODEL", Path(__file__).resolve().parent / "model"))
STEMMEN = ("man", "vrouw")
# De talen die de server mag vragen (language_id van Chatterbox Multilingual).
TALEN = ("nl", "en", "de", "es", "pt")

# Een aankondiging is hooguit een zin of twee; dit vangt een fout aan de
# andere kant die een heel artikel stuurt.
MAX_TEKENS = 400

# Een radiopresentator praat iets levendiger dan een voorlezer, maar niet
# theatraal. 0,5 is de standaard van Chatterbox; daarboven gaat het al snel
# overdrijven.
EXPRESSIE = float(os.environ.get("STEM_EXPRESSIE", "0.55"))
CFG = float(os.environ.get("STEM_CFG", "0.5"))

_model = None
_conds: dict = {}            # stem → voorgerekende klank, één keer per start
_slot = threading.Lock()     # één tegelijk: de GPU wordt er niet sneller van


def maak_voorbeelden() -> None:
    """De twee voorbeeldstemmen, uit de Nederlandse systeemstemmen van macOS.

    Synthetisch met opzet: Chatterbox neemt alleen de klank over, de intonatie
    maakt hij zelf. Zo hoeft er geen opname van een echt mens aan te pas te komen.
    """
    MAP.mkdir(parents=True, exist_ok=True)
    zin = ("Goedenavond, je luistert naar de radio. Het is kwart over acht en "
           "buiten is het een graad of negen. We gaan zo verder met muziek, "
           "blijf lekker hangen.")
    for stem, systeem in (("man", "Xander"), ("vrouw", "Ellen")):
        aiff = MAP / f"{stem}.aiff"
        subprocess.run(["say", "-v", systeem, "-o", str(aiff), zin], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                        str(aiff), str(MAP / f"{stem}.wav")], check=True)
        aiff.unlink(missing_ok=True)
        print(f"{stem}: {MAP / f'{stem}.wav'}")


def _laad() -> None:
    global _model
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    apparaat = "mps" if torch.backends.mps.is_available() else "cpu"
    t = time.time()
    if (MODEL_MAP / "t3_mtl23ls_v2.safetensors").exists():
        _model = ChatterboxMultilingualTTS.from_local(MODEL_MAP, apparaat)
    else:
        _model = ChatterboxMultilingualTTS.from_pretrained(device=apparaat)
    # Eerst allemaal voorrekenen, dan pas in één keer klaar melden: anders zegt
    # /gezond "klaar" terwijl de tweede stem er nog niet is.
    klaar = {}
    for stem in STEMMEN:
        bron = MAP / f"{stem}.wav"
        if not bron.exists():
            print(f"stem: geen voorbeeld voor {stem} ({bron})", flush=True)
            continue
        _model.prepare_conditionals(str(bron), exaggeration=EXPRESSIE)
        klaar[stem] = _model.conds
    _conds.update(klaar)
    print(f"stem: geladen op {apparaat} in {time.time() - t:.1f} s, stemmen {sorted(_conds)}", flush=True)
    # Eén keer warmdraaien: de eerste zin op MPS kost anders seconden extra,
    # en dat is precies de zin waar de app op wacht.
    if _conds:
        spreek("Hallo.", next(iter(_conds)))


def spreek(tekst: str, stem: str, taal: str = "nl") -> bytes:
    """Tekst → WAV (16 bit mono, de samplerate van het model), in `taal`."""
    import numpy as np
    if taal not in TALEN:
        raise ValueError(f"onbekende taal {taal}")
    with _slot:
        _model.conds = _conds[stem]
        wav = _model.generate(tekst, language_id=taal, exaggeration=EXPRESSIE, cfg_weight=CFG)
    data = wav.squeeze().detach().cpu().numpy()
    data = np.clip(data, -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(_model.sr))
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/gezond":
            return self._json(200, {"klaar": _model is not None and bool(_conds),
                                    "stemmen": sorted(_conds), "talen": list(TALEN)})
        self._json(404, {"fout": "onbekend"})

    def do_POST(self):
        if self.path != "/spreek":
            return self._json(404, {"fout": "onbekend"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            vraag = json.loads(self.rfile.read(min(n, 8192)) or b"{}")
        except Exception:
            return self._json(400, {"fout": "geen json"})
        tekst = str(vraag.get("tekst") or "").strip()
        stem = str(vraag.get("stem") or "")
        taal = str(vraag.get("taal") or "nl").strip().lower()
        if not tekst or len(tekst) > MAX_TEKENS:
            return self._json(400, {"fout": "tekst"})
        if taal not in TALEN:
            return self._json(400, {"fout": "taal"})
        if _model is None or not _conds:
            return self._json(503, {"fout": "nog niet klaar"})
        if stem not in _conds:
            stem = next(iter(_conds))
        t = time.time()
        try:
            wav = spreek(tekst, stem, taal)
        except Exception as e:
            print(f"stem: mislukt: {e}", flush=True)
            return self._json(500, {"fout": "mislukt"})
        print(f"stem: {stem}/{taal} {time.time() - t:.1f} s: {tekst[:60]}", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="wat", required=True)
    d = sub.add_parser("dienst")
    d.add_argument("--adres", default=os.environ.get("STEM_ADRES", "127.0.0.1"))
    d.add_argument("--poort", type=int, default=int(os.environ.get("STEM_POORT", "8426")))
    sub.add_parser("voorbeelden")
    a = p.parse_args(argv)
    if a.wat == "voorbeelden":
        maak_voorbeelden()
        return 0
    # Eerst luisteren, dan laden: /gezond zegt "nog niet klaar" in plaats van
    # dat de server een geweigerde verbinding ziet en denkt dat hij weg is.
    srv = ThreadingHTTPServer((a.adres, a.poort), _Handler)
    threading.Thread(target=_laad, daemon=True).start()
    print(f"stem: luistert op {a.adres}:{a.poort}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
