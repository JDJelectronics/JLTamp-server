"""Songtekst uitlijnen op de zang: een tekst zonder tijdcodes krijgt per regel
een begintijd.

Draait in venv-tekst (mlx-whisper, rekent op de GPU van de Mac). Aangeroepen
door karaoke_demucs.py als los proces, net als de BTC-akkoorden:

    venv-tekst/bin/python karaoke_tekst.py <zang.flac> <tekst.txt> <uit.json>

Werkwijze:
  1. Whisper luistert naar het ZANGSPOOR (uit de karaoke-scheiding): zonder
     begeleiding hoort hij de woorden veel beter, en in de stiltes tussen de
     coupletten is er niets om zich in te vergissen.
  2. Zijn woorden (met tijd) en de woorden van de songtekst worden op elkaar
     gelegd met een uitlijning zoals bij DNA (Needleman-Wunsch): woorden die
     (bijna) gelijk zijn tellen, overgeslagen woorden kosten een beetje. Wat
     Whisper verkeerd hoort of wat in de tekst ontbreekt (een herhaald refrein)
     breekt de rest dus niet.
  3. Elke regel begint bij zijn eerste gevonden woord. Een regel zonder
     gevonden woorden komt naar verhouding tussen zijn buren.

Uitvoer: {"regels": [{"timeMs", "line"}], "gevonden": fractie gevonden woorden,
"taal"}. Voortgang en uitkomst als JSON-regels op stdout.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import unicodedata

HIER = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("KARAOKE_TEKST_MODEL", os.path.join(HIER, "modellen", "whisper-turbo"))
# mlx-whisper leest het geluid via ffmpeg; onder launchd staat Homebrew niet in PATH.
for _map in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isfile(os.path.join(_map, "ffmpeg")) and _map not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _map + ":" + os.environ.get("PATH", "")


def _meld(**d) -> None:
    print(json.dumps(d), flush=True)


def normaal(w: str) -> str:
    """Kleine letters, zonder accenten en leestekens: "Don't!" → "dont"."""
    w = unicodedata.normalize("NFKD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", w)


def _lijkt(a: str, b: str) -> float:
    if a == b:
        return 2.0
    if len(a) >= 3 and len(b) >= 3 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75:
        return 1.2
    return -1.0


def uitlijnen(regels: list[str], woorden: list[tuple[str, float]]) -> tuple[list[float | None], float]:
    """Per regel de begintijd (of None), en welk deel van de tekstwoorden gevonden is."""
    tekst: list[tuple[str, int]] = []          # (woord, regelnummer)
    for i, r in enumerate(regels):
        for w in r.split():
            n = normaal(w)
            if n:
                tekst.append((n, i))
    gehoord = [(normaal(w), t) for w, t in woorden if normaal(w)]
    a, b = len(tekst), len(gehoord)
    if not a or not b:
        return [None] * len(regels), 0.0

    GAT = -0.4
    # Needleman-Wunsch, met de keuze per cel om terug te kunnen lopen.
    score = [[0.0] * (b + 1) for _ in range(a + 1)]
    pijl = [[0] * (b + 1) for _ in range(a + 1)]   # 0 diag, 1 omhoog (tekst over), 2 links (gehoord over)
    for i in range(1, a + 1):
        score[i][0] = i * GAT
        pijl[i][0] = 1
    for j in range(1, b + 1):
        score[0][j] = j * GAT * 0.5   # gehoorde woorden overslaan is goedkoop (intro-gebrabbel)
        pijl[0][j] = 2
    for i in range(1, a + 1):
        ti = tekst[i - 1][0]
        rij, vorige = score[i], score[i - 1]
        for j in range(1, b + 1):
            d = vorige[j - 1] + _lijkt(ti, gehoord[j - 1][0])
            o = vorige[j] + GAT
            l = rij[j - 1] + GAT * 0.5
            if d >= o and d >= l:
                rij[j], pijl[i][j] = d, 0
            elif o >= l:
                rij[j], pijl[i][j] = o, 1
            else:
                rij[j], pijl[i][j] = l, 2

    tijd_van: dict[int, float] = {}     # tekstwoord-index → tijd
    i, j = a, b
    while i > 0 and j > 0:
        p = pijl[i][j]
        if p == 0:
            if _lijkt(tekst[i - 1][0], gehoord[j - 1][0]) > 0:
                tijd_van[i - 1] = gehoord[j - 1][1]
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        else:
            j -= 1

    # Per regel: het eerste gevonden woord, teruggerekend naar het begin van de
    # regel (0,3 s per woord dat ervoor stond en niet gevonden is).
    starts: list[float | None] = [None] * len(regels)
    for k, (_w, r) in enumerate(tekst):
        if k in tijd_van and starts[r] is None:
            eerste = next(x for x, (_, rr) in enumerate(tekst) if rr == r)
            starts[r] = max(0.0, tijd_van[k] - 0.3 * (k - eerste))
    return starts, len(tijd_van) / a


def vul_aan(starts: list[float | None], regels: list[str], duur: float) -> list[float]:
    """Regels zonder tijd naar verhouding tussen hun buren; altijd oplopend."""
    n = len(starts)
    gewicht = [1 + len(r) / 28 for r in regels]
    uit = list(starts)
    # Oplopend maken: een anker dat vóór een eerder anker valt, telt niet.
    hoogste = -1.0
    for i, s in enumerate(uit):
        if s is not None:
            if s <= hoogste:
                uit[i] = None
            else:
                hoogste = s
    ankers = [i for i, s in enumerate(uit) if s is not None]
    if not ankers:
        return []
    # Voor het eerste anker: terug met 2,5 s per regel.
    for i in range(ankers[0] - 1, -1, -1):
        uit[i] = max(0.0, uit[i + 1] - 2.5)
    # Tussen ankers: naar gewicht.
    for a, b in zip(ankers, ankers[1:]):
        if b - a <= 1:
            continue
        totaal = sum(gewicht[a:b])
        t = uit[a]
        for i in range(a + 1, b):
            t += (uit[b] - uit[a]) * gewicht[i - 1] / totaal
            uit[i] = t
    # Na het laatste anker: 3 s per regel, niet voorbij het einde.
    for i in range(ankers[-1] + 1, n):
        uit[i] = min(duur - 1 if duur else 1e9, uit[i - 1] + 3.0)
    return [float(x) for x in uit]


def main(zang: str, tekstpad: str, uitpad: str) -> int:
    try:
        with open(tekstpad, encoding="utf-8") as f:
            alle = [r.strip() for r in f.read().splitlines()]
        regels = [r for r in alle if r]
        if not regels:
            raise ValueError("lege tekst")
        _meld(voortgang=0.05)
        import mlx_whisper
        res = mlx_whisper.transcribe(
            zang, path_or_hf_repo=MODEL, word_timestamps=True,
            condition_on_previous_text=False, verbose=None,
            # Een stukje van de tekst als hint: Whisper kent dan de taal en de
            # schrijfwijze van namen, en hoort minder vaak iets anders.
            initial_prompt=" ".join(regels[:4])[:200],
        )
        _meld(voortgang=0.85)
        woorden = [(w["word"], float(w["start"]))
                   for s in res.get("segments", []) for w in s.get("words", [])
                   if w.get("probability", 1) > 0.15]
        starts, gevonden = uitlijnen(regels, woorden)
        duur = float(res["segments"][-1]["end"]) if res.get("segments") else 0.0
        tijden = vul_aan(starts, regels, duur)
        uit = {"regels": [{"timeMs": int(round(t * 1000)), "line": r} for t, r in zip(tijden, regels)],
               "gevonden": round(gevonden, 3), "taal": res.get("language"),
               "woorden": len(woorden)}
        with open(uitpad, "w", encoding="utf-8") as f:
            json.dump(uit, f, ensure_ascii=False)
    except Exception as e:
        _meld(fout=f"{type(e).__name__}: {e}")
        return 1
    _meld(klaar=True, gevonden=uit["gevonden"], regels=len(uit["regels"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
