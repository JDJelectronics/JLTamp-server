"""Melodie: van de losse zang naar de noten die je moet zingen.

Voor zingen met punten (routers/melodie.py). De karaoke heeft de zang al los
van de muziek (`zang.m4a`); hier wordt die een rij noten {t, d, p}: begin en
duur in seconden en de midi-toonhoogte. De app legt daar de toonhoogte van de
microfoon naast.

Helemaal eigen rekenwerk, alleen numpy: geen model, geen librosa (dat viel op
een ARM-server al eens met een segmentatiefout om, zie akkoorden.py). Twee stappen:

  1. **Toonhoogte per 10 ms** met YIN (de Cheveigné & Kawahara, 2002), dezelfde
     methode als de stemmer in de app, maar in blokken met de FFT gerekend:
     een nummer van vier minuten is zo een paar seconden werk.
  2. **Noten maken**: de ruwe lijn gladstrijken, octaafsprongen van één vak
     terugzetten, en wat lang genoeg op één toon blijft een noot noemen.

De zang die Demucs oplevert is niet helemaal schoon: er lekt wat galm en soms
een instrument in. Daarom telt een vak alleen als het luid genoeg is ten
opzichte van de zang in het hele nummer én YIN er zeker van is.
"""
from __future__ import annotations

import numpy as np

RATE = 16000
# Elke 10 ms een meting, over een venster van 40 ms.
HOP = 160
W = 640
# Zang tussen ~70 Hz (een lage bas) en ~1050 Hz (een hoge sopraan).
F_MIN = 70.0
F_MAX = 1050.0
TAU_MIN = int(RATE / F_MAX)
TAU_MAX = int(RATE / F_MIN) + 1
# YIN: de eerste dip onder deze drempel is de periode.
DREMPEL = 0.15
# Zekerheid (1 − diepte van de dip) die een vak minstens moet hebben.
MIN_ZEKER = 0.75
# Zachter dan dit ten opzichte van de luidste zang (dB) = geen zang maar galm.
STIL_DB = -35.0
# Een noot duurt minstens zo lang; korter is een overgang of een medeklinker.
MIN_NOOT_S = 0.10
# Een gat van zo lang binnen één toon is nog dezelfde noot (een ademtje, een "t").
MAX_GAT_S = 0.06
# Zo ver (halve tonen) mag de lijn binnen één noot afwijken van zijn midden.
NOOT_MARGE = 0.7

BLOK = 512


def toonhoogte(x: np.ndarray, rate: int = RATE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mono float32 op `rate` Hz → (tijden, midi, zekerheid) per vak van HOP.

    `midi` is fractioneel en NaN waar er geen toon is.
    """
    if rate != RATE:
        raise ValueError(f"verwacht {RATE} Hz, kreeg {rate}")
    x = np.asarray(x, dtype=np.float32)
    lengte = W + TAU_MAX
    n_vakken = max(0, (x.size - lengte) // HOP + 1)
    tijden = (np.arange(n_vakken) * HOP + W / 2) / RATE
    if n_vakken == 0:
        leeg = np.zeros(0, dtype=np.float32)
        return tijden, leeg, leeg

    # Luidheid per vak, en de drempel ten opzichte van de luidste zang. Het
    # 99e percentiel en niet het maximum: één knal mag de drempel niet bepalen.
    vierkant = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    begins = np.arange(n_vakken) * HOP
    rms = np.sqrt((vierkant[begins + W] - vierkant[begins]) / W)
    db = 20 * np.log10(rms + 1e-9)
    luid = np.percentile(db, 99) if n_vakken else 0.0
    stem = db >= max(luid + STIL_DB, -60.0)

    midi = np.full(n_vakken, np.nan, dtype=np.float32)
    zeker = np.zeros(n_vakken, dtype=np.float32)
    nfft = 1 << int(np.ceil(np.log2(lengte + W)))
    taus = np.arange(TAU_MAX + 1)

    for b in range(0, n_vakken, BLOK):
        idx = np.nonzero(stem[b:b + BLOK])[0] + b
        if idx.size == 0:
            continue
        rijen = (idx * HOP)[:, None] + np.arange(lengte)[None, :]
        vak = x[rijen]                                       # (k, lengte)
        a = vak[:, :W]
        # r(τ) = Σ a[j]·x[j+τ], voor alle τ tegelijk via de FFT.
        r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(vak, nfft), nfft)[:, :TAU_MAX + 1]
        kw = np.concatenate([np.zeros((vak.shape[0], 1)), np.cumsum(vak.astype(np.float64) ** 2, axis=1)], axis=1)
        e0 = kw[:, W][:, None]
        e_tau = kw[:, taus + W] - kw[:, taus]
        d = np.maximum(e0 + e_tau - 2 * r, 0.0)
        # De cumulatief genormaliseerde verschilfunctie.
        som = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(som, 1e-12)

        for i, rij in zip(idx, cmnd):
            tau = _eerste_dip(rij)
            if tau is None:
                continue
            z = 1.0 - float(rij[tau])
            if z < MIN_ZEKER:
                continue
            precies = float(tau)
            if TAU_MIN < tau < TAU_MAX:
                p, q, s = rij[tau - 1], rij[tau], rij[tau + 1]
                noemer = p - 2 * q + s
                if noemer > 0:
                    precies = tau + 0.5 * (p - s) / noemer
            f = RATE / precies
            if F_MIN <= f <= F_MAX:
                midi[i] = 69 + 12 * np.log2(f / 440.0)
                zeker[i] = z
    return tijden, midi, zeker


def _eerste_dip(rij: np.ndarray) -> int | None:
    onder = np.nonzero(rij[TAU_MIN:TAU_MAX] < DREMPEL)[0]
    if onder.size:
        t = int(onder[0]) + TAU_MIN
        while t + 1 < TAU_MAX and rij[t + 1] < rij[t]:
            t += 1
        return t
    t = int(np.argmin(rij[TAU_MIN:TAU_MAX])) + TAU_MIN
    return t if rij[t] < 0.3 else None


def glad(midi: np.ndarray) -> np.ndarray:
    """Mediaan over vijf vakken (alleen waar er toon is), en een vak dat precies
    een octaaf naast zijn buren zit terug naar de buren: YIN pakt bij een hese
    of zachte toon soms de dubbele periode."""
    m = midi.copy()
    n = m.size
    if n == 0:
        return m
    vol = np.pad(m, 2, constant_values=np.nan)
    ramen = np.lib.stride_tricks.sliding_window_view(vol, 5)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mediaan = np.nanmedian(ramen, axis=1)
    toon = ~np.isnan(m)
    afstand = m - mediaan
    octaaf = toon & (np.abs(np.abs(afstand) - 12) < 1.0)
    m[octaaf] -= 12 * np.sign(afstand[octaaf])
    uit = np.where(toon, mediaan, np.nan)
    # Waar de mediaan nu nog een octaaf scheef staat (kort stuk), het teruggezette vak.
    uit[octaaf] = m[octaaf]
    return uit.astype(np.float32)


def noten(tijden: np.ndarray, midi: np.ndarray) -> list[dict]:
    """De gladde lijn → [{t, d, p}], p een geheel midi-nummer."""
    if tijden.size == 0:
        return []
    stap = float(tijden[1] - tijden[0]) if tijden.size > 1 else HOP / RATE
    max_gat = int(round(MAX_GAT_S / stap))
    uit: list[dict] = []

    i, n = 0, midi.size
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        begin = i
        waarden = [float(midi[i])]
        midden = waarden[0]
        laatst = i
        j = i + 1
        while j < n:
            if np.isnan(midi[j]):
                if j - laatst > max_gat:
                    break
                j += 1
                continue
            if abs(float(midi[j]) - midden) > NOOT_MARGE:
                # Blijft hij weg (drie vakken op rij), dan begint er een nieuwe noot.
                vooruit = midi[j:j + 3]
                if np.sum(np.abs(vooruit - midden) > NOOT_MARGE) >= min(3, vooruit.size):
                    break
                j += 1
                continue
            waarden.append(float(midi[j]))
            midden = float(np.median(waarden))
            laatst = j
            j += 1
        duur = (laatst - begin + 1) * stap
        if duur >= MIN_NOOT_S:
            uit.append({"t": round(float(tijden[begin]) - stap / 2, 3), "d": round(duur, 3),
                        "p": int(round(midden))})
        i = laatst + 1

    # Twee stukken op dezelfde toon met alleen een ademtje ertussen: één noot.
    samen: list[dict] = []
    for nt in uit:
        v = samen[-1] if samen else None
        if v and v["p"] == nt["p"] and nt["t"] - (v["t"] + v["d"]) <= MAX_GAT_S + stap:
            v["d"] = round(nt["t"] + nt["d"] - v["t"], 3)
        else:
            samen.append(dict(nt))
    return samen


def melodie(x: np.ndarray, rate: int = RATE) -> dict:
    """Mono zang op RATE Hz → {"noten": [...], "laag": midi, "hoog": midi}."""
    tijden, midi, _ = toonhoogte(x, rate)
    rij = noten(tijden, glad(midi))
    if rij:
        ps = sorted(nt["p"] for nt in rij)
        # Het bereik zonder de uitschieters: daar schaalt de app de balk op.
        laag = ps[round(0.03 * (len(ps) - 1))]
        hoog = ps[round(0.97 * (len(ps) - 1))]
    else:
        laag = hoog = 60
    return {"noten": rij, "laag": int(laag), "hoog": int(hoog)}


def _los(pad: str) -> int:
    """`python -m app.melodie <zang.m4a>`: de melodie als JSON op stdout.

    Zo draait het meten in een eigen proces. In het serverproces deelde het de
    Python-sluis (GIL) met de radio en de rest, en duurde een nummer van 0,6 s
    daar 25 tot 55 s.
    """
    import json
    import subprocess
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", pad, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    print(json.dumps(melodie(np.frombuffer(raw, dtype=np.float32)), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_los(sys.argv[1]))
