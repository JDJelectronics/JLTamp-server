"""Akkoorden herkennen met BTC — het getrainde model, voor de karaoke-dienst.

BTC is de "Bi-directional Transformer for Chord recognition" van Jonggwon Park
e.a. (ISMIR 2019, https://github.com/jayg996/BTC-ISMIR19, MIT). Een klein net
(3 MB aan gewichten) dat op een CQT-spectrogram per ~0,09 s zegt welk akkoord
er klinkt, en dat geleerd heeft van ruim 200 met de hand gelabelde nummers.
Het verschil met de oude herkenning (sjablonen op een chromagram, zie
routers/akkoorden.py) zit vooral in wat een sjabloon niet kan: een basnoot die
niet de grondtoon is, een melodienoot die "verkeerd" in het akkoord valt, een
volle mix met drums. Daar gokte de oude vaak een buurakkoord (Gm in plaats van
C, Am in plaats van F).

Er zijn twee gewichtensets, en we gebruiken ze allebei:

  * `btc_model.pt` — 25 klassen: 12 majeur, 12 mineur en N (geen akkoord).
    Voor `segmenten` (beginner en gemiddeld in de app), gemiddeld met wat
    het grote model als drieklank hoort (zie herken()).
  * `btc_model_large_voca.pt` — 170 klassen: 12 grondtonen × 14 soorten
    (min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7,
    sus2, sus4) plus X en N. Voor `segmentenExpert`; de app kent alleen
    7, maj7, m7, sus2 en sus4, dus de rest gaat naar de dichtstbijzijnde
    daarvan (zie EXPERT_SOORT).

Wat hier anders is dan BTC's eigen test.py, en waarom:

  1. **Overlappende vensters.** BTC kijkt naar vensters van 108 frames (10 s).
     test.py knipt het nummer in losse vensters; aan de rand van een venster
     mist het net de context, en daar sprong het akkoord soms even verkeerd.
     Hier drie keer, met de vensters een derde verschoven, en per frame het
     gewogen gemiddelde (een frame in het midden van een venster telt het
     zwaarst).
  2. **Gladstrijken.** Een Viterbi met een overgang die liever blijft dan
     wisselt, op de kansen van het net. Haalt het flikkeren weg (een frame
     G7 midden in een G) zonder echte wissels te verschuiven.
  3. **Samenvoegen** tot segmenten, met een minimale duur, net als de oude
     herkenning: korter is een doorgang, geen akkoord om te pakken.

Dit bestand importeert bij het laden alleen de standaardbibliotheek; torch,
numpy en librosa pas als er echt gerekend wordt. Het draait in `venv-akkoorden`
van de dienst (zie docs/AKKOORDEN.md), nooit in de server.

De netwerkdefinitie hieronder is overgenomen uit BTC (btc_model.py en
utils/transformer_modules.py), met dezelfde namen zodat de gewichten erop
passen, en opgeschoond voor een nieuwe numpy/torch (np.float bestaat niet
meer). Licentie van dat deel:

    MIT License — Copyright (c) 2019 Jonggwon Park

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from __future__ import annotations

import math
import os

# Zoals BTC getraind is: mono 22,05 kHz, CQT met 144 banden (24 per octaaf,
# zes octaven vanaf C1), sprong 2048, in stukken van 10 s → 108 frames per stuk.
SR = 22050
HOP = 2048
BANDEN = 144
PER_OCTAAF = 24
STUK_S = 10.0
VENSTER = 108
FRAME_S = STUK_S / VENSTER          # ≈ 0,0926 s, zoals BTC zelf rekent

# Het netwerk (run_config.yaml van BTC).
MODEL_INSTELLINGEN = dict(feature_size=144, timestep=108, hidden_size=128, num_layers=8,
                          num_heads=4, total_key_depth=128, total_value_depth=128,
                          filter_size=128)

# Zoveel vensters tegelijk door het net. Klein houden: op de CPU van de M4
# kostte een nummer van vier minuten met 8 per keer 0,2 s per model, met 16
# 0,9 s en met 64 1,2 s (de aandachtsmatrices vallen dan uit de cache).
PARTIJ = 8
GEWICHTEN_KLEIN = "btc_model.pt"
GEWICHTEN_GROOT = "btc_model_large_voca.pt"

# Spelling zoals de app en de oude herkenning: C C# D Eb E F F# G Ab A Bb B,
# en mineur zoals op de gitaar gebruikelijk (G#m, niet Abm).
GRONDTONEN = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINEUR = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")

# De 14 soorten van de grote set, in BTC's volgorde, en wat de app ervan laat
# zien. De app kent '', m, 7, maj7, m7, sus2 en sus4 (akkoordTheorie.ts):
#   dim  (1 b3 b5)    → m     twee van de drie tonen, en het klinkt mineur
#   aug  (1 3 #5)     → ''    idem, majeur
#   min6 / maj6       → m / ''  de sext is kleur; de drieklank klopt
#   minmaj7 (1 b3 5 7) → m
#   dim7 (1 b3 b5 bb7) → m
#   hdim7 (1 b3 b5 b7) → m7   drie van de vier tonen
BTC_SOORTEN = ("min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
               "maj7", "7", "dim7", "hdim7", "sus2", "sus4")
EXPERT_SOORT = {"min": "m", "maj": "", "dim": "m", "aug": "", "min6": "m", "maj6": "",
                "min7": "m7", "minmaj7": "m", "maj7": "maj7", "7": "7", "dim7": "m",
                "hdim7": "m7", "sus2": "sus2", "sus4": "sus4"}
EXPERT_ACHTER = ("", "m", "7", "maj7", "m7", "sus2", "sus4")
INTERVALLEN = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
               "m7": (0, 3, 7, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}

# Gladstrijken: kans dat het akkoord in het volgende frame (0,09 s) gelijk
# blijft. 0,9 is losjes; het net zelf is al rustig, dit haalt alleen het
# flikkeren weg. Gemeten op GuitarSet (zie docs/AKKOORDEN.md).
BLIJF_KANS = float(os.environ.get("KARAOKE_AKKOORDEN_BLIJF", "0.9"))
# Korter dan dit is een doorgang (zelfde getallen als de oude herkenning).
MIN_SEGMENT_S = 0.8
MIN_SEGMENT_EXPERT_S = 0.4


# ── het netwerk (uit BTC, zie de licentie bovenaan) ─────────────────────────

def _bouw_model(aantal_klassen: int):
    import numpy as np
    import torch
    import torch.nn as nn

    def bias_masker(lengte: int):
        m = np.triu(np.full([lengte, lengte], -np.inf), 1)
        return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(1)

    def tijdsignaal(lengte: int, kanalen: int, min_ts=1.0, max_ts=1.0e4):
        positie = np.arange(lengte)
        n = kanalen // 2
        stap = math.log(float(max_ts) / float(min_ts)) / (float(n) - 1)
        inv = min_ts * np.exp(np.arange(n).astype(float) * -stap)
        t = np.expand_dims(positie, 1) * np.expand_dims(inv, 0)
        s = np.concatenate([np.sin(t), np.cos(t)], axis=1)
        s = np.pad(s, [[0, 0], [0, kanalen % 2]], "constant")
        return torch.from_numpy(s.reshape([1, lengte, kanalen])).float()

    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            gem = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True)
            return self.gamma * (x - gem) / (sd + self.eps) + self.beta

    class SoftmaxOutputLayer(nn.Module):
        def __init__(self, hidden_size, output_size):
            super().__init__()
            self.output_size = output_size
            self.output_projection = nn.Linear(hidden_size, output_size)
            # Niet gebruikt bij het herkennen, maar wel in de gewichten.
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                                batch_first=True, bidirectional=True)

    class MultiHeadAttention(nn.Module):
        def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                     num_heads, bias_mask=None):
            super().__init__()
            self.num_heads = num_heads
            self.query_scale = (total_key_depth // num_heads) ** -0.5
            self.bias_mask = bias_mask
            self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
            self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
            self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
            self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        def _split(self, x):
            b, n, d = x.shape
            return x.view(b, n, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

        def forward(self, q, k, v):
            q = self._split(self.query_linear(q)) * self.query_scale
            k = self._split(self.key_linear(k))
            v = self._split(self.value_linear(v))
            logits = torch.matmul(q, k.permute(0, 1, 3, 2))
            if self.bias_mask is not None:
                logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].to(logits)
            ctx = torch.matmul(torch.softmax(logits, dim=-1), v)
            b, h, n, d = ctx.shape
            return self.output_linear(ctx.permute(0, 2, 1, 3).contiguous().view(b, n, h * d))

    class Conv(nn.Module):
        def __init__(self, input_size, output_size, kernel_size):
            super().__init__()
            self.pad = nn.ConstantPad1d((kernel_size // 2, (kernel_size - 1) // 2), 0)
            self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)

    class PositionwiseFeedForward(nn.Module):
        def __init__(self, input_depth, filter_size, output_depth):
            super().__init__()
            self.layers = nn.ModuleList([Conv(input_depth, filter_size, 3),
                                         Conv(filter_size, output_depth, 3)])

        def forward(self, x):
            # Ook na de laatste laag een ReLU: zo is BTC getraind.
            for laag in self.layers:
                x = torch.relu(laag(x))
            return x

    class SelfAttentionBlock(nn.Module):
        def __init__(self, c, bias_mask):
            super().__init__()
            h = c["hidden_size"]
            self.multi_head_attention = MultiHeadAttention(
                h, c["total_key_depth"], c["total_value_depth"], h, c["num_heads"], bias_mask)
            self.positionwise_convolution = PositionwiseFeedForward(h, c["filter_size"], h)
            self.layer_norm_mha = LayerNorm(h)
            self.layer_norm_ffn = LayerNorm(h)

        def forward(self, x):
            xn = self.layer_norm_mha(x)
            x = x + self.multi_head_attention(xn, xn, xn)
            return x + self.positionwise_convolution(self.layer_norm_ffn(x))

    class BiDirectionalSelfAttention(nn.Module):
        def __init__(self, c):
            super().__init__()
            masker = bias_masker(c["timestep"])
            self.attn_block = SelfAttentionBlock(c, masker)
            self.backward_attn_block = SelfAttentionBlock(c, torch.transpose(masker, 2, 3))
            self.linear = nn.Linear(c["hidden_size"] * 2, c["hidden_size"])

        def forward(self, x):
            return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))

    class BiDirectionalSelfAttentionLayers(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.timing_signal = tijdsignaal(c["timestep"], c["hidden_size"])
            self.embedding_proj = nn.Linear(c["feature_size"], c["hidden_size"], bias=False)
            self.self_attn_layers = nn.Sequential(
                *[BiDirectionalSelfAttention(c) for _ in range(c["num_layers"])])
            self.layer_norm = LayerNorm(c["hidden_size"])

        def forward(self, x):
            y = self.embedding_proj(x) + self.timing_signal[:, :x.shape[1], :].to(x)
            return self.layer_norm(self.self_attn_layers(y))

    class BTC(nn.Module):
        def __init__(self, c, klassen):
            super().__init__()
            self.self_attn_layers = BiDirectionalSelfAttentionLayers(c)
            self.output_layer = SoftmaxOutputLayer(c["hidden_size"], klassen)

        def forward(self, x):
            """(batch, 108, 144) → logits (batch, 108, klassen)."""
            return self.output_layer.output_projection(self.self_attn_layers(x))

    return BTC(MODEL_INSTELLINGEN, aantal_klassen)


def laad(pad: str, apparaat: str = "cpu"):
    """(model, gemiddelde, spreiding) uit een BTC-checkpoint."""
    import torch
    # weights_only=False: het checkpoint bevat ook numpy-getallen (mean/std).
    # Het is een bestand dat we zelf neerzetten, niet iets van buiten.
    ck = torch.load(pad, map_location="cpu", weights_only=False)
    klassen = ck["model"]["output_layer.output_projection.weight"].shape[0]
    model = _bouw_model(klassen)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(apparaat)
    return model, float(ck["mean"]), float(ck["std"])


# ── van geluid naar kansen ──────────────────────────────────────────────────

def kenmerken(y):
    """CQT zoals BTC getraind is: per stuk van 10 s, log-amplitude. (frames, 144)."""
    import numpy as np
    import librosa
    stuk = int(SR * STUK_S)
    delen = []
    for begin in range(0, y.size, stuk):
        s = y[begin:begin + stuk]
        n_frames = 1 + s.size // HOP
        if s.size < 2 * SR:
            # Een staartje: opvullen, anders klaagt de CQT over een te kort
            # signaal (hij halveert het tot zes keer; dan moet er nog 512 over zijn).
            s = np.pad(s, (0, 2 * SR - s.size))
        c = np.abs(librosa.cqt(s, sr=SR, n_bins=BANDEN, bins_per_octave=PER_OCTAAF,
                               hop_length=HOP))[:, :n_frames]
        delen.append(c)
    if not delen:
        return np.zeros((0, BANDEN), dtype="float32")
    return np.log(np.concatenate(delen, axis=1) + 1e-6).T.astype("float32")


def kansen(model, gem: float, sd: float, x, apparaat: str = "cpu"):
    """Per frame de kans op elk akkoord, (frames, klassen).

    Drie keer, met de vensters van 108 frames telkens een derde verschoven,
    en per frame het gewogen gemiddelde: een frame midden in een venster ziet
    vijf seconden voor- én achteruit, een frame aan de rand maar één kant op.
    Aangevuld wordt met stilte (wat de CQT van niets geeft), niet met nullen.
    """
    import numpy as np
    import torch
    n = x.shape[0]
    xn = (x - gem) / sd
    stil = (math.log(1e-6) - gem) / sd
    gewicht = np.sin(np.pi * (np.arange(VENSTER) + 0.5) / VENSTER).astype("float32")
    som, wsom = None, np.zeros(n, dtype="float32")
    for fase in (0, VENSTER // 3, 2 * VENSTER // 3):
        voor = (VENSTER - fase) % VENSTER
        na = (-(voor + n)) % VENSTER
        p = np.concatenate([np.full((voor, BANDEN), stil, "float32"), xn,
                            np.full((na, BANDEN), stil, "float32")])
        p = p.reshape(-1, VENSTER, BANDEN)
        uit = []
        with torch.no_grad():
            for i in range(0, p.shape[0], PARTIJ):
                logits = model(torch.from_numpy(p[i:i + PARTIJ]).to(apparaat))
                uit.append(torch.softmax(logits, -1).cpu().numpy())
        k = np.concatenate(uit).reshape(-1, uit[0].shape[-1])[voor:voor + n]
        w = np.tile(gewicht, p.shape[0])[voor:voor + n]
        som = k * w[:, None] if som is None else som + k * w[:, None]
        wsom += w
    return som / wsom[:, None]


def viterbi(k, blijf: float = BLIJF_KANS):
    """Meest waarschijnlijke pad door de kansen `k` (frames, klassen), met een
    overgang die met `blijf` gelijk blijft en anders gelijk verdeeld wisselt.
    In O(frames × klassen): de beste overstap is altijd vanuit de beste vorige."""
    import numpy as np
    n, m = k.shape
    logk = np.log(k + 1e-9)
    log_blijf = math.log(blijf)
    log_wissel = math.log((1 - blijf) / (m - 1))
    delta = logk[0].copy()
    terug = np.empty((n, m), dtype=np.int32)
    alle = np.arange(m, dtype=np.int32)
    for t in range(1, n):
        beste = int(np.argmax(delta))
        blijven = delta + log_blijf
        wissel = delta[beste] + log_wissel
        terug[t] = np.where(blijven >= wissel, alle, beste)
        delta = np.maximum(blijven, wissel) + logk[t]
    pad = np.empty(n, dtype=np.int32)
    pad[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        pad[t - 1] = terug[t, pad[t]]
    return pad


# ── van kansen naar namen ───────────────────────────────────────────────────

def namen_klein() -> list[str]:
    """De 25 klassen van btc_model.pt: C, C:min, C#, C#:min, …, N."""
    uit = []
    for r in range(12):
        uit += [GRONDTONEN[r], MINEUR[r]]
    return uit + ["N"]


def expert_kansen(k_groot):
    """170 klassen → 12 × 7 soorten die de app kent + N (85), door de kansen
    op te tellen. Zo telt een twijfel tussen Cmaj7 en Cmaj6 samen voor "C"
    in plaats van dat de twee elkaar wegdrukken."""
    import numpy as np
    doel = np.zeros((k_groot.shape[0], 12 * len(EXPERT_ACHTER) + 1), dtype="float32")
    for i in range(168):
        r, s = divmod(i, 14)
        j = EXPERT_ACHTER.index(EXPERT_SOORT[BTC_SOORTEN[s]]) * 12 + r
        doel[:, j] += k_groot[:, i]
    doel[:, -1] = k_groot[:, 168] + k_groot[:, 169]      # X en N
    return doel


def namen_expert() -> list[str]:
    return [GRONDTONEN[r] + a for a in EXPERT_ACHTER for r in range(12)] + ["N"]


def drieklank_kansen(k_expert):
    """85 expert-klassen → de 25 van het kleine model: elk rijk akkoord telt
    voor zijn drieklank (m en m7 → mineur, de rest → majeur; sus telt als
    majeur, zoals de app die op beginner/gemiddeld ook als drieklank toont)."""
    import numpy as np
    doel = np.zeros((k_expert.shape[0], 25), dtype="float32")
    for j, a in enumerate(EXPERT_ACHTER):
        mineur = 1 if a in ("m", "m7") else 0
        for r in range(12):
            doel[:, 2 * r + mineur] += k_expert[:, j * 12 + r]
    doel[:, 24] = k_expert[:, -1]
    return doel


def samenvoegen(stappen: list[tuple[float, float, str]], min_s: float) -> list[dict]:
    """Gelijke opeenvolgende stappen samen; te korte segmenten opgaan in de
    langste buur (zelfde regel als routers/akkoorden.py)."""
    segs: list[list] = []
    for a, b, k in stappen:
        if segs and segs[-1][2] == k:
            segs[-1][1] = b
        else:
            segs.append([a, b, k])
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


def _segmenten(pad, namen: list[str], duur: float, min_s: float) -> list[dict]:
    stappen = []
    for i, s in enumerate(pad):
        a = i * FRAME_S
        if a >= duur:
            break
        stappen.append((a, min((i + 1) * FRAME_S, duur), namen[int(s)]))
    return samenvoegen(stappen, min_s)


# Krumhansl-Kessler, zoals routers/akkoorden.py.
_KK_MAJ = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MIN = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def toonsoort(segmenten: list[dict]) -> str | None:
    """De toonsoort uit de akkoorden zelf: per toon hoe lang hij in een akkoord
    zat (grondtoon dubbel), en daar het best passende Krumhansl-profiel bij."""
    import numpy as np
    hist = np.zeros(12)
    for s in segmenten:
        k = s["akkoord"]
        if k == "N":
            continue
        mineur = k.endswith("m")
        r = (MINEUR if mineur else GRONDTONEN).index(k)
        for x, w in ((0, 2.0), (3 if mineur else 4, 1.0), (7, 1.0)):
            hist[(r + x) % 12] += w * (s["eind"] - s["start"])
    if not hist.any():
        return None
    best, naam = -2.0, None
    for profiel, namen in ((_KK_MAJ, GRONDTONEN), (_KK_MIN, MINEUR)):
        p = np.array(profiel)
        for r in range(12):
            c = float(np.corrcoef(hist, np.roll(p, r))[0, 1])
            if c > best:
                best, naam = c, namen[r]
    return naam


def tempo_en_fase(y) -> tuple[float | None, float]:
    """(bpm, eerste tel in s) voor het slagpatroon in de app — dezelfde aanpak
    als _tempo_en_fase in routers/akkoorden.py, zodat de app niets merkt."""
    import numpy as np
    import librosa
    hop = 512
    try:
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
        n = onset.size
        midden, half = n // 2, int(90 * SR / hop)
        bpm = float(np.atleast_1d(librosa.feature.tempo(
            onset_envelope=onset[max(0, midden - half):midden + half], sr=SR, hop_length=hop))[0])
        if not 50 <= bpm <= 220:
            return None, 0.0
        periode = 60.0 * SR / hop / bpm
        beste, som_beste = 0.0, -1.0
        for f in np.arange(0.0, periode, 0.5):
            idx = np.round(np.arange(f, n, periode)).astype(int)
            som = float(onset[idx[idx < n]].sum())
            if som > som_beste:
                beste, som_beste = f, som
        return round(bpm, 2), round(beste * hop / SR, 3)
    except Exception:
        return None, 0.0


# ── alles bij elkaar ────────────────────────────────────────────────────────

_geladen: dict[str, tuple] = {}


def herken(y, modelmap: str, apparaat: str = "cpu", melden=None) -> dict:
    """`y`: mono, SR, float32 → hetzelfde formaat als routers/akkoorden.herken().

    {segmenten, segmentenExpert, toonsoort, duur, bpm, telStart, methode}.
    `melden(fractie)` krijgt tussendoor 0..1.
    """
    import numpy as np
    melden = melden or (lambda f: None)
    duur = y.size / SR
    leeg = {"segmenten": [], "segmentenExpert": [], "toonsoort": None,
            "duur": round(duur, 2), "bpm": None, "telStart": 0.0, "methode": "btc"}
    if y.size < SR * 2:
        return leeg
    for naam in (GEWICHTEN_KLEIN, GEWICHTEN_GROOT):
        if naam not in _geladen:
            _geladen[naam] = laad(os.path.join(modelmap, naam), apparaat)
    melden(0.05)
    x = kenmerken(y)
    melden(0.5)
    if x.shape[0] < 4:
        return leeg
    model, gem, sd = _geladen[GEWICHTEN_KLEIN]
    k_klein = kansen(model, gem, sd, x, apparaat)
    melden(0.65)
    model, gem, sd = _geladen[GEWICHTEN_GROOT]
    k_groot = expert_kansen(kansen(model, gem, sd, x, apparaat))
    melden(0.8)
    # Drieklanken: het gemiddelde van beide modellen. Op GuitarSet iets beter
    # dan het kleine model alleen (78,2 tegen 77,0 %; met een melodie erover
    # 76,9 tegen 75,7 %), en het kleine model zegt in een zachte intro soms
    # "geen akkoord" waar het grote er wel een hoort. Bijkomend voordeel:
    # gemiddeld en expert spreken elkaar minder vaak tegen.
    k_drie = (k_klein + drieklank_kansen(k_groot)) / 2
    segmenten = _segmenten(viterbi(k_drie), namen_klein(), duur, MIN_SEGMENT_S)
    expert = _segmenten(viterbi(k_groot), namen_expert(), duur, MIN_SEGMENT_EXPERT_S)
    bpm, tel_start = tempo_en_fase(y)
    melden(1.0)
    return {"segmenten": segmenten, "segmentenExpert": expert, "toonsoort": toonsoort(segmenten),
            "duur": round(duur, 2), "bpm": bpm, "telStart": tel_start, "methode": "btc"}


def lees_audio(pad: str):
    """Mono, SR, float32. FLAC/WAV via soundfile; mp3 kan soundfile ook."""
    import numpy as np
    import librosa
    import soundfile as sf
    data, sr = sf.read(pad, dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y, dtype="float32")
