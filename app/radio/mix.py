"""Het mengpaneel: nummers aan elkaar, de stem erover, alles even hard.

Werkt in blokjes van een tiende seconde, als float32-stereo op 44,1 kHz. Er
staat nooit een heel nummer in het geheugen: een nummer wordt door een eigen
ffmpeg gedecodeerd en in stukjes van een halve seconde opgehaald (TrackBron),
met hooguit tien seconden voorraad. Zo blijft het geheugen klein en begrensd,
ook als er twee nummers tegelijk klinken tijdens een overvloei.

Wat de Menger doet, per blok:

  overvloei   vier seconden voor het einde van een nummer begint het volgende;
              het oude zakt weg (cosinus), het nieuwe komt op (sinus) — even
              hard opgeteld, geen gat en geen bult.
  ducking     staat er een stem bij een nummer, dan zakt de muziek vanaf het
              begin van de overgang in een kwart seconde naar -12 dB, de stem
              komt erover, en daarna komt de muziek in een seconde terug.
  volume      elk nummer krijgt zijn eigen versterking (uit gain_db, zie
              loudness.py), en een zachte begrenzer vangt wat dan nog boven de
              0 dB uit zou komen.

Een nummer dat eerder ophoudt dan zijn duur zegt, gaat gewoon direct over in
het volgende. Een nummer dat blijft hangen (de NAS die niet antwoordt) wordt
na acht seconden stilte overgeslagen. Er is geen pad waarop de Menger zelf
ophoudt met geluid maken: zonder nummer komt er stilte, en elk blok wordt
opnieuw gevraagd of er al iets is.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np   # komt mee met librosa (requirements.txt)

log = logging.getLogger("radio")

RATE = 44100
KANALEN = 2


class Bron(Protocol):
    def open(self) -> None: ...
    def lees(self, n: int) -> np.ndarray: ...   # float32 (m, 2); m < n = einde
    def sluit(self) -> None: ...


@dataclass
class Item:
    """Eén nummer in de uitzending, eventueel met een stem ervoor."""
    bron: Bron
    duur: int | None = None             # in frames; None = onbekend
    gain: float = 1.0                   # lineair
    stem: np.ndarray | None = None      # float32 (k, 2), al op niveau
    info: dict = field(default_factory=dict)


def db_naar_lineair(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def begrens(x: np.ndarray, drempel: float = 0.9) -> np.ndarray:
    """Zachte begrenzer: tot de drempel niets, daarboven een tanh-knik naar 1,0.
    Geen echte compressor, wel genoeg om nooit te klippen."""
    a = np.abs(x)
    boven = a > drempel
    if not boven.any():
        return x
    ruimte = 1.0 - drempel
    y = x.copy()
    y[boven] = np.sign(x[boven]) * (drempel + ruimte * np.tanh((a[boven] - drempel) / ruimte))
    return y


def naar_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def uit_pcm16(b: bytes) -> np.ndarray:
    b = b[: len(b) - len(b) % 4]
    return np.frombuffer(b, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0


class _Spoor:
    """Een nummer dat nu klinkt: hoever het is, en of het opkomt of wegzakt."""

    def __init__(self, item: Item, inloop: int = 0):
        self.item = item
        self.pos = 0
        self.inloop = inloop
        self.uit_len = 0          # > 0: zakt weg over zoveel frames
        self.uit_pos = 0

    def lees(self, n: int) -> np.ndarray:
        data = self.item.bron.lees(n)
        if len(data):
            data = data * np.float32(self.item.gain)
            if self.inloop and self.pos < self.inloop:
                t = np.clip((self.pos + np.arange(len(data))) / self.inloop, 0.0, 1.0)
                data = data * np.sin(t * np.pi / 2).astype(np.float32)[:, None]
        self.pos += len(data)
        return data

    def lees_uit(self, n: int) -> tuple[np.ndarray, bool]:
        """Het volgende stuk terwijl hij wegzakt; (data, klaar)."""
        wil = min(n, max(0, self.uit_len - self.uit_pos))
        data = self.lees(wil)
        t = np.clip((self.uit_pos + np.arange(len(data))) / max(1, self.uit_len), 0.0, 1.0)
        data = data * np.cos(t * np.pi / 2).astype(np.float32)[:, None]
        self.uit_pos += len(data)
        return data, self.uit_pos >= self.uit_len or len(data) < wil

    def resterend(self) -> int | None:
        return None if not self.item.duur else self.item.duur - self.pos


class Menger:
    """Maakt het geluid, blok na blok. Vraagt `volgende()` om een nieuw Item
    wanneer dat nodig is; `bij_start(item, frame)` hoort wanneer een nummer
    begint (frame = positie in de uitzending)."""

    def __init__(self, volgende: Callable[[], Item | None],
                 bij_start: Callable[[Item, int], None] | None = None,
                 overvloei_s: float = 4.0, duck_db: float = -12.0,
                 aanloop_s: float = 0.25, herstel_s: float = 1.0,
                 voorwarm_s: float = 12.0):
        self._volgende = volgende
        self._bij_start = bij_start or (lambda item, frame: None)
        self.overvloei = int(overvloei_s * RATE)
        self.duck = db_naar_lineair(duck_db)
        self.aanloop = max(1, int(aanloop_s * RATE))
        self.herstel = max(1, int(herstel_s * RATE))
        self.voorwarm = int(voorwarm_s * RATE)
        self.positie = 0                       # frames gemaakt, in totaal
        self.huidig: _Spoor | None = None
        self.uitgaand: list[_Spoor] = []
        self.straks: Item | None = None        # al opgevraagd en opengezet
        self._stem: np.ndarray | None = None
        self._stem_begin = 0                   # absolute frame waarop de stem start

    # ── items ──
    def _haal(self) -> Item | None:
        if self.straks is not None:
            item, self.straks = self.straks, None
            return item
        try:
            item = self._volgende()
        except Exception as e:
            log.warning("radio: volgend nummer vragen mislukt: %s", e)
            return None
        if item is not None:
            item.bron.open()
        return item

    def _voorwarmen(self) -> None:
        """Het volgende nummer al openen, zodat de decoder voorraad heeft als
        de overvloei begint."""
        if self.straks is not None or self.huidig is None:
            return
        rest = self.huidig.resterend()
        if rest is not None and rest <= self.overvloei + self.voorwarm:
            try:
                item = self._volgende()
            except Exception as e:
                log.warning("radio: volgend nummer vragen mislukt: %s", e)
                item = None
            if item is not None:
                item.bron.open()
                self.straks = item

    def _start(self, offset: int, overvloei: bool) -> None:
        item = self._haal()
        if item is None:
            self.huidig = None
            return
        self.huidig = _Spoor(item, inloop=self.overvloei if overvloei else 0)
        frame = self.positie + offset
        if item.stem is not None and len(item.stem):
            self._stem = item.stem
            self._stem_begin = frame + self.aanloop
        try:
            self._bij_start(item, frame)
        except Exception as e:
            log.warning("radio: bij_start mislukt: %s", e)

    def _sluit(self, sp: _Spoor) -> None:
        try:
            sp.item.bron.sluit()
        except Exception:
            pass

    def _meng_uit(self, sp: _Spoor, muziek: np.ndarray, vanaf: int) -> bool:
        """Mengt een wegzakkend spoor in muziek[vanaf:]; True als hij klaar is."""
        data, klaar = sp.lees_uit(len(muziek) - vanaf)
        muziek[vanaf: vanaf + len(data)] += data
        if klaar:
            self._sluit(sp)
        return klaar

    # ── het blok ──
    def blok(self, n: int) -> np.ndarray:
        muziek = np.zeros((n, KANALEN), dtype=np.float32)

        # 1. Wat al aan het wegzakken was.
        self.uitgaand = [sp for sp in self.uitgaand if not self._meng_uit(sp, muziek, 0)]

        # 2. Het nummer van nu, en zo nodig het volgende.
        i = 0
        starts = 0
        while i < n:
            if self.huidig is None:
                # Een rij kapotte bestanden mag de lus niet laten rondtollen:
                # na drie pogingen in één blok de rest van het blok stilte.
                starts += 1
                if starts > 3:
                    break
                self._start(i, overvloei=False)
                if self.huidig is None:
                    break                       # niets te spelen: stilte
            sp = self.huidig
            rest = sp.resterend()
            if rest is not None and rest <= self.overvloei and sp.uit_len == 0:
                # De overvloei begint hier. Altijd over de hele lengte: is
                # het nummer eerder op, dan houdt het wegzakken vanzelf op.
                sp.uit_len = max(1, self.overvloei)
                starts += 1
                self.huidig = None
                if not self._meng_uit(sp, muziek, i):
                    self.uitgaand.append(sp)
                self._start(i, overvloei=True)
                continue
            wil = n - i
            if rest is not None and 0 < rest - self.overvloei < wil:
                wil = rest - self.overvloei
            data = sp.lees(wil)
            muziek[i: i + len(data)] += data
            i += len(data)
            if len(data) < wil:
                # Eerder klaar dan gedacht (of stuk): meteen door.
                if sp.pos < 5 * RATE:
                    log.warning("radio: %s gaf bijna niets (%.1f s), overgeslagen",
                                sp.item.info.get("titel", "?"), sp.pos / RATE)
                self._sluit(sp)
                self.huidig = None

        self._voorwarmen()

        # 3. De stem, en de muziek eronder zachter.
        uit = muziek
        if self._stem is not None:
            a = self._stem_begin - self.aanloop
            b = self._stem_begin
            c = self._stem_begin + len(self._stem)
            d = c + self.herstel
            frames = self.positie + np.arange(n)
            env = np.interp(frames, [a, b, c, d], [1.0, self.duck, self.duck, 1.0]).astype(np.float32)
            uit = muziek * env[:, None]
            van = max(0, self._stem_begin - self.positie)
            s0 = max(0, self.positie - self._stem_begin)
            stuk = self._stem[s0: s0 + (n - van)]
            if van < n and len(stuk):
                uit[van: van + len(stuk)] += stuk
            if self.positie + n >= d:
                self._stem = None

        self.positie += n
        return begrens(uit)

    def sluit(self) -> None:
        for sp in self.uitgaand + ([self.huidig] if self.huidig else []):
            self._sluit(sp)
        if self.straks is not None:
            try:
                self.straks.bron.sluit()
            except Exception:
                pass
        self.uitgaand, self.huidig, self.straks = [], None, None


# ── De echte bron: een nummer via ffmpeg ─────────────────────────────────────

class TrackBron:
    """Een nummer, gedecodeerd door een eigen ffmpeg naar 16-bit stereo.

    Een leesdraad haalt stukjes van een halve seconde op in een begrensde rij
    (tien seconden). `lees` wacht nooit: is er (nog) niets, dan komen er nullen
    en telt de hapering op. Zo kan een trage NAS de uitzending niet laten
    stokken, en na `max_hapering_s` stilte achter elkaar is het nummer op.
    """

    STUK = RATE // 2 * 4                       # een halve seconde, in bytes

    def __init__(self, pad: str, ffmpeg: str = "ffmpeg", buffer_s: float = 10.0,
                 max_hapering_s: float = 8.0, opdracht: list[str] | None = None):
        self.pad = pad
        self.opdracht = opdracht or [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
            "-i", pad, "-vn", "-map", "0:a:0",
            "-f", "s16le", "-ac", str(KANALEN), "-ar", str(RATE), "pipe:1",
        ]
        self._rij: queue.Queue = queue.Queue(maxsize=max(1, int(buffer_s * 2)))
        self._rest = np.zeros((0, KANALEN), dtype=np.float32)
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._einde = False
        self._hapering = 0
        self._max_hapering = int(max_hapering_s * RATE)

    def open(self) -> None:
        if self._proc is not None or self._einde:
            return
        try:
            self._proc = subprocess.Popen(self.opdracht, stdin=subprocess.DEVNULL,
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            log.warning("radio: decoder starten mislukt voor %s: %s", self.pad, e)
            self._einde = True
            return
        threading.Thread(target=self._lees_lus, name="radio-decoder", daemon=True).start()

    def _lees_lus(self) -> None:
        proc = self._proc
        try:
            while not self._stop.is_set():
                stuk = proc.stdout.read(self.STUK)
                if not stuk:
                    break
                while not self._stop.is_set():
                    try:
                        self._rij.put(stuk, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception:
            pass
        finally:
            while True:
                try:
                    self._rij.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if self._stop.is_set():
                        break

    def lees(self, n: int) -> np.ndarray:
        if self._proc is None and not self._einde:
            self.open()
        delen = [self._rest]
        heb = len(self._rest)
        while heb < n and not self._einde:
            try:
                stuk = self._rij.get_nowait()
            except queue.Empty:
                break
            if stuk is None:
                self._einde = True
                break
            d = uit_pcm16(stuk)
            delen.append(d)
            heb += len(d)
        alles = np.concatenate(delen) if len(delen) > 1 else delen[0]
        if heb >= n:
            self._hapering = 0
            self._rest = alles[n:]
            return alles[:n]
        self._rest = np.zeros((0, KANALEN), dtype=np.float32)
        if self._einde:
            return alles
        # Hapering: aanvullen met stilte, en na te lang opgeven.
        self._hapering += n - heb
        if self._hapering > self._max_hapering:
            log.warning("radio: %s hapert al %.0f s, overgeslagen", self.pad,
                        self._hapering / RATE)
            self._einde = True
            self.sluit()
            return alles
        if heb:
            self._hapering = n - heb
        return np.concatenate([alles, np.zeros((n - heb, KANALEN), dtype=np.float32)])

    def sluit(self) -> None:
        self._stop.set()
        p = self._proc
        if p is not None:
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.stdout.close()
            except Exception:
                pass
            try:
                p.wait(timeout=2)
            except Exception:
                pass
        # De rij leeg, zodat het geheugen meteen terug is.
        try:
            while True:
                self._rij.get_nowait()
        except queue.Empty:
            pass


def decodeer_bestand(pad: str, ffmpeg: str = "ffmpeg", max_s: float = 30.0) -> np.ndarray | None:
    """Een kort bestand (de stem) in één keer naar float32-stereo. Begrensd op
    `max_s` seconden, zodat een verkeerd bestand het geheugen niet vult."""
    try:
        p = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", pad,
             "-t", str(max_s), "-f", "s16le", "-ac", str(KANALEN), "-ar", str(RATE), "pipe:1"],
            capture_output=True, timeout=30,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        return uit_pcm16(p.stdout)
    except Exception as e:
        log.info("radio: stem decoderen mislukt: %s", e)
        return None
