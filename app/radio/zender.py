"""De zender zelf: drie draden en een schakelaar.

  wachter       kijkt elke vijf seconden naar de schakelaar `radio` in meta.
                Aan → de zender start; uit → alles stopt (geen ffmpeg, geen
                rekenwerk). Ook na een herstart van de server komt de zender zo
                vanzelf terug: de schakelaar staat in de database.
  voorbereider  kiest nummers (rotatie.py), beslist over breaks (planning.py),
                meet zo nodig de luidheid, en laat de presentator zijn tekst en
                stem maken. Altijd twee nummers vooruit: een break kost 15 à 20
                seconden om te maken, en het nummer ervoor duurt minuten.
  speler        vraagt de Menger (mix.py) om blokjes geluid en schrijft die in
                het tempo van de klok naar de encoder (encoder.py). Valt de
                encoder om, dan een nieuwe — met een oplopende wachttijd, zodat
                een ffmpeg die steeds omvalt de processor niet opeet.

**Nooit stil.** Is er nog geen voorbereid nummer, dan kiest de speler er zelf
snel een (zonder break). Lukt een break niet, dan gaat het nummer zonder. Staat
een break klaar die over een ánder vorig nummer gaat dan er echt speelde (de
noodroute haalde iets naar voren), dan valt de stem weg in plaats van iets
verkeerds te zeggen.

**Niet de live zender.** Dit heeft niets te maken met een losse broadcaster;
het is een HLS-stream van deze server, voor de app en de speakers.

Schrijft alleen in DATA_DIR/radio (de HLS-stukken en de geschiedenis) en in de
cache van de presentator. De muziek wordt alleen gelezen.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config
from ..db import SessionLocal
from ..ids import album_key, track_key
from ..models import Track
from . import blokkeer, planning, programma, rotatie
from .encoder import Encoder, opdracht as encoder_opdracht
from .mix import RATE, KANALEN, Item, Menger, TrackBron, db_naar_lineair, decodeer_bestand, naar_pcm16

log = logging.getLogger("radio")

DOEL_LUFS = -16.0          # de muziek komt hierop uit; de stem van de presentator ook
_POOL_GELDIG = 30 * 60
_VOORRAAD = 2              # zoveel nummers staan er klaar
_GESCHIEDENIS = 200


# ── De speler: geluid naar de encoder, in het tempo van de klok ──────────────

class Speler:
    def __init__(self, menger: Menger, maak_encoder: Callable[[], Encoder],
                 blok_s: float = 0.1, voorsprong_s: float = 1.0):
        self.menger = menger
        self.maak_encoder = maak_encoder
        self.blok = max(1, int(blok_s * RATE))
        self.voorsprong = voorsprong_s
        self.encoder: Encoder | None = None
        self.encoder_sinds = 0.0
        self.herstarts = 0
        self.geschreven = 0          # frames, sinds de laatste (her)start van de encoder

    def levend(self) -> bool:
        return self.encoder is not None and self.encoder.levend()

    def lus(self, stop: threading.Event) -> None:
        wacht = 1.0
        t0 = time.monotonic()
        while not stop.is_set():
            if not self.levend():
                if self.encoder is not None:
                    code = self.encoder.proc.poll() if self.encoder.proc else None
                    self.herstarts += 1
                    log.warning("radio: encoder gestopt (code %s), opnieuw over %.0f s", code, wacht)
                    self.encoder.stop()
                    if stop.wait(wacht):
                        break
                    wacht = min(wacht * 2, 30.0)
                try:
                    self.encoder = self.maak_encoder()
                    self.encoder.start()
                except Exception as e:
                    log.warning("radio: encoder starten mislukt: %s", e)
                    self.encoder = None
                    if stop.wait(wacht):
                        break
                    wacht = min(wacht * 2, 30.0)
                    continue
                self.encoder_sinds = time.time()
                t0 = time.monotonic()
                self.geschreven = 0
            if time.time() - self.encoder_sinds > 60:
                wacht = 1.0                       # hij draait weer een tijd: terug naar kort

            vooruit = self.geschreven / RATE - (time.monotonic() - t0)
            if vooruit > self.voorsprong:
                stop.wait(min(vooruit - self.voorsprong, 0.05))
                continue
            if vooruit < -5:
                log.info("radio: %.1f s achter, klok gelijkgezet", -vooruit)
                t0 = time.monotonic() - self.geschreven / RATE

            try:
                geluid = self.menger.blok(self.blok)
            except Exception as e:
                # Een fout in het mengen mag de uitzending niet stoppen: dit
                # blokje stilte, en het volgende opnieuw proberen.
                log.exception("radio: mengen mislukt: %s", e)
                geluid = np.zeros((self.blok, KANALEN), dtype=np.float32)
            try:
                self.encoder.schrijf(naar_pcm16(geluid))
            except (OSError, ValueError):
                continue                          # de volgende ronde start een nieuwe
            self.geschreven += self.blok
        if self.encoder is not None:
            self.encoder.stop()
            self.encoder = None


# ── De zender ────────────────────────────────────────────────────────────────

def _gespeeld(d: dict) -> rotatie.Gespeeld:
    return rotatie.Gespeeld(id=int(d.get("id") or 0), sleutel=d.get("sleutel") or "",
                            artiesten=frozenset(d.get("artiesten") or ()), ts=float(d.get("ts") or 0))


_OPENBAAR = ("ratingKey", "titel", "artiest", "album", "jaar", "duur", "thumb", "start", "eind")


def openbaar(info: dict | None) -> dict | None:
    """Wat de app van een nummer te zien krijgt (zie routers/radio.py)."""
    if not info:
        return None
    uit = {k: info.get(k) for k in _OPENBAAR}
    # Zodat de app vanaf "net gehoord" naar het album kan.
    uit["albumKey"] = album_key(info["album_id"]) if info.get("album_id") else None
    return uit


def _thumb(t: Track) -> str:
    from ..serializers import _track_thumb
    return _track_thumb(t, track_key(t.id), album_key(t.album_id))


class Zender:
    def __init__(self, map_: Path | None = None, ffmpeg: str | None = None,
                 met_presentator: bool | None = None):
        self.map = Path(map_ or config.RADIO_DIR)
        self.hls = self.map / "hls"
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self.met_presentator = config.RADIO_PRESENTATOR if met_presentator is None else met_presentator
        self.vertraging = config.RADIO_VERTRAGING_S
        self._slot = threading.RLock()
        self._stop = threading.Event()
        self._draden: list[threading.Thread] = []
        self._klaar: deque[Item] = deque()
        self._gepland: list[tuple[rotatie.Gespeeld, float]] = []   # (wat, duur in s)
        self._pool: list[rotatie.Kandidaat] = []
        self._pool_ts = 0.0
        self._sinds = 0
        self._laatste_uur: str | None = None
        self._week = programma.laad_week(self.map)
        # Het programma van het laatst ingeplande nummer: verandert het, dan
        # kondigt de presentator het nieuwe aan.
        self._programma_gepland: str | None = None
        self._inst: dict = {}
        self._inst_ts = 0.0
        self._laatst_gepland: int | None = None
        self._laatst_uitgegeven: int | None = None
        self._nood_ts = 0.0
        self._beurt = random.randrange(2)
        self._rng = random.Random()
        self.geschiedenis: list[dict] = self._lees_geschiedenis()
        self.nu: dict | None = None
        self.laatste_break: dict | None = None
        self.gestart = 0.0
        self.fout: str | None = None
        from ..routers import presentator
        self._geheugen = presentator.Geheugen()
        if self.geschiedenis:
            self._laatst_gepland = self.geschiedenis[-1].get("id")
            self._laatst_uitgegeven = self._laatst_gepland
        self.menger = Menger(self._volgende, self._bij_start,
                             overvloei_s=config.RADIO_OVERVLOEI_S, duck_db=config.RADIO_DUCK_DB)
        self.speler = Speler(self.menger, self._maak_encoder)

    # ── aan en uit ──
    def start(self) -> None:
        self.hls.mkdir(parents=True, exist_ok=True)
        self._ruim_hls()
        self._stop.clear()
        self.gestart = time.time()
        # De pool meteen, niet pas in de voorbereider: anders vraagt de speler
        # de eerste seconden om een nummer uit een lege pool, en dat is stilte.
        try:
            self._ververs_pool()
        except Exception as e:
            log.warning("radio: pool bouwen mislukt: %s", e)
        self._draden = [
            threading.Thread(target=self._voorbereid_lus, name="radio-voorbereider", daemon=True),
            threading.Thread(target=self.speler.lus, args=(self._stop,), name="radio-speler", daemon=True),
        ]
        for d in self._draden:
            d.start()
        log.info("radio: zender gestart (%s)", self.hls)

    def stop(self) -> None:
        self._stop.set()
        for d in self._draden:
            d.join(timeout=10)
        self._draden = []
        self.menger.sluit()
        with self._slot:
            for item in self._klaar:
                item.bron.sluit()
            self._klaar.clear()
            self._gepland.clear()
        self._ruim_hls()
        self.nu = None
        log.info("radio: zender gestopt")

    def draait(self) -> bool:
        return bool(self._draden) and all(d.is_alive() for d in self._draden)

    def _ruim_hls(self) -> None:
        try:
            for p in self.hls.glob("*"):
                if p.suffix in (".ts", ".m3u8", ".tmp"):
                    p.unlink(missing_ok=True)
        except Exception:
            pass

    def _maak_encoder(self) -> Encoder:
        return Encoder(encoder_opdracht(self.ffmpeg, self.hls, config.RADIO_BITRATE,
                                        config.RADIO_SEGMENT_S, config.RADIO_LIJST))

    # ── geschiedenis ──
    def _lees_geschiedenis(self) -> list[dict]:
        try:
            d = json.loads((self.map / "geschiedenis.json").read_text(encoding="utf-8"))
            return d[-_GESCHIEDENIS:] if isinstance(d, list) else []
        except Exception:
            return []

    def _bewaar_geschiedenis(self) -> None:
        try:
            self.map.mkdir(parents=True, exist_ok=True)
            tmp = self.map / "geschiedenis.json.tmp"
            tmp.write_text(json.dumps(self.geschiedenis[-_GESCHIEDENIS:], ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, self.map / "geschiedenis.json")
        except Exception as e:
            log.info("radio: geschiedenis bewaren mislukt: %s", e)

    # ── wat de menger vraagt en meldt ──
    def _volgende(self) -> Item | None:
        with self._slot:
            item = self._klaar.popleft() if self._klaar else None
        if item is None:
            # Noodroute: niets voorbereid. Snel iets kiezen, zonder break, en
            # niet vaker dan eens per twee seconden (de menger vraagt elk blok).
            nu = time.monotonic()
            if nu - self._nood_ts < 2.0:
                return None
            self._nood_ts = nu
            try:
                item = self._plan_een(snel=True)
            except Exception as e:
                log.warning("radio: noodnummer mislukt: %s", e)
                item = None
            if item is not None:
                log.info("radio: niets voorbereid, noodnummer %s", item.info.get("titel"))
        if item is None:
            return None
        if item.stem is not None and item.info.get("vorig_id") != self._laatst_uitgegeven:
            log.info("radio: break ging over een ander vorig nummer, zonder stem verder")
            item.stem = None
            item.info.pop("break", None)
        self._laatst_uitgegeven = item.info.get("id")
        return item

    def _bij_start(self, item: Item, frame: int) -> None:
        start = time.time() + self.vertraging
        duur_s = (item.duur or 0) / RATE
        info = dict(item.info)
        info["start"] = round(start, 2)
        info["eind"] = round(start + duur_s, 2) if duur_s else None
        info["ts"] = start
        with self._slot:
            self._gepland = [(g, d) for g, d in self._gepland if g.id != info.get("id")]
            # Begon het vorige pas een paar tellen geleden, dan was het een
            # kapot bestand dat meteen werd overgeslagen: dat hoort niet in
            # "laatst gespeeld".
            if self.geschiedenis and start - float(self.geschiedenis[-1].get("start") or 0) < 5:
                self.geschiedenis.pop()
            self.nu = info
            brk = info.get("break")
            if brk:
                self.laatste_break = {**brk, "tijd": round(start, 2)}
            self.geschiedenis.append(info)
            self.geschiedenis = self.geschiedenis[-_GESCHIEDENIS:]
        self._bewaar_geschiedenis()

    def straks(self) -> dict | None:
        """Het nummer hierna: wat de menger al opende, of wat er klaarstaat."""
        s = self.menger.straks
        if s is not None:
            return s.info
        with self._slot:
            return self._klaar[0].info if self._klaar else None

    # ── voorbereiden ──
    def _voorbereid_lus(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._pool or time.time() - self._pool_ts > _POOL_GELDIG:
                    self._ververs_pool()
                with self._slot:
                    vol = len(self._klaar) >= _VOORRAAD
                if vol or not self._pool:
                    self._stop.wait(1.0 if self._pool else 30.0)
                    continue
                item = self._plan_een()
                if item is not None:
                    with self._slot:
                        self._klaar.append(item)
                else:
                    self._stop.wait(1.0)
            except Exception as e:
                log.exception("radio: voorbereiden mislukt: %s", e)
                self.fout = str(e)[:200]
                self._stop.wait(5.0)

    def _ververs_pool(self) -> None:
        db = SessionLocal()
        try:
            # Het schema opnieuw lezen (programma.json kan veranderd zijn) en de
            # pool vullen voor wat er de komende uren loopt.
            self._week = programma.laad_week(self.map)
            ids, sleutels, albums = blokkeer.filter_van(blokkeer.lees(self.map))
            pool = rotatie.bouw_pool(db, config.RADIO_BIBLIOTHEKEN or None,
                                     programma.komende(time.time(), 3, self._week), albums)
            pool = [k for k in pool if k.id not in ids and k.sleutel not in sleutels]
        finally:
            db.close()
        with self._slot:
            self._pool, self._pool_ts = pool, time.time()
        if not pool:
            self.fout = "geen nummers in de bibliotheek"

    def instellingen(self) -> dict:
        """Wat de beheerder in de app instelde (Beheer → Radio), hooguit 30 s
        oud. De zender leest het bij elk nummer, dus een wijziging werkt bij het
        volgende in te plannen nummer — zonder herstart en zonder dat de
        luisteraars eruit liggen."""
        if time.time() - self._inst_ts < 30 and self._inst:
            return self._inst
        inst = {"presentator": True, "programmering": True, "overvloei": config.RADIO_OVERVLOEI_S,
                "duck": -config.RADIO_DUCK_DB, "praat_elke": planning.MAX_TUSSEN,
                "artiest_afstand": rotatie.ARTIEST_AFSTAND,
                "herhaal_uur": rotatie.NUMMER_AFSTAND_S / 3600}
        try:
            from ..routers.admin_users import getal, schakelaar
            db = SessionLocal()
            try:
                inst = {
                    "presentator": schakelaar(db, "radio-presentator"),
                    "programmering": schakelaar(db, "radio-programmering"),
                    "overvloei": getal(db, "radio-overvloei"),
                    "duck": getal(db, "radio-duck"),
                    "praat_elke": getal(db, "radio-praat-elke"),
                    "artiest_afstand": getal(db, "radio-artiest-afstand"),
                    "herhaal_uur": getal(db, "radio-herhaal-uur"),
                }
            finally:
                db.close()
        except Exception as e:
            log.info("radio: instellingen niet te lezen (%s), standaard", e)
        self._inst, self._inst_ts = inst, time.time()
        # De menger meteen bijstellen: dat zijn gewone getallen in frames.
        self.menger.overvloei = int(float(inst["overvloei"]) * RATE)
        self.menger.duck = db_naar_lineair(-float(inst["duck"]))
        return inst

    def _verwachte_aanvang(self) -> float:
        """Wanneer een luisteraar het volgende in te plannen nummer ongeveer
        hoort. Grof (op de overvloei na), en dat is genoeg: de tijd wordt
        toch op vijf minuten afgerond."""
        nu = time.time()
        basis = nu + self.vertraging
        if self.nu and self.nu.get("eind"):
            basis = max(basis, float(self.nu["eind"]))
        for _, duur in self._gepland:
            basis += max(0.0, duur - config.RADIO_OVERVLOEI_S)
        return basis

    def _recent(self) -> list[rotatie.Gespeeld]:
        return [_gespeeld(d) for d in self.geschiedenis[-60:]] + [g for g, _ in self._gepland]

    def _plan_een(self, snel: bool = False) -> Item | None:
        """Kies, plan en maak het volgende nummer. `snel` (de noodroute):
        geen luidheid meten en geen break."""
        with self._slot:
            aanvang = self._verwachte_aanvang()
            inst = self.instellingen()
            deel = rotatie.dagdeel(time.localtime(aanvang).tm_hour)
            blok = programma.blok_op(aanvang, self._week)
            k = rotatie.kies(self._pool, self._recent(), aanvang, deel, self._rng,
                             programma=blok.programma if inst["programmering"] else None,
                             artiest_afstand=int(inst["artiest_afstand"]),
                             nummer_s=float(inst["herhaal_uur"]) * 3600)
            if k is None:
                return None
            soort = uur = None
            nieuw_programma = (inst["programmering"] and self._programma_gepland is not None
                               and blok.programma.naam != self._programma_gepland)
            vorig_programma, self._programma_gepland = self._programma_gepland, blok.programma.naam
            if not snel and self.met_presentator and inst["presentator"] and self._laatst_gepland:
                if nieuw_programma:
                    # Een nieuw programma wordt altijd aangekondigd; valt het op
                    # het hele uur, dan telt dit ook als de uur-break.
                    soort, uur = "programma", planning.uur_sleutel(aanvang)
                else:
                    soort, uur = planning.plan_break(self._sinds, aanvang, self._laatste_uur,
                                                     self._rng.random(), int(inst["praat_elke"]))
            vorig_id = self._laatst_gepland
            g = rotatie.Gespeeld(k.id, k.sleutel, k.artiesten, aanvang)
            self._gepland.append((g, 0.0))
            self._laatst_gepland = k.id
            if soort:
                self._sinds = 1
                if uur:
                    self._laatste_uur = uur
            else:
                self._sinds += 1

        item = self._maak_item(k, vorig_id, snel)
        if item is not None and inst["programmering"]:
            item.info["programma"] = blok.programma.naam
        with self._slot:
            if item is None:
                self._gepland = [(x, d) for x, d in self._gepland if x is not g]
                if self._laatst_gepland == k.id:
                    self._laatst_gepland = vorig_id
                    self._programma_gepland = vorig_programma
                return None
            self._gepland = [(x, (item.duur or 0) / RATE if x is g else d)
                             for x, d in self._gepland]
        if soort:
            if not self._zet_break(item, vorig_id, soort, aanvang):
                with self._slot:
                    self._sinds = planning.MAX_TUSSEN      # volgende keer opnieuw proberen
        return item

    def _maak_item(self, k: rotatie.Kandidaat, vorig_id: int | None, snel: bool) -> Item | None:
        db = SessionLocal()
        try:
            t = db.get(Track, k.id)
            if not t or not t.path or not os.path.exists(t.path):
                log.info("radio: nummer %s is er niet meer, overgeslagen", k.id)
                return None
            gain_db = t.gain_db
            if gain_db is None and not snel:
                from .. import loudness
                gain_db = loudness.measure_gain_db(t.path)
                if gain_db is not None:
                    try:
                        t.gain_db = gain_db
                        db.commit()
                    except Exception:
                        db.rollback()
            # gain_db brengt naar -18 LUFS (ReplayGain); de radio zit op -16.
            # Onbekend: een paar dB zachter, de meeste masters zijn luider.
            versterking = (gain_db + (DOEL_LUFS + 18.0)) if gain_db is not None else -4.0
            versterking = max(-15.0, min(10.0, versterking))
            artiest = (t.orig_artist or t.artist_name or "").strip()
            info = {
                "id": t.id, "ratingKey": track_key(t.id), "titel": t.title or "",
                "artiest": artiest, "album": t.album_title or "", "jaar": t.year,
                "duur": t.duration_ms or None, "thumb": _thumb(t),
                "sleutel": k.sleutel, "artiesten": sorted(k.artiesten), "vorig_id": vorig_id,
                "album_id": t.album_id,
            }
            return Item(bron=TrackBron(t.path, self.ffmpeg),
                        duur=int((t.duration_ms or 0) * RATE / 1000) or None,
                        gain=db_naar_lineair(versterking), info=info)
        finally:
            db.close()

    def _zet_break(self, item: Item, vorig_id: int | None, soort: str, aanvang: float) -> bool:
        """Maak de break voor dit nummer en hang hem eraan. False als het niet
        lukt; dan gaat het nummer gewoon zonder."""
        from ..routers import presentator
        if not vorig_id or not presentator.stem_beschikbaar():
            return False
        model = presentator.model_bereikbaar()
        db = SessionLocal()
        try:
            volgend, vorig = db.get(Track, item.info["id"]), db.get(Track, vorig_id)
            if not volgend:
                return False
            lt = time.localtime(aanvang)
            feiten, hoeken = presentator.feiten_voor(volgend, vorig, lt.tm_hour, lt.tm_min,
                                                     lt.tm_wday)
            if soort == "programma":
                p = programma.blok_op(aanvang, self._week).programma
                feiten["programma"] = {"naam": p.naam, "wat": p.sfeer}
        finally:
            db.close()
        stem = ("man", "vrouw")[self._beurt % 2]
        self._beurt += 1
        with presentator.een_plek(60) as ok:
            if not ok:
                log.info("radio: model bezet, break overgeslagen")
                return False
            stukken = presentator.maak_aankondiging(
                feiten, hoeken, stem, model=model, geheugen=self._geheugen,
                hoek={"uur": "uur", "programma": "programma"}.get(soort), dialoog=True)
        if not stukken:
            return False
        delen = []
        stilte = np.zeros((int(0.25 * RATE), KANALEN), dtype=np.float32)
        for i, s in enumerate(stukken):
            pcm = decodeer_bestand(str(presentator.CACHE / s["naam"]), self.ffmpeg)
            if pcm is None or not len(pcm):
                return False
            if i:
                delen.append(stilte)
            delen.append(pcm)
        item.stem = np.concatenate(delen)[: 30 * RATE]
        item.info["break"] = {
            "soort": soort,
            "tekst": " ".join(s["tekst"] for s in stukken),
            "stukken": [{"stem": s["stem"], "tekst": s["tekst"]} for s in stukken],
        }
        return True

    # ── voor de routes ──
    def blokkeer(self, id_: int | None = None, sleutel: str | None = None,
                 album_id: int | None = None) -> int:
        """Een net geblokkeerd nummer meteen uit de pool en uit wat er klaarstaat
        halen (de lijst zelf staat al op schijf, zie blokkeer.py). Geeft terug
        hoeveel klaargezette nummers eruit gingen. Wat de menger al geopend
        heeft (het nummer dat nu of zo meteen klinkt), speelt nog af."""
        weg = 0
        with self._slot:
            def raak(id_k, sleutel_k, album_k) -> bool:
                return ((id_ is not None and id_k == id_) or (sleutel and sleutel_k == sleutel)
                        or (album_id is not None and album_k == album_id))
            self._pool = [k for k in self._pool if not raak(k.id, k.sleutel, k.album_id)]
            houd: deque[Item] = deque()
            for item in self._klaar:
                i = item.info
                if raak(i.get("id"), i.get("sleutel"), i.get("album_id")):
                    item.bron.sluit()
                    weg += 1
                else:
                    houd.append(item)
            self._klaar = houd
            self._gepland = [(g, d) for g, d in self._gepland if g.id != id_ or id_ is None]
        return weg

    def toestand(self) -> dict:
        return {
            "draait": self.draait(),
            "encoder": self.speler.levend(),
            "sinds": round(self.gestart, 1) if self.gestart else None,
            "encoderSinds": round(self.speler.encoder_sinds, 1) if self.speler.encoder_sinds else None,
            "herstarts": self.speler.herstarts,
            "pool": len(self._pool),
            "klaar": len(self._klaar),
            "fout": self.fout,
        }

    def laatste(self, n: int = 10) -> list[dict]:
        with self._slot:
            return [openbaar(d) for d in reversed(self.geschiedenis[-n:])]


# ── De wachter ───────────────────────────────────────────────────────────────

_zender: Zender | None = None
_wachter: threading.Thread | None = None
_wachter_stop = threading.Event()
_wachter_slot = threading.Lock()


def huidige() -> Zender | None:
    return _zender


def _aan() -> bool:
    from ..routers.admin_users import schakelaar
    db = SessionLocal()
    try:
        return schakelaar(db, "radio")
    finally:
        db.close()


def controleer(aan: bool | None = None, maak: Callable[[], Zender] = Zender) -> None:
    """Breng de zender in de stand van de schakelaar. Eén ronde van de wachter."""
    global _zender
    with _wachter_slot:
        if aan is None:
            try:
                aan = _aan()
            except Exception as e:
                log.info("radio: schakelaar lezen mislukt: %s", e)
                return
        if aan and _zender is None:
            if not shutil.which("ffmpeg") and maak is Zender:
                log.warning("radio: staat aan, maar er is geen ffmpeg")
                return
            _zender = maak()
            _zender.start()
        elif aan and _zender is not None and not _zender.draait():
            log.warning("radio: een draad van de zender is gestopt, opnieuw")
            _zender.stop()
            _zender = maak()
            _zender.start()
        elif not aan and _zender is not None:
            z, _zender = _zender, None
            z.stop()


def _wachter_lus() -> None:
    while True:
        controleer()
        if _wachter_stop.wait(5.0):
            break


def start_wachter() -> None:
    global _wachter
    if _wachter is not None and _wachter.is_alive():
        return
    _wachter_stop.clear()
    _wachter = threading.Thread(target=_wachter_lus, name="radio-wachter", daemon=True)
    _wachter.start()


def stop_wachter() -> None:
    global _zender
    _wachter_stop.set()
    with _wachter_slot:
        if _zender is not None:
            z, _zender = _zender, None
            z.stop()
