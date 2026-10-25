"""De programmering: welk programma er wanneer op JLTamp Radio is.

Een radio die de hele dag hetzelfde draait, is een playlist. Wat een zender
een zender maakt, is dat de ochtend anders klinkt dan de avond en de
vrijdagavond anders dan de zondagochtend. Dus een week vol programma's: per
dag een paar blokken, elk met een eigen naam en een eigen filter (jaren,
genres). De rotatie (rotatie.kies) loot daarna binnen dat filter, met dezelfde
regels als altijd.

Twee dingen die je moet weten over de bibliotheek, want daar is dit op
afgesteld (gemeten op een grote bibliotheek):
- Het jaar is het jaar van het album. Een hit uit 1985 op een verzamelaar uit
  2012 telt hier als 2012. De decenniumblokken zijn dus wat krapper dan je
  zou willen, maar wat erin valt klopt meestal wel.
- Een kwart heeft geen genre en een vijfde geen jaar. Die vallen buiten elk
  filter en komen alleen in de blokken zonder filter langs.

Heeft een programma te weinig nummers (een lege bibliotheek, een te streng
filter), dan valt de zender terug op de gewone mix: liever een ander nummer
dan stilte (zie zender._plan_een).

Aanpassen zonder nieuwe code: zet een `programma.json` in de radiomap
(/data/radio). Zelfde vorm als hieronder (weekdag 0 = maandag), zie
`laad_week`. Een fout in dat bestand en de standaardweek blijft gelden.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("radio")


@dataclass(frozen=True)
class Programma:
    naam: str
    # Voor de presentator: in een paar woorden wat voor programma het is.
    sfeer: str = ""
    jaren: tuple[int, int] | None = None           # van, tot en met
    genres: tuple[str, ...] = ()                   # één ervan in het genre (deelstring)
    geen_genres: tuple[str, ...] = ()              # geen van deze
    # Hoe zwaar de bekende nummers (Top 40, hartjes, veel gedraaid) wegen
    # tegenover ontdekken. 1 = zoals de gewone mix; hoger = meer hits.
    hits: float = 1.0

    def past(self, jaar: int | None, genre: str) -> bool:
        if self.jaren:
            if not jaar or not (self.jaren[0] <= jaar <= self.jaren[1]):
                return False
        g = (genre or "").lower()
        if self.genres and not any(x in g for x in self.genres):
            return False
        if self.geen_genres and any(x in g for x in self.geen_genres):
            return False
        return True

    @property
    def heeft_filter(self) -> bool:
        return bool(self.jaren or self.genres or self.geen_genres)


# ── De bouwstenen ────────────────────────────────────────────────────────────

_FEEST = ("dance", "party", "schlager", "carnaval", "eurodance", "disco", "top 40")
_RUSTIG_NIET = ("hardstyle", "hardcore", "metal", "techno", "rap", "dirty south", "carnaval",
                "schlager", "drum")
_NL = ("nederlandstalig", "nederlands", "nederpop", "levenslied", "vlaams")

NACHT = Programma("De Nachtmix", "rustig doordraaien voor wie nog wakker is",
                  geen_genres=_RUSTIG_NIET + ("party", "dance"), hits=0.6)
GOEDEMORGEN = Programma("Goedemorgen JLTamp", "vrolijke meezingers om wakker van te worden",
                        jaren=(1990, 2030), genres=("pop", "dance", "nederlandstalig", "r&b", "top 40"),
                        geen_genres=("hardstyle", "hardcore", "metal"), hits=2.0)
WERKMIX = Programma("De Werkmix", "lekker doorlopende muziek voor onder het werk",
                    geen_genres=("hardstyle", "hardcore", "metal", "carnaval", "schlager"), hits=1.3)
THUISKOMEN = Programma("Thuiskomen", "de grootste hits van nu voor de weg naar huis",
                       jaren=(2015, 2030), geen_genres=("metal", "hardcore"), hits=2.5)
RUSTIG_AAN = Programma("Rustig Aan", "zachte nummers voor de avond",
                       genres=("singer", "folk", "acoustic", "soul", "r&b", "jazz", "blues", "pop"),
                       geen_genres=_RUSTIG_NIET + ("dance", "party", "electro"), hits=1.2)


def _jaren(naam: str, van: int, tot: int, sfeer: str) -> Programma:
    return Programma(naam, sfeer, jaren=(van, tot),
                     geen_genres=("kids", "films", "hardcore"), hits=1.8)


JAREN_60_70 = _jaren("Gouwe Ouwe", 1960, 1979, "de klassiekers uit de jaren zestig en zeventig")
JAREN_80 = _jaren("Jaren 80", 1980, 1989, "de hits uit de jaren tachtig")
JAREN_90 = _jaren("Jaren 90", 1990, 1999, "de hits uit de jaren negentig")
JAREN_00 = _jaren("De Jaren Nul", 2000, 2009, "de hits uit de jaren nul")
JAREN_10 = _jaren("Jaren 10", 2010, 2019, "de hits uit de jaren tien")
JAREN_80_90 = _jaren("Tachtig & Negentig", 1980, 1999, "de jaren tachtig en negentig door elkaar")

ROCK = Programma("Rock Avond", "gitaren en rock, van klassiek tot nu",
                 genres=("rock", "alternative", "indie", "metal"), hits=1.3)
NEDERLANDSTALIG = Programma("Nederlandstalig", "alleen Nederlandstalige muziek",
                            genres=_NL, hits=1.5)
DANCE_CLASSICS = Programma("Dance Classics", "dance uit de jaren negentig en nul",
                           jaren=(1988, 2012), genres=("dance", "electro", "house", "techno", "eurodance", "disco"),
                           hits=1.5)
BORREL = Programma("De Vrijdagmiddagborrel", "feesthits om het weekend in te luiden",
                   jaren=(1980, 2030), genres=_FEEST + ("pop", "nederlandstalig"),
                   geen_genres=("metal", "hardcore"), hits=2.5)
FOUT = Programma("Foute Vrijdag", "foute meezingers en feesthits",
                 genres=_FEEST + _NL, geen_genres=("metal",), hits=2.0)
ZATERDAG_KOFFIE = Programma("Zaterdagochtend Koffie", "rustig wakker worden met een bak koffie",
                            genres=("pop", "singer", "folk", "acoustic", "soul", "jazz"),
                            geen_genres=_RUSTIG_NIET + ("dance", "party"), hits=1.5)
TOP40_TERUG = Programma("Top 40 Terug in de Tijd", "grote hits uit alle decennia",
                        jaren=(1960, 2014), geen_genres=("kids", "films", "hardcore"), hits=3.0)
FEEST = Programma("Zaterdagavond Feest", "dansen en feesten tot laat",
                  genres=_FEEST + ("hardstyle", "rap", "latin"), hits=2.0)
ZONDAG_ZACHT = Programma("Zondag Zacht", "een rustige zondagochtend",
                         genres=("singer", "folk", "acoustic", "soul", "jazz", "blues", "pop"),
                         geen_genres=_RUSTIG_NIET + ("dance", "party", "electro"), hits=1.3)
ZONDAG_CLASSICS = Programma("Zondagmiddag Classics", "klassiekers uit de jaren zestig tot negentig",
                            jaren=(1960, 1999), geen_genres=("kids", "films"), hits=2.0)
URBAN = Programma("Urban Avond", "hiphop en R&B",
                  genres=("rap", "hip hop", "r&b", "urban"), hits=1.5)
NIEUW = Programma("Nieuw & Fris", "de nieuwste muziek",
                  jaren=(2024, 2030), geen_genres=("kids", "films"), hits=1.5)

# Weekdag (0 = maandag) → (beginuur, programma), op volgorde. Een dag begint
# om middernacht met de nacht; wat na het laatste blok komt, loopt door tot
# middernacht.
STANDAARD_WEEK: dict[int, list[tuple[int, Programma]]] = {
    0: [(0, NACHT), (6, GOEDEMORGEN), (10, WERKMIX), (13, JAREN_90), (17, THUISKOMEN), (20, RUSTIG_AAN)],
    1: [(0, NACHT), (6, GOEDEMORGEN), (10, WERKMIX), (13, JAREN_80), (17, THUISKOMEN), (20, ROCK)],
    2: [(0, NACHT), (6, GOEDEMORGEN), (10, WERKMIX), (13, JAREN_00), (17, THUISKOMEN), (20, NEDERLANDSTALIG)],
    3: [(0, NACHT), (6, GOEDEMORGEN), (10, NIEUW), (13, JAREN_60_70), (17, THUISKOMEN), (20, DANCE_CLASSICS)],
    4: [(0, NACHT), (6, GOEDEMORGEN), (10, WERKMIX), (13, JAREN_10), (16, BORREL), (20, FOUT)],
    5: [(0, NACHT), (8, ZATERDAG_KOFFIE), (11, TOP40_TERUG), (14, JAREN_80_90), (17, URBAN), (20, FEEST)],
    6: [(0, NACHT), (8, ZONDAG_ZACHT), (12, ZONDAG_CLASSICS), (16, NIEUW), (19, RUSTIG_AAN)],
}


# ── Laden en opzoeken ────────────────────────────────────────────────────────

def _uit_json(d: dict) -> Programma:
    jaren = d.get("jaren")
    return Programma(
        naam=str(d["naam"]), sfeer=str(d.get("sfeer", "")),
        jaren=(int(jaren[0]), int(jaren[1])) if jaren else None,
        genres=tuple(str(g).lower() for g in d.get("genres", ())),
        geen_genres=tuple(str(g).lower() for g in d.get("geen_genres", ())),
        hits=float(d.get("hits", 1.0)),
    )


def laad_week(map_: Path | None) -> dict[int, list[tuple[int, Programma]]]:
    """De week uit `programma.json` in de radiomap, of de standaardweek.

    Vorm: {"0": [{"uur": 6, "naam": "…", "sfeer": "…", "jaren": [1980, 1989],
    "genres": ["pop"], "geen_genres": ["metal"], "hits": 2.0}, …], "1": …}.
    Een dag die ontbreekt, krijgt de standaarddag."""
    if map_ is None:
        return STANDAARD_WEEK
    bestand = Path(map_) / "programma.json"
    if not bestand.exists():
        return STANDAARD_WEEK
    try:
        ruw = json.loads(bestand.read_text())
        week = dict(STANDAARD_WEEK)
        for dag, blokken in ruw.items():
            week[int(dag)] = sorted(((int(b["uur"]), _uit_json(b)) for b in blokken),
                                    key=lambda x: x[0])
        return week
    except Exception as e:
        log.warning("radio: programma.json klopt niet (%s), de standaardweek blijft", e)
        return STANDAARD_WEEK


@dataclass(frozen=True)
class Blok:
    programma: Programma
    start: float          # epoch-seconden
    eind: float


def blok_op(ts: float, week: dict[int, list[tuple[int, Programma]]] | None = None) -> Blok:
    """Het programma dat op tijdstip `ts` (lokale tijd) loopt, met begin en eind."""
    week = week or STANDAARD_WEEK
    lt = time.localtime(ts)
    middernacht = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    dag = week.get(lt.tm_wday) or [(0, WERKMIX)]
    huidig, start_uur, eind_uur = dag[0][1], dag[0][0], 24
    for i, (uur, p) in enumerate(dag):
        if uur <= lt.tm_hour:
            huidig, start_uur = p, uur
            eind_uur = dag[i + 1][0] if i + 1 < len(dag) else 24
    return Blok(huidig, middernacht + start_uur * 3600, middernacht + eind_uur * 3600)


def komende(ts: float, uren: float, week: dict[int, list[tuple[int, Programma]]] | None = None) -> list[Programma]:
    """De programma's die de komende `uren` lopen (voor het vullen van de pool)."""
    gezien: dict[str, Programma] = {}
    t, tot = ts, ts + uren * 3600
    while t <= tot:
        b = blok_op(t, week)
        gezien.setdefault(b.programma.naam, b.programma)
        t = b.eind + 1
    return list(gezien.values())


def voor_app(blok: Blok) -> dict:
    return {"naam": blok.programma.naam, "sfeer": blok.programma.sfeer,
            "start": blok.start, "eind": blok.eind}
