"""Een naam van buiten naast een nummer in de bibliotheek leggen.

Dit is het enige stuk code dat beslist of "Shakira x Burna Boy — Dai Dai" van
top40.nl hetzelfde nummer is als `Shakira & Burna Boy - Dai Dai (Extended)` op de
NAS. Het stond eerst in de playlist-importer, maar de Top 40 heeft precies
dezelfde vraag, en twee kopieën van deze regels lopen gegarandeerd uit elkaar.

Wat het moet overleven, gezien in de echte lijst en de echte bibliotheek:

  André Hazes            ~ Andre Hazes            accenten
  Shakira x Burna Boy    ~ Shakira & Burna Boy    x, &, feat., ft., met
  Anotr ft 54 Ultra      ~ ANOTR                  de gast staat soms in de naam
  Mr. Know It All        ~ Mr Know It All         leestekens
  Dai Dai                ~ Dai Dai (Radio Edit)   editie-achtervoegsels
  Choosin' Texas         ~ Choosin Texas          apostrofs

Wat het NIET mag doen: een remix of een live-versie doorlaten als het origineel
gevraagd wordt. Een lijst waarin de helft van de nummers de verkeerde versie is,
is erger dan een lijst met gaten — bij een gat kun je downloaden, bij een fout
nummer denkt iedereen dat het klopt.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

# Toevoegingen die een ándere opname aanduiden. Staat zoiets in de titel van het
# nummer in de bibliotheek maar niet in wat er gevraagd wordt, dan is het niet
# hetzelfde nummer — een remix is geen vervanging voor het origineel.
ANDERE_VERSIE = re.compile(
    r"\b(live|remix|rmx|instrumental|karaoke|acoustic|akoestisch|cover|"
    r"sped\s*up|slowed|reverb|mashup|edit\s*mix|bootleg|dub|demo|reprise|"
    r"a\s*cappella|acapella|"
    # 16-10-2026, uit de Top 1000: deze glipten erdoor als "het origineel".
    r"unplugged|naked|re-?record(ed|ing)?|new\s*version|broadway|cast\s*recording|"
    r"immortal|vocal\s*mix|monnow\s*valley)\b",
    re.I,
)
# Toevoegingen die dezelfde opname aanduiden en dus weg mogen bij het vergelijken.
ZELFDE_VERSIE = re.compile(
    r"\b(radio\s*edit|single\s*version|album\s*version|original\s*mix|"
    r"remaster(ed)?(\s*\d{4})?|\d{4}\s*remaster|explicit|clean|"
    r"official\s*(music\s*)?video|official\s*audio|lyrics?|audio|hd|hq|mv|visualizer|4k)\b",
    re.I,
)
# Waar een tweede artiest begint. "ft", "feat", "met", "x", "&", "vs", "with".
GAST = re.compile(
    r"\s*(?:\(|\[)?\s*\b(?:feat|ft|featuring|met|with|vs|versus|x|and|en|&|,|\+)\b\s*",
    re.I,
)


def ontdoe_accenten(s: str) -> str:
    """André → Andre. NFKD en dan alles weg wat op zichzelf geen letter is."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c))


def sleutel(s: str) -> str:
    """De vergelijkvorm: kleine letters, geen accenten, geen leestekens, geen
    haakjes-toevoegingen die over dezelfde opname gaan."""
    s = ontdoe_accenten(s).lower()
    s = ZELFDE_VERSIE.sub(" ", s)
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\b(feat|ft|featuring)\.?\b.*", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def is_andere_versie(titel: str) -> bool:
    """Staat er een woord in dat een ándere opname aanduidt? Alleen kijken naar
    wat tussen haakjes of na een streepje staat: een nummer dat "Live Forever"
    heet is geen live-opname."""
    t = ontdoe_accenten(titel or "").lower()
    t = ZELFDE_VERSIE.sub(" ", t)
    staart = " ".join(re.findall(r"\((.*?)\)|\[(.*?)\]", t) and
                      [x for p in re.findall(r"\((.*?)\)|\[(.*?)\]", t) for x in p if x] or [])
    na_streep = ""
    m = re.search(r"\s[-–—]\s(.*)$", t)
    if m:
        na_streep = m.group(1)
    return bool(ANDERE_VERSIE.search(staart + " " + na_streep))


def artiest_delen(artiest: str) -> list[str]:
    """"Shakira x Burna Boy" → ["shakira x burna boy", "shakira", "burna boy"].

    De hele naam blijft vooraan staan: een duo dat écht zo heet ("Simon &
    Garfunkel") moet als geheel kunnen matchen vóór de losse helften."""
    heel = sleutel(artiest)
    delen = [heel] if heel else []
    ruw = ontdoe_accenten(artiest or "").lower()
    ruw = re.sub(r"\(.*?\)|\[.*?\]", " ", ruw)
    for stuk in GAST.split(ruw):
        k = re.sub(r"[^a-z0-9]+", "", stuk)
        if k and k not in delen:
            delen.append(k)
    return delen


def zonder_artiest(titel: str, artiest: str) -> str:
    """"SIENNA SPIRO - Great Expectation" → "Great Expectation".

    Niet elk bestand is netjes getagd. In deze bibliotheek staan nummers waarvan
    de titel de artiest bevat, omdat de tag ooit uit de bestandsnaam is gehaald:
    "3 - SIENNA SPIRO - Great Expectation". Zonder dit vindt de Top 40 zo'n
    nummer nooit — hij zoekt op "Great Expectation" en de bibliotheek zegt dat
    het nummer anders heet. Dat overkwam vier van de veertig meteen de eerste
    avond, nadat de tagronde eroverheen was gegaan.

    Alleen de vóórkant wordt weggehaald, en alleen als het echt de artiest is:
    een nummer dat "Simon - The Man" heet blijft zo heten als de artiest iemand
    anders is.
    """
    t = (titel or "").strip()
    # Een leidend volgnummer hoort ook niet in een titel: "3 - Artiest - Titel".
    t = re.sub(r"^\s*\d{1,3}\s*[-.]\s*", "", t)
    a = sleutel(artiest)
    if not a:
        return t
    stukken = re.split(r"\s+[-–—]\s+", t, maxsplit=1)
    if len(stukken) == 2 and sleutel(stukken[0]) == a:
        return stukken[1].strip() or t
    return t


class Bibliotheek:
    """Alle nummers waar deze gebruiker bij mag, klaar om tegenaan te leggen.

    De titels worden één keer genormaliseerd. Dat lijkt een detail, maar de
    vorige versie normaliseerde bij elke niet-gevonden regel de héle bibliotheek
    opnieuw — met veertigduizend nummers en veertig regels zijn dat anderhalf
    miljoen reguliere expressies voor één lijst.

    Elk nummer staat onder twee sleutels als de tag rommelig is: zijn eigen
    titel, én de titel zonder de artiest ervoor. Zie zonder_artiest().
    """

    def __init__(self, tracks) -> None:
        self.op_titel: dict[str, list] = defaultdict(list)
        self.paren: list[tuple[str, object]] = []
        for t in tracks:
            k = sleutel(t.title)
            if not k:
                continue
            self.op_titel[k].append(t)
            self.paren.append((k, t))
            kaal = sleutel(zonder_artiest(t.title, getattr(t, "artist_name", "") or ""))
            if kaal and kaal != k:
                self.op_titel[kaal].append(t)
                self.paren.append((kaal, t))

    def _artiest_blob(self, t) -> str:
        return sleutel(getattr(t, "artist_name", "") or "") + "|" + \
               sleutel(getattr(t, "orig_artist", "") or "") + "|" + \
               sleutel(getattr(t, "album_artist", "") or "")

    def zoek(self, artiest: str, titel: str):
        """Het beste nummer, of None. Nooit een andere versie dan gevraagd."""
        kt = sleutel(titel)
        if not kt:
            return None
        wil_andere_versie = is_andere_versie(titel)

        kandidaten = list(self.op_titel.get(kt, []))
        if not kandidaten:
            # Titel zit in de andere vervat ("Dai Dai" ⊂ "Dai Dai (Extended)").
            # Begrensd, want dit loopt over de hele bibliotheek.
            for k, t in self.paren:
                if k and (k in kt or kt in k):
                    kandidaten.append(t)
                    if len(kandidaten) >= 24:
                        break
        if not kandidaten:
            return None

        if not wil_andere_versie:
            # Geen terugval op een andere versie als er geen zuivere is: dan
            # liever niets (en blijft het origineel openstaan) dan de
            # unplugged of de remix stilletjes als origineel in een lijst.
            kandidaten = [t for t in kandidaten if not is_andere_versie(t.title)]
            if not kandidaten:
                return None

        namen = artiest_delen(artiest)
        if namen:
            for naam in namen:                       # hele naam eerst, dan de delen
                for t in kandidaten:
                    blob = self._artiest_blob(t)
                    if naam and (naam in blob or any(naam == d for d in blob.split("|"))):
                        return t
            # Andersom: de bibliotheek heeft "ANOTR", de lijst "Anotr ft 54 Ultra".
            for t in kandidaten:
                for deel in self._artiest_blob(t).split("|"):
                    if deel and deel in namen[0]:
                        return t
            return None

        return kandidaten[0] if len(kandidaten) == 1 else None


def match_lijst(tracks, items: list[dict]) -> tuple[list, list[dict]]:
    """[{title, artist}] → (gevonden tracks in volgorde, niet gevonden items).

    Een nummer komt maar één keer in de uitkomst: een hitlijst met twee keer
    hetzelfde nummer leest als een fout, ook als de bron het twee keer noemt.
    """
    bib = Bibliotheek(tracks)
    gevonden: list = []
    gezien: set[int] = set()
    missers: list[dict] = []
    for it in items:
        titel = (it.get("title") or "").strip()
        artiest = (it.get("artist") or "").strip()
        if not titel:
            continue
        t = bib.zoek(artiest, titel)
        if t is None:
            missers.append({"title": titel, "artist": artiest, **(
                {"position": it["position"]} if it.get("position") else {})})
            continue
        if t.id in gezien:
            continue
        gezien.add(t.id)
        gevonden.append(t)
    return gevonden, missers
