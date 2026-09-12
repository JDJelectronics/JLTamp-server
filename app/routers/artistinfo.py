"""Een korte biografie bij een artiest, uit Wikipedia.

De app toont dit als een venster over de speler heen: foto, naam, en de inleiding
van het Wikipedia-artikel. De bibliotheek zelf weet dat niet — een muziekbestand
draagt hooguit een naam — dus het komt van buiten.

**Waarom Wikipedia en niet een taalmodel.** Dit gaat over bestaande personen.
Een model schrijft moeiteloos een geboortejaar en drie albums op die niet
bestaan, en aan de tekst is niet te zien welke regel klopt. Wikipedia kan er ook
naast zitten, maar dan staat het tenminste ergens gecontroleerd, en de link naar
het artikel gaat mee zodat je het kunt nakijken.

**De valkuil is de verkeerde pagina.** Zoeken op een artiestennaam levert net zo
goed een politicus op, of een televisieprogramma, of een van hun eigen liedjes.
Een biografie van de verkeerde persoon onder de naam van je zangeres is erger
dan geen biografie, en je ziet het niet altijd meteen. Daarom worden de treffers
nagelopen: pagina's die zichzelf als lied, album, film of serie omschrijven
vallen af, en er telt alleen een pagina die over een muzikant of groep gaat.
Blijft er niets over, dan komt er niets terug.

**Eén verzoek per artiest.** De zoekopdracht levert in dezelfde vraag ook de
omschrijvingen, de inleidingen en de foto's van de kandidaten. Dat scheelt zes
verzoeken per artiest, en dat is niet alleen netjes maar noodzakelijk: bij een
handvol snelle opvragingen achter elkaar antwoordt Wikipedia met 429.

Alles wordt op schijf gecachet, ook een misser — anders vraagt elke tik op
dezelfde artiest opnieuw het net op. Een misser verloopt sneller dan een
treffer, want een ontbrekend artikel kan er volgende maand wel zijn. Een
mislukte opvraging wordt NIET gecachet: anders bevriest één storing de uitkomst
voor een week.

Taal: eerst die van de app, anders Engels. Fries en Nederlands hebben een eigen
Wikipedia, dus dat werkt vaak gewoon.

Uit staat het met `JLTAMP_ARTIST_BIO=false`. Naar buiten gaat alleen de
artiestennaam — niets over de gebruiker, en nooit iets uit de bibliotheek.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..deps import require_user
from ..models import User

router = APIRouter(tags=["artist"])
log = logging.getLogger("artistinfo")

CACHE = config.DATA_DIR / "artistinfo"
CACHE.mkdir(parents=True, exist_ok=True)

# Wikipedia vraagt om een herkenbare naam met een adres erbij; een vage
# user-agent is precies wat er het eerst geweigerd wordt.
_UA = "JLTamp/1.0 (+https://github.com/JDJelectronics/JLTamp-server)"

_TREFFER_GELDIG = 60 * 24 * 3600     # een biografie verandert zelden
_MISSER_GELDIG = 7 * 24 * 3600       # een ontbrekend artikel kan er later zijn

# Minstens deze tijd tussen twee opvragingen, over alle gebruikers heen. Gemeten:
# vijf opvragingen vlak achter elkaar en Wikipedia antwoordt met 429, waarna
# alles er even uit ligt. Dit is een venster over de speler, geen batchtaak —
# een halve seconde wachten merkt niemand.
_MIN_TUSSENPOZE = 0.5
_slot = threading.Lock()
_laatste_oproep = 0.0

# Talen met een eigen Wikipedia waarin de app vertaald is. De rest valt terug
# op Engels.
_TALEN = {"nl", "en", "de", "es", "pt", "fy"}

# Waar een muziekpagina zich aan laat herkennen. Bewust ruim: liever een pagina
# te veel dan de juiste artiest missen omdat er "componist" stond in plaats van
# "musicus".
#
# GEEN \b voor deze woorden, en dat is met opzet. Het Nederlands en het Duits
# plakken samenstellingen aan elkaar: "rockband", "muziekgroep", "popzangeres".
# Met een woordgrens ervoor valt "band" in "rockband" buiten de boot, en dan
# zakt de eigen pagina van Paramore af en wint hun zangeres. Precies dat gebeurde.
# Alleen de korte, dubbelzinnige woorden houden hun grenzen: "dj" zit anders in
# "Djibouti" en "duo" in "duurzaam".
_MUZIEK = re.compile(
    r"(?:"
    r"music|musician|singer|songwriter|band|rapper|composer|guitarist|"
    r"drummer|bassist|pianist|vocalist|producer|orchestra|choir|"
    # "zang" en niet "zanger": dan vallen zangduo, zangeres en zangkoor er
    # allemaal onder. Op "Nederlands zangduo" — de omschrijving van Suzan &
    # Freek — kwam de vorige lijst niet uit, en dan viel de app terug op het
    # Engelse artikel over een Nederlands duo.
    r"muziek|muzikant|zang|componist|gitarist|orkest|koor|"
    r"groep|formatie|liedjesschrijver|"
    r"musik|musiker|sänger|sängerin|komponist|schlagzeuger|"
    r"música|músico|cantante|compositor|banda|grupo|cantor|cantora|"
    r"muzyk|sjonger|sjongeres"
    r"|duo|trio|kwartet|quartet"
    r")|(?:\b(?:dj)\b)",
    re.IGNORECASE,
)

# ... en waar het juist GEEN artiestenpagina is. Dit gaat vóór op de lijst
# hierboven, want een liedjespagina noemt bijna altijd de zanger en zou anders
# als treffer doorgaan. Zoeken op "Sober" bij Kelly Clarkson leverde precies dat.
# Doorverwijspagina's horen hier ook thuis: bij "Lola Young" staat die bovenaan,
# en die haalt de muziekwoorden binnen van de artiesten die hij opsomt.
#
# "muziekalbum" staat er apart in en niet als "album": de omschrijving van een
# albumpagina luidt "muziekalbum van Muse", en dat bevat óók het woord "muziek".
# Op volgorde kijken helpt daar dus niet — dit moet gewoon een eigen term zijn.
_GEEN_ARTIEST = re.compile(
    r"(?:"
    r"muziekalbum|muziekvideo|studioalbum|verzamelalbum|doorverwijspagina|"
    r"televisieserie|televisieprogramma|videospel|"
    r"disambiguation|studio album|extended play|soundtrack|compilation album|"
    r"begriffskl|desambiguaci|fernsehserie|"
    r"canción|película|serie de televisión"
    r")|(?:\b(?:song|single|film|movie|television|tv series|video game|novel|"
    r"album by|lied|nummer van|single van|album van|film uit|roman|lied von)\b)",
    re.IGNORECASE,
)


# Namen die geen artiest zijn maar een verzamelaanduiding. Zonder deze controle
# gaat "Various Artists" gewoon de zoekmachine in en komt er een willekeurige
# zangeres terug die toevallig bovenaan stond — met foto en al, en niets wijst
# erop dat het de verkeerde is. Dezelfde lijst als in lyrics.py, waar hetzelfde
# probleem speelde: een verzamelalbum zet dit als albumartiest neer en de tracks
# erven het.
_GEEN_NAAM = {"", "various artists", "various", "va", "verzamelaars",
              "diverse artiesten", "unknown artist", "unknown", "soundtrack"}


def _is_verzamelnaam(naam: str) -> bool:
    return re.sub(r"[^a-z ]+", "", (naam or "").lower()).strip() in _GEEN_NAAM


def _aan() -> bool:
    v = os.environ.get("JLTAMP_ARTIST_BIO", "true").strip().lower()
    return v in ("1", "true", "yes", "on")


def _sleutel(naam: str, taal: str) -> str:
    schoon = re.sub(r"[^a-z0-9]+", "-", naam.lower()).strip("-") or "leeg"
    return f"{schoon[:80]}.{taal}.json"


def _haal_json(url: str) -> dict | None:
    """None betekent: de opvraging is MISLUKT. Een geslaagde opvraging zonder
    resultaat geeft gewoon een (lege) dict terug. Dat onderscheid draagt tot in
    de cache, want een 429 mag geen week lang als 'niet gevonden' blijven staan.
    """
    global _laatste_oproep
    with _slot:
        wacht = _MIN_TUSSENPOZE - (time.monotonic() - _laatste_oproep)
        if wacht > 0:
            time.sleep(wacht)
        _laatste_oproep = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        log.info("wikipedia opvraging mislukt (%s): %s", type(e).__name__, e)
        return None


def _normaliseer(tekst: str) -> str:
    """Kleine letters, accenten en leestekens eraf, spaties samengetrokken."""
    plat = unicodedata.normalize("NFKD", tekst or "")
    plat = "".join(c for c in plat if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", plat.lower())).strip()


def _hoort_bij(naam: str, titel: str) -> bool:
    """Gaat deze pagina wel over de artiest die we zochten?

    Dit ontbrak, en dat liep meteen mis: op de Friese Wikipedia bestaat geen
    artikel over Leona Lewis, dus gaf de zoekmachine losjes verwante pagina's
    terug — en de eerste die "gitarist" in zijn omschrijving had, was Jimmy Page.
    Een muziekpagina, keurig door alle andere controles heen, en volstrekt de
    verkeerde. Op een kleine wiki is dat eerder regel dan uitzondering.

    De titel van een artiestenpagina is de naam, soms met een verduidelijking
    erachter: "Kane (band)", "Nirvana (Amerikaanse band)". Die haakjes gaan er
    dus af voor de vergelijking.

    Liever te streng dan te ruim: "P!nk" valt hierdoor af tegen de titel "Pink
    (zangeres)". Niets tonen is een teleurstelling, de verkeerde persoon tonen
    is een fout die niemand opmerkt.
    """
    n = _normaliseer(naam)
    t = _normaliseer(re.sub(r"\s*\([^)]*\)\s*$", "", titel or ""))
    if not n or not t:
        return False
    if n == t:
        return True
    # Bevatting mag, maar niet met flintertjes: "Kane" mag matchen met
    # "Kane band", "Ed" niet met "Ed Sheeran".
    korte = min(n, t, key=len)
    return len(korte) >= 4 and (n in t or t in n)


def _is_artiest(p: dict) -> bool:
    """Gaat deze pagina over een muzikant of groep?

    `description` is het regeltje onder de titel ("Amerikaans zangeres"); dat is
    de betrouwbaarste aanwijzing en gaat dus voor. Maar lang niet elke pagina
    heeft er een — en juist daar ging het mis: bij "Lola Young" won een liedje
    van haar, want zonder omschrijving keek de vorige versie alleen of er ergens
    in de inleiding een muziekwoord stond, en dat staat er in een liedjespagina
    natuurlijk ook.

    Zonder omschrijving beslist daarom de EERSTE ZIN, en wel op volgorde: een
    artikel begint met wat het onderwerp ís.

        "One Thing is een nummer van Lola Young, Engelse singer-songwriter"
         → "nummer" komt eerst  → geen artiest
        "Lola Young is een Engelse singer-songwriter, bekend van de single ..."
         → "singer-songwriter" komt eerst → artiest

    Beide zinnen bevatten allebei soorten woorden; alleen de volgorde scheidt ze.
    """
    omschrijving = p.get("description") or ""
    if omschrijving:
        if _GEEN_ARTIEST.search(omschrijving):
            return False
        return bool(_MUZIEK.search(omschrijving))

    inleiding = (p.get("extract") or "").strip()
    eerste_zin = re.split(r"(?<=[.!?])\s", inleiding, maxsplit=1)[0][:300] if inleiding else ""
    if not eerste_zin:
        return False
    pos = _MUZIEK.search(eerste_zin)
    neg = _GEEN_ARTIEST.search(eerste_zin)
    if not pos:
        return False
    return not neg or pos.start() < neg.start()


def _zoek(taal: str, naam: str) -> tuple[bool, dict | None]:
    """(geslaagd, pagina). geslaagd=False betekent: Wikipedia gaf geen antwoord.

    Eén verzoek levert de vijf beste treffers mét omschrijving, inleiding en
    foto. Ze worden in de volgorde van de zoekmachine nagelopen; de eerste die
    op een artiest lijkt wint. Zit er geen artiest bij, dan liever niets dan de
    verkeerde persoon.
    """
    q = urllib.parse.quote(naam)
    data = _haal_json(
        f"https://{taal}.wikipedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrsearch={q}&gsrlimit=5"
        f"&prop=extracts|pageimages|description"
        f"&exintro=1&explaintext=1&exlimit=5"
        f"&piprop=original|thumbnail&pithumbsize=800"
    )
    if data is None:
        return False, None
    paginas = ((data.get("query") or {}).get("pages")) or {}
    for p in sorted(paginas.values(), key=lambda x: x.get("index", 99)):
        if _hoort_bij(naam, p.get("title") or "") and _is_artiest(p):
            return True, p
    return True, None


# Een verwijsnotitie bovenaan een artikel: "Dizze side giet oer de Sweedske band
# ABBA. Foar oare betsjuttings, sjoch: Abba (betsjuttingsside)." Die hoort bij de
# wiki, niet bij de artiest, en staat wél in de inleiding die we ophalen — dus
# het eerste wat je leest is een voetnoot over naamgeving.
_VERWIJSNOTITIE = re.compile(
    r"(betsjuttingsside|doorverwijspagina|disambiguation|begriffskl|desambiguaci)"
    r"|^(dizze side giet oer|dit artikel gaat over|voor andere betekenissen"
    r"|for other uses|this article is about|dieser artikel behandelt)",
    re.IGNORECASE,
)


def _zonder_verwijsnotitie(tekst: str) -> str:
    """De inleiding zonder de verwijsnotities die er soms bovenop staan."""
    alineas = [a for a in (tekst or "").split("\n")]
    while alineas and (not alineas[0].strip() or _VERWIJSNOTITIE.search(alineas[0].strip())):
        alineas.pop(0)
    return "\n".join(alineas).strip()


# Bestanden die in een artikel staan maar geen foto van de artiest zijn: het
# bandlogo, een handtekening, een landkaart, een vlag. Vectorbestanden zijn het
# bijna altijd; de rest herkennen we aan de naam.
_GEEN_FOTO = re.compile(
    r"logo|signature|handtekening|autograaf|wapen|flag|vlag|icon|"
    r"map|kaart|locator|handprint|star|walk of fame|disc|award",
    re.IGNORECASE,
)


def _fotos(taal: str, titel: str) -> list[str]:
    """De foto's uit het artikel, voor het galerijtje boven de tekst.

    De samenvatting geeft er maar één — de leadfoto — en dat is voor een
    artiestenpagina te mager. Dit is een tweede verzoek, maar alleen bij een
    cachemisser, en de uitkomst gaat mee in hetzelfde bestand.
    """
    t = urllib.parse.quote((titel or "").replace(" ", "_"), safe="")
    data = _haal_json(f"https://{taal}.wikipedia.org/api/rest_v1/page/media-list/{t}")
    if not data:
        return []
    uit: list[str] = []
    for item in data.get("items") or []:
        if item.get("type") != "image":
            continue
        naam = item.get("title") or ""
        if _GEEN_FOTO.search(naam):
            continue
        bron = ((item.get("srcset") or [{}])[0]).get("src") or ""
        if not bron or bron.lower().endswith(".svg"):
            continue
        # Wikipedia levert protocol-relatieve adressen ("//upload.wikimedia...").
        # Zo'n adres laadt op het toestel niet; er moet https: voor.
        if bron.startswith("//"):
            bron = "https:" + bron
        if bron.startswith("https://") and bron not in uit:
            uit.append(bron)
        if len(uit) >= 6:
            break
    return uit


def _uit_pagina(p: dict, taal: str) -> dict:
    titel = p.get("title") or ""
    fotos = _fotos(taal, titel)
    afb = (p.get("original") or {}).get("source") or (p.get("thumbnail") or {}).get("source")
    # De hoofdafbeelding van een artikel is lang niet altijd een foto: bij ABBA
    # op de Friese Wikipedia is het het bandlogo, een zwart vierkant. Dat kwam
    # in de kop te staan én bepaalde de achtergrondkleur van het venster, dus
    # het hele ding werd zwart. Ziet de hoofdafbeelding eruit als een logo of
    # een tekening, dan nemen we de eerste échte foto uit het artikel.
    if (not afb or _GEEN_FOTO.search(afb) or afb.lower().split("?")[0].endswith(".svg")) and fotos:
        afb = fotos[0]
    return {
        "naam": titel,
        "bio": _zonder_verwijsnotitie(p.get("extract") or ""),
        "beschrijving": (p.get("description") or "").strip(),
        "afbeelding": afb,
        "bron": "wikipedia",
        "bronUrl": f"https://{taal}.wikipedia.org/wiki/{urllib.parse.quote(titel.replace(' ', '_'))}",
        "taal": taal,
        "afbeeldingen": fotos,
    }


# ── Vertalen met het model op de eigen machine ───────────────────────────────
#
# De Friese Wikipedia is klein: over de meeste artiesten staat daar niets, en
# dan kreeg een Friestalige gebruiker een Nederlandse of Engelse tekst. Nu wordt
# het artikel dat er wél is vertaald naar de taal van de app.
#
# Het model draait op de machine van de server zelf (Ollama). Geen dienst van
# buiten, en de tekst verlaat het huis niet. Duurt ongeveer tien seconden per
# artiest, één keer — daarna staat het in dezelfde cache als de rest.
#
# Wat je ervan mag verwachten: begrijpelijk, niet foutloos. Fries is een kleine
# taal en een model dat er weinig van gezien heeft maakt fouten in naamvallen en
# leenwoorden. Voor Duits, Spaans en Portugees is het merkbaar beter. Uit te
# zetten met JLTAMP_VERTAAL=false, en dan krijg je gewoon de brontekst.
_AI = os.environ.get("JLTAMP_AI_URL", "http://host.docker.internal:11434").rstrip("/")
_AI_MODEL = os.environ.get("JLTAMP_AI_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")

_TAALNAAM = {"nl": "Nederlands", "en": "Engels", "de": "Duits", "es": "Spaans",
             "pt": "Portugees", "fy": "Fries"}


def _vertalen_aan() -> bool:
    return os.environ.get("JLTAMP_VERTAAL", "true").strip().lower() in ("1", "true", "yes", "on")


def _vertaal(tekst: str, naar: str) -> str:
    """De tekst in de gevraagde taal, of onveranderd als het niet lukt.

    Mislukken is hier geen fout maar een normale uitkomst: de machine kan uit
    staan of het model kan er te lang over doen. Dan is een Nederlandse zin nog
    altijd beter dan geen zin.
    """
    doel = _TAALNAAM.get(naar)
    if not tekst or not doel or not _vertalen_aan():
        return tekst
    opdracht = (
        f"Vertaal de volgende tekst naar het {doel}. "
        "Geef ALLEEN de vertaling terug, zonder inleiding, uitleg of aanhalingstekens. "
        "Behoud namen van personen, bands en albums exact zoals ze er staan.\n\n"
        f"{tekst}"
    )
    try:
        verzoek = urllib.request.Request(
            f"{_AI}/api/generate",
            data=json.dumps({
                "model": _AI_MODEL,
                "prompt": opdracht,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.2, "num_predict": 700},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(verzoek, timeout=90) as r:
            uit = (json.loads(r.read().decode()).get("response") or "").strip()
    except Exception as e:
        log.info("vertalen mislukt (%s): %s", naar, e)
        return tekst
    # Een model dat toch gaat uitleggen levert iets veel langers op dan de
    # brontekst. Dan liever het origineel dan een lap tekst.
    if not uit or len(uit) > len(tekst) * 2.2:
        return tekst
    return uit


@router.get("/artist/info")
def artist_info(
    name: str,
    lang: str = "en",
    user: User = Depends(require_user),
):
    """Biografie bij een artiestennaam. Lege `bio` = niets gevonden."""
    naam = (name or "").strip()
    if not naam:
        raise HTTPException(status_code=400, detail="name is verplicht")

    taal = (lang or "en").strip().lower()[:2]
    if taal not in _TALEN:
        taal = "en"

    leeg = {"naam": naam, "bio": "", "beschrijving": "", "afbeelding": None,
            "afbeeldingen": [], "bron": "", "bronUrl": "", "taal": taal,
            "bronTaal": ""}

    if not _aan() or _is_verzamelnaam(naam):
        return leeg

    bestand = CACHE / _sleutel(naam, taal)
    if bestand.exists():
        try:
            gecacht = json.loads(bestand.read_text(encoding="utf-8"))
            ouderdom = time.time() - float(gecacht.get("opgehaald") or 0)
            geldig = _TREFFER_GELDIG if gecacht.get("bio") else _MISSER_GELDIG
            if ouderdom < geldig:
                return {k: v for k, v in gecacht.items() if k != "opgehaald"}
        except Exception:
            pass   # onleesbare cache → gewoon opnieuw ophalen

    # Eerst in de taal van de app, dan terugvallen. Fries gaat via Nederlands:
    # de Friese Wikipedia is klein en heeft over de meeste artiesten niets, en
    # voor een Friestalige lezer is een Nederlands artikel dichterbij dan een
    # Engels. Gemeten: van Leona Lewis, Metejoor, Kane en Shakira staat er geen
    # van vieren op fy.wikipedia.
    ketting = ["fy", "nl", "en"] if taal == "fy" else [taal, "en"]
    # Sommige artiestennamen dragen een toevoeging: "Josh Woodward (Instrumental
    # Versions)". Daar bestaat geen artikel over; over Josh Woodward wel. Dus
    # eerst de naam zoals hij is, en anders zonder die toevoeging.
    kaal = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*$", "", naam).strip()
    namen = [naam] if kaal == naam else [naam, kaal]

    geslaagd, pagina, gebruikte_taal = False, None, taal
    for n in namen:
        for t in dict.fromkeys(ketting):      # volgorde houden, dubbelen eruit
            ok, p = _zoek(t, n)
            geslaagd = geslaagd or ok
            if ok and p:
                pagina, gebruikte_taal = p, t
                break
        if pagina:
            break

    if not geslaagd:
        # Wikipedia gaf geen antwoord. Niets cachen — anders staat deze artiest
        # een week lang als "niets gevonden" genoteerd door één hapering.
        return leeg

    uit = _uit_pagina(pagina, gebruikte_taal) if pagina else dict(leeg, taal=gebruikte_taal)

    # Gevonden in een andere taal dan gevraagd? Dan vertalen. De bronvermelding
    # blijft naar het oorspronkelijke artikel wijzen, want dáár staat het.
    if uit.get("bio") and gebruikte_taal != taal:
        uit["bio"] = _vertaal(uit["bio"], taal)
        if uit.get("beschrijving"):
            uit["beschrijving"] = _vertaal(uit["beschrijving"], taal)
        uit["taal"] = taal
        uit["bronTaal"] = gebruikte_taal

    try:
        bestand.write_text(json.dumps({**uit, "opgehaald": time.time()}), encoding="utf-8")
    except Exception as e:
        log.debug("artistinfo cache schrijven mislukt: %s", e)

    return uit
