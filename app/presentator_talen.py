"""De presentator in de taal van de app: nl, en, de, es, pt.

Alles wat de presentator in een bepaalde taal ZEGT zonder dat het model eraan
te pas komt, staat hier: de tijd in woorden, het dagdeel, de dag, de
temperatuur, het woordje tussen titel en artiest ("van", "by", "von"), hoe een
terugblik begint, en de vaste zinnen voor als het model het niet redt.

**Nooit een Nederlandse zin in een Engelse aankondiging.** Daarom staat hier
ook `klinkt_nederlands`: een eenvoudige controle op woorden die alleen in het
Nederlands voorkomen. Een aankondiging in een andere taal waar die in staan,
gaat niet door. En de terugval op Piper (die alleen Nederlands kent) geldt
alleen voor nl — zie presentator.py.

Nederlands is de standaard: oude apps sturen geen taal mee, en JLTamp Radio
(één stream voor iedereen) praat altijd Nederlands.
"""
from __future__ import annotations

import re

TALEN = ("nl", "en", "de", "es", "pt")
STANDAARD = "nl"

TAALNAAM = {"nl": "Dutch", "en": "English", "de": "German", "es": "Spanish",
            "pt": "Portuguese"}


def taal_van(waarde: str | None) -> str:
    """"pt-BR" → "pt", "EN" → "en", "fy" of iets onbekends → "nl"."""
    t = re.split(r"[-_]", (waarde or "").strip().lower())[0]
    return t if t in TALEN else STANDAARD


# ── De tijd ──────────────────────────────────────────────────────────────────

_UREN = {
    "en": ["twelve", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven"],
    "de": ["zwölf", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht",
           "neun", "zehn", "elf"],
    "es": ["doce", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
           "nueve", "diez", "once"],
    "pt": ["doze", "uma", "duas", "três", "quatro", "cinco", "seis", "sete", "oito",
           "nove", "dez", "onze"],
}


def tijd(uur: int, minuut: int, taal: str) -> str:
    """De tijd zoals je hem zegt, afgerond op vijf minuten, in deze taal."""
    if taal == "nl":
        from .routers.presentator import tijd_in_woorden
        return tijd_in_woorden(uur, minuut)
    m = int(round(minuut / 5.0) * 5)
    if m == 60:
        uur, m = uur + 1, 0
    u = _UREN[taal]
    h, v = u[uur % 12], u[(uur + 1) % 12]
    if taal == "en":
        return {
            0: f"{h} o'clock", 5: f"five past {h}", 10: f"ten past {h}",
            15: f"quarter past {h}", 20: f"twenty past {h}", 25: f"twenty-five past {h}",
            30: f"half past {h}", 35: f"twenty-five to {v}", 40: f"twenty to {v}",
            45: f"quarter to {v}", 50: f"ten to {v}", 55: f"five to {v}",
        }[m]
    if taal == "de":
        h = "ein" if h == "eins" and m == 0 else h
        return {
            0: f"{h} Uhr", 5: f"fünf nach {h}", 10: f"zehn nach {h}",
            15: f"Viertel nach {h}", 20: f"zwanzig nach {h}", 25: f"fünf vor halb {v}",
            30: f"halb {v}", 35: f"fünf nach halb {v}", 40: f"zwanzig vor {v}",
            45: f"Viertel vor {v}", 50: f"zehn vor {v}", 55: f"fünf vor {v}",
        }[m]
    if taal == "es":
        lid_h = "la" if h == "una" else "las"
        lid_v = "la" if v == "una" else "las"
        return {
            0: f"{lid_h} {h} en punto", 5: f"{lid_h} {h} y cinco", 10: f"{lid_h} {h} y diez",
            15: f"{lid_h} {h} y cuarto", 20: f"{lid_h} {h} y veinte",
            25: f"{lid_h} {h} y veinticinco", 30: f"{lid_h} {h} y media",
            35: f"{lid_v} {v} menos veinticinco", 40: f"{lid_v} {v} menos veinte",
            45: f"{lid_v} {v} menos cuarto", 50: f"{lid_v} {v} menos diez",
            55: f"{lid_v} {v} menos cinco",
        }[m]
    # pt
    para = f"para a {v}" if v == "uma" else f"para as {v}"
    return {
        0: f"{h} em ponto", 5: f"{h} e cinco", 10: f"{h} e dez", 15: f"{h} e quinze",
        20: f"{h} e vinte", 25: f"{h} e vinte e cinco", 30: f"{h} e meia",
        35: f"vinte e cinco {para}", 40: f"vinte {para}",
        45: f"quinze {para}", 50: f"dez {para}", 55: f"cinco {para}",
    }[m]


def het_is(tijd_tekst: str, taal: str) -> str:
    """"Het is kwart over negen." — de opening van een break rond het hele uur."""
    if taal == "en":
        return f"It's {tijd_tekst}."
    if taal == "de":
        return f"Es ist {tijd_tekst}."
    if taal == "es":
        return f"{'Es' if tijd_tekst.startswith('la una') else 'Son'} {tijd_tekst}."
    if taal == "pt":
        return f"{'É' if tijd_tekst.startswith('uma') else 'São'} {tijd_tekst}."
    return f"Het is {tijd_tekst}."


_DAGDELEN = {
    "nl": ("nacht", "ochtend", "middag", "avond"),
    "en": ("night", "morning", "afternoon", "evening"),
    "de": ("Nacht", "Morgen", "Nachmittag", "Abend"),
    "es": ("noche", "mañana", "tarde", "noche"),
    "pt": ("noite", "manhã", "tarde", "noite"),
}

_DAGEN = {
    "nl": ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
    "pt": ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira",
           "sábado", "domingo"],
}

_GRADEN = {"nl": "graden", "en": "degrees", "de": "Grad", "es": "grados", "pt": "graus"}


def dagdeel(uur: int, taal: str) -> str:
    d = _DAGDELEN.get(taal, _DAGDELEN["nl"])
    if uur < 6:
        return d[0]
    if uur < 12:
        return d[1]
    if uur < 18:
        return d[2]
    return d[3]


def dag(weekdag: int, taal: str) -> str:
    return _DAGEN.get(taal, _DAGEN["nl"])[weekdag]


def graden(temp: float, taal: str) -> str:
    return f"{round(temp)} {_GRADEN.get(taal, 'graden')}"


# ── Zinnen zonder model ──────────────────────────────────────────────────────

_VAN = {"nl": "van", "en": "by", "de": "von", "es": "de", "pt": "de"}


def wie(n: dict, taal: str = "nl") -> str:
    """"Titel van Artiest" in deze taal."""
    return f"{n['titel']} {_VAN.get(taal, 'van')} {n['artiest']}" if n.get("artiest") else n["titel"]


# Hoe de terugblik begint. Door de code gekozen: het model zelf koos negen van
# de tien keer "Je hoorde net".
TERUG_BEGIN = {
    "nl": ["Dat was", "Je hoorde net", "Net hoorde je", "Na", "Dat was dus",
           "Dat was zojuist", "Zojuist hoorde je", "Daar was", "Dat was even",
           "Je luisterde naar"],
    "en": ["That was", "You just heard", "That was just", "There you had",
           "We just heard", "Before that,"],
    "de": ["Das war", "Gerade lief", "Eben lief", "Das war gerade", "Gerade gehört:"],
    "es": ["Eso fue", "Acabas de escuchar", "Hace un momento sonó", "Sonó",
           "Acabamos de escuchar"],
    "pt": ["Essa foi", "Você acabou de ouvir", "Acabou de tocar", "Tocou agora",
           "Acabamos de ouvir"],
}


def vaste_zinnen(vorm: str, v: str, w: str | None, taal: str) -> list[str]:
    """De zinnen voor als het model twee keer iets verzint. Saai, maar ze
    kloppen altijd. `v` = het volgende nummer ("Titel van Artiest"), `w` = het
    vorige of None."""
    if w and vorm == "terug":
        return {
            "nl": [f"Dat was {w}.", f"Je luisterde naar {w}.", f"Zojuist {w}.", f"Daar was {w}."],
            "en": [f"That was {w}.", f"You just heard {w}.", f"There you had {w}."],
            "de": [f"Das war {w}.", f"Gerade lief {w}.", f"Eben gehört: {w}."],
            "es": [f"Eso fue {w}.", f"Acabas de escuchar {w}.", f"Sonó {w}."],
            "pt": [f"Essa foi {w}.", f"Você acabou de ouvir {w}.", f"Tocou {w}."],
        }[taal]
    if w and vorm != "vooruit":
        return {
            "nl": [f"Dat was {w}. Nu {v}.", f"Je hoorde {w}, en nu komt {v}.",
                   f"Na {w} is het tijd voor {v}.", f"Dat was {w}. We gaan door met {v}.",
                   f"Net hoorde je {w}. Hier is {v}."],
            "en": [f"That was {w}. Now {v}.", f"You just heard {w}, and here's {v}.",
                   f"After {w}, it's time for {v}.", f"That was {w}. Next up, {v}."],
            "de": [f"Das war {w}. Jetzt {v}.", f"Gerade lief {w}, und jetzt kommt {v}.",
                   f"Nach {w} geht es weiter mit {v}."],
            "es": [f"Eso fue {w}. Ahora, {v}.", f"Acabas de escuchar {w}, y ahora viene {v}.",
                   f"Después de {w}, seguimos con {v}."],
            "pt": [f"Essa foi {w}. Agora, {v}.", f"Você ouviu {w}, e agora vem {v}.",
                   f"Depois de {w}, seguimos com {v}."],
        }[taal]
    return {
        "nl": [f"Dan nu {v}.", f"Hier is {v}.", f"En nu {v}.", f"We gaan verder met {v}.",
               f"Daar is {v}.", f"Luister even naar {v}."],
        "en": [f"Here's {v}.", f"And now, {v}.", f"Up next, {v}.", f"Let's go on with {v}."],
        "de": [f"Hier ist {v}.", f"Und jetzt {v}.", f"Weiter geht's mit {v}."],
        "es": [f"Aquí está {v}.", f"Y ahora, {v}.", f"Seguimos con {v}."],
        "pt": [f"Aqui está {v}.", f"E agora, {v}.", f"Seguimos com {v}."],
    }[taal]


# ── Voor het model (alles behalve nl; de Nederlandse opdracht staat in
#    presentator.py zoals hij altijd was) ─────────────────────────────────────

# Hoe een dj in die taal praat. In de taal zelf: zo hoort het model meteen de
# toon, en het is een tweede duwtje richting de juiste taal.
SPREEKTAAL = {
    "en": "Talk like a relaxed English-speaking radio DJ: short, casual, everyday words, "
          "'you' and 'we'.",
    "de": "Sprich wie ein entspannter deutscher Radio-DJ: kurz, locker, Alltagssprache, "
          "'du' und 'wir'.",
    "es": "Habla como un locutor de radio relajado: frases cortas, lenguaje cotidiano, "
          "de tú.",
    "pt": "Fale como um locutor de rádio descontraído: frases curtas, linguagem do dia a "
          "dia, informal.",
}

HOEK_UITLEG_EN = {
    "volgend": "say something about the next song from the facts (album, year or genre)",
    "nieuw": "say that the next song was just added to the collection",
    "artiest": "say one thing about the artist of the next song, from 'overArtiest'",
    "tijd": "mention in passing what time it is ('tijd') or the part of the day ('dagdeel')",
    "weer": "mention in passing the temperature outside",
    "meeluisteraars": "greet who is listening along (first name)",
    "uur": "start by saying what time it is, exactly as in 'tijd', like on the radio "
           "around the hour",
    "programma": "say that the show in 'programma' starts now, with its name exactly as "
                 "written, and in a few words what it plays ('wat')",
}


# ── Is het per ongeluk Nederlands? ───────────────────────────────────────────

# Woorden die in een aankondiging vaak vallen en in geen van de andere vier
# talen een gewoon woord zijn. Twee ervan in een niet-Nederlandse tekst, en hij
# gaat niet door.
_NL_WOORDEN = {"het", "een", "naar", "hoorde", "nummer", "nummers", "volgende", "zojuist",
               "jij", "wij", "deze", "nog", "gaan", "komt", "luisterde", "dus", "lekker",
               "uur", "ochtend", "middag", "avond", "graden", "luister", "nieuw",
               "gewoon"}


def klinkt_nederlands(tekst: str, namen: list[str] = ()) -> bool:
    """Staan er Nederlandse woorden in, buiten de titels en namen om?"""
    t = (tekst or "").lower()
    for n in namen:
        if n:
            t = t.replace(n.lower(), " ")
    woorden = set(re.findall(r"[a-zà-ÿ]+", t))
    return len(woorden & _NL_WOORDEN) >= 2


# Woorden waarmee het model graag iets aandikt dat niet in de feiten staat, in
# alle vijf de talen tegelijk (een Duits lofwoord in een Duitse tekst wordt zo
# net zo goed gevangen als een Nederlands in een Nederlandse).
LOF = re.compile(
    r"\b(hit|hits|klassieker|klassiekers|tijdloze?|legendarische?|iconische?|"
    r"wereldhit|succes|succesvolle?|populaire?|bekendste|beroemde?|"
    r"sfeer|sfeervolle?|perfect|perfecte|intieme?|karakteristieke?|"
    r"meeslepende?|krachtige?|emotionele?|tijdloos|vibe|"
    # en
    r"classic|classics|legendary|iconic|timeless|success|successful|popular|famous|"
    r"best-known|atmospheric|intimate|powerful|emotional|smash|chart-topper|"
    # de
    r"klassiker|legendäre?[nrs]?|ikonische?[nrs]?|zeitlose?[nrs]?|erfolg|erfolgreiche?[nrs]?|"
    r"berühmte?[nrs]?|beliebte?[nrs]?|perfekte?[nrs]?|stimmungsvolle?[nrs]?|"
    # es
    r"clásicos?|legendari[oa]s?|icónic[oa]s?|éxito|éxitos|famos[oa]s?|populares|"
    r"atemporal|perfect[oa]|"
    # pt
    r"clássicos?|lendári[oa]s?|icônic[oa]s?|icónic[oa]s?|sucesso|sucessos|perfeit[oa])\b",
    re.I)
