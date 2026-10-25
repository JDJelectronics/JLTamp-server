"""LLM-ondersteunde query-verrijking (optioneel, fail-safe).

Een korte, vrije prompt ("zaterdagochtend koffie, niks moeten") draagt weinig
woorden die bge-m3 kan matchen, waardoor vibe-zoekopdrachten worden gekaapt door
tracks met díe woorden in de titel. Deze module vraagt het lokale model
(Ollama) om beschrijvende STIJL/SFEER-woorden (genre/mood/era/energy), die aan de
embedding-query worden toegevoegd — de originele prompt blijft leidend voor
keyword-/artiest-matching.

Robuust: elke fout, timeout of uitgezette toggle -> lege string -> de engine valt
terug op het bestaande gedrag. Resultaten worden gecached per prompt.
"""
from __future__ import annotations

import functools
import json
import urllib.request

from . import config

# Het grote model (zie config.LLM_MODEL): het kleine qwen3:1.7b gaf met
# format=json op elke vraag een leeg antwoord (gemeten 22-10-2026), dus deze
# verrijking deed al die tijd niets. Het grote staat toch al warm voor de
# presentator en antwoordt dan in ~0,7 s.
#
# Nederlandse begrippen staan erbij omdat het model ze anders letterlijk nam:
# "foute party jaren 90" werd "alternative rock, rebellious". En era alleen als
# erom gevraagd wordt — anders zette het overal "2010s" neer en duwde dat de
# zoekopdracht naar één decennium.
_PROMPT = (
    "Je krijgt een muziek-zoekopdracht van een Nederlandse luisteraar. Geef ALLEEN een JSON-object "
    "met de sleutels genre, mood, era, energy. Vul per sleutel een paar korte Engelse EN Nederlandse "
    "beschrijvende woorden in die de STIJL en SFEER vangen (bv. \"acoustic akoestisch mellow rustig\").\n"
    "Let op Nederlandse begrippen: \"fout\"/\"foute\" = guilty pleasure meezingers en feesthits "
    "(cheesy, party, singalong); \"Nederlandstalig\" = zet \"Dutch Nederlandstalig\" in genre; "
    "\"smartlap\" = levenslied; \"gezellig\" = feel-good.\n"
    "era ALLEEN invullen als de opdracht een tijd noemt of sterk suggereert (\"jaren 80\", \"oldies\"); "
    "anders \"\". GEEN songtitels, GEEN artiestennamen. Onbekend veld = \"\".\n\n"
    "Opdracht: {q}\nJSON:"
)


@functools.lru_cache(maxsize=1024)
def enrich_query(prompt: str) -> str:
    """Extra sfeer-/stijlwoorden voor de embedding-query, of "" bij twijfel/fout."""
    prompt = (prompt or "").strip()
    if not config.USE_LLM_INTENT or not prompt:
        return ""
    body = json.dumps({
        "model": config.LLM_MODEL,
        "prompt": _PROMPT.format(q=prompt),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 120},
    }).encode()
    try:
        req = urllib.request.Request(
            config.OLLAMA_URL.rstrip("/") + "/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as r:
            resp = json.load(r).get("response", "")
        data = json.loads(resp)
        terms = " ".join(
            str(data.get(k, "") or "") for k in ("genre", "mood", "era", "energy"))
        cleaned = " ".join(terms.split())          # normaliseer witruimte
        return cleaned[:200]                        # nooit een prompt-bom
    except Exception as e:                           # noqa: BLE001
        print(f"⚠️  LLM-verrijking overgeslagen: {type(e).__name__}: {e}")
        return ""
