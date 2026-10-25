"""Nummers die nooit meer op JLTamp Radio komen.

Een grote bibliotheek heeft ook rare nummers: een skit die geen skit
heet, een sample-cd, iets wat daar ooit per ongeluk in is beland. De beheerder
zet zo'n nummer met één druk op deze lijst (POST /radio/blokkeer), en dan komt
het niet meer langs — ook niet als ander bestand van hetzelfde nummer
(de flac naast de mp3), want er wordt ook op artiest|titel geblokkeerd.

Een eigen bestandje in de radiomap, geen tabel: het is een korte lijst die
alleen de radio leest, en zo hoeft er niets aan de database te veranderen.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("radio")

_slot = threading.Lock()


def _bestand(map_: Path) -> Path:
    return Path(map_) / "geblokkeerd.json"


def lees(map_: Path) -> list[dict]:
    """De lijst: [{id, sleutel, titel, artiest, ts}], nieuwste laatst."""
    try:
        data = json.loads(_bestand(map_).read_text())
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as e:
        log.warning("radio: geblokkeerd.json onleesbaar (%s), lege lijst", e)
        return []


def _schrijf(map_: Path, lijst: list[dict]) -> None:
    pad = _bestand(map_)
    pad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pad.with_suffix(".tmp")
    tmp.write_text(json.dumps(lijst, ensure_ascii=False, indent=1))
    tmp.replace(pad)


def voeg_toe(map_: Path, id_: int, sleutel: str, titel: str, artiest: str) -> list[dict]:
    with _slot:
        lijst = [x for x in lees(map_) if x.get("id") != id_]
        lijst.append({"id": id_, "sleutel": sleutel, "titel": titel, "artiest": artiest,
                      "ts": int(time.time())})
        _schrijf(map_, lijst)
    log.info("radio: geblokkeerd: %s — %s (%s)", titel, artiest, id_)
    return lijst


def voeg_album_toe(map_: Path, album_id: int, titel: str) -> list[dict]:
    """Een heel album eruit. Voor de reuzen: duizenden losse radio-opnames
    onder één albumnaam, bijvoorbeeld."""
    with _slot:
        lijst = [x for x in lees(map_) if x.get("album_id") != album_id]
        lijst.append({"album_id": album_id, "titel": titel, "ts": int(time.time())})
        _schrijf(map_, lijst)
    log.info("radio: album geblokkeerd: %s (%s)", titel, album_id)
    return lijst


def haal_weg(map_: Path, id_: int | None = None, album_id: int | None = None) -> list[dict]:
    with _slot:
        lijst = [x for x in lees(map_)
                 if not ((id_ and x.get("id") == id_) or (album_id and x.get("album_id") == album_id))]
        _schrijf(map_, lijst)
    return lijst


def filter_van(lijst: list[dict]) -> tuple[set[int], set[str], set[int]]:
    """(ids, sleutels, album_ids) om de pool mee te filteren."""
    return ({int(x["id"]) for x in lijst if x.get("id")},
            {x["sleutel"] for x in lijst if x.get("sleutel")},
            {int(x["album_id"]) for x in lijst if x.get("album_id")})
