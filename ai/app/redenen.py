"""Waarom dit nummer? — de reden per nummer in een lijst die de AI maakte.

De engine weet bij het kiezen precies waarom een nummer erin komt: het ligt vlak
bij een favoriet, het is een van je meest gedraaide, het past bij de prompt en
heeft het goede tempo. Die kennis was weg zodra de lijst er stond. Hier wordt
hij bewaard, per lijst, zodat de app onder elk nummer één regel kan tonen.

Geen taalmodel: de redenen komen uit gegevens die er echt zijn (een favoriet,
een telling, een gemeten tempo). Een LLM zou er mooiere zinnen van maken, maar
ook dingen die niet waar zijn — en een reden die niet klopt is erger dan geen.

Opslag: één JSON-bestand in de datamap, begrensd tot de recentste lijsten en
atomisch weggeschreven. Er staat alleen artiest en titel in, van de gebruiker
zelf; wie de redenen mag lezen, beslist JLTamp (zie /ai/redenen in main.py).
"""
from __future__ import annotations

import json
import threading
import time

from . import config

_BESTAND = config.DATA_DIR / "redenen.json"
# Vijftig regels per lijst: 400 lijsten is een paar MB, en ruim genoeg om elke
# lijst van elke gebruiker van de afgelopen weken te dekken.
_MAX_LIJSTEN = 400

_lock = threading.Lock()
_data: dict | None = None


def _laad() -> dict:
    global _data
    if _data is None:
        try:
            _data = json.loads(_BESTAND.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _data = {}
    return _data


def bewaar(playlist_key: str, redenen: dict[str, str]) -> None:
    """Onthoud {ratingKey: reden} voor deze lijst. Mag nooit een lijst breken."""
    if not playlist_key or not redenen:
        return
    try:
        with _lock:
            data = _laad()
            data[str(playlist_key)] = {"at": int(time.time()), "redenen": redenen}
            if len(data) > _MAX_LIJSTEN:
                oudste = sorted(data, key=lambda k: data[k].get("at", 0))
                for k in oudste[: len(data) - _MAX_LIJSTEN]:
                    data.pop(k, None)
            _BESTAND.parent.mkdir(parents=True, exist_ok=True)
            tijdelijk = _BESTAND.with_suffix(".tmp")
            tijdelijk.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tijdelijk.replace(_BESTAND)
    except OSError as e:
        print(f"⚠️  redenen niet bewaard voor {playlist_key}: {e}")


def lees(playlist_key: str) -> dict[str, str]:
    with _lock:
        entry = _laad().get(str(playlist_key))
    return dict(entry.get("redenen", {})) if entry else {}
