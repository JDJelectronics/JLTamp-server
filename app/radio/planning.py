"""Wanneer de presentator iets zegt.

Elke twee à drie nummers een break, en rond het hele uur één met de tijd erin
("het is vijf uur"), zoals op de radio. Puur rekenwerk zonder klok of database,
zodat het te testen is: de zender geeft mee hoeveel nummers er sinds de vorige
break waren en wanneer het nummer ongeveer begint.

`sinds` telt de nummers sinds de laatste break, met het nummer waar die break
voor stond als eerste. Twee nummers na een break mag het weer (de helft van de
keren), na drie moet het.
"""
from __future__ import annotations

import time

MIN_TUSSEN = 2
MAX_TUSSEN = 3

# Zo ver rond het hele uur telt het als "rond het uur": een break om 16:57 of
# om 17:03 zegt allebei dat het vijf uur is (tijd_in_woorden rondt af).
_RAND_MIN = 4


def uur_sleutel(aanvang: float) -> str | None:
    """Welk heel uur er rond `aanvang` valt, als sleutel ("2026-295-17"), of None
    als het niet rond een heel uur is."""
    lt = time.localtime(aanvang)
    if lt.tm_min >= 60 - _RAND_MIN:
        lt = time.localtime(aanvang + (60 - lt.tm_min) * 60)
    elif lt.tm_min > _RAND_MIN:
        return None
    return f"{lt.tm_year}-{lt.tm_yday}-{lt.tm_hour}"


def plan_break(sinds: int, aanvang: float, laatste_uur: str | None,
               toeval: float, max_tussen: int = MAX_TUSSEN) -> tuple[str | None, str | None]:
    """(soort, uursleutel) voor het nummer dat om `aanvang` begint.

    soort is None (geen break), "uur" (break met de tijd) of "gewoon".
    `laatste_uur` is de sleutel van de laatste uur-break, zodat er per uur maar
    één is. `toeval` is een getal in [0, 1) — van buiten, voor de tests.
    """
    sleutel = uur_sleutel(aanvang)
    if sleutel and sleutel != laatste_uur and sinds >= 1:
        return "uur", sleutel
    min_tussen = max(1, max_tussen - 1)
    if sinds >= max_tussen:
        return "gewoon", None
    if sinds >= min_tussen and toeval < 0.5:
        return "gewoon", None
    return None, None
