"""Genres zoals een luisteraar ze kiest, niet zoals de tags ze noemen.

De bibliotheek kent 197 verschillende genre-tags, en een groot deel is ruis of
dubbel: "hip hop" én "rap" én "rap/hip hop"; "nederlandstalig", "nederlandse
muziek", "nederlandstalige muziek" en "nederlands"; "films", "film scores" en
"soundtrack". En tags die niets zeggen: "hi-res" (een bibliotheek), "music",
"other". Een radio op genre met die lijst als keuzemenu is een zoekplaatje.

Hier staan de groepen waar je echt uit kiest. Een nummer kan in meer dan één
groep vallen ("indie pop" is pop én alternatief) — dat is geen fout, zo luistert
iemand ook.
"""
from __future__ import annotations

import re

# id: (naam in de app, tags die erbij horen)
GROEPEN: dict[str, tuple[str, tuple[str, ...]]] = {
    "pop": ("Pop", ("pop", "international pop", "dance pop", "k-pop", "europop")),
    "nederlandstalig": ("Nederlandstalig", ("nederlandstalig", "nederlandse muziek",
                                            "nederlandstalige muziek", "nederlands",
                                            "levenslied", "nederpop")),
    "rock": ("Rock", ("rock", "classic rock", "soft rock", "punk", "punk rock")),
    "alternatief": ("Alternatief & indie", ("alternative", "indie", "indie rock",
                                            "indie pop", "grunge")),
    "hiphop": ("Hip-hop & rap", ("hip hop", "hip-hop", "rap", "dirty south", "trap",
                                 "gangsta rap")),
    "dance": ("Dance & electronic", ("dance", "electro", "electronic", "electronica",
                                     "techno", "house", "edm", "trance", "dubstep",
                                     "drum and bass")),
    "hardstyle": ("Hardstyle & hardcore", ("hardstyle", "hardcore", "gabber")),
    "rnb": ("R&B & soul", ("r&b", "rnb", "soul", "funk", "neo soul")),
    "metal": ("Metal", ("metal", "heavy metal")),
    "blues": ("Blues", ("blues",)),
    "folk": ("Folk & singer-songwriter", ("folk", "singer & songwriter",
                                          "singer-songwriter", "acoustic")),
    "latin": ("Latin", ("latin", "latin music", "reggaeton", "salsa", "bachata")),
    "reggae": ("Reggae", ("reggae", "dancehall", "ska")),
    "country": ("Country", ("country",)),
    "jazz": ("Jazz", ("jazz",)),
    "klassiek": ("Klassiek", ("classical", "klassiek", "opera")),
    "film": ("Film & games", ("soundtrack", "films", "film scores", "games", "score")),
    "schlager": ("Schlager & carnaval", ("schlager", "party schlager", "carnaval")),
    "oldies": ("Oldies", ("oldies",)),
    "wereld": ("Wereldmuziek", ("african music", "asian music", "national folk", "world")),
    "kids": ("Kinderliedjes", ("kids", "children", "kindermuziek")),
}

# Tags die over de bestanden gaan, niet over de muziek.
_NEGEER = {"music", "other", "hi-res", "unknown", "overig", "misc", "various", ""}


def groepen_van(genre: str | None) -> set[str]:
    """In welke groepen een nummer met deze genre-tag valt (misschien geen)."""
    uit: set[str] = set()
    for deel in re.split(r"[,;/|]", (genre or "").lower()):
        tag = deel.strip()
        if tag in _NEGEER:
            continue
        for gid, (_naam, tags) in GROEPEN.items():
            # "hard rock" hoort bij rock, "death metal" bij metal.
            if tag in tags or any(tag.endswith(" " + t) for t in tags):
                uit.add(gid)
    return uit
