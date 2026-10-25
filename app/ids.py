"""ratingKey <-> (type, numeric id) mapping. Prefixed so one metadata route
resolves any entity and playlist uris can carry track keys. Kept trivial."""
from __future__ import annotations


def track_key(i: int) -> str:    return f"t{i}"
def album_key(i: int) -> str:    return f"al{i}"
def artist_key(i: int) -> str:   return f"ar{i}"
def playlist_key(i: int) -> str: return f"pl{i}"


def parse_key(rk: str) -> tuple[str, int] | None:
    """'t123' -> ('track',123); 'al45' -> ('album',45); 'ar3' -> ('artist',3);
    'pl7' -> ('playlist',7). Returns None if unparseable."""
    if not rk:
        return None
    rk = str(rk).strip()
    # Two-char prefixes first (al/ar/pl all differ from the 1-char 't').
    for pfx, kind in (("al", "album"), ("ar", "artist"), ("pl", "playlist")):
        if rk.startswith(pfx) and rk[len(pfx):].isdigit():
            return kind, int(rk[len(pfx):])
    if rk.startswith("t") and rk[1:].isdigit():
        return "track", int(rk[1:])
    # Bare numeric → treat as a track id (tolerant).
    if rk.isdigit():
        return "track", int(rk)
    return None
