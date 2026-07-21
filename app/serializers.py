"""ORM objects -> Plex-shaped `MediaContainer.Metadata` dicts. Field names and
nesting match exactly what the JLTamp app parses (see the contract spec):
transformTrack reads Media[0].Part[0].key, grandparentTitle, parentThumb, etc.
"""
from __future__ import annotations

from sqlalchemy import select

from .ids import track_key, album_key, artist_key, playlist_key
from .models import Artist, Album, Track, Playlist, UserTrackState


def track_states(db, user_id: int, tracks) -> dict:
    """Load one user's play state for a batch of tracks: {track_id: state}.
    Callers pass `state=states.get(t.id)` into track_dict so viewCount /
    lastViewedAt / viewOffset are that user's, not the previous listener's."""
    ids = [t.id for t in tracks]
    if not ids:
        return {}
    rows = db.execute(
        select(UserTrackState).where(
            UserTrackState.user_id == user_id, UserTrackState.track_id.in_(ids)
        )
    ).scalars()
    return {r.track_id: r for r in rows}


def art_ref(rk: str | None) -> str | None:
    """A `thumb`/`art` path the app hands back to us via /photo/:/transcode or
    directly. `None` when the entity has no art (so the app shows a placeholder)."""
    return f"/art/{rk}" if rk else None


def artist_dict(a: Artist) -> dict:
    rk = artist_key(a.id)
    has_art = bool(a.art_path or a.online_art_path)
    return {
        "ratingKey": rk,
        "key": f"/library/metadata/{rk}/children",
        "type": "artist",
        "title": a.name,
        "thumb": art_ref(rk) if has_art else None,
        "art": art_ref(rk) if has_art else None,
        "summary": a.summary or "",
        "addedAt": a.added_at or 0,
    }


def album_dict(al: Album) -> dict:
    rk = album_key(al.id)
    ar_rk = artist_key(al.artist_id)
    has_art = bool(al.art_path or al.online_art_path)
    return {
        "ratingKey": rk,
        "key": f"/library/metadata/{rk}/children",
        "type": "album",
        "title": al.title,
        "parentTitle": al.artist_name,
        "parentRatingKey": ar_rk,
        "parentKey": f"/library/metadata/{ar_rk}/children",
        "thumb": art_ref(rk) if has_art else None,
        "parentThumb": art_ref(ar_rk),
        "art": art_ref(rk) if has_art else None,
        "year": al.year,
        "leafCount": al.track_count,
        "duration": al.duration_ms,
        "addedAt": al.added_at or 0,
    }


def track_dict(t: Track, playlist_item_id: int | None = None, state=None) -> dict:
    """`state` is the caller's UserTrackState for this track, when it has one.
    Play count / last-played / resume are per user, so a caller that does not
    load them reports zeroes rather than another user's history."""
    rk = track_key(t.id)
    al_rk = album_key(t.album_id)
    ar_rk = artist_key(t.artist_id)
    part_key = f"/library/parts/{rk}/file{t.ext or ''}"
    media = [{
        "id": t.id,
        "duration": t.duration_ms,
        "bitrate": t.bitrate,
        "container": t.container,
        "audioCodec": t.codec,
        "audioChannels": t.channels,
        "Part": [{
            "id": t.id,
            "key": part_key,
            "duration": t.duration_ms,
            "container": t.container,
            "size": t.size,
            "file": t.path,
        }],
    }]
    d = {
        "ratingKey": rk,
        "key": f"/library/metadata/{rk}",
        "type": "track",
        "title": t.title,
        "grandparentTitle": t.artist_name,          # artist
        "grandparentRatingKey": ar_rk,
        "grandparentThumb": art_ref(ar_rk),
        "parentTitle": t.album_title,               # album
        "parentRatingKey": al_rk,
        "parentThumb": art_ref(al_rk),
        "originalTitle": t.orig_artist or None,
        "duration": t.duration_ms,
        "index": t.track_no,
        "parentIndex": t.disc_no,
        "year": t.year,
        "genre": t.genre or "",
        "thumb": art_ref(al_rk),                    # tracks show album cover
        "art": art_ref(al_rk),
        "addedAt": t.added_at or 0,
        "lastViewedAt": (state.last_played_at if state else 0) or 0,
        "viewCount": (state.play_count if state else 0) or 0,
        "viewOffset": (state.view_offset_ms if state else 0) or 0,
        "userRating": (state.rating if state else 0) or 0,
        "Media": media,
    }
    # Loudness-normalisation gain (dB), when analysed — the app levels volume
    # with it. Omitted when NULL so the client leaves the volume untouched.
    if t.gain_db is not None:
        d["gainDb"] = t.gain_db
    if playlist_item_id is not None:
        d["playlistItemID"] = playlist_item_id
    return d


def playlist_dict(pl: Playlist, leaf_count: int, duration_ms: int) -> dict:
    rk = playlist_key(pl.id)
    return {
        "ratingKey": rk,
        "key": f"/playlists/{rk}/items",
        "type": "playlist",
        "title": pl.title,
        "playlistType": "audio",
        "smart": False,
        "leafCount": leaf_count,
        "duration": duration_ms,
        "composite": f"/art/{rk}",
        "addedAt": pl.created_at or 0,
        "updatedAt": pl.updated_at or 0,
    }


def container(items: list[dict], *, total: int | None = None, extra: dict | None = None,
              node: str = "Metadata") -> dict:
    mc = {
        "size": len(items),
        node: items,
    }
    if total is not None:
        mc["totalSize"] = total
    if extra:
        mc.update(extra)
    return {"MediaContainer": mc}
