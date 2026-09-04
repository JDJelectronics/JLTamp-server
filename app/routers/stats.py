"""Listening statistics — the Tautulli side of the server.

Tautulli watches a Plex server from the outside and rebuilds history by polling.
Here the server IS the source: every timeline heartbeat already writes a PlayEvent
(see history.py), so the stats are exact rather than sampled — no polling, no
missed plays between polls.

Scope rules, everywhere in this router: an admin may look at the whole server (and
at one user via ?user_id=), a normal user only ever sees their own listening.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func

from ..db import SessionLocal
from ..deps import require_user
from ..ids import track_key, artist_key, album_key
from ..models import Album, Artist, PlayEvent, Track, User
from ..serializers import art_ref, user_thumb_ref

router = APIRouter()

# What counts as "a play" in the stats: the same threshold history.py uses to
# bump a play count, so the numbers here agree with the ones on the track.
from .history import PLAY_THRESHOLD  # noqa: E402


def _scope_user(db, user: User, user_id: int | None) -> int | None:
    """Which user's events to read. None = the whole server (admin only)."""
    if user.is_admin:
        return user_id  # None → everyone
    if user_id is not None and user_id != user.id:
        raise HTTPException(403, "Not your listening history")
    return user.id


def _since(days: int) -> int:
    return int(time.time()) - max(1, days) * 86400


def _events(scoped_user: int | None, days: int):
    stmt = select(PlayEvent).where(PlayEvent.started_at >= _since(days))
    if scoped_user is not None:
        stmt = stmt.where(PlayEvent.user_id == scoped_user)
    return stmt


def _counted(stmt):
    """Only real listens — a 4-second skip is not a play."""
    return stmt.where(PlayEvent.completed.is_(True))


@router.get("/stats/history")
def history_log(user: User = Depends(require_user), user_id: int | None = None,
                days: int = 30, limit: int = 100, start: int = 0, q: str = ""):
    """The raw log: what was played, by whom, when, for how long.

    `q` searches the title, artist and album as they were recorded — which is
    the point of the denormalised columns: "when did I play that song" has to
    keep working after the file moved or the library was rescanned.
    """
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)

        def matching(stmt):
            term = (q or "").strip()
            if not term:
                return stmt
            like = f"%{term}%"
            return stmt.where(
                PlayEvent.track_title.ilike(like)
                | PlayEvent.artist_name.ilike(like)
                | PlayEvent.album_title.ilike(like)
            )

        stmt = matching(_events(scoped, days)).order_by(PlayEvent.started_at.desc()) \
            .offset(max(0, start)).limit(min(500, limit))
        rows = list(db.execute(stmt).scalars())
        total = db.execute(
            matching(
                select(func.count(PlayEvent.id)).where(
                    PlayEvent.started_at >= _since(days),
                    *([PlayEvent.user_id == scoped] if scoped is not None else []),
                )
            )
        ).scalar() or 0

        people = {u.id: u for u in db.execute(select(User)).scalars()}
        names = {i: (u.display_name or u.email) for i, u in people.items()}
        thumbs = {i: user_thumb_ref(u) for i, u in people.items()}
        items = [{
            "id": ev.id,
            "ratingKey": track_key(ev.track_id),
            "title": ev.track_title,
            "artist": ev.artist_name,
            "album": ev.album_title,
            "user": names.get(ev.user_id, "?"),
            "userThumb": thumbs.get(ev.user_id),
            "userId": ev.user_id,
            "device": ev.device,
            "startedAt": ev.started_at,
            "listenedMs": ev.listened_ms,
            "durationMs": ev.duration_ms,
            "completed": bool(ev.completed),
            "percent": round(100 * ev.listened_ms / ev.duration_ms) if ev.duration_ms else 0,
        } for ev in rows]
        return {"MediaContainer": {"size": len(items), "totalSize": total, "History": items}}
    finally:
        db.close()


@router.get("/stats/summary")
def summary(user: User = Depends(require_user), user_id: int | None = None, days: int = 30):
    """Headline numbers for the dashboard."""
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        base = _counted(_events(scoped, days)).subquery()

        plays = db.execute(select(func.count()).select_from(base)).scalar() or 0
        listened = db.execute(select(func.sum(base.c.listened_ms))).scalar() or 0
        tracks = db.execute(select(func.count(func.distinct(base.c.track_id)))).scalar() or 0
        artists = db.execute(select(func.count(func.distinct(base.c.artist_name)))).scalar() or 0
        users = db.execute(select(func.count(func.distinct(base.c.user_id)))).scalar() or 0

        return {"days": days, "plays": plays, "listenedMs": int(listened),
                "listenedHours": round(listened / 3_600_000, 1),
                "uniqueTracks": tracks, "uniqueArtists": artists, "users": users}
    finally:
        db.close()


@router.get("/stats/top")
def top(user: User = Depends(require_user), metric: str = "artists",
        user_id: int | None = None, days: int = 30, limit: int = 10):
    """Top artists / albums / tracks / users / devices, by number of plays."""
    columns = {
        "artists": PlayEvent.artist_name,
        "albums": PlayEvent.album_title,
        "tracks": PlayEvent.track_title,
        "devices": PlayEvent.device,
        "users": PlayEvent.user_id,
    }
    col = columns.get(metric)
    if col is None:
        raise HTTPException(400, f"Unknown metric: {metric}")

    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        if metric == "users" and not user.is_admin:
            raise HTTPException(403, "Admin only")

        stmt = (
            select(col, func.count(PlayEvent.id).label("plays"),
                   func.sum(PlayEvent.listened_ms).label("ms"))
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by(col).order_by(func.count(PlayEvent.id).desc()).limit(min(50, limit))
        )
        if scoped is not None:
            stmt = stmt.where(PlayEvent.user_id == scoped)

        rows = db.execute(stmt).all()
        names = {}
        if metric == "users":
            names = {u.id: (u.display_name or u.email) for u in db.execute(select(User)).scalars()}

        items = [{
            "title": names.get(value, str(value)) if metric == "users" else (value or "Unknown"),
            "plays": plays,
            "listenedMs": int(ms or 0),
        } for value, plays, ms in rows if value not in (None, "")]
        return {"metric": metric, "days": days, "size": len(items), "Top": items}
    finally:
        db.close()


@router.get("/stats/users")
def users_overview(user: User = Depends(require_user), days: int = 30):
    """Admin-only per-user leaderboard: who listened, how much, when last. Powers
    the dashboard's user picker and the 'all users' overview. A normal user gets
    only their own row (so the endpoint is safe to call from any client)."""
    db = SessionLocal()
    try:
        counts = (
            select(
                PlayEvent.user_id,
                func.count(PlayEvent.id).label("plays"),
                func.sum(PlayEvent.listened_ms).label("ms"),
                func.max(PlayEvent.started_at).label("last"),
            )
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by(PlayEvent.user_id)
        )
        if not user.is_admin:
            counts = counts.where(PlayEvent.user_id == user.id)
        stats = {row.user_id: row for row in db.execute(counts).all()}

        # Admins see every user (even those with zero plays); a normal user sees
        # only themselves.
        if user.is_admin:
            people = db.execute(select(User).where(User.is_active.is_(True))).scalars()
        else:
            people = [user]

        items = []
        for u in people:
            row = stats.get(u.id)
            items.append({
                "id": u.id,
                "name": u.display_name or u.email,
                # A face is quicker to pick out of twenty than a name is, and on
                # this server twenty is the real number.
                "thumb": user_thumb_ref(u),
                "email": u.email if user.is_admin else None,
                "isAdmin": u.is_admin,
                "plays": int(row.plays) if row else 0,
                "listenedMs": int(row.ms or 0) if row else 0,
                "lastPlayedAt": int(row.last) if row else 0,
            })
        items.sort(key=lambda x: x["plays"], reverse=True)
        return {"days": days, "size": len(items), "Users": items}
    finally:
        db.close()


@router.get("/stats/activity")
def activity(user: User = Depends(require_user), user_id: int | None = None, days: int = 30):
    """Plays and listening time per day — the dashboard's chart. Days with no
    listening are returned as zeroes so the chart has no holes in it."""
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        stmt = (
            select(
                func.strftime("%Y-%m-%d", PlayEvent.started_at, "unixepoch").label("day"),
                func.count(PlayEvent.id),
                func.sum(PlayEvent.listened_ms),
            )
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by("day")
        )
        if scoped is not None:
            stmt = stmt.where(PlayEvent.user_id == scoped)
        found = {day: (plays, int(ms or 0)) for day, plays, ms in db.execute(stmt).all()}

        start = _since(days)
        series = []
        for i in range(days + 1):
            day = time.strftime("%Y-%m-%d", time.gmtime(start + i * 86400))
            plays, ms = found.get(day, (0, 0))
            series.append({"day": day, "plays": plays, "listenedMs": ms})
        return {"days": days, "size": len(series), "Activity": series}
    finally:
        db.close()


# ── year in review ───────────────────────────────────────────────────────────
# The same PlayEvent rows the dashboard uses, read as a story instead of a
# table: what you played most, when you listened, and the couple of numbers
# that are fun precisely because nobody was collecting them on purpose.
#
# Bounded to one calendar year and computed on the fly — a year of one
# listener's events is a few thousand rows, which SQLite groups in
# milliseconds. No cache to invalidate, no table to migrate.

def _year_bounds(year: int) -> tuple[int, int]:
    from datetime import datetime, timezone
    start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return start, end


# Seasons, northern hemisphere and meteorological (whole months) rather than
# astronomical — "the summer of 2025" means June, July and August to a person,
# not "from the solstice". Winter is the odd one: it starts in the PREVIOUS
# year, so "winter 2025" is Dec 2024 → Feb 2025, which is the winter people
# would call by that name.
_SEASON_MONTHS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
}


def _period_bounds(period: str, year: int) -> tuple[int, int]:
    from datetime import datetime, timezone
    import calendar

    if period == "winter":
        start = int(datetime(year - 1, 12, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime(year, 2, calendar.monthrange(year, 2)[1], 23, 59, 59,
                           tzinfo=timezone.utc).timestamp())
        return start, end

    months = _SEASON_MONTHS.get(period)
    if not months:
        return _year_bounds(year)

    first, last = months
    start = int(datetime(year, first, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(year, last, calendar.monthrange(year, last)[1], 23, 59, 59,
                       tzinfo=timezone.utc).timestamp())
    return start, end


PERIODS = ("year", "spring", "summer", "autumn", "winter")


@router.get("/stats/wrapped")
def wrapped(user: User = Depends(require_user), year: int | None = None,
            period: str = "year", user_id: int | None = None):
    """One listener's year — or one season of it: totals, top artists/tracks/
    albums, the day and the hour they listened most, and where they listened."""
    from datetime import datetime, timezone

    if year is None:
        year = datetime.now(timezone.utc).year
    if year < 1970 or year > 2200:
        raise HTTPException(400, "Not a year")
    if period not in PERIODS:
        raise HTTPException(400, f"Unknown period: {period}")
    start, end = _period_bounds(period, year)

    db = SessionLocal()
    try:
        # "Your year" is personal, so an omitted user_id means ME — not the
        # whole server. The dashboard's _scope_user reads a missing user_id as
        # "everyone" for an admin, which is right for a dashboard and wrong
        # here: it silently mixed the household's listening into the owner's
        # own year. An admin can still pass ?user_id= to look at someone else,
        # and a normal user asking for anyone but themselves still gets a 403.
        scoped = _scope_user(db, user, user_id if user_id is not None else user.id)

        def base():
            s = select(PlayEvent).where(PlayEvent.started_at >= start,
                                        PlayEvent.started_at <= end,
                                        PlayEvent.completed.is_(True))
            if scoped is not None:
                s = s.where(PlayEvent.user_id == scoped)
            return s.subquery()

        b = base()

        totals = db.execute(select(
            func.count(b.c.id), func.sum(b.c.listened_ms),
            func.count(func.distinct(b.c.track_id)),
            func.count(func.distinct(b.c.artist_name)),
            func.count(func.distinct(b.c.album_title)),
            func.min(b.c.started_at),
        )).one()
        plays, ms, tracks, artists, albums, first_at = totals

        def top(col, limit=5):
            rows = db.execute(
                select(col, func.count(b.c.id), func.sum(b.c.listened_ms))
                .group_by(col).order_by(func.count(b.c.id).desc()).limit(limit)
            ).all()
            return [{"title": v or "Unknown", "plays": int(n or 0), "listenedMs": int(m or 0)}
                    for v, n, m in rows if v not in (None, "")]

        def named_art(model, name_col, names: list[str], key_fn) -> dict:
            """Cover art for a handful of names, in ONE query.

            The history table stores artist and album as text (denormalised on
            purpose, so history survives a file moving), which is fine to read
            and useless to illustrate. Looking each name back up gives the story
            cards something to show. Bounded to the five names already on the
            card, so this is five rows, not a join over the library.
            """
            if not names:
                return {}
            rows = db.execute(select(model).where(name_col.in_(names))).scalars().all()
            out = {}
            for r in rows:
                if getattr(r, "art_path", None) or getattr(r, "online_art_path", None):
                    out.setdefault(getattr(r, name_col.key), art_ref(key_fn(r.id)))
            return out

        def with_art(rows: list[dict], art: dict) -> list[dict]:
            for r in rows:
                r["thumb"] = art.get(r["title"])
            return rows

        def top_track_rows(limit=5):
            """Top tracks grouped by track_id, not by title.

            Titles collide — every compilation has a "Intro", and two different
            recordings of the same song would be merged into one bogus row. The
            id is also what makes "turn this into a playlist" possible at all.
            Tracks that have since left the library are dropped: the denormalised
            title still reads fine in history, but there is nothing to play.
            """
            rows = db.execute(
                select(b.c.track_id, func.count(b.c.id), func.sum(b.c.listened_ms))
                .group_by(b.c.track_id).order_by(func.count(b.c.id).desc()).limit(limit)
            ).all()
            out = []
            for tid, n, m in rows:
                t = db.get(Track, tid) if tid else None
                if not t:
                    continue
                out.append({
                    "title": t.title or "Unknown",
                    "artist": t.orig_artist or t.artist_name or "",
                    "ratingKey": track_key(t.id),
                    "thumb": art_ref(track_key(t.id)),
                    "plays": int(n or 0),
                    "listenedMs": int(m or 0),
                })
            return out

        # Busiest day and hour. strftime on a unix column keeps this in SQLite
        # rather than pulling a year of rows into Python to bucket them.
        day_col = func.strftime("%Y-%m-%d", b.c.started_at, "unixepoch", "localtime")
        day_rows = db.execute(
            select(day_col, func.count(b.c.id), func.sum(b.c.listened_ms))
            .group_by(day_col).order_by(func.sum(b.c.listened_ms).desc()).limit(1)
        ).all()
        hour_col = func.strftime("%H", b.c.started_at, "unixepoch", "localtime")
        hour_rows = db.execute(
            select(hour_col, func.count(b.c.id))
            .group_by(hour_col).order_by(func.count(b.c.id).desc()).limit(1)
        ).all()
        # Per-month minutes, for a twelve-bar chart. Months with nothing in them
        # are filled in as zero so the chart keeps its shape.
        month_col = func.strftime("%m", b.c.started_at, "unixepoch", "localtime")
        months = {m: 0 for m in range(1, 13)}
        for m, tot in db.execute(
            select(month_col, func.sum(b.c.listened_ms)).group_by(month_col)
        ).all():
            months[int(m)] = int(tot or 0)

        # The track played most, and its share — "you played this 47 times, that
        # is 1 in every 20 songs you heard" reads better than a bare count.
        top_tracks = top_track_rows(10)

        # Artists and albums get their cover art resolved by name so the story
        # cards have something to look at.
        top_artists = top(b.c.artist_name, 5)
        top_albums = top(b.c.album_title, 5)
        with_art(top_artists, named_art(Artist, Artist.name,
                                        [a["title"] for a in top_artists], artist_key))
        with_art(top_albums, named_art(Album, Album.title,
                                       [a["title"] for a in top_albums], album_key))

        top_share = round(100 * top_tracks[0]["plays"] / plays, 1) if (top_tracks and plays) else 0.0

        return {
            "year": year,
            "period": period,
            "plays": int(plays or 0),
            "listenedMs": int(ms or 0),
            "minutes": int((ms or 0) / 60000),
            "tracks": int(tracks or 0),
            "artists": int(artists or 0),
            "albums": int(albums or 0),
            "firstPlayAt": int(first_at or 0),
            "topArtists": top_artists,
            "topTracks": top_tracks,
            "topAlbums": top_albums,
            "topTrackShare": top_share,
            "devices": top(b.c.device, 5),
            "busiestDay": ({"date": day_rows[0][0], "plays": int(day_rows[0][1] or 0),
                            "listenedMs": int(day_rows[0][2] or 0)} if day_rows else None),
            "favouriteHour": (int(hour_rows[0][0]) if hour_rows else None),
            "monthlyMs": [months[m] for m in range(1, 13)],
            "from": start,
            "to": end,
        }
    finally:
        db.close()


@router.get("/stats/onthisday")
def on_this_day(user: User = Depends(require_user), user_id: int | None = None,
                limit: int = 10):
    """What you were playing on this calendar day in earlier years.

    Wrapped is a party once a year; this is the small pleasure that works the
    other 364 days, out of exactly the same rows. Returns nothing at all until
    there IS an earlier year — a feature that greets a new listener with an
    empty box is worse than one that waits.

    The day is the LISTENER's, not UTC's: strftime with 'localtime', the same
    way the busiest-day figure is worked out.
    """
    from datetime import datetime, timezone

    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id if user_id is not None else user.id)
        today = datetime.now().strftime("%m-%d")
        this_year = datetime.now(timezone.utc).year

        day = func.strftime("%m-%d", PlayEvent.started_at, "unixepoch", "localtime")
        year = func.strftime("%Y", PlayEvent.started_at, "unixepoch", "localtime")

        stmt = (select(year, PlayEvent.track_id, PlayEvent.track_title,
                       PlayEvent.artist_name, func.count(PlayEvent.id))
                .where(day == today, PlayEvent.completed.is_(True))
                .group_by(year, PlayEvent.track_id)
                .order_by(func.count(PlayEvent.id).desc()))
        if scoped is not None:
            stmt = stmt.where(PlayEvent.user_id == scoped)

        years: dict[int, list] = {}
        for y, tid, title, artist, plays in db.execute(stmt).all():
            y = int(y)
            if y >= this_year:
                continue           # today itself is not a memory yet
            bucket = years.setdefault(y, [])
            if len(bucket) >= limit:
                continue
            t = db.get(Track, tid) if tid else None
            bucket.append({
                "title": t.title if t else title,
                "artist": (t.orig_artist or t.artist_name) if t else artist,
                "ratingKey": track_key(t.id) if t else None,
                "thumb": art_ref(track_key(t.id)) if t else None,
                "plays": int(plays or 0),
                # A track that has since left the library still belongs in the
                # memory, it just cannot be played from it.
                "playable": t is not None,
            })

        return {"day": today,
                "years": [{"year": y, "tracks": years[y]} for y in sorted(years, reverse=True)]}
    finally:
        db.close()


@router.post("/stats/wrapped/playlist")
def wrapped_playlist(user: User = Depends(require_user), year: int | None = None,
                     period: str = "year", limit: int = 30, title: str = ""):
    """Turn a wrapped period into a real playlist of the tracks you played most.

    Reading your year is nice; playing it back is the point. Ordered by plays,
    so the playlist opens on the track you wore out.

    Always the CALLER's own listening — an admin looking at someone else's year
    still gets their own top tracks here, because writing another person's
    taste into your playlists is not something a stats view should do.
    """
    from datetime import datetime, timezone
    from ..ids import playlist_key
    from ..models import Playlist, PlaylistItem
    from ..deps import accessible_library_ids

    if year is None:
        year = datetime.now(timezone.utc).year
    if period not in PERIODS:
        raise HTTPException(400, f"Unknown period: {period}")
    start, end = _period_bounds(period, year)
    limit = max(1, min(100, limit))

    db = SessionLocal()
    try:
        rows = db.execute(
            select(PlayEvent.track_id, func.count(PlayEvent.id).label("plays"))
            .where(PlayEvent.user_id == user.id,
                   PlayEvent.started_at >= start, PlayEvent.started_at <= end,
                   PlayEvent.completed.is_(True))
            .group_by(PlayEvent.track_id)
            .order_by(func.count(PlayEvent.id).desc())
            .limit(limit)
        ).all()

        allowed = accessible_library_ids(db, user)
        track_ids: list[int] = []
        for tid, _plays in rows:
            t = db.get(Track, tid) if tid else None
            if not t:
                continue    # played once, since removed from the library
            if allowed is not None and t.library_id not in allowed:
                continue    # access was revoked since it was played
            track_ids.append(t.id)

        if not track_ids:
            raise HTTPException(404, "Nothing played in that period")

        now = int(time.time())
        name = (title or "").strip() or _default_playlist_title(period, year)
        pl = Playlist(user_id=user.id, title=name, created_at=now, updated_at=now)
        db.add(pl)
        db.flush()
        for pos, tid in enumerate(track_ids):
            db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos, added_by=user.id))
        db.commit()

        return {"ratingKey": playlist_key(pl.id), "title": pl.title, "size": len(track_ids)}
    finally:
        db.close()


def _default_playlist_title(period: str, year: int) -> str:
    """The name it gets when the client does not pass one. English here, and
    the clients pass a translated title — but a playlist created from a script
    or a curl should still read like something a person named."""
    if period == "year":
        return f"My Top Songs {year}"
    return f"My {period.capitalize()} {year}"


@router.delete("/stats")
def clear_stats(user: User = Depends(require_user)):
    """Wipe the caller's own listening history.

    Every /stats view (summary, top, activity, history) is derived from
    PlayEvent, so deleting the caller's rows empties their Statistics tab.
    Deliberately scoped to the caller only — it never removes another user's
    history, not even for an admin — and leaves resume points, likes and
    playlists untouched.
    """
    db = SessionLocal()
    try:
        n = db.query(PlayEvent).filter(PlayEvent.user_id == user.id).delete()
        db.commit()
        return {"deleted": int(n or 0)}
    finally:
        db.close()
