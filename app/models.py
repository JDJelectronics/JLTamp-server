"""Database model. A music library is small enough that a handful of flat
tables + denormalised names (so we rarely need joins on hot list endpoints)
keeps everything fast and simple.

ratingKey scheme: Plex uses one opaque global id space. We mirror that with a
type prefix so a single `/library/metadata/{ratingKey}` route resolves to the
right entity and playlist `uri`s can reference track keys:
    track   -> "t{id}"      album -> "al{id}"
    artist  -> "ar{id}"     playlist -> "pl{id}"
Helpers for this live in ids.py.

Multi-user (2026-07): users log in with their own email (invite-only). Music is
organised into admin-defined libraries (Plex-style); each user is granted access
to specific libraries. Playlists and liked songs are per-user.
"""
from __future__ import annotations

from sqlalchemy import Integer, String, Float, Boolean, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Meta(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, default="")


# ── Users / auth ─────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    # null until the invited user sets their password (invite-only signup).
    password_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    display_name: Mapped[str] = mapped_column(String, default="")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # set for a pending invite; consumed when the user sets a password.
    invite_token: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # "Forgot password": a one-shot token, mailed to the user, that EXPIRES.
    # Kept apart from invite_token so a reset can never revive a withdrawn invite
    # or vice versa.
    reset_token: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    reset_expires_at: Mapped[int] = mapped_column(Integer, default=0)
    thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, default=0)
    # Email me a digest when a manual scan adds new music. Per-user, default on;
    # the user can switch it off from Settings on the web.
    notify_new_music: Mapped[bool] = mapped_column(Boolean, default=True)
    # Preferred language ('nl', 'en', 'fy', 'de', 'es-ES', 'pt-BR', 'pt-PT').
    # Chosen/auto-detected at signup; used to localize outgoing mail.
    lang: Mapped[str | None] = mapped_column(String, nullable=True)


class Session(Base):
    """A bearer token -> user mapping. Tokens are sent as `X-Plex-Token`
    (header or query) exactly like before, but now resolve to a specific user."""
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[int] = mapped_column(Integer, default=0)
    label: Mapped[str] = mapped_column(String, default="")  # e.g. "web", "mobile"


# ── Libraries (Plex-style, admin-managed) ────────────────────────────────────
class Library(Base):
    __tablename__ = "libraries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String, default="music")  # future: podcasts etc.
    created_at: Mapped[int] = mapped_column(Integer, default=0)
    # last scan bookkeeping (shown in admin UI)
    last_scan_at: Mapped[int] = mapped_column(Integer, default=0)
    last_scan_ms: Mapped[int] = mapped_column(Integer, default=0)
    track_count: Mapped[int] = mapped_column(Integer, default=0)


class LibraryFolder(Base):
    """A library can contain several folders (all under the read-only mounts)."""
    __tablename__ = "library_folders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True)
    path: Mapped[str] = mapped_column(String)  # absolute path inside the container


class UserLibraryAccess(Base):
    """Which libraries a user may see (Plex-style sharing). Admin implicitly
    sees everything, so admins need no rows here."""
    __tablename__ = "user_library_access"
    __table_args__ = (UniqueConstraint("user_id", "library_id", name="uq_user_library"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True)


# ── Music entities ───────────────────────────────────────────────────────────
class Artist(Base):
    __tablename__ = "artists"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True, default=0)
    name: Mapped[str] = mapped_column(String, index=True)
    sort_name: Mapped[str] = mapped_column(String, index=True, default="")
    summary: Mapped[str] = mapped_column(String, default="")
    art_path: Mapped[str | None] = mapped_column(String, nullable=True)  # abs path to an image file
    # online-enriched artist photo (Deezer etc.), cached under the data dir.
    online_art_path: Mapped[str | None] = mapped_column(String, nullable=True)
    enriched: Mapped[bool] = mapped_column(Boolean, default=False)  # metadata lookup done
    added_at: Mapped[int] = mapped_column(Integer, default=0)  # epoch seconds


class Album(Base):
    __tablename__ = "albums"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True, default=0)
    title: Mapped[str] = mapped_column(String, index=True)
    sort_title: Mapped[str] = mapped_column(String, index=True, default="")
    artist_id: Mapped[int] = mapped_column(ForeignKey("artists.id"), index=True)
    artist_name: Mapped[str] = mapped_column(String, default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    genre: Mapped[str] = mapped_column(String, default="", index=True)
    art_path: Mapped[str | None] = mapped_column(String, nullable=True)
    online_art_path: Mapped[str | None] = mapped_column(String, nullable=True)
    enriched: Mapped[bool] = mapped_column(Boolean, default=False)
    track_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    added_at: Mapped[int] = mapped_column(Integer, default=0)


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True, default=0)
    title: Mapped[str] = mapped_column(String, index=True)
    sort_title: Mapped[str] = mapped_column(String, index=True, default="")
    album_id: Mapped[int] = mapped_column(ForeignKey("albums.id"), index=True)
    album_title: Mapped[str] = mapped_column(String, default="")
    artist_id: Mapped[int] = mapped_column(ForeignKey("artists.id"), index=True)
    artist_name: Mapped[str] = mapped_column(String, default="")   # album/grouping artist (grandparentTitle)
    album_artist: Mapped[str] = mapped_column(String, default="")
    orig_artist: Mapped[str] = mapped_column(String, default="")   # the track's own artist (Plex originalTitle)
    track_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    disc_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    genre: Mapped[str] = mapped_column(String, default="", index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # File / media info
    path: Mapped[str] = mapped_column(String, unique=True)   # absolute path on disk
    ext: Mapped[str] = mapped_column(String, default="")     # ".flac" etc (lowercase)
    container: Mapped[str] = mapped_column(String, default="")
    codec: Mapped[str] = mapped_column(String, default="")
    bitrate: Mapped[int] = mapped_column(Integer, default=0)   # kbps
    channels: Mapped[int] = mapped_column(Integer, default=2)
    samplerate: Mapped[int] = mapped_column(Integer, default=44100)
    bits: Mapped[int] = mapped_column(Integer, default=16)
    size: Mapped[int] = mapped_column(Integer, default=0)      # bytes
    mtime: Mapped[float] = mapped_column(Float, default=0.0)   # file mtime for incremental rescan

    art_path: Mapped[str | None] = mapped_column(String, nullable=True)  # usually the album cover
    added_at: Mapped[int] = mapped_column(Integer, default=0)
    last_played_at: Mapped[int] = mapped_column(Integer, default=0, index=True)
    play_count: Mapped[int] = mapped_column(Integer, default=0)
    # Loudness-normalisation gain in dB (ReplayGain track gain / EBU R128),
    # relative to the reference level. NULL = not analysed yet. The app levels
    # playback volume with this so quiet and loud tracks sound equally loud.
    gain_db: Mapped[float | None] = mapped_column(Float, nullable=True)


# ── Per-user playlists & likes ───────────────────────────────────────────────
class Playlist(Base):
    __tablename__ = "playlists"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, default=0)
    title: Mapped[str] = mapped_column(String)
    created_at: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[int] = mapped_column(Integer, default=0)


class PlaylistItem(Base):
    __tablename__ = "playlist_items"
    # id == Plex's `playlistItemID` (used by DELETE /playlists/{rk}/items/{id})
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    playlist_id: Mapped[int] = mapped_column(ForeignKey("playlists.id"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)


class LikedTrack(Base):
    """Per-user 'liked songs'."""
    __tablename__ = "liked_tracks"
    __table_args__ = (UniqueConstraint("user_id", "track_id", name="uq_user_liked"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    created_at: Mapped[int] = mapped_column(Integer, default=0)


class UserTrackState(Base):
    """Per-user playback state: play count, last-played time, and resume offset.
    History is per user (not global on Track) so multi-user plays never collide.
    Powers 'Recently played', 'Most played', play counts and resume."""
    __tablename__ = "user_track_state"
    __table_args__ = (UniqueConstraint("user_id", "track_id", name="uq_user_track_state"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    play_count: Mapped[int] = mapped_column(Integer, default=0)
    last_played_at: Mapped[int] = mapped_column(Integer, default=0)
    view_offset_ms: Mapped[int] = mapped_column(Integer, default=0)
    # Plex-style star rating, 0-10 (0 = unrated). Per user, like everything else
    # here: one listener's 5 stars is not another's.
    rating: Mapped[float] = mapped_column(Float, default=0.0)


class PlayEvent(Base):
    """One listening session: this user played this track, then, for this long.

    UserTrackState is the *current* state (counter, resume point) — it cannot
    tell you what you listened to last Tuesday. This is the append-only log that
    can: it powers the history list, the per-day charts and the top-artist stats.
    Denormalised artist/album/title so history survives a track being removed
    from disk (the NAS is read-only, but files do move).
    """
    __tablename__ = "play_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    artist_name: Mapped[str] = mapped_column(String, default="", index=True)
    album_title: Mapped[str] = mapped_column(String, default="")
    track_title: Mapped[str] = mapped_column(String, default="")
    started_at: Mapped[int] = mapped_column(Integer, default=0, index=True)
    ended_at: Mapped[int] = mapped_column(Integer, default=0)
    # Wall-clock seconds this session actually lasted, and the furthest point
    # reached — a 3s skip and a full listen are not the same event.
    listened_ms: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    device: Mapped[str] = mapped_column(String, default="")
    session_id: Mapped[str] = mapped_column(String, default="", index=True)


Index("ix_uts_user_lastplayed", UserTrackState.user_id, UserTrackState.last_played_at)
Index("ix_uts_user_playcount", UserTrackState.user_id, UserTrackState.play_count)
Index("ix_pe_user_started", PlayEvent.user_id, PlayEvent.started_at)
Index("ix_album_artist_title", Album.artist_id, Album.sort_title)
Index("ix_track_album_disc_no", Track.album_id, Track.disc_no, Track.track_no)
Index("ix_pli_playlist_pos", PlaylistItem.playlist_id, PlaylistItem.position)
Index("ix_artist_lib_sort", Artist.library_id, Artist.sort_name)
Index("ix_album_lib_sort", Album.library_id, Album.sort_title)
Index("ix_track_lib", Track.library_id)
