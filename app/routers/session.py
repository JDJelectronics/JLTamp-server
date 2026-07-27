"""Listen Together — synced group listening.

The key idea: everyone streams the SAME track from THIS server independently
(they all have library access), so we never send audio between phones — we only
synchronise POSITION. A session lives purely in memory (ephemeral: it ends when
the host leaves or the server restarts). A host creates a session and gets a
short code; guests join with it. Playback events (track / play / pause / seek /
queue) are broadcast over a WebSocket. Clock sync via ping/pong lets each client
convert to a shared server-time, and every start is SCHEDULED (startAtServerTs a
few hundred ms ahead) so all clients begin on the same beat.

Auth: the WebSocket can't send custom headers reliably, so the token comes as the
`X-Plex-Token` query param (same convention Android Auto / Cast use).
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from ..deps import _user_for_token, require_user
from ..models import User
from ..serializers import user_thumb_ref

router = APIRouter(prefix="/session", tags=["session"])


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Participant:
    user_id: int
    name: str
    thumb: str | None
    ws: WebSocket
    is_host: bool = False


@dataclass
class Session:
    code: str
    host_user_id: int
    created_at: int
    # ── Authoritative playback state ──
    track: dict | None = None      # the track JSON the host's client sent
    position_ms: int = 0           # position captured at anchor_ts
    anchor_ts: int = 0             # server time (ms) the position was captured / scheduled-start
    is_playing: bool = False
    queue: list = field(default_factory=list)
    participants: dict[int, Participant] = field(default_factory=dict)  # user_id → Participant

    def effective_position(self) -> int:
        """Where a late joiner should be RIGHT NOW. anchor_ts may be in the
        future (a scheduled start), which correctly yields a position just below
        position_ms until the start time is reached."""
        if self.is_playing and self.anchor_ts:
            return max(0, self.position_ms + (_now_ms() - self.anchor_ts))
        return self.position_ms


class SessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def _gen_code(self) -> str:
        # No 0/O/1/I/L — unambiguous when read aloud or typed.
        alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
        while True:
            code = "".join(secrets.choice(alphabet) for _ in range(5))
            if code not in self.sessions:
                return code

    def create(self, host: User) -> Session:
        s = Session(code=self._gen_code(), host_user_id=host.id, created_at=_now_ms())
        self.sessions[s.code] = s
        return s

    def get(self, code: str) -> Session | None:
        return self.sessions.get((code or "").upper())

    def remove(self, code: str) -> None:
        self.sessions.pop((code or "").upper(), None)


MANAGER = SessionManager()


def _participants_payload(s: Session) -> list:
    return [
        {"userId": p.user_id, "name": p.name, "thumb": p.thumb, "isHost": p.is_host}
        for p in s.participants.values()
    ]


async def _broadcast(s: Session, msg: dict, exclude: int | None = None) -> None:
    data = json.dumps(msg)
    dead: list[int] = []
    for uid, p in list(s.participants.items()):
        if uid == exclude:
            continue
        try:
            await p.ws.send_text(data)
        except Exception:
            dead.append(uid)
    for uid in dead:
        s.participants.pop(uid, None)


# ── REST ────────────────────────────────────────────────────────────────────
@router.post("/create")
def create_session(user: User = Depends(require_user)):
    s = MANAGER.create(user)
    return {"code": s.code}


@router.get("/{code}")
def session_info(code: str, user: User = Depends(require_user)):
    s = MANAGER.get(code)
    if not s:
        raise HTTPException(404, "No such session")
    return {
        "code": s.code,
        "hostUserId": s.host_user_id,
        "participants": _participants_payload(s),
        "isPlaying": s.is_playing,
        "hasTrack": s.track is not None,
    }


# ── WebSocket ───────────────────────────────────────────────────────────────
@router.websocket("/{code}/ws")
async def session_ws(websocket: WebSocket, code: str):
    token = websocket.query_params.get("X-Plex-Token")
    user = _user_for_token(token)
    if not user:
        await websocket.close(code=4401)  # unauthorized
        return

    code = (code or "").upper()
    s = MANAGER.get(code)
    if not s:
        await websocket.close(code=4404)  # no such session
        return

    await websocket.accept()

    name = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or f"User {user.id}"
    )
    # A URL, not the raw thumb_path. This handed clients a SERVER FILESYSTEM path
    # (e.g. /data/avatars/user_3.jpg), which no client can load — so every
    # Listen Together participant showed a broken avatar. The presence handler
    # below always built a proper URL; this one was simply missed.
    thumb = user_thumb_ref(user)
    is_host = user.id == s.host_user_id
    s.participants[user.id] = Participant(
        user_id=user.id, name=name, thumb=thumb, ws=websocket, is_host=is_host
    )

    # 1. Hand the newcomer the current authoritative state so they sync instantly.
    try:
        await websocket.send_text(json.dumps({
            "t": "state",
            "serverTs": _now_ms(),
            "track": s.track,
            "positionMs": s.effective_position(),
            "isPlaying": s.is_playing,
            "queue": s.queue,
            "hostUserId": s.host_user_id,
            "youAreHost": is_host,
            "youId": user.id,
            "participants": _participants_payload(s),
        }))
    except Exception:
        pass
    # 2. Tell everyone the roster changed.
    await _broadcast(s, {"t": "participants", "participants": _participants_payload(s)})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("t")

            # Clock sync — reply immediately with server time.
            if t == "ping":
                await websocket.send_text(json.dumps(
                    {"t": "pong", "id": msg.get("id"), "serverTs": _now_ms()}
                ))
                continue

            # Anyone may add to the shared queue.
            if t == "queueAdd" and msg.get("track") is not None:
                s.queue.append(msg["track"])
                await _broadcast(s, {"t": "queue", "queue": s.queue})
                continue

            # Anyone may send an emoji reaction — echoed to everyone (incl. sender
            # so their own reaction floats up too).
            if t == "reaction":
                await _broadcast(s, {
                    "t": "reaction", "emoji": str(msg.get("emoji", ""))[:8],
                    "userId": user.id, "name": name,
                })
                continue

            # Only the host drives transport (MVP). Guest transport msgs ignored.
            if not is_host:
                continue

            # Transport events go to the GUESTS only — the host already applied
            # them locally, so echoing them back is pure redundant traffic.
            if t == "track":
                s.track = msg.get("track")
                s.queue = msg.get("queue", s.queue)
                s.position_ms = int(msg.get("positionMs", 0))
                # Honour the host's real play state instead of assuming True.
                # A host promoted mid-session republishes its current track while
                # it may be PAUSED; hardcoding True made every guest start
                # playing against a silent host. Defaults to True so an older
                # client that omits the field behaves exactly as before.
                s.is_playing = bool(msg.get("isPlaying", True))
                s.anchor_ts = _now_ms() + 700  # scheduled start ~700ms ahead
                await _broadcast(s, {
                    "t": "track", "track": s.track, "queue": s.queue,
                    "positionMs": s.position_ms, "isPlaying": s.is_playing,
                    "startAtServerTs": s.anchor_ts,
                    "serverTs": _now_ms(),
                }, exclude=user.id)
            elif t == "play":
                s.position_ms = int(msg.get("positionMs", s.effective_position()))
                s.is_playing = True
                s.anchor_ts = _now_ms() + 350
                await _broadcast(s, {
                    "t": "play", "positionMs": s.position_ms,
                    "startAtServerTs": s.anchor_ts, "serverTs": _now_ms(),
                }, exclude=user.id)
            elif t == "pause":
                s.position_ms = int(msg.get("positionMs", s.effective_position()))
                s.is_playing = False
                s.anchor_ts = _now_ms()
                await _broadcast(s, {"t": "pause", "positionMs": s.position_ms, "serverTs": _now_ms()},
                                 exclude=user.id)
            elif t == "seek":
                s.position_ms = int(msg.get("positionMs", 0))
                s.is_playing = bool(msg.get("isPlaying", s.is_playing))
                s.anchor_ts = _now_ms() + (350 if s.is_playing else 0)
                await _broadcast(s, {
                    "t": "seek", "positionMs": s.position_ms, "isPlaying": s.is_playing,
                    "startAtServerTs": s.anchor_ts if s.is_playing else 0, "serverTs": _now_ms(),
                }, exclude=user.id)
            elif t == "sync":
                # Host heartbeat: its TRUE current position, anchored to NOW (no
                # scheduled start), so guests continuously track steady-state
                # playback instead of free-running between transport events.
                s.position_ms = int(msg.get("positionMs", s.effective_position()))
                s.is_playing = bool(msg.get("isPlaying", s.is_playing))
                s.anchor_ts = _now_ms()
                await _broadcast(s, {
                    "t": "sync", "positionMs": s.position_ms,
                    "isPlaying": s.is_playing, "serverTs": _now_ms(),
                }, exclude=user.id)
            elif t == "queue":
                s.queue = msg.get("queue", [])
                await _broadcast(s, {"t": "queue", "queue": s.queue})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Only clean up if the roster still points at THIS socket. On an abrupt
        # drop the client reconnects and re-registers under the same user_id BEFORE
        # this ghost's finally runs; without this guard we'd pop the freshly
        # reconnected participant (silent desync / wrong host-migration / room
        # teardown). If a newer socket owns the slot, leave everything alone.
        cur = s.participants.get(user.id)
        if cur is not None and cur.ws is websocket:
            # Was this the host at the moment they left? (They may have been promoted
            # to host mid-session, so check the live host id, not the join-time flag.)
            was_host = (user.id == s.host_user_id)
            s.participants.pop(user.id, None)
            if not s.participants:
                # Empty room → tear the session down.
                MANAGER.remove(code)
            elif was_host:
                # Host left but listeners remain → promote the longest-present one
                # (dict preserves insertion order) instead of ending the session.
                new_host = next(iter(s.participants.values()))
                s.host_user_id = new_host.user_id
                new_host.is_host = True
                await _broadcast(s, {
                    "t": "host", "hostUserId": new_host.user_id,
                    "participants": _participants_payload(s),
                })
            else:
                await _broadcast(s, {"t": "participants", "participants": _participants_payload(s)})


# ── Local presence + discovery ───────────────────────────────────────────────
# Everyone logged into this server opens a lightweight presence WebSocket. The
# server groups clients by the network it sees them coming from (same LAN /24, or
# the same NAT public IP), so people on the same Wi-Fi can see each other and send
# a DIRECT invite — no code or QR. The invite just carries a session code the
# invited client joins.
presence_router = APIRouter(prefix="/presence", tags=["presence"])


@dataclass
class Peer:
    user_id: int
    name: str
    thumb: str | None
    net: str
    ws: WebSocket
    session_code: str | None = None


class PresenceManager:
    def __init__(self) -> None:
        self.peers: dict[int, Peer] = {}   # user_id → Peer (one presence per user)

    @staticmethod
    def net_of(ip: str) -> str:
        """Discovery grouping key. JLTamp is a single-household, invite-only server:
        everyone who is logged in is trusted, so Listen Together discovery is ONE
        shared pool — you see every other signed-in device, web or app.

        (The previous per-/24 split isolated web users, who arrive via the Caddy
        reverse proxy as 172.21.x, from phones on the LAN reaching the server
        directly as 192.168.x — so web and app could never discover each other.
        `ip` is kept in the signature for callers/back-compat but ignored.)"""
        return "all"


PRESENCE = PresenceManager()


def _presence_client_ip(ws: WebSocket) -> str:
    xff = ws.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return ws.client.host if ws.client else ""


def _peer_list(net: str, exclude: int | None = None) -> list:
    return [
        {"userId": p.user_id, "name": p.name, "thumb": p.thumb,
         "inSession": p.session_code is not None}
        for p in PRESENCE.peers.values()
        if p.net == net and p.user_id != exclude
    ]


async def _presence_push_lists(net: str) -> None:
    """Send every peer on this network its OWN peer list (excluding itself), so no
    one ever sees themselves in 'people on your network'."""
    for p in list(PRESENCE.peers.values()):
        if p.net == net:
            try:
                await p.ws.send_text(json.dumps({"t": "peers", "peers": _peer_list(net, exclude=p.user_id)}))
            except Exception:
                pass


@presence_router.websocket("/ws")
async def presence_ws(websocket: WebSocket):
    token = websocket.query_params.get("X-Plex-Token")
    user = _user_for_token(token)
    if not user:
        await websocket.close(code=4401)
        return
    await websocket.accept()

    net = PresenceManager.net_of(_presence_client_ip(websocket))
    name = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or f"User {user.id}"
    )
    thumb = user_thumb_ref(user)
    PRESENCE.peers[user.id] = Peer(user_id=user.id, name=name, thumb=thumb, net=net, ws=websocket)

    # Refresh everyone on this network (incl. the newcomer) with their own list.
    await _presence_push_lists(net)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("t")

            if t == "ping":
                await websocket.send_text(json.dumps({"t": "pong"}))
                continue

            # Advertise which session (if any) this user is hosting/in, so peers
            # see who's already listening.
            if t == "setSession":
                p = PRESENCE.peers.get(user.id)
                if p:
                    p.session_code = (msg.get("code") or None)
                    await _presence_push_lists(net)
                continue

            # Direct invite: relay a code to a peer ON THE SAME NETWORK. `kind`
            # says whether it's a Listen Together session or a shared playlist.
            if t == "invite":
                target = PRESENCE.peers.get(msg.get("toUserId"))
                code = msg.get("code")
                if target and target.net == net and code:
                    try:
                        await target.ws.send_text(json.dumps({
                            "t": "invited", "fromUserId": user.id,
                            "fromName": name, "code": code,
                            "kind": msg.get("kind") or "session",
                            "title": msg.get("title") or "",
                        }))
                    except Exception:
                        pass
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Same reconnect guard as session_ws: only evict if the roster still points
        # at THIS socket, so a reconnect (which re-registers first) isn't removed by
        # the ghost's finally.
        cur = PRESENCE.peers.get(user.id)
        if cur is not None and cur.ws is websocket:
            PRESENCE.peers.pop(user.id, None)
            await _presence_push_lists(net)
