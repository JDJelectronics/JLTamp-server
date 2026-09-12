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
    """Eén VERBINDING, niet één persoon.

    Dit was `user_id → Participant`, en dat werkt precies zolang iedereen op één
    toestel zit. Doet iemand mee vanaf zijn telefoon én zijn tablet — of test
    iemand met twee eigen toestellen — dan schreef de tweede verbinding de
    eerste domweg over: de eerste bleef open maar stond niet meer in de lijst,
    dus kreeg niets meer door. En omdat de host aan een gebruiker hing in plaats
    van aan een verbinding, kreeg dat tweede toestel ook nog eens "jij bent de
    host" te horen. Twee hosts, geen gast, niemand die iemand volgt: dat is het
    "hij doet niks" dat van buiten niet te verklaren was.

    Vandaar een eigen sleutel per verbinding. De gebruiker staat er nog steeds
    bij — voor de naam, de foto en het doorgeven van het hostschap — maar het
    ROUTEREN gaat per verbinding.
    """
    conn_id: str
    user_id: int
    name: str
    thumb: str | None
    ws: WebSocket
    is_host: bool = False
    # Welk toestel dit is ("Pixel 9", "iPhone", "SM-X205"). Alleen nodig om twee
    # verbindingen van dezelfde persoon uit elkaar te houden.
    device: str | None = None


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
    participants: dict[str, Participant] = field(default_factory=dict)  # conn_id → Participant
    # De verbinding die de baas is. Blijft leeg tot de eerste verbinding van de
    # gebruiker die de sessie aanmaakte binnenkomt; zie de websocket hieronder.
    host_conn_id: str | None = None

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
    # `id` is de verbinding, `userId` de persoon. Twee toestellen van dezelfde
    # persoon geven twee regels met hetzelfde userId — daarom heeft de client
    # `id` nodig als sleutel.
    return [
        {"id": p.conn_id, "userId": p.user_id, "name": p.name, "thumb": p.thumb,
         "isHost": p.is_host, "device": p.device}
        for p in s.participants.values()
    ]


async def _broadcast(s: Session, msg: dict, exclude: str | None = None) -> None:
    data = json.dumps(msg)
    dead: list[str] = []
    for cid, p in list(s.participants.items()):
        if cid == exclude:
            continue
        try:
            await p.ws.send_text(data)
        except Exception:
            dead.append(cid)
    for cid in dead:
        s.participants.pop(cid, None)


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
    conn_id = secrets.token_hex(8)
    device = (websocket.query_params.get("device") or "").strip()[:40] or None
    # De eerste verbinding van de gebruiker die de sessie aanmaakte wordt de
    # host-verbinding. Komt diezelfde persoon later nog eens binnen (tweede
    # toestel), dan is dat gewoon een luisteraar erbij — niet opeens een tweede
    # host die de eerste overschrijft.
    if s.host_conn_id is None and user.id == s.host_user_id:
        s.host_conn_id = conn_id
    is_host = conn_id == s.host_conn_id
    s.participants[conn_id] = Participant(
        conn_id=conn_id, user_id=user.id, name=name, thumb=thumb, ws=websocket,
        is_host=is_host, device=device,
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
            # De eigen VERBINDING. Zit dezelfde persoon op twee toestellen, dan
            # is het account geen antwoord meer op "ben ik de host" — de client
            # vergeleek daarop en dan riepen beide toestellen zichzelf uit.
            "youConnId": conn_id,
            "hostConnId": s.host_conn_id,
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

            # Recomputed every message, never trusted from connect time: after a
            # promotion or a handover the old flag lies, and it is what gates
            # transport authority.
            is_host = conn_id == s.host_conn_id

            # Clock sync — reply immediately with server time.
            if t == "ping":
                await websocket.send_text(json.dumps(
                    {"t": "pong", "id": msg.get("id"), "serverTs": _now_ms()}
                ))
                continue

            # Anyone may add to the shared queue.
            if t == "queueAdd" and msg.get("track") is not None:
                track = msg["track"]
                # Stamp who added it. The clients show this for as long as the
                # track is still queued, so "the owner added X" stays on screen
                # until it actually plays instead of flashing past in a toast.
                if isinstance(track, dict):
                    track = {**track, "addedBy": name, "addedById": user.id}
                s.queue.append(track)
                await _broadcast(s, {"t": "queue", "queue": s.queue})
                continue

            # Anyone may send a reaction — an emoji OR a short typed line —
            # echoed to everyone including the sender, so their own message
            # floats up too and they can see it went out.
            #
            # One channel rather than a second "chat" message type: guests and
            # host already both have permission to use this one, and a typed
            # line is the same thing as an emoji, only longer. The cap moved
            # from 8 to 140 characters and newlines are collapsed, because this
            # renders as a floating line over the player, not a chat log.
            # The host may hand the session to someone else. Same broadcast the
            # disconnect-promotion path uses, so clients need no new handling:
            # whoever becomes host republishes their track and everyone
            # re-anchors to it.
            if t == "makeHost":
                if not is_host:
                    continue
                try:
                    target = int(msg.get("userId") or 0)
                except (TypeError, ValueError):
                    continue
                # Zoek een VERBINDING van die persoon. Zit hij op twee toestellen,
                # dan wordt de langst aanwezige de host — dat is de verbinding die
                # er als eerste was, en de volgorde van de lijst houdt dat vast.
                doel = next((q for q in s.participants.values() if q.user_id == target), None)
                if doel is None or doel.conn_id == s.host_conn_id:
                    continue
                for q in s.participants.values():
                    q.is_host = (q.conn_id == doel.conn_id)
                s.host_user_id = target
                s.host_conn_id = doel.conn_id
                await _broadcast(s, {
                    "t": "host", "hostUserId": target,
                    "hostConnId": s.host_conn_id,
                    "participants": _participants_payload(s),
                })
                continue

            # De host stopt ermee — en dan stoppen we met z'n allen.
            #
            # Tot nu toe was weggaan altijd "weggaan": ging de host eruit, dan
            # promoveerde de server de langst aanwezige luisteraar en liep de
            # sessie door. Voor een host die de verbinding verliest is dat precies
            # goed, maar voor een host die op "sessie beëindigen" tikt niet: de
            # gast bleef achter in een sessie die niemand meer bedoelde, en moest
            # zelf nog een keer op weg-drukken.
            if t == "end":
                if not is_host:
                    continue
                await _broadcast(s, {"t": "ended"}, exclude=conn_id)
                for q in list(s.participants.values()):
                    if q.conn_id == conn_id:
                        continue
                    try:
                        await q.ws.close(code=4000)
                    except Exception:
                        pass
                s.participants.clear()
                MANAGER.remove(code)
                try:
                    await websocket.close(code=4000)
                except Exception:
                    pass
                return

            if t == "reaction":
                text = " ".join(str(msg.get("emoji", "")).split())[:140]
                if not text:
                    continue
                await _broadcast(s, {
                    "t": "reaction", "emoji": text,
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
                }, exclude=conn_id)
            elif t == "play":
                s.position_ms = int(msg.get("positionMs", s.effective_position()))
                s.is_playing = True
                s.anchor_ts = _now_ms() + 350
                await _broadcast(s, {
                    "t": "play", "positionMs": s.position_ms,
                    "startAtServerTs": s.anchor_ts, "serverTs": _now_ms(),
                }, exclude=conn_id)
            elif t == "pause":
                s.position_ms = int(msg.get("positionMs", s.effective_position()))
                s.is_playing = False
                s.anchor_ts = _now_ms()
                await _broadcast(s, {"t": "pause", "positionMs": s.position_ms, "serverTs": _now_ms()},
                                 exclude=conn_id)
            elif t == "seek":
                s.position_ms = int(msg.get("positionMs", 0))
                s.is_playing = bool(msg.get("isPlaying", s.is_playing))
                s.anchor_ts = _now_ms() + (350 if s.is_playing else 0)
                await _broadcast(s, {
                    "t": "seek", "positionMs": s.position_ms, "isPlaying": s.is_playing,
                    "startAtServerTs": s.anchor_ts if s.is_playing else 0, "serverTs": _now_ms(),
                }, exclude=conn_id)
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
                }, exclude=conn_id)
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
        cur = s.participants.get(conn_id)
        if cur is not None and cur.ws is websocket:
            # Was this the host at the moment they left? (They may have been promoted
            # to host mid-session, so check the live host connection, not the
            # join-time flag.)
            was_host = (conn_id == s.host_conn_id)
            s.participants.pop(conn_id, None)
            if not s.participants:
                # Empty room → tear the session down.
                MANAGER.remove(code)
            elif was_host:
                # Host left but listeners remain → promote the longest-present one
                # (dict preserves insertion order) instead of ending the session.
                new_host = next(iter(s.participants.values()))
                s.host_user_id = new_host.user_id
                s.host_conn_id = new_host.conn_id
                new_host.is_host = True
                await _broadcast(s, {
                    "t": "host", "hostUserId": new_host.user_id,
                    "hostConnId": new_host.conn_id,
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
    """Eén AANWEZIG TOESTEL, niet één persoon.

    Zelfde verhaal als bij de deelnemers van een sessie: dit hing aan een
    gebruiker, dus je telefoon en je tablet schreven elkaar over. Wie jou wilde
    uitnodigen bereikte dan alleen het toestel dat zich het laatst meldde, en in
    de lijst "mensen op je netwerk" stond je maar één keer — terwijl je op twee
    plekken zat.
    """
    conn_id: str
    user_id: int
    name: str
    thumb: str | None
    net: str
    ws: WebSocket
    session_code: str | None = None
    device: str | None = None


class PresenceManager:
    def __init__(self) -> None:
        self.peers: dict[str, Peer] = {}   # conn_id → Peer (één per TOESTEL)

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


def _peer_list(net: str, exclude: str | None = None) -> list:
    # `id` is het toestel, `userId` de persoon. Zit iemand op twee toestellen,
    # dan staan er twee regels met hetzelfde userId — de client zet het toestel
    # eronder zodat je ze uit elkaar houdt.
    return [
        {"id": p.conn_id, "userId": p.user_id, "name": p.name, "thumb": p.thumb,
         "device": p.device, "inSession": p.session_code is not None}
        for p in PRESENCE.peers.values()
        if p.net == net and p.conn_id != exclude
    ]


async def _presence_push_lists(net: str) -> None:
    """Send every peer on this network its OWN peer list (excluding itself), so no
    one ever sees themselves in 'people on your network'."""
    for p in list(PRESENCE.peers.values()):
        if p.net == net:
            try:
                await p.ws.send_text(json.dumps({"t": "peers", "peers": _peer_list(net, exclude=p.conn_id)}))
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
    conn_id = secrets.token_hex(8)
    device = (websocket.query_params.get("device") or "").strip()[:40] or None
    PRESENCE.peers[conn_id] = Peer(
        conn_id=conn_id, user_id=user.id, name=name, thumb=thumb, net=net,
        ws=websocket, device=device,
    )

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
                p = PRESENCE.peers.get(conn_id)
                if p:
                    p.session_code = (msg.get("code") or None)
                    await _presence_push_lists(net)
                continue

            # Direct invite: relay a code to a peer ON THE SAME NETWORK. `kind`
            # says whether it's a Listen Together session or a shared playlist.
            if t == "invite":
                # Bij voorkeur naar het TOESTEL dat is aangetikt; kent de client
                # dat nog niet (oudere versie), dan naar elk toestel van die
                # persoon — beter twee keer gevraagd dan niemand bereikt.
                doelen: list[Peer] = []
                gekozen = PRESENCE.peers.get(msg.get("toId") or "")
                if gekozen is not None:
                    doelen = [gekozen]
                else:
                    try:
                        wie = int(msg.get("toUserId") or 0)
                    except (TypeError, ValueError):
                        wie = 0
                    doelen = [q for q in PRESENCE.peers.values() if q.user_id == wie]
                code = msg.get("code")
                for target in doelen:
                    if not (target.net == net and code):
                        continue
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
        cur = PRESENCE.peers.get(conn_id)
        if cur is not None and cur.ws is websocket:
            PRESENCE.peers.pop(conn_id, None)
            await _presence_push_lists(net)
