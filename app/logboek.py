"""Het logboek van Beheer: elke wijziging op een beheerpad, met wie en wanneer.

Een kale ASGI-middleware, geen BaseHTTPMiddleware: die laatste leest de body
zelf en laat dan niets over voor de route. Hier wordt de body gebufferd en
daarna opnieuw afgespeeld, zodat de route hem gewoon krijgt.
"""
from __future__ import annotations

import json
import time
from urllib.parse import parse_qs

from .db import SessionLocal
from .models import BeheerLog, Session as AuthSession

# Welke paden een wijziging in Beheer zijn.
PADEN = ("/admin", "/libraries", "/app/requirement", "/charts/looks", "/charts/top40/refresh")
# Wat er niet in hoort: het veelvuldige of het eigen werk van de server.
NIET = ("/admin/status", "/admin/overzicht", "/admin/live", "/admin/rescan/stop")
GEHEIM = ("token", "password", "wachtwoord", "pass", "secret", "image")


def _schoon(waarde):
    if isinstance(waarde, dict):
        return {k: ("•••" if any(g in k.lower() for g in GEHEIM) else _schoon(v)) for k, v in waarde.items()}
    if isinstance(waarde, list):
        return [_schoon(v) for v in waarde[:20]]
    if isinstance(waarde, str) and len(waarde) > 200:
        return waarde[:200] + "…"
    return waarde


class LogboekMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        pad = scope.get("path") or ""
        if not pad.startswith(PADEN) or pad.startswith(NIET):
            return await self.app(scope, receive, send)

        delen, meer = [], True
        while meer:
            bericht = await receive()
            delen.append(bericht.get("body", b""))
            meer = bericht.get("more_body", False)
        body = b"".join(delen)
        verzonden = {"klaar": False}

        async def opnieuw():
            if not verzonden["klaar"]:
                verzonden["klaar"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        status = {"code": 0}

        async def stuur(bericht):
            if bericht.get("type") == "http.response.start":
                status["code"] = bericht.get("status", 0)
            await send(bericht)

        try:
            await self.app(scope, opnieuw, stuur)
        finally:
            try:
                self._schrijf(scope, pad, body, status["code"])
            except Exception:
                pass

    def _schrijf(self, scope, pad, body, code):
        koppen = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        token = koppen.get("x-plex-token") or (parse_qs((scope.get("query_string") or b"").decode()).get("X-Plex-Token") or [""])[0]
        try:
            inhoud = json.dumps(_schoon(json.loads(body.decode() or "null")), ensure_ascii=False)[:1000] if body else ""
        except Exception:
            inhoud = ""
        q = (scope.get("query_string") or b"").decode()
        if q:
            q = "&".join(p for p in q.split("&") if not p.lower().startswith("x-plex-token"))
        db = SessionLocal()
        try:
            uid = 0
            if token:
                s = db.get(AuthSession, token)
                uid = s.user_id if s else 0
            db.add(BeheerLog(at=int(time.time()), user_id=uid, methode=scope.get("method", ""),
                             pad=pad + (f"?{q}" if q else ""), status=code, inhoud=inhoud))
            db.commit()
        finally:
            db.close()
