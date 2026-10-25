"""Admin: manage users (invite-only), per-user library access (Plex-style
sharing). Only admins may call these."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete, func

from ..db import SessionLocal
from ..models import (User, UserLibraryAccess, Library, Meta, Session as AuthSession,
                      Playlist, PlaylistItem, LikedTrack, Track, UserTrackState,
                      PlayEvent, SubsonicKey, SubsonicStar)
from ..security import new_invite, norm_email
from .. import config, mailer
from ..deps import require_admin, create_session
from ..instellingen import servernaam as _servernaam

router = APIRouter()


class InviteBody(BaseModel):
    email: str
    display_name: str | None = None
    is_admin: bool = False
    library_ids: list[int] = []


class AccessBody(BaseModel):
    library_ids: list[int]


class NameBody(BaseModel):
    name: str | None = None


class FlagBody(BaseModel):
    value: bool


def _is_owner(user: User) -> bool:
    """The account that OWNS this server. It is admin forever — no admin, not even
    a second one that got invited or compromised, may touch its rights."""
    return norm_email(user.email) == norm_email(config.ADMIN_EMAIL)


def _active_admin_count(db) -> int:
    return db.query(User).filter(
        User.is_admin == True, User.is_active == True  # noqa: E712
    ).count()


def _guard_owner(user: User) -> None:
    if _is_owner(user):
        raise HTTPException(status_code=403, detail="De eigenaar van de server kan niet worden gewijzigd")


def _guard_last_admin(db, user: User) -> None:
    """Block removing the final working admin — even a non-owner one — so the
    server can never end up with nobody able to administer it."""
    if user.is_admin and user.is_active and _active_admin_count(db) <= 1:
        raise HTTPException(status_code=400, detail="Dit is de laatste beheerder — die kan niet worden verwijderd of gedegradeerd")


def _user_dict(db, u: User) -> dict:
    grants = [r.library_id for r in db.execute(
        select(UserLibraryAccess).where(UserLibraryAccess.user_id == u.id)).scalars()]
    return {
        "id": u.id,
        "email": u.email,
        "displayName": u.display_name,
        "isAdmin": u.is_admin,
        "isOwner": _is_owner(u),
        "isActive": u.is_active,
        "isReview": bool(getattr(u, "is_review", False)),
        "pending": u.password_hash is None,
        "inviteToken": u.invite_token,
        "libraryIds": grants,
        "createdAt": u.created_at,
        "hasThumb": bool(getattr(u, "thumb_path", None)),
        "rechten": rechten(db, u.id),
    }


def _set_access(db, user_id: int, library_ids: list[int]):
    db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.user_id == user_id))
    valid = {l.id for l in db.execute(select(Library)).scalars()}
    for lid in set(library_ids):
        if lid in valid:
            db.add(UserLibraryAccess(user_id=user_id, library_id=lid))


class AutoAccessBody(BaseModel):
    library_ids: list[int]


# ── Serverinstellingen ──────────────────────────────────────────────────────
#
# Dingen die voor de hele server gelden en die je wilt kunnen omzetten zonder
# aan de container te komen. Ze staan in `meta`, want een omgevingsvariabele
# aanpassen betekent opnieuw opstarten — en dan liggen alle luisteraars eruit
# voor een schakelaartje.
#
# De omgevingsvariabele blijft wel de beginwaarde: staat er niets in `meta`, dan
# geldt wat er in docker-compose staat. Pas als je hier iets zet, wint dit.
META_OPEN_REG = "open_registration"

# ── De schakelaars van de server ────────────────────────────────────────────
#
# Standen die voor de HELE server gelden en die je onderweg wilt kunnen omzetten
# zonder een nieuwe versie in twee winkels. Ze staan alle vier in `meta`, en ze
# staan hier bij elkaar zodat er eentje bijzetten één regel is.
#
# Per schakelaar: de sleutel in meta, wat hij doet als hij AAN staat, en wat de
# stand is als er nog nooit aan gedraaid is.
#
#   winkelstand   de app laat niets zien over wat er met een zoekopdracht
#                 gebeurt die niets oplevert — het zoekscherm zegt alleen
#                 "niets gevonden", zoals elke muziekapp. Standaard UIT.
#
#   wensen        of het zoekscherm überhaupt iets doet met wat er niet
#                 gevonden is. Dit is de gewone aan/uit; de winkelstand
#                 hierboven is de tijdelijke. Standaard AAN.
#
#   albums        of er ook hele albums gewenst mogen worden, of alleen losse
#                 nummers. Standaard AAN.
#
#   importeren    of een geplakte Spotify-lijst geïmporteerd mag
#                 worden. Stond tot nu toe alleen in de omgeving van de
#                 container (JLTAMP_IMPORTERS) en was dus alleen met een
#                 herstart om te zetten. Die omgeving is nu de STANDAARD; deze
#                 schakelaar gaat erboven.
SCHAKELAARS: dict[str, tuple[str, bool]] = {
    "winkelstand": ("winkelstand", False),
    "wensen": ("wensen_aan", True),
    "albums": ("albumwensen_aan", True),
    "importeren": ("importeren_aan", config.IMPORTERS),
    # De viraal-tegel in het blok. Staat aan; uit haalt hem weg zonder dat er
    # iets in de winkels hoeft te veranderen.
    "viraal": ("viraal_aan", True),
    # De mail die na een scan meldt welke muziek erbij kwam. Staat aan. Uit zet
    # hem voor IEDEREEN stil zonder aan de persoonlijke voorkeur van gebruikers
    # (notify_new_music) te komen, dus weer aanzetten herstelt precies wie hem
    # had. Handig als er in partijen muziek bijkomt: elke partij is een scan, en
    # dus een mail naar iedereen.
    "nieuwe-muziek-mail": ("nieuwe_muziek_mail_aan", True),
    # De hoofdkraan voor ALLE uitgaande post: uitnodigingen, wachtwoord-resets,
    # de welkomstmail én de nieuwe-muziek-mail. Staat aan. Uit betekent dat er
    # niets meer de deur uitgaat — ook geen wachtwoord-reset, dus dat is een
    # stand waar je bewust voor kiest en weer uit komt. Wie alleen de mail na
    # een scan kwijt wil, heeft aan `nieuwe-muziek-mail` genoeg.
    "mail": ("mail_aan", True),
    # De feestmodus: gasten vragen via een link nummers aan (zie feest.py).
    # Staat UIT tot de gastheer er bewust voor kiest.
    "feest": ("feest_aan", False),
    # Karaoke in de speler (zie routers/karaoke.py). Staat UIT: een nummer
    # scheiden kost rekentijd en schijfruimte, en er moet naast de server eerst
    # een dienst voor draaien. Pas aanzetten als die er is.
    "karaoke": ("karaoke_aan", False),
    # De presentator tussen de nummers (routers/presentator.py). Staat uit tot
    # Piper en het model klaarstaan.
    "presentator": ("presentator_aan", False),
    # Akkoorden bij de songtekst (routers/akkoorden.py). Staat UIT: elk nummer
    # waar iemand de gitaarknop bij aanzet kost een halve minuut rekenen op de
    # server, en de herkenning is een schatting — eerst zelf proberen.
    "akkoorden": ("akkoorden_aan", False),
    # De speelhoek op home (Raad het nummer, Zing verder, Wie is de artiest).
    "spellen": ("spellen_aan", True),
    # De DJ-decks in het meespeelmenu van de speler.
    "dj": ("dj_aan", True),
    # Onderhoudsmodus: de app toont gewone gebruikers een scherm "even
    # onderhoud". Beheerders kunnen gewoon door. Zie beheer.py.
    "onderhoud": ("onderhoud_aan", False),
    # Elke nacht een kopie van de database (beheer.py, /data/backups).
    "auto-backup": ("auto_backup_aan", True),
    # JLTamp Radio: één 24/7-stream voor iedereen (app/radio). Staat UIT: aan
    # betekent een ffmpeg die dag en nacht loopt. Uit = er draait niets.
    "radio": ("radio_aan", False),
    # De presentatoren op de radio. Uit = alleen muziek (de app-presentator,
    # "presentator" hierboven, staat daar los van).
    "radio-presentator": ("radio_presentator_aan", True),
    # De Subsonic-API (routers/subsonic.py): bestaande Subsonic-apps zoals
    # Symfonium of Feishin met deze server laten praten. Staat aan; uit = /rest/
    # zegt tegen elke app "staat uit", en de rest van de server merkt niets.
    "subsonic": ("subsonic_aan", True),
    # Het weekschema (app/radio/programma.py). Uit = de hele dag de gewone mix.
    "radio-programmering": ("radio_programma_aan", True),
}

# ── Getallen ─────────────────────────────────────────────────────────────────
#
# Net als de schakelaars, maar dan een aantal. Per getal: de sleutel in meta,
# de standaard, en de grenzen waarbinnen hij mag liggen.
GETALLEN: dict[str, tuple[str, int, int, int]] = {
    # Elke zoveel uur alle bibliotheken scannen; 0 = uit (zie beheer.py).
    "auto-scan-uur": ("auto_scan_uur", 0, 0, 168),
    # Hoeveel automatische back-ups er blijven staan.
    "backups-bewaren": ("backups_bewaren", 7, 1, 60),
    # JLTamp Radio — de zender leest ze live, zonder herstart.
    "radio-overvloei": ("radio_overvloei", 4, 0, 12),          # seconden
    "radio-duck": ("radio_duck", 12, 0, 30),                   # zoveel dB zachter onder de stem
    "radio-praat-elke": ("radio_praat_elke", 3, 2, 10),        # uiterlijk elk zoveelste nummer een break
    "radio-artiest-afstand": ("radio_artiest_afstand", 8, 2, 30),   # nummers tussen dezelfde artiest
    "radio-herhaal-uur": ("radio_herhaal_uur", 6, 1, 48),      # uren voor een nummer terug mag komen
}


# ── Rechten per gebruiker ────────────────────────────────────────────────────
#
# Bovenop de serverschakelaars: een schakelaar zet iets voor iedereen aan of
# uit, een recht neemt het één persoon af. Standaard mag alles; er wordt alleen
# opgeslagen wat afwijkt, in meta onder "recht:<id>".
RECHTEN = ("karaoke", "dj", "spellen")


def rechten(db, uid: int) -> dict:
    rij = db.get(Meta, f"recht:{uid}")
    try:
        opgeslagen = json.loads(rij.value) if rij and rij.value else {}
    except Exception:
        opgeslagen = {}
    uit = {r: opgeslagen.get(r, True) is not False for r in RECHTEN}
    uit["daglimiet"] = int(opgeslagen.get("daglimiet") or 0)
    return uit


def recht(db, user, naam: str) -> bool:
    if user is None:
        return True
    return rechten(db, user.id).get(naam, True)


def getal(db, naam: str) -> int:
    sleutel, standaard, laag, hoog = GETALLEN[naam]
    rij = db.get(Meta, sleutel)
    try:
        return max(laag, min(hoog, int(rij.value))) if rij else standaard
    except (TypeError, ValueError):
        return standaard


def alle_getallen(db) -> dict[str, int]:
    return {naam: getal(db, naam) for naam in GETALLEN}


def schakelaar(db, naam: str) -> bool:
    """De stand van één schakelaar, of zijn standaard als er nooit aan gedraaid is."""
    sleutel, standaard = SCHAKELAARS.get(naam, (naam, False))
    rij = db.get(Meta, sleutel)
    if rij is None or rij.value not in ("0", "1"):
        return standaard
    return rij.value == "1"


def zet_schakelaar(db, naam: str, aan: bool) -> None:
    sleutel = SCHAKELAARS.get(naam, (naam, False))[0]
    rij = db.get(Meta, sleutel)
    if not rij:
        rij = Meta(key=sleutel)
        db.add(rij)
    rij.value = "1" if aan else "0"


def winkelstand(db) -> bool:
    """Blijft bestaan omdat er op een paar plekken naar verwezen wordt."""
    return schakelaar(db, "winkelstand")


def open_registratie(db) -> bool:
    """Mag iedereen zich aanmelden, of is het op uitnodiging?"""
    rij = db.get(Meta, META_OPEN_REG)
    if rij and rij.value in ("0", "1"):
        return rij.value == "1"
    return bool(config.OPEN_REGISTRATION)


@router.get("/admin/server/settings")
def lees_serverinstellingen(_: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        return {
            "openRegistration": open_registratie(db),
            "serverName": _servernaam(),
            # Alles in één keer, met de namen uit SCHAKELAARS.
            "schakelaars": {naam: schakelaar(db, naam) for naam in SCHAKELAARS},
            "getallen": alle_getallen(db),
            # `winkelstand` los erbij voor een app die de lijst nog niet kent.
            "winkelstand": schakelaar(db, "winkelstand"),
        }
    finally:
        db.close()


@router.post("/admin/server/settings")
def zet_serverinstellingen(body: FlagBody, _: User = Depends(require_admin)):
    """Open registratie aan of uit.

    Aan betekent: iedereen die het adres van deze server kent, kan een account
    maken. Dat is een deur die je bewust openzet — vandaar dat hij hier staat en
    niet ergens in een instellingenlijstje.
    """
    db = SessionLocal()
    try:
        rij = db.get(Meta, META_OPEN_REG)
        if not rij:
            rij = Meta(key=META_OPEN_REG)
            db.add(rij)
        rij.value = "1" if body.value else "0"
        db.commit()
        return {"openRegistration": bool(body.value)}
    finally:
        db.close()


class SchakelaarBody(BaseModel):
    naam: str
    value: bool


@router.post("/admin/server/schakelaar")
def zet_serverschakelaar(body: SchakelaarBody, _: User = Depends(require_admin)):
    """Eén schakelaar omzetten. Zie SCHAKELAARS hierboven."""
    if body.naam not in SCHAKELAARS:
        raise HTTPException(400, "Onbekende schakelaar")
    db = SessionLocal()
    try:
        zet_schakelaar(db, body.naam, body.value)
        db.commit()
        return {"schakelaars": {naam: schakelaar(db, naam) for naam in SCHAKELAARS}}
    finally:
        db.close()


class RechtenBody(BaseModel):
    rechten: dict = {}
    daglimiet: int = 0


@router.post("/admin/users/{user_id}/rechten")
def zet_rechten(user_id: int, body: RechtenBody, _: User = Depends(require_admin)):
    """Wat deze ene gebruiker wel en niet mag. Alleen afwijkingen worden bewaard."""
    db = SessionLocal()
    try:
        if not db.get(User, user_id):
            raise HTTPException(404, "Gebruiker niet gevonden")
        waarde = {r: False for r in RECHTEN if body.rechten.get(r) is False}
        if body.daglimiet and body.daglimiet > 0:
            waarde["daglimiet"] = min(int(body.daglimiet), 500)
        sleutel = f"recht:{user_id}"
        rij = db.get(Meta, sleutel)
        if not rij:
            rij = Meta(key=sleutel)
            db.add(rij)
        rij.value = json.dumps(waarde)
        db.commit()
        return {"rechten": rechten(db, user_id)}
    finally:
        db.close()


class GetalBody(BaseModel):
    naam: str
    value: int


@router.post("/admin/server/getal")
def zet_servergetal(body: GetalBody, _: User = Depends(require_admin)):
    """Eén getal zetten. Zie GETALLEN hierboven; buiten de grenzen wordt afgerond."""
    if body.naam not in GETALLEN:
        raise HTTPException(400, "Onbekend getal")
    sleutel, _std, laag, hoog = GETALLEN[body.naam]
    db = SessionLocal()
    try:
        rij = db.get(Meta, sleutel)
        if not rij:
            rij = Meta(key=sleutel)
            db.add(rij)
        rij.value = str(max(laag, min(hoog, int(body.value))))
        db.commit()
        return {"getallen": alle_getallen(db)}
    finally:
        db.close()


@router.post("/admin/server/winkelstand")
def zet_winkelstand(body: FlagBody, _: User = Depends(require_admin)):
    """De winkelstand los. Blijft bestaan voor een app die de lijst nog niet kent."""
    db = SessionLocal()
    try:
        zet_schakelaar(db, "winkelstand", body.value)
        db.commit()
        return {"winkelstand": bool(body.value)}
    finally:
        db.close()


@router.get("/admin/auto-access")
def get_auto_access(_: User = Depends(require_admin)):
    """Welke bibliotheken nieuwe accounts automatisch krijgen."""
    import json
    db = SessionLocal()
    try:
        row = db.get(Meta, "auto_grant_libs")
        ids = json.loads(row.value) if row and row.value else []
        return {"library_ids": ids}
    finally:
        db.close()


@router.post("/admin/auto-access")
def set_auto_access(body: AutoAccessBody, _: User = Depends(require_admin)):
    """Stel in welke bibliotheken nieuwe accounts automatisch krijgen."""
    import json
    db = SessionLocal()
    try:
        valid = {l.id for l in db.execute(select(Library)).scalars()}
        ids = sorted({int(x) for x in body.library_ids if int(x) in valid})
        row = db.get(Meta, "auto_grant_libs")
        if not row:
            row = Meta(key="auto_grant_libs")
            db.add(row)
        row.value = json.dumps(ids)
        db.commit()
        return {"library_ids": ids}
    finally:
        db.close()


@router.get("/admin/users")
def list_users(_: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        # Wat de beheerder wil weten zonder elk account open te klikken: wanneer
        # was iemand er voor het laatst, op welk toestel, en luistert hij echt.
        # Alles in een paar groepsvragen in plaats van per gebruiker.
        from ..models import PlayEvent, LikedTrack
        week = int(time.time()) - 7 * 86400
        # Sessies van de AI-dienst (die namens iemand weeklijsten maakt) tellen
        # niet als "er geweest": anders is iedereen elke week actief.
        mens = AuthSession.label.notin_(("ai-service", "bekijk-als"))
        laatst = dict(db.execute(select(AuthSession.user_id, func.max(AuthSession.last_seen_at))
                                 .where(mens).group_by(AuthSession.user_id)).all())
        toestel = {}
        for uid, label, gezien in db.execute(select(AuthSession.user_id, AuthSession.label, AuthSession.last_seen_at)
                                             .where(mens).order_by(AuthSession.last_seen_at)).all():
            toestel[uid] = label or toestel.get(uid, "")
        plays = {uid: (n, ms or 0) for uid, n, ms in db.execute(
            select(PlayEvent.user_id, func.count(PlayEvent.id), func.sum(PlayEvent.listened_ms))
            .group_by(PlayEvent.user_id)).all()}
        plays_week = dict(db.execute(select(PlayEvent.user_id, func.count(PlayEvent.id))
                                     .where(PlayEvent.started_at >= week).group_by(PlayEvent.user_id)).all())
        likes = dict(db.execute(select(LikedTrack.user_id, func.count(LikedTrack.id))
                                .group_by(LikedTrack.user_id)).all())
        uit = []
        for u in db.execute(select(User).order_by(User.id)).scalars():
            d = _user_dict(db, u)
            n, ms = plays.get(u.id, (0, 0))
            d.update({"lastSeen": int(laatst.get(u.id) or 0), "device": toestel.get(u.id, ""),
                      "plays": n, "playsWeek": int(plays_week.get(u.id, 0)),
                      "listenedMin": int(ms // 60000), "likes": int(likes.get(u.id, 0))})
            uit.append(d)
        return {"users": uit}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/session")
def mint_user_session(user_id: int, admin: User = Depends(require_admin)):
    """Issue a session token for another user. Admin-only.

    The AI engine runs background jobs — weekly discovery playlists — that must
    create playlists *as* each user, but a background job holds no user's login
    token. This lets the admin engine obtain one. It is deliberately narrow:
    admin auth is required, the token is labelled so it is distinguishable in
    the sessions table, and it grants exactly what any of that user's own
    tokens would. It does NOT widen what a non-admin can do.
    """
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user or not user.is_active:
            raise HTTPException(404, "user not found or inactive")
        token = create_session(db, user, label="ai-service")
        return {"token": token, "user_id": user.id,
                "email": user.email, "display_name": user.display_name}
    finally:
        db.close()


@router.post("/admin/users/invite")
def invite_user(body: InviteBody, admin: User = Depends(require_admin)):
    email = norm_email(body.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Ferkeard e-mailadres")
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=409, detail="E-mail al yn gebrûk")
        token = new_invite()
        # Single-admin server by the owner's rule: invited users are NEVER admin.
        # There is exactly one admin — the owner — and no invite can create a
        # second one. (body.is_admin is intentionally ignored.)
        user = User(email=email, password_hash=None,
                    display_name=(body.display_name or email.split("@")[0]).strip(),
                    is_admin=False, is_active=True,
                    invite_token=token, created_at=int(time.time()))
        db.add(user)
        db.flush()
        _set_access(db, user.id, body.library_ids)
        db.commit()

        # Mail the link ourselves when SMTP is set up. `mailed` tells the admin
        # UI whether it still has to hand the link over manually.
        mailed = mailer.send_invite(
            email, token,
            inviter=(admin.display_name or admin.email),
            server_name=_servernaam(),
        )
        return {"user": _user_dict(db, user), "inviteToken": token,
                "invitePath": f"/invite/{token}", "mailed": mailed}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/reinvite")
def reinvite(user_id: int, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        user.invite_token = new_invite()
        user.password_hash = None  # force them to set a new password
        db.commit()
        mailed = mailer.send_invite(
            user.email, user.invite_token,
            inviter=(admin.display_name or admin.email),
            server_name=_servernaam(),
        )
        return {"inviteToken": user.invite_token,
                "invitePath": f"/invite/{user.invite_token}", "mailed": mailed}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/access")
def set_access(user_id: int, body: AccessBody, _: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        if not db.get(User, user_id):
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        _set_access(db, user_id, body.library_ids)
        db.commit()
        return {"ok": True, "libraryIds": body.library_ids}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/admin")
def set_admin(user_id: int, body: FlagBody, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        if body.value:
            # Promotion: any account may be made admin, but only once it has
            # actually joined (a pending invite has no password yet) and is
            # active — you can't hand admin rights to an account nobody controls.
            if user.password_hash is None or not user.is_active:
                raise HTTPException(status_code=400, detail="Alleen een actieve, aangemelde gebruiker kan beheerder worden")
        if not body.value:
            # Demotion: never the owner, never yourself, never the last admin.
            _guard_owner(user)
            if user.id == admin.id:
                raise HTTPException(status_code=400, detail="Kinsto dysels net degradearje")
            _guard_last_admin(db, user)
        user.is_admin = body.value
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/review")
def set_review(user_id: int, body: FlagBody, admin: User = Depends(require_admin)):
    """Markeer een account als reviewaccount.

    Dat betekent op dit moment één ding: het blok "Van JLTamp" blijft bij dat
    account weg. Zie de toelichting bij `User.is_review` — een demo waarmee een
    winkel de app nakijkt, hoort een kale app te zien en geen aanbevelingen die
    op een luistergeschiedenis leunen die er nog niet is.

    Geen beperking op wie: dit is geen recht dat je weggeeft, het is een
    schakelaar die iets wegláát. De beheerder mag hem op elk account zetten, ook
    op zijn eigen — om te kijken hoe het eruitziet.
    """
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Gebruiker niet gevonden")
        user.is_review = bool(body.value)
        db.commit()
        return {"ok": True, "isReview": user.is_review}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/active")
def set_active(user_id: int, body: FlagBody, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        if not body.value:
            _guard_owner(user)
            if user.id == admin.id:
                raise HTTPException(status_code=400, detail="Kinsto dysels net útskeakelje")
            _guard_last_admin(db, user)
        user.is_active = body.value
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/name")
def set_display_name(user_id: int, body: NameBody, admin: User = Depends(require_admin)):
    """De naam waarmee iemand in de app verschijnt.

    Zonder dit stond er het e-mailadres: wie is uitgenodigd en nooit een naam
    heeft ingevuld heet "someone@example.com" — in de deelnemerslijst van samen
    luisteren, op de kaart "luistert nu ook", overal. Dat is geen naam maar een
    adres, en de beheerder kon er niets aan doen.
    """
    naam = (body.name or "").strip()[:60]
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        # Leeg maken mag: dan valt de app terug op het deel vóór de @.
        user.display_name = naam or None
        db.commit()
        return {"ok": True, "displayName": user.display_name}
    finally:
        db.close()


@router.delete("/admin/users/{user_id}")
def delete_user(user_id: int, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        _guard_owner(user)
        if user.id == admin.id:
            raise HTTPException(status_code=400, detail="Kinsto dysels net wiskje")
        _guard_last_admin(db, user)
        db.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
        db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.user_id == user_id))
        db.execute(delete(LikedTrack).where(LikedTrack.user_id == user_id))
        db.execute(delete(SubsonicKey).where(SubsonicKey.user_id == user_id))
        db.execute(delete(SubsonicStar).where(SubsonicStar.user_id == user_id))
        db.execute(delete(Meta).where(Meta.key == f"subsonic_queue:{user_id}"))
        # A playlist's items and every per-user row must go too, or they orphan.
        pl_ids = [p.id for p in db.execute(
            select(Playlist).where(Playlist.user_id == user_id)).scalars()]
        if pl_ids:
            db.execute(delete(PlaylistItem).where(PlaylistItem.playlist_id.in_(pl_ids)))
        db.execute(delete(Playlist).where(Playlist.user_id == user_id))
        db.execute(delete(UserTrackState).where(UserTrackState.user_id == user_id))
        db.execute(delete(PlayEvent).where(PlayEvent.user_id == user_id))
        db.delete(user)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── import liked/rated tracks from an existing Plex server (READ-ONLY) ────────
class PlexImportBody(BaseModel):
    plex_url: str = "http://192.168.1.10:32400"
    plex_token: str
    min_rating: float = 1.0


@router.post("/admin/import-plex-likes")
def import_plex_likes(body: PlexImportBody, admin: User = Depends(require_admin)):
    """Import the calling admin's Plex track ratings as JLTamp 'likes'. Only ever
    issues GET requests to Plex — it never writes to or modifies Plex."""
    if not config.IMPORTERS:
        raise HTTPException(403, "The importers are disabled on this server")
    import json as _json
    import urllib.parse
    import urllib.request

    base = body.plex_url.rstrip("/")

    def pget(path: str):
        req = urllib.request.Request(
            f"{base}{path}",
            headers={"X-Plex-Token": body.plex_token, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return _json.loads(r.read().decode("utf-8", "replace"))

    try:
        secs = pget("/library/sections").get("MediaContainer", {}).get("Directory", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kin Plex net berikke: {e}")
    music_keys = [d["key"] for d in secs if d.get("type") == "artist"]

    liked: list[tuple[str, str]] = []
    rq = urllib.parse.quote(f"userRating>={body.min_rating}", safe=">=")
    for k in music_keys:
        start = 0
        while True:
            try:
                data = pget(f"/library/sections/{k}/all?type=10&{rq}"
                            f"&X-Plex-Container-Start={start}&X-Plex-Container-Size=200")
            except Exception:
                break
            mc = data.get("MediaContainer", {})
            tracks = mc.get("Metadata", []) or []
            for t in tracks:
                liked.append((t.get("title", ""), t.get("grandparentTitle", "") or t.get("originalTitle", "")))
            start += len(tracks)
            if len(tracks) < 200 or start >= int(mc.get("totalSize", start) or start):
                break

    def norm(s: str) -> str:
        return (s or "").strip().lower()

    db = SessionLocal()
    try:
        index: dict[tuple, int] = {}
        title_only: dict[str, int] = {}
        for tr in db.execute(select(Track)).scalars():
            nt = norm(tr.title)
            title_only[nt] = tr.id
            for a in {tr.artist_name, tr.orig_artist, tr.album_artist}:
                if a:
                    index[(nt, norm(a))] = tr.id

        matched: set[int] = set()
        for title, artist in liked:
            nt = norm(title)
            tid = index.get((nt, norm(artist))) or title_only.get(nt)
            if tid:
                matched.add(tid)

        # Import into the "Liked Songs" PLAYLIST, because that is what the app
        # renders. Writing only `liked_tracks` is why an earlier import landed
        # invisibly and needed reconcile_liked_playlists.py afterwards — the rows
        # were there, the user just never saw them. The table is still mirrored
        # so it does not drift out of step with the playlist.
        from .likes import liked_playlist

        pl = liked_playlist(db, admin.id, create=True)
        in_playlist = {r.track_id for r in db.execute(
            select(PlaylistItem).where(PlaylistItem.playlist_id == pl.id)).scalars()}
        in_table = {lt.track_id for lt in db.execute(
            select(LikedTrack).where(LikedTrack.user_id == admin.id)).scalars()}
        pos = db.execute(
            select(PlaylistItem.position).where(PlaylistItem.playlist_id == pl.id)
            .order_by(PlaylistItem.position.desc())).scalars().first()
        pos = (pos + 1) if pos is not None else 0
        now = int(time.time())
        added = 0
        for tid in matched:
            if tid not in in_playlist:
                db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos,
                                    added_by=admin.id, added_at=int(time.time())))
                pos += 1
                added += 1
            if tid not in in_table:
                db.add(LikedTrack(user_id=admin.id, track_id=tid, created_at=now))
        if added:
            pl.updated_at = now
        db.commit()
        return {"plexRated": len(liked), "matched": len(matched), "imported": added}
    finally:
        db.close()
