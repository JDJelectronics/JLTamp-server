"""Plex-compatible library browsing, metadata, genres and search — now
multi-library and per-user access filtered. Each JLTamp library is exposed as a
Plex 'section'; a user only sees the sections (and their content) they've been
granted. Admins see everything.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, Depends, Query, HTTPException
from sqlalchemy import select, func, or_

from .. import config
from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key
from ..models import Artist, Album, Track, Library, User
from ..serializers import artist_dict, album_dict, track_dict, track_states, container

router = APIRouter()


# ── access helpers ───────────────────────────────────────────────────────────
def _allowed(db, user: User) -> list[int] | None:
    """Library ids the user may see; None = all (admin)."""
    return accessible_library_ids(db, user)


def _scope(stmt, model, allowed: list[int] | None, section: str | None = None):
    """Restrict a query to the user's accessible libraries (and an optional
    single section/library key)."""
    if allowed is not None:
        stmt = stmt.where(model.library_id.in_(allowed or [-1]))
    if section and section.isdigit():
        stmt = stmt.where(model.library_id == int(section))
    return stmt


def _can_see(db, user: User, library_id: int) -> bool:
    allowed = _allowed(db, user)
    return allowed is None or library_id in allowed


# ── identity (also the reachability probe) ───────────────────────────────────
@router.get("/identity")
def identity(_: User = Depends(require_user)):
    return {"MediaContainer": {
        "machineIdentifier": config.SERVER_ID,
        "friendlyName": config.SERVER_NAME,
        "version": "0.2.0",
    }}


# ── sections = the user's accessible libraries ───────────────────────────────
@router.get("/library/sections")
def sections(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        libs = list(db.execute(select(Library).order_by(Library.id)).scalars())
        directory = []
        for lib in libs:
            if allowed is not None and lib.id not in allowed:
                continue
            directory.append({
                "key": str(lib.id),
                "type": "artist",       # a music library is type 'artist' in Plex
                "title": lib.name,
                "agent": "jltamp",
                "scanner": "JLTamp Music",
            })
        return {"MediaContainer": {"size": len(directory), "Directory": directory}}
    finally:
        db.close()


def _apply_genre(stmt, model, genre: str | None):
    if genre:
        stmt = stmt.where(func.lower(model.genre) == genre.strip().lower())
    return stmt


# ── sorting (Plex `sort=field:dir`) ──────────────────────────────────────────
# Plex clients sort a section by passing e.g. sort=addedAt:desc. Unknown fields
# fall back to the natural order rather than erroring, which is what Plex does.
_SORT_FIELDS = {
    8: {  # artists
        "titleSort": (Artist.sort_name, Artist.name),
        "title": (Artist.name,),
        "addedAt": (Artist.id,),
    },
    9: {  # albums
        "titleSort": (Album.sort_title,),
        "title": (Album.title,),
        "addedAt": (Album.added_at,),
        "year": (Album.year,),
        "artist": (Album.artist_name, Album.sort_title),
    },
    10: {  # tracks
        "titleSort": (Track.sort_title, Track.title),
        "title": (Track.title,),
        "addedAt": (Track.added_at,),
        "year": (Track.year,),
        "artist": (Track.artist_name, Track.album_title, Track.disc_no, Track.track_no),
        "album": (Track.album_title, Track.disc_no, Track.track_no),
        "duration": (Track.duration_ms,),
    },
}
_DEFAULT_ORDER = {
    8: (Artist.sort_name, Artist.name),
    9: (Album.artist_name, Album.sort_title),
    10: (Track.artist_name, Track.album_title, Track.disc_no, Track.track_no, Track.title),
}


def _apply_sort(stmt, type: int, sort: str | None):
    fields = _SORT_FIELDS.get(type, {})
    if sort:
        name, _, direction = sort.partition(":")
        cols = fields.get(name.strip())
        if cols:
            desc = direction.strip().lower().startswith("desc")
            order = []
            for c in cols:
                # Untagged rows (no year) sort LAST either way — a year-sorted
                # list that opens with 200 "unknown" albums is useless. DESC puts
                # NULLs last already in SQLite; ASC needs the nudge.
                if not desc and c.nullable:
                    order.append(c.is_(None))
                order.append(c.desc() if desc else c.asc())
            return stmt.order_by(*order)
    return stmt.order_by(*_DEFAULT_ORDER[type])


def _apply_year(stmt, model, year: int | None, decade: int | None):
    if year:
        stmt = stmt.where(model.year == year)
    if decade:
        stmt = stmt.where(model.year >= decade, model.year < decade + 10)
    return stmt


def _int_param(raw, default=None):
    """A paging parameter is whatever the caller typed — this is a Plex-compatible
    API, so the caller is not always our own app. A bare int() on it turned any
    non-numeric `X-Plex-Container-Start` into an unhandled ValueError, i.e. a 500
    on a malformed request. Fall back to the default instead."""
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _page(db, stmt, start: int, size: int | None):
    if start:
        stmt = stmt.offset(start)
    if size is not None:
        stmt = stmt.limit(size)
    return list(db.execute(stmt).scalars())


# Namen die geen artiest zijn maar een bak waar verzamelalbums onder hangen.
#
# Bij het bladeren op genre stond deze altijd bovenaan, bij élk genre: op een
# verzamelaar staan nu eenmaal nummers van alle soorten, dus telt hij overal het
# zwaarst mee. Dat maakte de rij artiesten onder "Rock" en onder "Kids" precies
# even nietszeggend. De platen zelf blijven gewoon vindbaar — die staan bij de
# albums op datzelfde scherm.
GEEN_ARTIEST = (
    "various artists", "various", "va", "diverse artiesten",
    "verschillende artiesten", "unknown artist", "unknown",
)


# ── /library/sections/{key}/all — artists(8) / albums(9) / tracks(10) ────────
@router.get("/library/sections/{key}/all")
def section_all(request: Request, key: str, user: User = Depends(require_user),
                type: int = 8, genre: str | None = None, limit: int | None = None,
                sort: str | None = None, year: int | None = None,
                decade: int | None = None, artist: str | None = None):
    start = _int_param(request.query_params.get("X-Plex-Container-Start"), 0) or 0
    size = _int_param(request.query_params.get("X-Plex-Container-Size"), limit or None)
    # A negative start is meaningless — clamp it. A negative size used to reach
    # SQLite as `LIMIT -1`, which it reads as "no limit", so keep that meaning:
    # silently serving an EMPTY page would be a worse answer than the old one.
    start = max(0, start)
    if size is not None and size < 0:
        size = limit or None

    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        if key and key.isdigit() and not _can_see(db, user, int(key)):
            raise HTTPException(403, "Gjin tagong")
        if type == 8:  # artists
            base = _scope(select(Artist), Artist, allowed, key)
            count = _scope(select(func.count(Artist.id)), Artist, allowed, key)
            if genre:
                # ⚠️ Hier werd `genre` gewoon GENEGEERD.
                #
                # Tik je in het zoekscherm op "Rock", dan vroeg de app de
                # artiesten van dat genre op — en kreeg hij de eerste twintig
                # artiesten van de hele bibliotheek terug, op alfabet. Dus stond
                # er onder elk genre precies dezelfde rij namen. Dat was niet
                # "matig gefilterd" maar helemaal niet gefilterd: de parameter
                # kwam binnen en werd nergens gebruikt.
                #
                # Een artiest heeft zelf geen genre in de database; zijn nummers
                # wel. Dus wordt er geteld hoeveel nummers van dít genre bij hem
                # horen, en dat bepaalt meteen de volgorde: wie het meest van dit
                # genre heeft staat bovenaan. Bij "Hardstyle" is dat de artiest
                # van wie je twintig hardstyle-nummers hebt, en niet degene die
                # toevallig met een A begint.
                g = genre.strip().lower()
                tel = (
                    select(Track.artist_id.label("aid"),
                           func.count(Track.id).label("n"))
                    .where(func.lower(Track.genre) == g)
                    .group_by(Track.artist_id)
                    .subquery()
                )
                base = base.join(tel, tel.c.aid == Artist.id) \
                    .where(func.lower(Artist.name).notin_(GEEN_ARTIEST))
                count = count.join(tel, tel.c.aid == Artist.id) \
                    .where(func.lower(Artist.name).notin_(GEEN_ARTIEST))
                total = db.execute(count).scalar() or 0
                # Een uitdrukkelijke sorteerwens van de app gaat vóór; staat er
                # niets, dan die van hierboven.
                base = (_apply_sort(base, 8, sort) if sort
                        else base.order_by(tel.c.n.desc(), Artist.sort_name))
                items = [artist_dict(a) for a in _page(db, base, start, size)]
            else:
                total = db.execute(count).scalar() or 0
                items = [artist_dict(a) for a in _page(db, _apply_sort(base, 8, sort), start, size)]
        elif type == 9:  # albums
            def narrow(s):
                s = _scope(s, Album, allowed, key)
                if genre:
                    # Ook hier ruimer dan eerst. Er stond `lower(Album.genre) ==
                    # genre`: alleen albums waarvan de tag van het ALBUM precies
                    # zo heet. Een verzamelaar staat vaak onder "Various" en een
                    # album onder "Pop" terwijl de nummers "Rock" zijn — dan
                    # vond je dat album niet, terwijl het er wel degelijk bij
                    # hoort. Nu telt het album mee als zijn eigen tag klopt óf
                    # als er nummers van dit genre op staan.
                    g = genre.strip().lower()
                    van_nummers = select(Track.album_id).where(func.lower(Track.genre) == g)
                    s = s.where(or_(func.lower(Album.genre) == g,
                                    Album.id.in_(van_nummers)))
                s = _apply_year(s, Album, year, decade)
                if artist:
                    s = s.where(func.lower(Album.artist_name) == artist.strip().lower())
                return s
            total = db.execute(narrow(select(func.count(Album.id)))).scalar() or 0
            base = _apply_sort(narrow(select(Album)), 9, sort)
            items = [album_dict(a) for a in _page(db, base, start, size)]
        else:  # tracks (10)
            def narrow(s):
                s = _apply_year(_apply_genre(_scope(s, Track, allowed, key), Track, genre),
                                Track, year, decade)
                if artist:
                    s = s.where(func.lower(Track.artist_name) == artist.strip().lower())
                return s
            total = db.execute(narrow(select(func.count(Track.id)))).scalar() or 0
            base = _apply_sort(narrow(select(Track)), 10, sort)
            rows = _page(db, base, start, size)
            states = track_states(db, user.id, rows)
            items = [track_dict(t, state=states.get(t.id)) for t in rows]
        return container(items, total=total)
    finally:
        db.close()


# ── recently added ───────────────────────────────────────────────────────────
@router.get("/library/sections/{key}/recentlyAdded")
def recently_added(key: str, user: User = Depends(require_user), type: int = 10, limit: int = 30):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        if type == 9:
            base = _scope(select(Album), Album, allowed, section).order_by(Album.added_at.desc()).limit(limit)
            items = [album_dict(a) for a in db.execute(base).scalars()]
        else:
            base = _scope(select(Track), Track, allowed, section).order_by(Track.added_at.desc()).limit(limit)
            rows = list(db.execute(base).scalars())
            states = track_states(db, user.id, rows)
            items = [track_dict(t, state=states.get(t.id)) for t in rows]
        return container(items)
    finally:
        db.close()


# ── genres ───────────────────────────────────────────────────────────────────
#
# Twee dingen die onthouden worden, en waarom.
#
# **Staat het bestand er ook echt?** In de database staat bij een album een pad
# naar een hoes, maar dat pad kan wijzen naar iets dat er niet meer is — een
# album dat opnieuw getagd werd, een hoes die is opgeruimd. Dan geeft /art een
# 404 en staat er op het zoekscherm een leeg vlak; "Pop" was zo'n geval. Dus
# wordt er gekeken voor we een hoes beloven.
#
# Maar die hoezen staan op de NAS, via NFS, en daar kost één keer kijken zes
# milliseconden. Twaalfhonderd keer kijken is zeven seconden, en dat is wat het
# zoekscherm ging kosten bij het openen. Vandaar deze twee lagen:
#
#   `_BESTAAT`  — per pad, zolang de server draait. Een pad dat er niet is komt
#                 er ook niet meer; en een hoes die er later bijkomt krijgt een
#                 ánder pad (de verrijking schrijft naar /data), dus die valt er
#                 niet onder.
#   `_GENRES`   — het hele antwoord, tien minuten. Genres veranderen alleen na
#                 een scan, en dan is tien minuten wachten niet erg.
_BESTAAT: dict[str, bool] = {}
_GENRE_CACHE_S = 600
_GENRES: dict[tuple, tuple[float, dict]] = {}


def _bestaat(pad: str) -> bool:
    if pad not in _BESTAAT:
        _BESTAAT[pad] = os.path.exists(pad)
    return _BESTAAT[pad]


@router.get("/library/sections/{key}/genre")
def genres(key: str, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        # Per combinatie van toegang: twee gebruikers met verschillende
        # bibliotheken krijgen verschillende genres, dus die mogen niet
        # elkaars antwoord terugkrijgen.
        cache_sleutel = (tuple(sorted(allowed)) if allowed is not None else None, section)
        bewaard = _GENRES.get(cache_sleutel)
        if bewaard and (time.time() - bewaard[0]) < _GENRE_CACHE_S:
            return bewaard[1]
        stmt = _scope(select(Track.genre, func.count(Track.id)), Track, allowed, section) \
            .where(Track.genre != "").group_by(func.lower(Track.genre)).order_by(func.lower(Track.genre))
        rows = db.execute(stmt).all()

        # Een hoes per genre, zodat de tegels in het zoekscherm een plaat laten
        # zien in plaats van een gekleurd vlakje.
        #
        # Hier stond `min(Album.id)`: de laagste albumnummer van dat genre. Dat
        # is deterministisch en dus rustig, maar het gaf op het scherm zes keer
        # dezelfde foto — één oude verzamelaar met nummers uit tien genres is
        # voor al die genres het laagste nummer, en dus de hoes van allemaal.
        #
        # Nu wordt er gekozen op wat een genre écht vertegenwoordigt: het album
        # met de MEESTE nummers in dat genre, en bij voorkeur een album waarvan
        # het genre zélf klopt (dat sluit de verzamelaars uit, die staan onder
        # "Various" of hebben een ander genre dan het nummer). En als twee
        # genres op hetzelfde album uitkomen, wijkt de tweede uit naar zijn
        # volgende keuze — een rij tegels met steeds dezelfde foto is geen rij.
        art_stmt = _scope(
            select(func.lower(Track.genre), Album.id, func.lower(Album.genre),
                   func.count(Track.id), Album.art_path, Album.online_art_path),
            Track, allowed, section,
        ) \
            .join(Album, Album.id == Track.album_id) \
            .where(Track.genre != "") \
            .where(Album.art_path.isnot(None) | Album.online_art_path.isnot(None)) \
            .group_by(func.lower(Track.genre), Album.id)

        kandidaten: dict[str, list[tuple]] = {}
        paden: dict[int, str] = {}
        for g, aid, albumgenre, aantal, pad, online in db.execute(art_stmt).all():
            # Sorteersleutel: eerst of het albumgenre klopt, dan hoeveel nummers
            # van dit genre erop staan, dan het nummer zodat het niet wisselt
            # zonder reden.
            kandidaten.setdefault(g, []).append(
                (0 if (albumgenre or "") == g else 1, -aantal, aid)
            )
            # Zelfde volgorde als resolve_art_path in artwork.py.
            paden[aid] = pad or online or ""

        def _heeft_hoes(aid: int) -> bool:
            pad = paden.get(aid) or ""
            return bool(pad) and _bestaat(pad)

        # Eerst allemaal tegelijk kijken, dan pas kiezen.
        #
        # Nakijken of een hoes bestaat kost over NFS zo'n zes milliseconden, en
        # dat is wachten op het netwerk, niet rekenen. Achter elkaar duurde het
        # daardoor vijf seconden voordat het zoekscherm iets liet zien. Zestien
        # tegelijk vragen maakt daar een halve seconde van — en daarna zit het
        # in `_BESTAAT` en hoeft het nooit meer.
        #
        # Alleen de bovenste paar per genre: verder dan dat komt de keuze
        # hieronder toch niet.
        vooruit = {
            paden.get(aid)
            for g in kandidaten
            for _, _, aid in sorted(kandidaten[g])[:6]
        }
        vooruit = {p for p in vooruit if p and p not in _BESTAAT}
        if vooruit:
            with ThreadPoolExecutor(max_workers=16) as pool:
                for pad, er in zip(vooruit, pool.map(os.path.exists, vooruit)):
                    _BESTAAT[pad] = er

        # Niet één hoes maar een handjevol, en allemaal verschillend.
        #
        # De tegel op het zoekscherm wisselt ze langzaam af. Eerst was alleen de
        # EERSTE hoes per genre uniek en mochten de andere gedeeld worden — "die
        # komt toch pas over een halve minuut langs". Dat klopte niet: je kijkt
        # naar twaalf tegels tegelijk, dus je ziet die halve minuut gewoon
        # gebeuren, en dan staat dezelfde plaat op twee plekken.
        #
        # Nu wordt élke hoes maar één keer vergeven, over alle genres heen — en
        # op het PAD en niet op het albumnummer, want twee albums kunnen dezelfde
        # afbeelding delen en dan zie je hem alsnog dubbel.
        #
        # De grootste genres eerst. Eerst stond hier "de genres met de minste
        # keus eerst", met als gedachte: een genre met precies één album mag dat
        # album hebben. Netjes bedacht, maar het verkeerde doel — het scherm laat
        # de dertig grootste genres zien, en juist díe kwamen als laatste aan de
        # beurt en moesten het doen met wat er nog lag.
        # Zes per genre: vier staan er tegelijk op de tegel (een mozaïek van
        # twee bij twee) en twee liggen klaar als wisselgeld — elke paar tellen
        # vervangt er één een vakje. Meer dan zes heeft geen zin: dan is elke
        # hoes zo kort in beeld dat het onrustig wordt, en het kost per genre
        # weer een paar hoezen die een ander genre dan niet meer heeft.
        PER_GENRE = 6
        telling = {g.lower(): c for g, c in rows if g}
        art_map: dict[str, list[str]] = {}
        bezet: set[str] = set()

        for g in sorted(kandidaten, key=lambda k: -telling.get(k, 0)):
            rij = [aid for _, _, aid in sorted(kandidaten[g])]
            # Stoppen zodra we er genoeg hebben. Hier stond een lijstcomprehensie
            # met de bestandscontrole in de voorwaarde, en die liep élk album van
            # het genre na — voor "Pop" zijn dat er honderden, en elke keer kijken
            # kost over NFS twee milliseconden. Daar ging het openen van het
            # zoekscherm zes seconden van duren.
            gekozen: list[str] = []
            for aid in rij:
                pad = paden.get(aid) or ""
                if not pad or pad in bezet or not _bestaat(pad):
                    continue
                bezet.add(pad)
                gekozen.append(f"/art/al{aid}")
                if len(gekozen) >= PER_GENRE:
                    break
            if gekozen:
                art_map[g] = gekozen

        # ── Te weinig? Dan de hoezen van de nummers zelf erbij ──
        #
        # Een genre als "Other" heeft tweeduizend nummers maar staat op twee
        # albums mét hoes, en die waren allebei al vergeven. Dan viel het terug op
        # een gedeelde plaat, en stonden er vier tegels met precies hetzelfde
        # beeld — waar het hier nou juist om begonnen was.
        #
        # Een nummer kan zijn eigen afbeelding hebben (uit de tag), en dat is een
        # veel grotere vijver. Alleen voor de genres die tekortkomen, en op het
        # PAD ontdubbeld, zodat vijftig nummers van hetzelfde album niet vijftig
        # keer dezelfde hoes opleveren.
        tekort = [g for g in kandidaten if len(art_map.get(g, [])) < PER_GENRE]
        if tekort:
            t_stmt = _scope(
                select(func.lower(Track.genre), func.min(Track.id), Track.art_path),
                Track, allowed, section,
            ) \
                .where(Track.art_path.isnot(None)) \
                .where(func.lower(Track.genre).in_(tekort)) \
                .group_by(func.lower(Track.genre), Track.art_path)
            los: dict[str, list[tuple[int, str]]] = {}
            for g, tid, pad in db.execute(t_stmt).all():
                if pad:
                    los.setdefault(g, []).append((tid, pad))
            for g in tekort:
                heeft = art_map.setdefault(g, [])
                # Niet eindeloos blijven kijken: bij een genre met duizenden
                # nummers is de vraag "bestaat dit bestand" duur, en twintig
                # pogingen is ruim genoeg om er drie te vinden.
                pogingen = 0
                for tid, pad in los.get(g, []):
                    if len(heeft) >= PER_GENRE or pogingen >= 30:
                        break
                    if pad in bezet:
                        continue
                    pogingen += 1
                    if not _bestaat(pad):
                        continue
                    bezet.add(pad)
                    heeft.append(f"/art/t{tid}")
                if not heeft:
                    art_map.pop(g, None)

        directory = []
        for g, c in rows:
            if not g or len(g) < 2:
                continue
            entry = {"key": g, "title": g, "count": c}
            refs = art_map.get(g.lower()) or []
            if refs:
                # `thumb` blijft er staan voor wie er maar één verwacht.
                entry["thumb"] = refs[0]          # → getThumbnailUrl on the client
                entry["thumbs"] = list(refs)
            directory.append(entry)
        antwoord = {"MediaContainer": {"size": len(directory), "Directory": directory}}
        _GENRES[cache_sleutel] = (time.time(), antwoord)
        return antwoord
    finally:
        db.close()


# ── folder view — expose artists as browsable folders ────────────────────────
@router.get("/library/sections/{key}/folder")
def folder(key: str, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        rows = db.execute(_scope(select(Artist), Artist, allowed, section)
                          .order_by(Artist.sort_name)).scalars()
        return container([artist_dict(a) for a in rows])
    finally:
        db.close()


# ── single metadata + children (access-checked) ──────────────────────────────
@router.get("/library/metadata/{rk}")
def metadata(rk: str, user: User = Depends(require_user)):
    # Meerdere sleutels in één keer, zoals Plex: /library/metadata/t1,t2,t3.
    # De app haalt zo de nummers van een radiozender op. Zonder dit gaf de server
    # 200 met een lege lijst, en meldde elke "Start radio" dat er geen afspeelbare
    # nummers waren. Volgorde blijft die van de vraag; wat je niet mag zien valt weg.
    if "," in rk:
        return _metadata_many([k for k in rk.split(",") if k][:500], user)
    parsed = parse_key(rk)
    db = SessionLocal()
    try:
        if parsed:
            kind, i = parsed
            if kind == "track":
                t = db.get(Track, i)
                if t and _can_see(db, user, t.library_id):
                    states = track_states(db, user.id, [t])
                    return container([track_dict(t, state=states.get(t.id))])
            elif kind == "album":
                a = db.get(Album, i)
                if a and _can_see(db, user, a.library_id):
                    return container([album_dict(a)])
            elif kind == "artist":
                a = db.get(Artist, i)
                if a and _can_see(db, user, a.library_id):
                    return container([artist_dict(a)])
        return container([])
    finally:
        db.close()


def _metadata_many(keys: list[str], user: User) -> dict:
    """Metadata voor een lijst sleutels ("t1", "al2", "ar3"), in die volgorde."""
    db = SessionLocal()
    try:
        soort = {"track": Track, "album": Album, "artist": Artist}
        ids: dict[str, list[int]] = {"track": [], "album": [], "artist": []}
        volgorde: list[tuple[str, int]] = []
        for k in keys:
            parsed = parse_key(k.strip())
            if parsed and parsed[0] in soort:
                ids[parsed[0]].append(parsed[1])
                volgorde.append(parsed)
        rijen: dict[tuple[str, int], object] = {}
        for kind, lijst in ids.items():
            if lijst:
                model = soort[kind]
                for r in db.execute(select(model).where(model.id.in_(set(lijst)))).scalars():
                    rijen[(kind, r.id)] = r
        zichtbaar: dict[int, bool] = {}

        def mag(lib_id: int) -> bool:
            if lib_id not in zichtbaar:
                zichtbaar[lib_id] = _can_see(db, user, lib_id)
            return zichtbaar[lib_id]

        tracks = [rijen[p] for p in volgorde
                  if p[0] == "track" and p in rijen and mag(rijen[p].library_id)]
        states = track_states(db, user.id, tracks) if tracks else {}
        uit = []
        for p in volgorde:
            r = rijen.get(p)
            if r is None or not mag(r.library_id):
                continue
            if p[0] == "track":
                uit.append(track_dict(r, state=states.get(r.id)))
            elif p[0] == "album":
                uit.append(album_dict(r))
            else:
                uit.append(artist_dict(r))
        return container(uit)
    finally:
        db.close()


@router.get("/library/metadata/{rk}/children")
def metadata_children(rk: str, user: User = Depends(require_user)):
    parsed = parse_key(rk)
    db = SessionLocal()
    try:
        if parsed:
            kind, i = parsed
            if kind == "artist":
                ar = db.get(Artist, i)
                if not ar or not _can_see(db, user, ar.library_id):
                    return container([])
                rows = db.execute(select(Album).where(Album.artist_id == i)
                                  .order_by(Album.year, Album.sort_title)).scalars()
                return container([album_dict(a) for a in rows])
            if kind == "album":
                al = db.get(Album, i)
                if not al or not _can_see(db, user, al.library_id):
                    return container([])
                rows = list(db.execute(select(Track).where(Track.album_id == i)
                                       .order_by(Track.disc_no, Track.track_no, Track.title)).scalars())
                states = track_states(db, user.id, rows)
                return container([track_dict(t, state=states.get(t.id)) for t in rows])
        return container([])
    finally:
        db.close()


# ── search (scoped to accessible libraries) ──────────────────────────────────
#
# Wat hier eerst stond was één regel per soort:
#
#     lower(Track.title) LIKE "%<alles wat je typte>%"
#
# en dat is op drie manieren te weinig.
#
# 1. Een nummer werd alleen op zijn TITEL gevonden. Zoek je op "martin garrix",
#    dan kreeg je de artiest wel te zien maar stond er onder "Nummers" niets —
#    terwijl je daar juist zijn nummers verwacht.
# 2. De hele zoekterm moest letterlijk en aaneengesloten voorkomen. "garrix
#    animals" gaf niets, want zo staat het nergens; "animals garrix" evenmin.
# 3. Er werd niet gerangschikt. Zoek je op "love", dan kwamen er twintig
#    willekeurige nummers terug — niet het nummer dat écht "Love" heet.
#
# En een vierde, die je pas merkt als je hem tegenkomt: accenten. "Beyoncé" en
# "André Hazes" staan met accent in de tags, en niemand typt die. `lower()` in
# SQLite haalt daar geen streepje af.
#
# De opzet nu: de zoekterm wordt in WOORDEN geknipt, en elk woord moet ergens
# voorkomen — in de titel, de artiest, of het album. In welke volgorde je ze
# typt maakt niet uit. Daarna wordt er gerangschikt, zodat wat er het meest op
# lijkt bovenaan staat.


def _vouw(tekst: str) -> str:
    """Alles plat: kleine letters, geen accenten, geen leestekens.

    "Beyoncé — Don't Hurt Yourself" wordt "dont hurt yourself" met "beyonce"
    ervoor. Let op het weglaten van de apostrof in plaats van er een spatie van
    te maken: zo vindt zowel "dont hurt" als "don t hurt" hetzelfde, en dat is
    precies hoe mensen het typen. Streepjes worden wél een spatie, want
    "Sixteen-Year-Old" is drie woorden.
    """
    t = unicodedata.normalize("NFKD", tekst or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# Per kale letter: welke tekens er in de tags voor kunnen staan. Zowel klein als
# groot, want `lower()` van SQLite kent alleen het gewone alfabet — een "É"
# blijft daar een "É".
_ACCENTEN = {
    "a": "aàáâãäåÀÁÂÃÄÅ",
    "e": "eéèêëÉÈÊË",
    "i": "iíìîïÍÌÎÏ",
    "o": "oóòôõöøÓÒÔÕÖØ",
    "u": "uúùûüÚÙÛÜ",
    "y": "yýÿÝ",
    "c": "cçÇ",
    "n": "nñÑ",
    "s": "sšŠ",
    "z": "zžŽ",
}


def _patroon(woord: str) -> str:
    """Een patroon dat door een accent heen kijkt.

    SQLite kan zelf geen accenten wegdenken, en honderdduizend nummers rij voor
    rij door Python halen is geen optie. `GLOB` wél: dat kent, anders dan `LIKE`,
    tekenklassen. Van "beyonce" wordt

        *b[eéèêëÉÈÊË]yonc[eéèêëÉÈÊË]*

    en dat vindt "Beyoncé" én "Beyonce", zonder dat het verder iets extra's
    binnenlaat. Eerder stond hier een LIKE met `_` op elke klinker; dat werd
    `%b______%` en dat is geen zeef meer.

    Wat hier NIET doorheen komt: een leesteken midden in een woord dat je zelf
    niet typt. Zoek je op "dont", dan staat er in de tag "Don't" en dat vindt hij
    niet. Typ je "don" of "don t", dan wel. Dat is bewust: de rek die daarvoor
    nodig is maakt de zeef zo ruim dat de goede treffer eruit kan vallen.
    """
    return "*" + "".join(
        f"[{_ACCENTEN[c]}]" if c in _ACCENTEN else c for c in woord
    ) + "*"


def _woorden(query: str) -> list[str]:
    """De zoekterm in losse woorden, hooguit zes. Meer is geen zoekopdracht."""
    return [w for w in _vouw(query).split(" ") if w][:6]


def _elk_woord(woorden: list[str], velden) -> list:
    """Per woord: het moet in MINSTENS ÉÉN van de velden zitten.

    De voorwaarden worden met EN aan elkaar geregen — twee woorden die allebei
    ergens voorkomen. Vandaar dat "garrix animals" en "animals garrix" hetzelfde
    vinden, en dat een derde woord de lijst korter maakt in plaats van langer.
    """
    uit = []
    for w in woorden:
        pat = _patroon(w)
        uit.append(or_(*[func.lower(v).op("GLOB")(pat) for v in velden]))
    return uit


def _score(naam: str, omgeving: str, woorden: list[str], hele: str) -> int:
    """Hoe goed past dit bij wat er getypt is? Hoger is beter.

    De volgorde die eruit komt is de volgorde die je verwacht: eerst wat exact
    zo heet, dan wat ermee begint, dan wat het woord ergens in de titel heeft, en
    pas daarna de treffers die alleen via de artiest of het album binnenkwamen.
    Zoek je op "love", dan staat "Love" boven "Lovely" boven "Love Me Like You
    Do" boven "Crazy Little Thing Called Love".
    """
    n = _vouw(naam)
    if not n:
        return 0
    if n == hele:
        return 1000
    s = 0
    if n.startswith(hele + " "):
        s += 600
    elif n.startswith(hele):
        s += 450
    elif hele and hele in n:
        s += 250
    for w in woorden:
        if re.search(rf"\b{re.escape(w)}\b", n):
            s += 90
        elif re.search(rf"\b{re.escape(w)}", n):
            s += 60
        elif w in n:
            s += 30
        elif re.search(rf"\b{re.escape(w)}\b", omgeving):
            s += 15
    # Bij gelijke stand wint de kortste titel: een korte titel die past is bijna
    # altijd bedoeld, een lange die toevallig het woord bevat bijna nooit.
    s += max(0, 40 - len(n) // 2)
    return s


@router.get("/hubs/search")
def hub_search(query: str = Query(""), user: User = Depends(require_user),
               limit: int = 20, sectionId: str | None = None,
               prefer: str | None = None):
    woorden = _woorden(query)
    hele = " ".join(woorden)
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        if not woorden:
            return {"MediaContainer": {"size": 0, "Hub": []}}

        # Ruim ophalen en daarna zelf rangschikken. De grove zeef hierboven laat
        # te veel door om het aan de database over te laten, en een LIMIT zonder
        # ORDER BY zou juist de beste treffer kunnen afkappen.
        RUIM = 400

        t_stmt = _scope(select(Track), Track, allowed, sectionId)
        for v in _elk_woord(woorden, [Track.title, Track.artist_name,
                                      Track.album_artist, Track.orig_artist,
                                      Track.album_title]):
            t_stmt = t_stmt.where(v)
        kandidaten = list(db.execute(t_stmt.limit(RUIM)).scalars())

        a_stmt = _scope(select(Artist), Artist, allowed, sectionId)
        for v in _elk_woord(woorden, [Artist.name]):
            a_stmt = a_stmt.where(v)
        art_kandidaten = list(db.execute(a_stmt.limit(RUIM)).scalars())

        al_stmt = _scope(select(Album), Album, allowed, sectionId)
        for v in _elk_woord(woorden, [Album.title, Album.artist_name]):
            al_stmt = al_stmt.where(v)
        alb_kandidaten = list(db.execute(al_stmt.limit(RUIM)).scalars())

        # De echte controle. De zeef hierboven keek door accenten heen en liet
        # daardoor ook onzin door; hier moet elk woord er echt in staan.
        def houdt(naam: str, omgeving: str) -> bool:
            hooi = f"{_vouw(naam)} {omgeving}"
            return all(w in hooi for w in woorden)

        def van_deze_artiest(naam: str) -> int:
            """Is de zoekterm de ARTIEST? Dan telt dat zwaar mee.

            Zonder dit kwam er bij "beyonce" bovenaan "Beyoncé Interlude" en
            "46 - Beyoncé - NILE" te staan: nummers met haar naam in de titel,
            van iemand anders. Haar eigen nummers stonden eronder, want daar zit
            de treffer in een veld dat minder telt. Maar wie een artiest intypt
            zoekt die artiest, niet een nummer dat naar hem verwijst.
            """
            n = _vouw(naam or "")
            if not n or not hele:
                return 0
            if n == hele:
                return 400
            if n.startswith(hele + " ") or n.startswith(hele):
                return 200
            return 0

        tracks, artists, albums = [], [], []
        for t in kandidaten:
            omg = _vouw(f"{t.artist_name} {t.album_artist} {t.orig_artist} {t.album_title}")
            if not houdt(t.title, omg):
                continue
            punten = _score(t.title, omg, woorden, hele)
            punten += max(van_deze_artiest(t.artist_name),
                          van_deze_artiest(t.album_artist),
                          van_deze_artiest(t.orig_artist))
            # Vaak gedraaid telt mee, maar licht: het mag een betere titeltreffer
            # niet inhalen.
            punten += min(60, (t.play_count or 0) * 3)
            tracks.append((punten, t))
        for a in art_kandidaten:
            if not houdt(a.name, ""):
                continue
            artists.append((_score(a.name, "", woorden, hele), a))
        for al in alb_kandidaten:
            omg = _vouw(al.artist_name or "")
            if not houdt(al.title, omg):
                continue
            albums.append((_score(al.title, omg, woorden, hele)
                           + van_deze_artiest(al.artist_name), al))

        tracks.sort(key=lambda p: -p[0])
        artists.sort(key=lambda p: -p[0])
        albums.sort(key=lambda p: -p[0])

        # ── Eén regel per nummer ──
        #
        # Zoeken op "martin garrix" gaf "Higher Ground" twee keer, "MAD" twee
        # keer: dezelfde opname, één keer als single en één keer op "Top 100 over
        # 2020". Op dat verzamelalbum staat als artiest "Various Artists" met
        # Martin Garrix als oorspronkelijke artiest, en dus waren het voor de
        # database twee verschillende dingen — terwijl er op het scherm twee keer
        # exact dezelfde regel staat, met dezelfde hoes en dezelfde speelduur.
        #
        # Twee dingen maken het één regel. De sleutel gebruikt de ECHTE artiest
        # (de oorspronkelijke als die er is, anders de albumartiest), zodat de
        # single en de verzamelaarsversie dezelfde sleutel krijgen. En welke van
        # de twee blijft staan, hangt af van `_rang` hieronder.
        voorkeur = int(prefer) if (prefer or "").isdigit() else None

        def _rang(obj) -> int:
            """Welk exemplaar houden we, als er meer van zijn?

            De bibliotheek waar de gebruiker nu in staat gaat voor — dat is de
            knop "Standaard kwaliteit / High-Res Audio", en een nummer dat er als
            MP3 én als FLAC staat hoort in de kwaliteit te komen die je koos.
            Daarna: liever niet van een verzamelalbum. Een single is het nummer
            zelf; op "Top 100 over 2020" is het een gast.
            """
            r = 0
            if voorkeur is not None and getattr(obj, "library_id", None) == voorkeur:
                r += 4
            if not (getattr(obj, "orig_artist", "") or "").strip():
                r += 2
            return r

        def _een_per_stuk(paren, sleutel):
            beste: dict[str, tuple] = {}
            volgorde: list[str] = []
            for punten, obj in paren:
                k = sleutel(obj)
                if k not in beste:
                    beste[k] = (punten, obj)
                    volgorde.append(k)
                    continue
                staand = beste[k][1]
                if _rang(obj) > _rang(staand):
                    # De punten van de eerste blijven staan: anders zou het
                    # nummer in de lijst verspringen omdat er een ánder bestand
                    # van gekozen is, en daar is de volgorde niet van.
                    beste[k] = (beste[k][0], obj)
            return [beste[k] for k in volgorde]

        def _echte_artiest(t) -> str:
            return _vouw((t.orig_artist or "").strip() or t.artist_name or t.album_artist)

        tracks = _een_per_stuk(tracks, lambda t: f"{_vouw(t.title)}|{_echte_artiest(t)}")
        albums = _een_per_stuk(albums, lambda a: f"{_vouw(a.title)}|{_vouw(a.artist_name)}")
        # Artiesten net zo: dezelfde naam staat in beide bibliotheken, en dan
        # kreeg je "Adele" twee keer onder elkaar — dat leest als twee
        # verschillende artiesten.
        artists = _een_per_stuk(artists, lambda a: _vouw(a.name))
        tracks = [t for _, t in tracks[:limit]]
        artists = [a for _, a in artists[:limit]]
        albums = [al for _, al in albums[:limit]]

        tstates = track_states(db, user.id, tracks)
        hubs = [
            {"type": "track", "Metadata": [track_dict(t, state=tstates.get(t.id)) for t in tracks]},
            {"type": "artist", "Metadata": [artist_dict(a) for a in artists]},
            {"type": "album", "Metadata": [album_dict(a) for a in albums]},
        ]
        # Niets gevonden? Dan onthouden we waar naar gezocht werd, zodat de
        # eigenaar later ziet wat er in de collectie ontbreekt. Zie wishlist.py
        # voor het samenvoegen van half getypte zoekopdrachten.
        if not any(h["Metadata"] for h in hubs):
            from .wishlist import noteer_misser
            noteer_misser(db, user.id, query)
        return {"MediaContainer": {"size": len(hubs), "Hub": hubs}}
    finally:
        db.close()


@router.get("/library/sections/{key}/search")
def section_search(key: str, query: str = Query(""), user: User = Depends(require_user)):
    q = f"%{query.strip().lower()}%"
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        tracks = list(db.execute(_scope(select(Track), Track, allowed, section)
                                 .where(func.lower(Track.title).like(q)).limit(30)).scalars())
        artists = db.execute(_scope(select(Artist), Artist, allowed, section)
                             .where(func.lower(Artist.name).like(q)).limit(15)).scalars()
        albums = db.execute(_scope(select(Album), Album, allowed, section)
                            .where(func.lower(Album.title).like(q)).limit(15)).scalars()
        states = track_states(db, user.id, tracks)
        items = ([track_dict(t, state=states.get(t.id)) for t in tracks]
                 + [artist_dict(a) for a in artists] + [album_dict(a) for a in albums])
        return container(items)
    finally:
        db.close()
