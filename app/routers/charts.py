"""De lijsten van buiten — wekelijks, voor iedereen zichtbaar.

Er zijn er twee — zie EXTERNE_LIJSTEN:

  top40   de Nederlandse Top 40, van top40.nl
  viraal  wat er op dit moment rondgaat

Hoe het werkt en waarom het zo verdeeld is:

Hoe het werkt:

  1. Een script buiten de server zorgt dat de lijst hier binnenkomt (POST
     /charts/lijst/{feed}). Viraal kan de server zelf ophalen (/ophalen), want
     die bron is een Spotify-lijst en die leest de importer al.
  2. Hier wordt elke regel tegen de bibliotheek gelegd (zie matching.py) en de
     bijbehorende openbare lijst opnieuw gevuld, in lijstvolgorde.
  3. Wat er niet is gaat als antwoord terug (`missing`). Komt het later alsnog
     in de bibliotheek, dan vult /refresh de lijst opnieuw uit de bewaarde namen.

De lijsten worden elke week opnieuw gevuld in plaats van bijgewerkt: een
hitlijst is een momentopname, en wat eruit valt hoort eruit te vallen. De lijst
zelf (en dus de ratingKey die in de app bekend is) blijft wel bestaan.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy import Integer, String, func, select

from ..db import SessionLocal
from ..deps import require_admin, require_user, optional_user, accessible_library_ids
from ..ids import playlist_key
from ..matching import match_lijst
from ..serializers import container, track_dict, track_states
from ..models import (
    LikedTrack, Meta, PlayEvent, Playlist, PlaylistItem, Track, User,
)

router = APIRouter(tags=["charts"])

# De lijst wordt hieraan herkend, niet aan zijn titel: de eigenaar mag hem
# hernoemen zonder dat de wekelijkse ronde zijn eigen lijst kwijt is.
FEED_TOP40 = "top40"
STANDAARD_TITEL = "JLTamp Top 40"
# Meer dan dit accepteren we niet in één keer: dit is een hitlijst, geen import.
MAX_REGELS = 100
# Waar de laatst binnengekomen lijst bewaard blijft (artiest + titel per positie).
#
# Waarom dat nodig is: een playlist verwijst naar nummer-id's, en de scanner mag
# die vervangen — een bestand hernoemd, een map opnieuw ingelezen, en de rij is
# een andere rij. Dat gebeurde de eerste avond al: na een scan stonden er van de
# negenendertig nog achttien in de lijst, met gaten in de volgorde. De lijst met
# namen gaat daar niet stuk, dus die bewaren we en leggen we opnieuw naast de
# bibliotheek zodra dat nodig is (zie /charts/top40/refresh).
META_TOP40 = "chart_top40"

# Sinds er een tweede lijst van buiten komt, staat hier wélke dat zijn.
#
# Ze werken allemaal hetzelfde: een script buiten de server haalt ze op, POST ze
# hierheen, en hier worden ze naast de bibliotheek gelegd. Het verschil zit
# alleen in de bron en dus in de naam. Eén register in plaats van een tweede
# kopie van dit bestand: elke lijst die erbij komt is één regel.
#
#   feed → (standaardtitel, waar de namen bewaard blijven)
FEED_VIRAAL = "viraal"
EXTERNE_LIJSTEN: dict[str, tuple[str, str]] = {
    FEED_TOP40: (STANDAARD_TITEL, META_TOP40),
    FEED_VIRAAL: ("Viraal", "chart_viraal"),
}
# Waar Viraal vandaan komt.
#
# Dit is Spotify's eigen "Viral Hits" — de lijst die bijhoudt wat er op dit
# moment rondgaat, en dat is in de praktijk wat er op TikTok gebeurt. Het is
# met opzet niet TikTok zelf: die hebben geen muziek-API, hun geluiden zijn
# versneld en gepitcht, en hun merk is hun merk. Dit is dezelfde inhoud uit een
# bron die we wél mogen lezen en die we al kunnen lezen.
#
# Instelbaar, zodat er geen nieuwe versie van de server nodig is als Spotify
# deze lijst ooit hernoemt of opheft.
BRON_VIRAAL = "https://open.spotify.com/playlist/37i9dQZF1DX2L0iB23Enbq"
META_VIRAAL_BRON = "chart_viraal_bron"


def _openbare_lijst(db, eigenaar: User, titel: str, feed: str = FEED_TOP40) -> Playlist:
    """De ene lijst van deze feed, of maak hem aan."""
    pl = db.execute(
        select(Playlist).where(Playlist.feed == feed).order_by(Playlist.id)
    ).scalars().first()
    if pl:
        return pl
    nu = int(time.time())
    pl = Playlist(user_id=eigenaar.id, title=titel, created_at=nu, updated_at=nu,
                  is_public=True, feed=feed)
    db.add(pl)
    db.flush()
    return pl


def _bewaar_lijst(db, items: list[dict], feed: str = FEED_TOP40) -> None:
    rij = db.get(Meta, EXTERNE_LIJSTEN[feed][1])
    waarde = json.dumps(items, ensure_ascii=False)
    if rij:
        rij.value = waarde
    else:
        db.add(Meta(key=EXTERNE_LIJSTEN[feed][1], value=waarde))


def _lees_lijst(db, feed: str = FEED_TOP40) -> list[dict]:
    rij = db.get(Meta, EXTERNE_LIJSTEN[feed][1])
    if not rij or not rij.value:
        return []
    try:
        uit = json.loads(rij.value)
        return uit if isinstance(uit, list) else []
    except Exception:
        return []


def _vul_lijst(db, user: User, items: list[dict], titel: str,
               feed: str = FEED_TOP40) -> dict:
    """Leg de lijst naast de bibliotheek en zet hem in de playlist. Eén plek,
    zodat een verse lijst en een herstelronde gegarandeerd hetzelfde doen."""
    allowed = accessible_library_ids(db, user)
    q = db.query(Track)
    if allowed is not None:
        q = q.filter(Track.library_id.in_(allowed or [-1]))
    gevonden, missers = match_lijst(q.all(), items)

    pl = _openbare_lijst(db, user, titel, feed)
    # Leeggooien en opnieuw vullen: een hitlijst is een momentopname.
    db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pl.id).delete()
    for pos, t in enumerate(gevonden):
        db.add(PlaylistItem(playlist_id=pl.id, track_id=t.id, position=pos,
                            added_by=user.id, added_at=int(time.time())))
    pl.updated_at = int(time.time())
    pl.is_public = True
    return {"playlist": pl, "gevonden": gevonden, "missers": missers}


# ── de lijsten die de server zelf samenstelt ─────────────────────────────────
#
# De Top 40 komt van buiten; deze drie komen uit je eigen server. Ze staan naast
# elkaar op het beginscherm onder "Van JLTamp", en ze meten met opzet iets
# anders: wat de wereld draait, wat júllie draaien, wat er net binnen is en wat
# al jaren blijft hangen.
LIJST_LENGTE = 40


def _stemmen(db, vanaf: int, tot: int | None = None, uid: int | None = None) -> list[tuple[int, int]]:
    """(track_id, stemmen) over een periode, meest gedraaid eerst.

    Eén stem per gebruiker per nummer per dag. Zonder die regel is dit geen
    hitlijst maar een weergave van wie het vaakst op herhaling heeft gestaan:
    één iemand die een nummer vijftig keer draait zou de hele lijst kapen.

    Een beluistering telt pas mee vanaf dertig seconden. Doorklikken is geen
    luisteren.
    """
    dag = func.strftime("%Y-%m-%d", func.datetime(PlayEvent.started_at, "unixepoch"))
    stem = func.count(func.distinct(func.cast(PlayEvent.user_id, String) + ":" + dag))
    q = (
        select(PlayEvent.track_id, stem.label("stemmen"))
        .where(PlayEvent.started_at >= vanaf, PlayEvent.listened_ms >= 30000)
        .group_by(PlayEvent.track_id)
        .order_by(stem.desc())
        .limit(LIJST_LENGTE * 3)
    )
    if tot is not None:
        q = q.where(PlayEvent.started_at < tot)
    if uid is not None:
        q = q.where(PlayEvent.user_id == uid)
    return [(r[0], r[1]) for r in db.execute(q).all()]


def _bestaande(db, ids: list[int]) -> list[Track]:
    """De nummers die er nog zijn, in de volgorde waarin ze gevraagd zijn."""
    if not ids:
        return []
    gevonden = {t.id: t for t in db.query(Track).filter(Track.id.in_(ids)).all()}
    return [gevonden[i] for i in ids if i in gevonden]


def _bouw_populair(db, uid: int | None = None) -> list[Track]:
    """Wat er deze week het meest gedraaid is — door jou, of anders door het huis."""
    nu = int(time.time())
    ids = [tid for tid, _ in _stemmen(db, nu - 7 * 86400, uid=uid)]
    return _bestaande(db, ids)[:LIJST_LENGTE]


def _bouw_klassiekers(db, uid: int | None = None) -> list[Track]:
    """Wat er het afgelopen jaar het meest gedraaid is, zonder de laatste week.

    De week eruit laten is geen detail: anders is dit dezelfde lijst als
    "Populair", met dezelfde nummers in dezelfde volgorde.
    """
    nu = int(time.time())
    ids = [tid for tid, _ in _stemmen(db, nu - 365 * 86400, nu - 7 * 86400, uid=uid)]
    return _bestaande(db, ids)[:LIJST_LENGTE]


def _bouw_nieuw(db, uid: int | None = None) -> list[Track]:
    """Wat er het laatst aan de bibliotheek is toegevoegd.

    Hoogstens twee nummers per album: één scan van een doos met verzamelaars
    zou de lijst anders volledig vullen met hetzelfde album.
    """
    uit: list[Track] = []
    per_album: dict[int, int] = {}
    rijen = (
        db.query(Track)
        .filter(Track.added_at > 0)
        .order_by(Track.added_at.desc())
        .limit(LIJST_LENGTE * 20)
        .all()
    )
    for t in rijen:
        al = t.album_id or 0
        if per_album.get(al, 0) >= 2:
            continue
        per_album[al] = per_album.get(al, 0) + 1
        uit.append(t)
        if len(uit) >= LIJST_LENGTE:
            break
    return uit


def _bouw_nacht(db, uid: int | None = None) -> list[Track]:
    """Wat er 's nachts gedraaid wordt, tussen elf en vijf.

    Een andere lijst dan "Populair", en dat is het punt: wat je om twee uur
    's nachts opzet is zelden wat je om twee uur 's middags opzet. De uren komen
    uit de tijd van het toestel zoals die is opgeslagen — geen tijdzones, geen
    correcties; het gaat om het patroon, niet om de klok.
    """
    nu = int(time.time())
    uur = func.cast(func.strftime("%H", func.datetime(PlayEvent.started_at, "unixepoch", "localtime")), Integer)
    dag = func.strftime("%Y-%m-%d", func.datetime(PlayEvent.started_at, "unixepoch"))
    stem = func.count(func.distinct(func.cast(PlayEvent.user_id, String) + ":" + dag))
    q = (
        select(PlayEvent.track_id, stem.label("stemmen"))
        .where(
            PlayEvent.started_at >= nu - 180 * 86400,
            PlayEvent.listened_ms >= 30000,
            (uur >= 23) | (uur < 5),
        )
        .group_by(PlayEvent.track_id)
        .order_by(stem.desc())
        .limit(LIJST_LENGTE * 3)
    )
    if uid is not None:
        q = q.where(PlayEvent.user_id == uid)
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


def _bouw_stof(db, uid: int | None = None) -> list[Track]:
    """Favorieten waar al een jaar niet naar geluisterd is.

    Iedereen heeft een hartje gegeven aan nummers die hij daarna nooit meer
    opzette. Dit is de lijst die daarnaar terugwijst — niet wat je veel draait,
    maar wat je ooit mooi vond en kwijt bent geraakt.

    Een nummer telt mee als er in het afgelopen jaar geen enkele beluistering op
    staat, door niemand. Daardoor is de lijst voor iedereen hetzelfde, net als de
    andere lijsten van het huis.
    """
    nu = int(time.time())
    recent = select(func.distinct(PlayEvent.track_id)).where(PlayEvent.started_at >= nu - 365 * 86400)
    if uid is not None:
        recent = recent.where(PlayEvent.user_id == uid)
    # Nummers met de meeste hartjes eerst: wat door drie mensen leuk gevonden is
    # en toch stof vangt, is een betere vondst dan een eenling.
    harten = func.count(LikedTrack.user_id)
    q = (
        select(LikedTrack.track_id, harten.label("harten"))
        .where(LikedTrack.track_id.not_in(recent))
        .group_by(LikedTrack.track_id)
        .order_by(harten.desc(), func.random())
        .limit(LIJST_LENGTE * 2)
    )
    if uid is not None:
        # Jouw hartjes, en jouw luistergeschiedenis. Een nummer dat jij ooit
        # mooi vond maar dat iemand anders vorige week draaide, ben JIJ nog
        # steeds kwijt.
        q = q.where(LikedTrack.user_id == uid)
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


def _bouw_opdezedag(db, uid: int | None = None) -> list[Track]:
    """Wat je op deze datum in eerdere jaren draaide.

    De lijst die het meest verandert: morgen staat er iets heel anders in. Het
    jaar van nu telt niet mee — anders is het gewoon "vandaag gedraaid", en dat
    weet je zelf nog wel.
    """
    nu = time.gmtime()
    dagmaand = time.strftime("%m-%d", nu)
    ditjaar = time.strftime("%Y", nu)
    datum = func.strftime("%m-%d", func.datetime(PlayEvent.started_at, "unixepoch"))
    jaar = func.strftime("%Y", func.datetime(PlayEvent.started_at, "unixepoch"))
    aantal = func.count(PlayEvent.id)
    q = (
        select(PlayEvent.track_id, aantal.label("keer"))
        .where(datum == dagmaand, jaar != ditjaar, PlayEvent.listened_ms >= 30000)
        .group_by(PlayEvent.track_id)
        .order_by(aantal.desc())
        .limit(LIJST_LENGTE * 2)
    )
    if uid is not None:
        q = q.where(PlayEvent.user_id == uid)
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


def _bouw_samen(db, uid: int | None = None) -> list[Track]:
    """Nummers die meerdere mensen op deze server leuk vinden.

    De enige lijst die iets zegt over het huis in plaats van over één persoon:
    waar de smaak van twee of meer mensen elkaar raakt. Bij één luisteraar op de
    server blijft hij leeg, en dat hoort ook — dan valt er niets te delen.
    """
    harten = func.count(func.distinct(LikedTrack.user_id))
    q = (
        select(LikedTrack.track_id, harten.label("harten"))
        .group_by(LikedTrack.track_id)
        .having(harten >= 2)
        .order_by(harten.desc(), func.random())
        .limit(LIJST_LENGTE * 2)
    )
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


def _bouw_ochtend(db, uid: int | None = None) -> list[Track]:
    """Wat er 's ochtends vroeg opgaat, tussen vijf en negen.

    De tegenhanger van "Na middernacht". Dezelfde manier van tellen, ander uur —
    en in de praktijk een heel andere lijst.
    """
    nu = int(time.time())
    uur = func.cast(func.strftime("%H", func.datetime(PlayEvent.started_at, "unixepoch", "localtime")), Integer)
    dag = func.strftime("%Y-%m-%d", func.datetime(PlayEvent.started_at, "unixepoch"))
    stem = func.count(func.distinct(func.cast(PlayEvent.user_id, String) + ":" + dag))
    q = (
        select(PlayEvent.track_id, stem.label("stemmen"))
        .where(
            PlayEvent.started_at >= nu - 180 * 86400,
            PlayEvent.listened_ms >= 30000,
            uur >= 5, uur < 9,
        )
        .group_by(PlayEvent.track_id)
        .order_by(stem.desc())
        .limit(LIJST_LENGTE * 3)
    )
    if uid is not None:
        q = q.where(PlayEvent.user_id == uid)
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


def _bouw_lang(db, uid: int | None = None) -> list[Track]:
    """Alles boven de zeven minuten.

    Voor als je niet elke drie minuten iets nieuws wilt: de lange nummers uit de
    collectie, op lengte gesorteerd. Boven het half uur valt het af — dat zijn
    geen nummers meer maar dj-sets, hoorspelen en per ongeluk aan elkaar geplakte
    kanten van een plaat.
    """
    q = (
        db.query(Track)
        .filter(Track.duration_ms >= 7 * 60 * 1000, Track.duration_ms <= 30 * 60 * 1000)
        .order_by(func.random())
        .limit(LIJST_LENGTE * 2)
    )
    rijen = q.all()
    rijen.sort(key=lambda t: -(t.duration_ms or 0))
    return rijen[:LIJST_LENGTE]



# ── Het seizoen ──────────────────────────────────────────────────────────────
SEIZOENEN = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Lente", 4: "Lente", 5: "Lente",
    6: "Zomer", 7: "Zomer", 8: "Zomer",
    9: "Herfst", 10: "Herfst", 11: "Herfst",
}
# Welke maanden bij het seizoen van nu horen, als "%m"-tekst voor SQLite.
SEIZOEN_MAANDEN = {
    "Winter": ("12", "01", "02"),
    "Lente": ("03", "04", "05"),
    "Zomer": ("06", "07", "08"),
    "Herfst": ("09", "10", "11"),
}


def _seizoen_nu() -> str:
    return SEIZOENEN[time.gmtime().tm_mon]


def _titel_seizoen(persoonlijk: bool = False) -> str:
    """"Zomer in huis", en over drie maanden "Herfst in huis".

    Persoonlijk heet hij anders: "Jouw zomer". Het is dan namelijk iets anders —
    niet wat er hier in huis op ging, maar wat jij draaide. Een lijst die "in
    huis" heet terwijl hij over jou gaat, belooft het verkeerde.
    """
    return f"Jouw {_seizoen_nu().lower()}" if persoonlijk else f"{_seizoen_nu()} in huis"


def _bouw_seizoen(db, uid: int | None = None) -> list[Track]:
    """Wat er in dit seizoen in eerdere jaren het meest gedraaid werd.

    "Op deze dag" kijkt naar één datum en verandert dus elke ochtend; dit kijkt
    naar drie maanden en verandert vier keer per jaar. Dat is met opzet een
    andere maat: het ene is een herinnering aan een dag, het andere is hoe het
    hier klinkt als het buiten zo is.

    Het lopende jaar telt niet mee — anders staat er wat je vorige week al
    draaide, en dat is geen vondst.

    Levert dat te weinig op (het eerste jaar dat de server draait, of een
    seizoen waarin niemand iets nieuws opzette), dan mag het lopende jaar er
    alsnog bij. Een lege tegel is erger dan een tegel met wat bekends erin.
    """
    maanden = SEIZOEN_MAANDEN[_seizoen_nu()]
    ditjaar = time.strftime("%Y", time.gmtime())
    maand = func.strftime("%m", func.datetime(PlayEvent.started_at, "unixepoch"))
    jaar = func.strftime("%Y", func.datetime(PlayEvent.started_at, "unixepoch"))
    aantal = func.count(PlayEvent.id)

    def _haal(zonder_ditjaar: bool) -> list[int]:
        voorwaarden = [maand.in_(maanden), PlayEvent.listened_ms >= 30000]
        if zonder_ditjaar:
            voorwaarden.append(jaar != ditjaar)
        if uid is not None:
            voorwaarden.append(PlayEvent.user_id == uid)
        q = (
            select(PlayEvent.track_id, aantal.label("keer"))
            .where(*voorwaarden)
            .group_by(PlayEvent.track_id)
            .order_by(aantal.desc())
            .limit(LIJST_LENGTE * 2)
        )
        return [r[0] for r in db.execute(q).all()]

    rijen = _bestaande(db, _haal(True))
    if len(rijen) < 10:
        rijen = _bestaande(db, _haal(False))
    return rijen[:LIJST_LENGTE]


# Woorden die een kerstnummer verraden. Titel, album of artiest — één treffer
# is genoeg. Bewust ruim: een lijst die twee nummers mist is minder erg dan een
# lijst die er niet is.
KERSTWOORDEN = (
    "kerst", "christmas", "xmas", "noel", "noël", "jingle bell", "santa",
    "sinterklaas", "silent night", "stille nacht", "winter wonderland",
    "feliz navidad", "mistletoe", "sleigh", "advent",
)


def _bouw_kerst(db, uid: int | None = None) -> list[Track]:
    """Alles wat naar kerst ruikt. Hoogstens twee per album, anders is de hele
    lijst één kerstplaat."""
    return _op_woorden(db, KERSTWOORDEN)


def _op_woorden(db, woorden: tuple, per_album: int = 2) -> list[Track]:
    """Nummers waarvan de titel of het album een van deze woorden bevat.

    Op naam en niet op genre: de bibliotheek is met de hand gegroeid en de
    genres zijn een rommeltje, maar "Christmas" in een albumtitel is een
    behoorlijk betrouwbaar teken.
    """
    from sqlalchemy import or_
    voorwaarden = []
    for w in woorden:
        patroon = f"%{w}%"
        voorwaarden.append(func.lower(Track.title).like(patroon))
        voorwaarden.append(func.lower(Track.album_title).like(patroon))
    rijen = (
        db.query(Track)
        .filter(or_(*voorwaarden))
        .order_by(func.random())
        .limit(LIJST_LENGTE * 5)
        .all()
    )
    uit: list[Track] = []
    per: dict[int, int] = {}
    for t in rijen:
        al = t.album_id or 0
        if per.get(al, 0) >= per_album:
            continue
        per[al] = per.get(al, 0) + 1
        uit.append(t)
        if len(uit) >= LIJST_LENGTE:
            break
    return uit


def _meest_geliefd(db, uid: int | None = None) -> list[Track]:
    """Waar de meeste hartjes op zitten. Het feestje van het huis.

    De terugval voor feestdagen die je niet aan een woord kunt herkennen. Een
    lijst met de nummers waar hier de meeste mensen een hartje aan gaven, is
    geen slechte feestlijst — en het is in elk geval een echte.
    """
    harten = func.count(LikedTrack.user_id)
    q = (
        select(LikedTrack.track_id, harten.label("harten"))
        .group_by(LikedTrack.track_id)
        .order_by(harten.desc(), func.random())
        .limit(LIJST_LENGTE * 2)
    )
    if uid is not None:
        q = q.where(LikedTrack.user_id == uid)
    return _bestaande(db, [r[0] for r in db.execute(q).all()])[:LIJST_LENGTE]


NIEUWJAARSWOORDEN = (
    "new year", "nieuwjaar", "auld lang syne", "happy new year", "countdown",
    "vuurwerk", "fireworks", "champagne", "celebration", "party",
)

# Voor Koningsdag valt er weinig te herkennen aan een titel — "oranje" staat
# nergens in. Wat er wél is: een handvol woorden die in dit huis bij een feestje
# horen, en anders gewoon waar de meeste hartjes op zitten.
KONINGSWOORDEN = (
    "oranje", "hollandse", "hollands", "feest", "meezing", "polonaise",
    "carnaval", "hup holland", "koningsdag",
)


def _bouw_nieuwjaar(db, uid: int | None = None) -> list[Track]:
    """Oud en nieuw: aftellen, vuurwerk, en anders het jaar dat je gedraaid hebt.

    Eerst op naam. Levert dat te weinig op — en dat doet het in de meeste
    verzamelingen — dan wordt het wat er dit jaar het meest gedraaid is. Dat is
    ook een manier om het jaar uit te luiden, en die klopt altijd.
    """
    uit = _op_woorden(db, NIEUWJAARSWOORDEN)
    if len(uit) >= 12:
        return uit
    nu = int(time.time())
    return _bestaande(db, [tid for tid, _ in _stemmen(db, nu - 365 * 86400, uid=uid)])[:LIJST_LENGTE]


def _bouw_koningsdag(db, uid: int | None = None) -> list[Track]:
    """Koningsdag: meezingers, en anders waar hier de meeste hartjes op zitten."""
    uit = _op_woorden(db, KONINGSWOORDEN)
    if len(uit) >= 12:
        return uit
    return _meest_geliefd(db, uid)


def _bouw_nooit(db, uid: int | None = None) -> list[Track]:
    """Nummers die hier staan en waar nog nooit iemand naar geluisterd heeft.

    Niet zomaar willekeurig uit de bibliotheek: dat levert vooral bonustracks
    van dozen met verzamelaars op waar niemand op zit te wachten. Dit gaat om
    artiesten die hier wél gedraaid worden — de kanten van hun platen die je
    steeds overslaat omdat je altijd hetzelfde nummer aanzet.

    Een nummer telt mee als er nul beluisteringen op staan, door niemand. Zodra
    iemand het één keer opzet, valt het er vanzelf uit; deze lijst gaat over wat
    nog niemand kent.
    """
    # De artiesten waar hier naar geluisterd wordt.
    artiesten = (
        select(func.distinct(Track.artist_name))
        .join(PlayEvent, PlayEvent.track_id == Track.id)
        .where(Track.artist_name.isnot(None))
    )
    gespeeld = select(func.distinct(PlayEvent.track_id))
    if uid is not None:
        # Jouw artiesten, en nummers die JIJ nog nooit hebt gehoord. Dat iemand
        # anders in huis ze wel kent, maakt ze voor jou niet minder nieuw.
        artiesten = artiesten.where(PlayEvent.user_id == uid)
        gespeeld = gespeeld.where(PlayEvent.user_id == uid)
    q = (
        db.query(Track)
        .filter(Track.artist_name.in_(artiesten))
        .filter(Track.id.not_in(gespeeld))
        # Geen flarden van tien seconden en geen dj-sets van een uur.
        .filter(Track.duration_ms >= 60 * 1000, Track.duration_ms <= 12 * 60 * 1000)
        .order_by(func.random())
        .limit(LIJST_LENGTE * 3)
    )
    uit: list[Track] = []
    per_artiest: dict[str, int] = {}
    for t in q.all():
        naam = (t.artist_name or "").lower()
        # Hoogstens twee per artiest: anders is het de onbeluisterde helft van
        # één verzamelbox en niet een lijst van het huis.
        if per_artiest.get(naam, 0) >= 2:
            continue
        per_artiest[naam] = per_artiest.get(naam, 0) + 1
        uit.append(t)
        if len(uit) >= LIJST_LENGTE:
            break
    return uit


# feed → (titel, hoe hij gevuld wordt)
# Korte namen met opzet: op de banner staat er al "VAN JLTAMP" boven, en een
# naam van vier woorden wordt daar afgekapt tot onzin ("Klassieker het huis").
# Alles wat er te bouwen valt. De publieke selectie staat eronder; de rest
# bestaat alleen nog in persoonlijke vorm (zie /charts/mine).
HUISLIJSTEN_BRON: dict[str, tuple[object, object]] = {
    "populair": ("Populair", _bouw_populair),
    "nieuw": ("Net binnen", _bouw_nieuw),
    "klassiekers": ("Klassiekers", _bouw_klassiekers),
    "nacht": ("Na middernacht", _bouw_nacht),
    "stof": ("Stof afblazen", _bouw_stof),
    "opdezedag": ("Op deze dag", _bouw_opdezedag),
    "samen": ("Samen", _bouw_samen),
    "ochtend": ("Ochtendspits", _bouw_ochtend),
    # Heette "Het lange werk". Dat klonk als huiswerk; dit is juist de lijst
    # voor als je NIET elke drie minuten iets nieuws wilt.
    "lang": ("Tijd genoeg", _bouw_lang),
    "kerst": ("Kerst", _bouw_kerst),
    "nieuwjaar": ("Oud en nieuw", _bouw_nieuwjaar),
    "koningsdag": ("Koningsdag", _bouw_koningsdag),
    # De titel is hier een functie: hij verandert mee met het seizoen.
    "seizoen": (_titel_seizoen, _bouw_seizoen),
    "nooit": ("Nooit gehoord", _bouw_nooit),
}

# ⚠️ Hier stonden `HUISLIJSTEN` en `_zet_huislijst`: die maakten van een paar
# feeds een échte openbare afspeellijst (Kerst, Koningsdag, Oud en nieuw, Net
# binnen, Samen, Tijd genoeg). Openbaar betekent dat ze bij IEDEREEN tussen de
# afspeellijsten stonden, en dat is precies wat er niet moest gebeuren — zie het
# blok verderop. Ze worden nu, net als de rest, ter plekke berekend en nergens
# opgeslagen. `opruimen()` haalt de exemplaren weg die er nog van staan.
#
# `_openbare_lijst` blijft: de Top 40 is en blijft een echte lijst.


# ── Persoonlijk: hetzelfde blok, maar dan voor jou ──────────────────────────
#
# "Van JLTamp" was een blok met lijsten van het HUIS: wat er hier het meest
# gedraaid wordt, wat er 's nachts opgaat, welke favorieten stof vangen. Dat is
# aardig, maar het is geen aanbeveling — het is een gemiddelde. De eigenaar wil dat
# het blok een aanbeveling VOOR DE GEBRUIKER is, en dat is iets anders: wat JIJ
# 's nachts opzet, welke van JOUW hartjes je kwijt bent, wat JIJ in deze tijd
# van het jaar draaide.
#
# Dus bouwt de server per gebruiker zijn eigen versie. De lijst is van hem
# (`user_id`), staat niet openbaar, en draagt het voorvoegsel `mijn:` in `feed`
# zodat hij nooit te verwarren is met de huisversie — die blijft bestaan voor de
# lijsten die per definitie van het huis zijn (de Top 40, Net binnen, Samen).
#
# **De terugval is het belangrijkste deel.** Iemand die net meedoet heeft geen
# geschiedenis, en een lege tegel is erger dan een tegel met iets bekends. Komt
# er te weinig uit zijn eigen luisteren, dan wordt het de huisversie — hij ziet
# dan hetzelfde als vroeger in plaats van niets.
# Het voorvoegsel dat die persoonlijke lijsten vroeger droegen. Staat er alleen
# nog om ze te kunnen opruimen; er wordt niets meer mee gemaakt.
MIJN = "mijn:"
# En het lijstje "deze wil ik niet meer" van toen weggooien nog iets betekende.
NIET_MEER = "geen_huislijst:"


# ── Het blok "Van JLTamp" ────────────────────────────────────────────────────
#
# ⚠️ Dit blok maakt GEEN afspeellijsten. Niet voor het huis en niet voor een
# gebruiker. Lees dat als een regel en niet als een detail.
#
# Hoe het wél was: voor elke gebruiker werden er acht afspeellijsten aangemaakt
# ("Jouw herfst", "Na middernacht", "Stof afblazen"...), met `user_id` op die
# gebruiker en een `feed` die met `mijn:` begon. Dat leek netjes — het zijn
# lijsten, dus maak er lijsten van — maar het klopte op twee manieren niet.
#
# Ze stonden tussen zijn eigen afspeellijsten, waar ze niet horen. En gooide hij
# er één weg, dan ging het blok stuk: de lijst was weg, de tegel was weg, en
# omdat weggooien ook werd onthouden ("die wil hij niet meer") kwam hij ook niet
# terug. Een blok dat kapot kan door iets wat iemand in zijn eigen bibliotheek
# doet, is verkeerd gebouwd.
#
# Zie het zo: zie het als een ADVERTENTIEBLOK. Het staat los van de
# gebruiker. Er mogen best persoonlijke dingen in staan — "Jouw herfst" gaat
# over wat jij draaide — maar het is een aanbeveling die de server toont, geen
# spullen die in zijn kast worden gezet. Zijn afspeellijsten maakt hij zelf.
#
# Dus bestaat een tegel alleen nog als ANTWOORD. `/charts/blok` geeft de tegels,
# `/charts/blok/{feed}/items` geeft de nummers, en in de database staat er
# alleen een rijtje track-id's in `meta` — een cache, geen bezit. Er is niets
# om weg te gooien, dus er is niets dat stuk kan.
#
# De Top 40 is de uitzondering en blijft een echte openbare afspeellijst: die is
# van het huis en wordt wekelijks gevuld.
# Hij staat in het blok als verwijzing naar die lijst.

# Zo vaak opnieuw bouwen. Vaker heeft geen zin: deze lijsten kijken naar weken
# en maanden, en het kost een handvol zoekopdrachten.
VERSE_UREN = 6
# Minder dan dit uit je eigen geschiedenis en we vallen terug op het huis.
GENOEG = 8

PERSOONLIJK = ("populair", "klassiekers", "nacht", "ochtend",
               "stof", "opdezedag", "seizoen", "nooit")
# Wat er per definitie van het huis is: daar valt niets aan te personaliseren.
VAN_HET_HUIS = ("nieuw", "samen", "lang", "kerst", "nieuwjaar", "koningsdag")
BLOK_FEEDS = PERSOONLIJK + VAN_HET_HUIS

# De cache. `blok:huis` is voor iedereen hetzelfde en wordt dus één keer
# gebouwd; `blok:<uid>` bevat alleen de feeds waar deze gebruiker genoeg eigen
# geschiedenis voor heeft.
BLOK_HUIS = "blok:huis"
BLOK_MIJN = "blok:"


def _cache_lees(db, sleutel_naam: str) -> dict:
    rij = db.get(Meta, sleutel_naam)
    if not rij:
        return {}
    try:
        uit = json.loads(rij.value)
        return uit if isinstance(uit, dict) else {}
    except Exception:
        return {}


def _cache_schrijf(db, sleutel_naam: str, data: dict) -> None:
    waarde = json.dumps(data, ensure_ascii=False)
    # `db.get` en niet `query().first()` zou hier ook kunnen, maar dan vindt hij
    # een rij die in dezelfde sessie net is toegevoegd en nog niet is
    # weggeschreven niet terug — en dan komt er een tweede bij met dezelfde
    # sleutel. Dat gaf een `UNIQUE constraint failed: meta.key` bij het eerste
    # verzoek na het opstarten. Vandaar ook de `flush`: na deze regel is het weg.
    rij = db.query(Meta).filter(Meta.key == sleutel_naam).first()
    if rij:
        rij.value = waarde
    else:
        db.add(Meta(key=sleutel_naam, value=waarde))
    db.flush()


def _titel_van(feed: str, persoonlijk: bool) -> str:
    titel = HUISLIJSTEN_BRON[feed][0]
    return titel(persoonlijk) if callable(titel) else titel


def _huis_blok(db, force: bool = False) -> dict:
    """De versie van het huis, voor iedereen dezelfde."""
    nu = int(time.time())
    cache = _cache_lees(db, BLOK_HUIS)
    veranderd = False
    for feed in BLOK_FEEDS:
        oud = cache.get(feed)
        if oud and not force and (nu - int(oud.get("t") or 0)) < VERSE_UREN * 3600:
            continue
        tracks = HUISLIJSTEN_BRON[feed][1](db, None)
        cache[feed] = {"t": nu, "titel": _titel_van(feed, False),
                       "ids": [t.id for t in tracks]}
        veranderd = True
    if veranderd:
        _cache_schrijf(db, BLOK_HUIS, cache)
    return cache


def _mijn_blok(db, uid: int, force: bool = False) -> dict:
    """Wat er van déze gebruiker zelf is.

    Alleen de feeds waar genoeg eigen geschiedenis voor is. Komt er te weinig
    uit, dan staat er niets in en valt het blok hieronder terug op het huis —
    een nieuwe gebruiker ziet dan hetzelfde als iedereen in plaats van niets.
    """
    nu = int(time.time())
    cache = _cache_lees(db, BLOK_MIJN + str(uid))
    veranderd = False
    for feed in PERSOONLIJK:
        oud = cache.get(feed)
        if oud and not force and (nu - int(oud.get("t") or 0)) < VERSE_UREN * 3600:
            continue
        tracks = HUISLIJSTEN_BRON[feed][1](db, uid)
        if len(tracks) >= GENOEG:
            cache[feed] = {"t": nu, "titel": _titel_van(feed, True),
                           "ids": [t.id for t in tracks]}
        else:
            # Wel de tijd onthouden, anders wordt het elke keer opnieuw
            # geprobeerd voor iemand die nog nauwelijks geluisterd heeft.
            cache[feed] = {"t": nu, "titel": _titel_van(feed, True), "ids": []}
        veranderd = True
    if veranderd:
        _cache_schrijf(db, BLOK_MIJN + str(uid), cache)
    return cache


def _lijst_van(db, feed: str) -> Playlist | None:
    """De ene openbare afspeellijst van deze externe lijst, als hij er al is."""
    return db.execute(
        select(Playlist).where(Playlist.feed == feed).order_by(Playlist.id)
    ).scalars().first()


def _extern_ids(db, feed: str) -> tuple[str, list[int]]:
    """Een lijst van buiten (Top 40, Viraal) als tegel.

    Die zijn en blijven échte openbare afspeellijsten — van het huis, wekelijks
    gevuld. Maar ze horen in het blok wél hetzelfde
    te werken als de rest: dezelfde tegel, hetzelfde scherm, dezelfde drie
    puntjes om ze over te nemen. Dus worden ze hier gewoon uitgelezen.
    """
    pl = _lijst_van(db, feed)
    if not pl:
        return EXTERNE_LIJSTEN[feed][0], []
    ids = list(db.execute(
        select(PlaylistItem.track_id)
        .where(PlaylistItem.playlist_id == pl.id)
        .order_by(PlaylistItem.position)
    ).scalars())
    return pl.title, ids


def _blok_ids(db, user: User, feed: str) -> tuple[str, list[int]]:
    """Titel en track-id's van één tegel, voor deze gebruiker."""
    if feed in EXTERNE_LIJSTEN:
        return _extern_ids(db, feed)
    if feed in PERSOONLIJK:
        mijn = _mijn_blok(db, user.id).get(feed) or {}
        if mijn.get("ids"):
            return str(mijn.get("titel") or feed), list(mijn["ids"])
    huis = _huis_blok(db).get(feed) or {}
    return str(huis.get("titel") or _titel_van(feed, False)), list(huis.get("ids") or [])


def _tegel(feed: str, titel: str, aantal: int, nu: int) -> dict:
    """Een tegel in de vorm die de app van een afspeellijst kent.

    Met opzet dezelfde velden: het beginscherm tekende deze tegels al vanuit
    `/playlists`, en door hier hetzelfde terug te geven hoefde daar bijna niets
    aan te veranderen. `key` wijst naar de nummers, en dat is het enige verschil
    — die komen niet uit een afspeellijst maar worden ter plekke opgezocht.
    """
    return {
        "ratingKey": f"blok:{feed}",
        "key": f"/charts/blok/{feed}/items",
        "type": "playlist",
        "title": titel,
        "feed": feed,
        "leafCount": aantal,
        "public": True,
        "updatedAt": nu,
    }


@router.get("/charts/blok")
def lees_blok(user: User = Depends(require_user)):
    """De tegels van "Van JLTamp" voor deze gebruiker.

    Bouwt niets in zijn account. Wat hier uitkomt bestaat alleen als antwoord.
    """
    # Ook hier, en niet alleen in /charts/looks. Die vertelt de app dat hij het
    # blok niet moet tekenen, maar een beoordelaar die het verkeer bekijkt kreeg
    # de inhoud nog steeds terug. Twee sloten op dezelfde deur.
    if bool(getattr(user, "is_review", False)):
        return {"MediaContainer": {"size": 0, "Metadata": []}}

    db = SessionLocal()
    try:
        nu = int(time.time())
        uit = []

        # De lijsten van buiten voorop, in de volgorde van het register: eerst
        # de Top 40, dan Viraal. Het blijven echte afspeellijsten in de
        # database, maar in het blok zijn het tegels als alle andere: dezelfde
        # vorm, hetzelfde scherm als je erop tikt.
        from .admin_users import schakelaar
        for feed in EXTERNE_LIJSTEN:
            if feed == FEED_VIRAAL and not schakelaar(db, "viraal"):
                continue
            titel, ids = _extern_ids(db, feed)
            if not ids:
                continue
            pl = _lijst_van(db, feed)
            tegel = _tegel(feed, titel, len(ids), nu)
            # Het weeknummer onder de banner komt van wanneer de lijst gevuld is.
            tegel["updatedAt"] = int((pl.updated_at if pl else 0) or nu)
            uit.append(tegel)

        # Eén keer bouwen en dan de feeds erdoorheen lopen. `_blok_ids` per feed
        # aanroepen deed het bouwen veertien keer opnieuw — en schreef dus ook
        # veertien keer aan dezelfde cacheregel.
        huis = _huis_blok(db)
        mijn = _mijn_blok(db, user.id)
        for feed in BLOK_FEEDS:
            eigen = (mijn.get(feed) or {}) if feed in PERSOONLIJK else {}
            bron = eigen if eigen.get("ids") else (huis.get(feed) or {})
            ids = list(bron.get("ids") or [])
            if not ids:
                # Een tegel zonder nummers is een deur naar een lege kamer. Op
                # sommige dagen is "Op deze dag" gewoon leeg.
                continue
            uit.append(_tegel(feed, str(bron.get("titel") or feed), len(ids), nu))
        db.commit()
        # Wat de beheerder in Beheer → Beginscherm verborgen heeft.
        try:
            verborgen = set(json.loads((db.get(Meta, "blok_verborgen") or Meta(value="[]")).value or "[]"))
        except Exception:
            verborgen = set()
        uit = [x for x in uit if x.get("feed") not in verborgen]
        return {"MediaContainer": {"size": len(uit), "Metadata": uit}}
    finally:
        db.close()


@router.get("/charts/blok/{feed}/items")
def lees_blok_items(feed: str, user: User = Depends(require_user)):
    """De nummers van één tegel."""
    if feed not in EXTERNE_LIJSTEN and feed not in HUISLIJSTEN_BRON:
        raise HTTPException(404, "Onbekende lijst")
    db = SessionLocal()
    try:
        _, ids = _blok_ids(db, user, feed)
        db.commit()
        if not ids:
            return container([])
        allowed = accessible_library_ids(db, user)
        stmt = select(Track).where(Track.id.in_(ids))
        if allowed is not None:
            stmt = stmt.where(Track.library_id.in_(allowed or [-1]))
        per_id = {t.id: t for t in db.execute(stmt).scalars()}
        # De volgorde van de lijst aanhouden: `IN` geeft hem in id-volgorde
        # terug, en bij "Op deze dag" of de Top 40 is de volgorde het verhaal.
        tracks = [per_id[i] for i in ids if i in per_id]
        states = track_states(db, user.id, tracks)
        return container([track_dict(t, state=states.get(t.id)) for t in tracks])
    finally:
        db.close()


@router.post("/charts/blok/{feed}/bewaar")
def bewaar_blok(feed: str, titel: str = Body("", embed=True),
                user: User = Depends(require_user)):
    """Neem deze tegel over als je EIGEN afspeellijst.

    Het blok maakt niets in iemands account — maar als je zelf zegt "die wil ik
    houden", dan is dat precies andersom: dan is het jouw keuze en jouw lijst.
    Er komt een gewone afspeellijst uit, zonder `feed`, die verder niets meer met
    het blok te maken heeft. Verandert de tegel volgende week, dan verandert deze
    lijst niet mee — hij is van jou.

    Een kopie en geen verwijzing, dus. Dat is ook het antwoord op "wat als het
    blok verdwijnt": dan heb je nog steeds je lijst.
    """
    if feed not in EXTERNE_LIJSTEN and feed not in HUISLIJSTEN_BRON:
        raise HTTPException(404, "Onbekende lijst")
    db = SessionLocal()
    try:
        naam, ids = _blok_ids(db, user, feed)
        db.commit()
        if not ids:
            raise HTTPException(400, "Deze lijst is leeg")
        nu = int(time.time())
        pl = Playlist(user_id=user.id, title=(titel or "").strip() or naam,
                      created_at=nu, updated_at=nu, is_public=False, feed="")
        db.add(pl)
        db.flush()
        # Alleen wat deze gebruiker mag zien: de tegel is al op zijn toegang
        # gebouwd, maar tussen bouwen en bewaren kan er een bibliotheek afgaan.
        allowed = accessible_library_ids(db, user)
        stmt = select(Track.id).where(Track.id.in_(ids))
        if allowed is not None:
            stmt = stmt.where(Track.library_id.in_(allowed or [-1]))
        mag = set(db.execute(stmt).scalars())
        pos = 0
        for tid in ids:
            if tid not in mag:
                continue
            db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos,
                                added_by=user.id, added_at=int(time.time())))
            pos += 1
        db.commit()
        return {"ratingKey": playlist_key(pl.id), "title": pl.title, "count": pos}
    finally:
        db.close()


# ── Oude versies van de app ─────────────────────────────────────────────────
#
# Deze twee riepen vroeger het bouwen van de persoonlijke lijsten aan en het
# verbergen ervan. Ze doen nu niets meer, maar ze blijven bestaan: op een
# toestel dat nog niet is bijgewerkt geeft een 404 hier een foutmelding op het
# beginscherm, en dat is erger dan een antwoord waar niets in staat.


@router.post("/charts/mine")
def bouw_persoonlijk(force: bool = False, user: User = Depends(require_user)):
    return {"lists": []}


@router.get("/charts/mine/hidden")
def welke_niet(user: User = Depends(require_user)):
    return {"hidden": []}


@router.post("/charts/mine/hidden")
def zet_niet(feed: str = Body("", embed=True), value: bool = Body(True, embed=True),
             user: User = Depends(require_user)):
    return {"hidden": []}


# ── Opruimen: wat er vroeger wél werd aangemaakt ────────────────────────────
#
# Draait bij het opstarten, en doet daarna niets meer omdat er niets meer te
# vinden is. Wat eruit gaat:
#
#   • elke `mijn:`-afspeellijst van elke gebruiker — dat zijn de acht per
#     persoon die hier vroeger werden gemaakt;
#   • de openbare huislijsten van het blok (Kerst, Koningsdag, Oud en nieuw,
#     Net binnen, Samen, Tijd genoeg): die stonden bij iedereen tussen de
#     afspeellijsten omdat ze openbaar zijn. De Top 40 blijft;
#   • het lijstje "deze wil ik niet meer", want er valt niets meer weg te gooien.
#
# De eigenaar vroeg hier expliciet om: "kijk alle gebruikers na en haal wat van het
# blok JLTamp is uit hun lijsten".


def opruimen() -> dict:
    db = SessionLocal()
    try:
        weg_feeds = [MIJN + f for f in HUISLIJSTEN_BRON] + list(VAN_HET_HUIS)
        lijsten = list(db.execute(
            select(Playlist).where(Playlist.feed.in_(weg_feeds))
        ).scalars())
        for pl in lijsten:
            db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pl.id).delete()
            db.delete(pl)
        # En ook alles wat met `mijn:` begint maar hierboven niet in de lijst
        # stond — een feed die ooit bestond en nu niet meer.
        rest = list(db.execute(
            select(Playlist).where(Playlist.feed.like(MIJN + "%"))
        ).scalars())
        for pl in rest:
            db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pl.id).delete()
            db.delete(pl)
        vergeten = db.query(Meta).filter(Meta.key.like(NIET_MEER + "%")).delete()
        db.commit()
        return {"lijsten": len(lijsten) + len(rest), "voorkeuren": vergeten}
    finally:
        db.close()


@router.post("/charts/opruimen")
def opruimen_nu(user: User = Depends(require_admin)):
    return opruimen()


# ── De huisstijl van de tegels ───────────────────────────────────────────────
#
# Welke vorm, welke kleur en welke animatie een huislijst op het beginscherm
# krijgt, stond in de app. Dat betekent dat "met nieuwjaar iets anders dan Het
# lange werk" wachten is op een nieuwe versie in twee winkels — en dat duurt
# dagen tot weken. Daar is een feestdag te kort voor.
#
# Dus staat het hier. De app haalt dit op, legt het over zijn eigen instellingen
# heen, en bewaart het op het toestel voor de volgende keer. Staat er niets
# ingesteld (de normale toestand), dan gebruikt de app gewoon wat hij zelf al
# wist — dit eindpunt is een uitzondering, geen vervanging.
#
# Wat je kunt zetten per feed: kleur, tegel, rond, span, familie, teken,
# beweging, tempo, richting, breed, collage, tekstOnder, tekenHoek. Namen die de
# app niet kent, negeert hij; hij valt dan terug op wat hij zelf had. Dat is met
# opzet: een typefout hier mag nooit een leeg beginscherm geven.
META_LOOKS = "feed_looks"


LEEG_STIJL = {"looks": {}, "volgorde": [], "aan": True, "specials": []}


def schakelaar_veilig(naam: str) -> bool:
    """Een schakelaar met een eigen databaseverbinding (voor het vroege antwoord
    aan een reviewaccount, waar er nog geen open is)."""
    from .admin_users import schakelaar
    db = SessionLocal()
    try:
        return schakelaar(db, naam)
    finally:
        db.close()


@router.get("/charts/looks")
def lees_looks(user: User | None = Depends(optional_user)):
    """Hoe het blok "Van JLTamp" eruitziet. Zonder inloggen: dit is geen inhoud.

    Leeg antwoord = de app doet wat hij zelf al deed.

    - `looks`     — per feed de vorm, kleur en animatie.
    - `volgorde`  — in welke volgorde de lijsten staan.
    - `aan`       — staat het blok er überhaupt. `false` haalt het hele blok weg.
    - `specials`  — extra tegels die geen afspeellijst zijn: "Jouw jaar 2026",
                    een kerstlijst, dat werk. Zie `zet_looks` hieronder.
    - `albumWensen` — mag er ook om een heel album gevraagd worden.
    - `karaoke`   — staat de karaokeknop in de speler. Zie routers/karaoke.py.
    - `wensen`    — mag het zoekscherm een wensknop laten zien als er niets
                    gevonden is. Altijd `false` op deze server; wat er niet
                    gevonden wordt komt vanzelf op het verlanglijstje.
    - `herkennen` — staat de microfoon in de zoekbalk er. Deze server heeft
                    geen herkenning, dus altijd `false`.
    - `akkoorden` — staat de gitaarknop in de songtekst (zie akkoorden.py).
                    Standaard uit; uit bij een account voor een beoordeling.
    """
    # Een account voor een winkelbeoordeling krijgt het blok niet te zien. Hier
    # en niet in de app: dan geldt het meteen voor elke versie die er al is, en
    # hoeft er niets in de winkel te veranderen om een beoordeling rustiger te
    # laten verlopen.
    from .admin_users import schakelaar
    if user is not None and bool(getattr(user, "is_review", False)):
        uit = dict(LEEG_STIJL)
        uit["aan"] = False
        uit["wensen"] = False
        uit["albumWensen"] = False
        # Ook het importeren. Dat stond hier niet bij en werd alleen door /api
        # gemeld — en /api kent geen gebruiker, dus een beoordelaar zag de kaart
        # "Afspeellijsten importeren" gewoon staan. Hier weet de server wél wie
        # het vraagt.
        uit["importeren"] = False
        uit["karaoke"] = False
        uit["herkennen"] = False
        uit["akkoorden"] = False
        # De speelhoek wél: hij staat in de notities voor de winkels en
        # gebruikt alleen de demobibliotheek. Uitzetten liet een beloofde
        # functie verdwijnen, en daar keurt een beoordelaar op af.
        uit["spellen"] = schakelaar_veilig("spellen")
        uit["dj"] = schakelaar_veilig("dj")
        return uit

    db = SessionLocal()
    try:
        wensen = schakelaar(db, "wensen") and not schakelaar(db, "winkelstand")
        albumwensen = wensen and schakelaar(db, "albums")
        importeren = schakelaar(db, "importeren") and not schakelaar(db, "winkelstand")
        # Karaoke: niet onder de winkelstand, want daar valt niets aan uit te
        # leggen dat niet ook gewoon in de app staat. Wel uit voor een
        # beoordelaar (hierboven), zoals alles wat het huis aanzet.
        from .admin_users import recht
        # Publiek is er geen route achter de wensknop (/wishlist/noteer): wat er
        # niet gevonden wordt komt vanzelf op het verlanglijstje, en verder niets.
        wensen = albumwensen = False
        karaoke = schakelaar(db, "karaoke") and recht(db, user, "karaoke")
        herkennen = False
        # Net als karaoke niet onder de winkelstand: wat hij doet staat
        # gewoon in de app. De beoordelaar is hierboven al afgevangen.
        akkoorden = schakelaar(db, "akkoorden")
        # Onderhoudsmodus: gewone gebruikers krijgen een scherm "even onderhoud";
        # een beheerder ziet alleen een balkje en kan gewoon door.
        onderhoud_rij = db.get(Meta, "onderhoud_tekst")
        extra = {"spellen": schakelaar(db, "spellen") and recht(db, user, "spellen"),
                 "dj": schakelaar(db, "dj") and recht(db, user, "dj"),
                 "onderhoud": schakelaar(db, "onderhoud"),
                 # Alleen gewone gebruikers worden tegengehouden; de beheerder
                 # moet er juist bij kunnen.
                 "onderhoudBlok": schakelaar(db, "onderhoud") and not (user is not None and user.is_admin),
                 "onderhoudTekst": (onderhoud_rij.value if onderhoud_rij else "") or ""}
        rij = db.get(Meta, META_LOOKS)
        if not rij or not rij.value:
            return {**LEEG_STIJL, "wensen": wensen, "albumWensen": albumwensen, "importeren": importeren, "karaoke": karaoke, "herkennen": herkennen, "akkoorden": akkoorden, **extra}
        try:
            uit = json.loads(rij.value)
        except Exception:
            return {**LEEG_STIJL, "wensen": wensen, "albumWensen": albumwensen, "importeren": importeren, "karaoke": karaoke, "herkennen": herkennen, "akkoorden": akkoorden, **extra}
        if not isinstance(uit, dict):
            return {**LEEG_STIJL, "wensen": wensen, "albumWensen": albumwensen, "importeren": importeren, "karaoke": karaoke, "herkennen": herkennen, "akkoorden": akkoorden, **extra}
        return {
            "looks": uit.get("looks") or {},
            "volgorde": uit.get("volgorde") or [],
            "aan": uit.get("aan", True) is not False,
            "specials": uit.get("specials") or [],
            "wensen": wensen,
            "albumWensen": albumwensen,
            "importeren": importeren,
            "karaoke": karaoke,
            "herkennen": herkennen,
            "akkoorden": akkoorden,
            **extra,
        }
    finally:
        db.close()


@router.post("/charts/looks")
def zet_looks(
    looks: dict = Body(default={}, embed=True),
    volgorde: list = Body(default=[], embed=True),
    aan: bool | None = Body(default=None, embed=True),
    specials: list | None = Body(default=None, embed=True),
    vervang: bool = Body(default=False, embed=True),
    user: User = Depends(require_admin),
):
    """Zet of vul de huisstijl aan. Alleen de beheerder.

    Standaard wordt er per feed BIJGEWERKT: wat je niet noemt blijft staan.
    Met `vervang: true` gooi je alles weg en zet je precies dit neer — handig om
    na de feestdagen in één keer terug te gaan naar normaal (`{"looks": {},
    "vervang": true}`).

    Voorbeeld — met nieuwjaar een lijst in de vorm van een ster:

        {"looks": {"nieuwjaar": {"kleur": "#F3C64B", "rond": "ster",
                                 "familie": "bijzonder", "span": 1}},
         "volgorde": ["top40", "nieuwjaar", "populair"]}

    Het hele blok weghalen:

        {"aan": false}

    Een terugblik en een kerstlijst erbij, alleen in december:

        {"specials": [
          {"id": "jaar2026", "soort": "wrapped", "titel": "Jouw jaar 2026",
           "kleur": "#C13AC0", "rond": "ster", "van": "2026-12-01",
           "tot": "2027-01-15"},
          {"id": "kerst", "soort": "lijst", "titel": "Kerst", "ratingKey": "p12",
           "kleur": "#F94D67", "rond": "bloem", "van": "12-01", "tot": "12-27"}
        ]}

    `van` en `tot` mogen een volledige datum zijn (eenmalig) of `MM-DD` (elk
    jaar). Buiten dat venster laat de app de tegel weg — dan hoeft niemand er
    met kerst 's avonds nog achteraan.
    """
    if not isinstance(looks, dict):
        raise HTTPException(400, "looks moet een object zijn")
    db = SessionLocal()
    try:
        rij = db.get(Meta, META_LOOKS)
        huidig = dict(LEEG_STIJL)
        huidig["looks"] = {}
        huidig["volgorde"] = []
        huidig["specials"] = []
        if rij and rij.value and not vervang:
            try:
                oud = json.loads(rij.value)
                if isinstance(oud, dict):
                    huidig["looks"] = oud.get("looks") or {}
                    huidig["volgorde"] = oud.get("volgorde") or []
                    huidig["aan"] = oud.get("aan", True) is not False
                    huidig["specials"] = oud.get("specials") or []
            except Exception:
                pass
        for feed, look in looks.items():
            if not isinstance(look, dict):
                continue
            basis = huidig["looks"].get(feed) or {}
            basis.update(look)
            huidig["looks"][str(feed)] = basis
        if volgorde:
            huidig["volgorde"] = [str(f) for f in volgorde]
        if aan is not None:
            huidig["aan"] = bool(aan)
        if specials is not None:
            huidig["specials"] = [x for x in specials if isinstance(x, dict)]
        waarde = json.dumps(huidig, ensure_ascii=False)
        if rij:
            rij.value = waarde
        else:
            db.add(Meta(key=META_LOOKS, value=waarde))
        db.commit()
        return {"ok": True, **huidig}
    finally:
        db.close()


@router.post("/charts/rebuild")
def bouw_huislijsten(user: User = Depends(require_admin)):
    """Bouw het blok "Van JLTamp" opnieuw.

    Draai dit na een scan met nieuwe muziek. Er wordt hier niets meer AANGEMAAKT — dit
    zet alleen de cache van het huis opnieuw (zie `_huis_blok`), zodat er na een
    scan met nieuwe muziek niet zes uur op gewacht hoeft te worden.

    De persoonlijke kant blijft staan: die hangt aan wat iemand zelf gedraaid
    heeft en verandert niet door een scan.
    """
    db = SessionLocal()
    try:
        cache = _huis_blok(db, force=True)
        db.commit()
        return {"lists": [{"feed": f, "title": v.get("titel"), "count": len(v.get("ids") or [])}
                          for f, v in cache.items()]}
    finally:
        db.close()


def _zet_lijst(feed: str, items: list[dict], title: str, user: User) -> dict:
    """Vul de openbare lijst van deze feed met deze regels, in deze volgorde.

    items: [{position, artist, title}, …] — position is optioneel en dient
    alleen om de melding leesbaar te houden; de vólgorde in de lijst bepaalt de
    volgorde in de playlist.
    """
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "Geen regels meegestuurd")
    if len(items) > MAX_REGELS:
        raise HTTPException(400, f"Te veel regels (max {MAX_REGELS})")

    db = SessionLocal()
    try:
        standaard = EXTERNE_LIJSTEN[feed][0]
        titel = (title or standaard).strip() or standaard
        uit = _vul_lijst(db, user, items, titel, feed)
        # De namen bewaren, zodat de lijst na een scan opnieuw te vullen is
        # zonder dat de bron er weer aan te pas hoeft te komen.
        _bewaar_lijst(db, items, feed)
        db.commit()

        return {
            "ratingKey": playlist_key(uit["playlist"].id),
            "title": uit["playlist"].title,
            "total": len(items),
            "matched": len(uit["gevonden"]),
            "missing": uit["missers"],
        }
    finally:
        db.close()


def _ververs_lijst(feed: str, user: User) -> dict:
    """Vul de lijst opnieuw uit de bewaarde namen.

    Draai dit na een scan met nieuwe muziek. Twee redenen: nieuwe nummers
    staan er dan meteen in, én een scan die
    nummer-id's heeft vervangen wordt hiermee rechtgezet — anders zakt de lijst
    stilletjes in elkaar, met gaten in de volgorde.
    """
    db = SessionLocal()
    try:
        items = _lees_lijst(db, feed)
        if not items:
            return {"matched": 0, "total": 0, "note": "geen bewaarde lijst"}
        pl = _lijst_van(db, feed)
        titel = (pl.title if pl else "") or EXTERNE_LIJSTEN[feed][0]
        uit = _vul_lijst(db, user, items, titel, feed)
        db.commit()
        return {
            "ratingKey": playlist_key(uit["playlist"].id),
            "total": len(items),
            "matched": len(uit["gevonden"]),
            "missing": len(uit["missers"]),
        }
    finally:
        db.close()


def _lees_lijst_stand(feed: str) -> dict:
    """Waar staat de lijst, en hoe vol is hij? Voor de app en voor de ronde."""
    db = SessionLocal()
    try:
        pl = _lijst_van(db, feed)
        if not pl:
            return {"ratingKey": None, "title": EXTERNE_LIJSTEN[feed][0],
                    "count": 0, "updatedAt": 0}
        n = db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pl.id).count()
        return {
            "ratingKey": playlist_key(pl.id),
            "title": pl.title,
            "count": n,
            "updatedAt": pl.updated_at or 0,
        }
    finally:
        db.close()


def _bekende_feed(feed: str) -> str:
    if feed not in EXTERNE_LIJSTEN:
        raise HTTPException(404, "Onbekende lijst")
    return feed


# ── de lijsten van buiten ────────────────────────────────────────────────────
#
# Eén stel routes voor alle lijsten die van buiten komen. Het script dat ze
# aanlevert kiest met de feed in het pad wélke: `top40` of `viraal`.
@router.post("/charts/lijst/{feed}")
def zet_lijst(
    feed: str,
    items: list[dict] = Body(..., embed=True),
    title: str = Body("", embed=True),
    user: User = Depends(require_admin),
):
    return _zet_lijst(_bekende_feed(feed), items, title, user)


@router.post("/charts/lijst/{feed}/ophalen")
def haal_lijst_op(feed: str, user: User = Depends(require_admin)):
    """Haal de lijst zélf op bij zijn bron en vul hem.

    Voor Viraal. De Top 40 kan dit niet: die komt van buiten binnen via
    POST /charts/lijst/top40. Hier is de bron een Spotify-lijst, en die kunnen
    we al lezen omdat de importer dat ook doet (met een Spotify-koppeling, zie
    /admin/spotify/koppel).
    """
    _bekende_feed(feed)
    if feed != FEED_VIRAAL:
        raise HTTPException(400, "Deze lijst haalt zichzelf niet op")

    db = SessionLocal()
    try:
        rij = db.get(Meta, META_VIRAAL_BRON)
        bron = (rij.value if rij else "") or BRON_VIRAAL
    finally:
        db.close()

    # Binnenin importeren: charts.py wordt bij het opstarten eerder geladen dan
    # importer.py, en bovenaan zou dat een kringetje worden.
    from .importer import _fetch_spotify
    naam, items = _fetch_spotify(bron)
    if not items:
        raise HTTPException(502, f"Niets gevonden op de bron ({naam})")
    # De naam van de bron nemen we NIET over. Die heet bij Spotify "Viral Hits"
    # en dat is hun lijst; bij ons heet de tegel Viraal. Zie ook de standaard
    # in EXTERNE_LIJSTEN.
    uit = _zet_lijst(feed, items, "", user)
    uit["bron"] = naam
    return uit


@router.post("/charts/lijst/{feed}/bron")
def zet_bron(feed: str, url: str = Body("", embed=True),
             user: User = Depends(require_admin)):
    """Waar Viraal vandaan komt. Leeg zet hem terug op de standaard."""
    _bekende_feed(feed)
    if feed != FEED_VIRAAL:
        raise HTTPException(400, "Deze lijst heeft geen instelbare bron")
    db = SessionLocal()
    try:
        waarde = (url or "").strip()
        rij = db.query(Meta).filter(Meta.key == META_VIRAAL_BRON).first()
        if rij:
            rij.value = waarde
        elif waarde:
            db.add(Meta(key=META_VIRAAL_BRON, value=waarde))
        db.commit()
        return {"bron": waarde or BRON_VIRAAL}
    finally:
        db.close()


@router.post("/charts/lijst/{feed}/refresh")
def ververs_lijst(feed: str, user: User = Depends(require_admin)):
    return _ververs_lijst(_bekende_feed(feed), user)


@router.get("/charts/lijst/{feed}")
def lees_lijst_stand(feed: str, user: User = Depends(require_user)):
    return _lees_lijst_stand(_bekende_feed(feed))


# De oude Top 40-routes blijven bestaan: scripts die ze aanroepen worden niet
# tegelijk met de server bijgewerkt, en een app die al geïnstalleerd is kent
# alleen deze.
@router.post("/charts/top40")
def zet_top40(
    items: list[dict] = Body(..., embed=True),
    title: str = Body(STANDAARD_TITEL, embed=True),
    user: User = Depends(require_admin),
):
    return _zet_lijst(FEED_TOP40, items, title, user)


@router.post("/charts/top40/refresh")
def ververs_top40(user: User = Depends(require_admin)):
    return _ververs_lijst(FEED_TOP40, user)


@router.get("/charts/top40")
def lees_top40(user: User = Depends(require_user)):
    return _lees_lijst_stand(FEED_TOP40)

