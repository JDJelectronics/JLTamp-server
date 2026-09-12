"""Waar de tellen van een nummer vallen, uit het geluid zelf.

Dit bestaat omdat niets anders het kan. Een gloed die met de muziek meeklopt
heeft twee dingen nodig: het tempo én de plek van de tel. Deezer geeft alleen
het eerste, en dan loopt een gloed wel in de goede snelheid maar naast de
muziek. En Deezer kent lang niet alles: op een greep uit deze bibliotheek wist
hij van één op de veertien nummers het tempo, want er staat veel Nederlands en
obscuur materiaal in. Analyse van het geluid werkt overal, ook bij een opname
die nergens in een catalogus staat.

**Wat er is geprobeerd en niet werkte.** Eerst een eigen beatvolger in puur
Python (energieomhullende, autocorrelatie, fase erbij zoeken). Snel — tienden
van seconden — maar afgezet tegen zes nummers met bekend tempo kwamen er twee
goed uit. Half- en dubbeltempo-missers, de klassieke valkuil. Vandaar librosa,
dat hier al jaren voor gemaakt is.

**Deezer als startpunt.** Waar librosa het tempo van een nummer verkeerd inzet,
is dat meestal het halve of het dubbele. Een goed beginpunt haalt dat er vaak
uit. Op acht nummers waarvan Deezer het tempo kende, kwam librosa er alleen
op vijf uit en met dat startpunt op zes. Kent Deezer het tempo niet, dan gaat
librosa zonder startpunt aan de slag — en dat is nog altijd oneindig veel beter
dan geen tellen.

**Het blijft een schatting.** Beide methodes zijn dat, en waar ze het oneens
zijn is niet te zeggen wie gelijk heeft. Reken op ongeveer drie van de vier
nummers goed. Voor een gloed is dat prima; voor iets waar een beslissing van
afhangt zou het dat niet zijn.

De uitkomst gaat in de cache naast de andere afgeleide gegevens. Analyseren
duurt een halve tot een paar seconden per nummer, één keer.
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger("beats")

# Het HELE nummer wordt nu geanalyseerd, niet een stuk uit het midden.
#
# Eerst werd er vanaf twintig seconden een minuut beluisterd; buiten dat stuk
# trok de app het raster door met de mediane tussenpoos. Dat gaat goed zolang
# het tempo gelijk blijft, maar het zegt niets over hoe LUID de muziek op dat
# moment is — en een gloed die niet met de muziek meeademt, gaat niet echt mee.
# Vandaar ook de energiecurve hieronder.
SR = 22050

# Hoe fijn de luidheid wordt gemeten. Vier metingen per seconde is genoeg om een
# intro van een refrein te onderscheiden en houdt het antwoord klein: een nummer
# van vier minuten wordt zo een lijst van ongeveer duizend getallen.
STAP_MS = 250


def _decodeer(pad: str):
    """Het geluid als reeks monsters. ffmpeg zit al in het image voor het
    omzetten van FLAC naar mp3, dus er komt geen decodeerbibliotheek bij."""
    import numpy as np

    uit = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", pad,
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, timeout=300,
    ).stdout
    return np.frombuffer(uit, dtype="<f4").copy()


def _energie(y, np) -> list[float]:
    """Hoe luid het nummer is, per STAP_MS, geschaald naar 0 tot 1.

    Niet gedeeld door de hoogste piek: één knal maakt dan de rest van het nummer
    plat. Er wordt geschaald op een hoog percentiel, en daarboven afgekapt — dan
    ligt het refrein rond de één en blijft het verschil met een couplet te zien.
    """
    vak = int(SR * STAP_MS / 1000)
    if vak <= 0 or y.size < vak:
        return []
    aantal = y.size // vak
    blokken = y[: aantal * vak].reshape(aantal, vak)
    rms = np.sqrt(np.mean(np.square(blokken, dtype=np.float64), axis=1))
    top = float(np.percentile(rms, 95)) or float(rms.max() or 1.0)
    if top <= 0:
        return []
    genormeerd = np.clip(rms / top, 0.0, 1.0)
    return [round(float(v), 3) for v in genormeerd]


def tellen_van(pad: str, start_bpm: float = 0.0) -> dict:
    """{bpm, tellen, energie, stap} — tellen in milliseconden vanaf het begin,
    en hoe luid het nummer is per `stap` milliseconden.

    Leeg bij een bestand dat niet te lezen is of te kort om iets over te zeggen.
    """
    try:
        import numpy as np
        import librosa
    except Exception as e:                       # pragma: no cover
        log.warning("beatanalyse niet beschikbaar: %s", e)
        return {"bpm": 0.0, "tellen": [], "energie": [], "stap": STAP_MS}

    try:
        y = _decodeer(pad)
    except Exception as e:
        log.info("decoderen mislukt (%s): %s", pad, e)
        return {"bpm": 0.0, "tellen": [], "energie": [], "stap": STAP_MS}

    if y.size < SR * 5:
        return {"bpm": 0.0, "tellen": [], "energie": [], "stap": STAP_MS}

    try:
        kw = {"start_bpm": float(start_bpm)} if 40 <= start_bpm <= 220 else {}
        tempo, tellen = librosa.beat.beat_track(y=y, sr=SR, units="time", **kw)
        bpm = float(np.atleast_1d(tempo)[0])
        # Het hele nummer is geanalyseerd, dus de tijden kloppen al.
        ms = [int(round(float(t) * 1000)) for t in tellen]
        energie = _energie(y, np)
    except Exception as e:
        log.info("beatanalyse mislukt (%s): %s", pad, e)
        return {"bpm": 0.0, "tellen": [], "energie": [], "stap": STAP_MS}

    if len(ms) < 8 or not (40 <= bpm <= 220):
        return {"bpm": 0.0, "tellen": [], "energie": energie, "stap": STAP_MS}

    return {"bpm": round(bpm, 2), "tellen": ms, "energie": energie, "stap": STAP_MS}
