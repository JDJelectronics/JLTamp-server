"""Offline checks: vector store round-trip, growth, mismatch guard, scoring."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.embed_store import EmbeddingStore, DimensionMismatch, normalise
from app import scoring
from app.jltamp_client import Track

tmp = Path(tempfile.mkdtemp())
fails = []


def check(name, cond, detail=""):
    print(f"  {'✅' if cond else '❌'} {name}{'' if cond else f'  → {detail}'}")
    if not cond:
        fails.append(name)


print("\n── Vector store ──")
store = EmbeddingStore(tmp / "vec", dim=8, model_id="test-model")
rng = np.random.default_rng(42)
vecs = {f"t{i}": rng.normal(size=8).astype(np.float32) for i in range(10)}
store.add_many(vecs)
store.save()
check("10 vectoren opgeslagen", len(store) == 10, len(store))

got = store.matrix(["t3"])[0][0]
expect = normalise(vecs["t3"])
check("vector komt genormaliseerd terug", np.allclose(got, expect, atol=1e-6))
check("norm is 1", abs(float(np.linalg.norm(got)) - 1.0) < 1e-5)

# Self-similarity must be the maximum.
sims = store.search(vecs["t3"], list(vecs))
best = max(sims, key=sims.get)
check("meest gelijkend op zichzelf is zichzelf", best == "t3", best)
check("cosine van zichzelf ≈ 1", abs(sims["t3"] - 1.0) < 1e-5, sims["t3"])

# Growth past the initial capacity must preserve earlier rows.
big = {f"b{i}": rng.normal(size=8).astype(np.float32) for i in range(5000)}
store.add_many(big)
store.save()
check("5010 vectoren na groei", len(store) == 5010, len(store))
still = store.matrix(["t3"])[0][0]
check("oude vector overleeft hergroei", np.allclose(still, expect, atol=1e-6))

# Reopen: data must survive a restart.
reopened = EmbeddingStore(tmp / "vec", dim=8, model_id="test-model")
check("index overleeft herstart", len(reopened) == 5010, len(reopened))
check("vector overleeft herstart",
      np.allclose(reopened.matrix(["t3"])[0][0], expect, atol=1e-6))

# The guard that the old engine lacked.
try:
    EmbeddingStore(tmp / "vec", dim=16, model_id="test-model")
    check("andere dimensie wordt geweigerd", False, "geen fout opgegooid")
except DimensionMismatch:
    check("andere dimensie wordt geweigerd", True)

try:
    EmbeddingStore(tmp / "vec", dim=8, model_id="ander-model")
    check("ander model wordt geweigerd", False, "geen fout opgegooid")
except DimensionMismatch:
    check("ander model wordt geweigerd", True)

print("\n── Zoeken zonder de grote kopie ──")
# search() en top_keys() vermenigvuldigen nu één keer over de hele memmap en
# indexeren daarna de scores, in plaats van eerst 307 MB aan rijen te kopiëren.
# Dat moet exact hetzelfde opleveren als de kopieerweg, anders is de winst
# waardeloos.
all_keys = list(vecs) + list(big)
probe = rng.normal(size=8).astype(np.float32)
mat, present = store.matrix(all_keys)
expect = dict(zip(present, (mat @ normalise(probe)).tolist()))
got = store.search(probe, all_keys)
check("dezelfde sleutels", set(got) == set(expect), len(got))
check("dezelfde scores",
      all(abs(got[k] - expect[k]) < 1e-6 for k in expect),
      max(abs(got[k] - expect[k]) for k in expect) if expect else "leeg")

ranked = sorted(expect, key=expect.get, reverse=True)[:5]
check("top_keys geeft dezelfde volgorde",
      [k for k, _ in store.top_keys(probe, all_keys, 5)] == ranked)
check("top_keys van niets is leeg", store.top_keys(probe, all_keys, 0) == [])
check("onbekende sleutels tellen niet mee",
      store.search(probe, ["bestaat-niet"]) == {})
check("de dichtstbijzijnde van zichzelf is zichzelf",
      store.top_keys(vecs["t3"], all_keys, 1)[0][0] == "t3")

print("\n── Prompt-analyse ──")
check("jaartal uit 'jaren 80'", scoring.extract_year("jaren 80 hits") == 1980)
check("jaartal uit '90s'", scoring.extract_year("90s rock") == 1990)
check("expliciet jaartal", scoring.extract_year("hits uit 1984") == 1984)
check("geen jaartal", scoring.extract_year("chill muziek") is None)
check("context 'gym'", "gym" in scoring.active_contexts("gym playlist"))
check("uitsluiting", scoring.exclusions("feest zonder metal") == ["metal"])
check("playlist-naam leesbaar",
      scoring.playlist_name("maak een playlist voor chill avond") == "🤖 Chill Avond",
      scoring.playlist_name("maak een playlist voor chill avond"))

print("\n── Scoring ──")


def mk(key, artist, title, **kw):
    t = Track(rating_key=key, title=title, artist=artist,
              orig_artist=kw.get("orig_artist", ""), album=kw.get("album", ""),
              year=kw.get("year"), genre=kw.get("genre", ""), duration_ms=200000,
              play_count=kw.get("play_count", 0), last_played_at=0,
              rating=0.0)
    t.skips = kw.get("skips", 0)
    t.liked = kw.get("liked", False)
    t.features = kw.get("features", {})
    return t


tracks = [
    mk("t1", "Adele", "Hello", genre="pop", year=2015),
    mk("t2", "Metallica", "One", genre="metal", year=1988),
    mk("t3", "Peppa Pig", "Bing Bong", genre="kids"),
    mk("t4", "Calvin Harris", "Summer", genre="dance", year=2014,
       features={"bpm": 128, "energy": 0.15}),
    mk("t5", "Ludovico Einaudi", "Nuvole Bianche", genre="classical",
       features={"bpm": 62, "energy": 0.03}),
]
sim = {t.rating_key: 0.5 for t in tracks}

res = scoring.score_tracks("chill muziek", tracks, sim)
check("kindermuziek eruit zonder erom te vragen",
      all(t.rating_key != "t3" for t, _ in res))

res = scoring.score_tracks("kids muziek voor peuters", tracks, sim)
check("kindermuziek erin mét vraag erom",
      any(t.rating_key == "t3" for t, _ in res))

res = scoring.score_tracks("feest zonder metal", tracks, sim)
check("uitsluiting werkt", all(t.rating_key != "t2" for t, _ in res))

by_key = {t.rating_key: s for t, s in scoring.score_tracks("gym", tracks, sim)}
check("gym: 128 BPM scoort boven 62 BPM",
      by_key.get("t4", 0) > by_key.get("t5", 0),
      f"t4={by_key.get('t4'):.2f} t5={by_key.get('t5'):.2f}")

sleep = {t.rating_key: s for t, s in scoring.score_tracks("slapen", tracks, sim)}
check("slapen: 62 BPM scoort boven 128 BPM",
      sleep.get("t5", 0) > sleep.get("t4", 0),
      f"t5={sleep.get('t5'):.2f} t4={sleep.get('t4'):.2f}")

skipped = [mk("s1", "A", "X"), mk("s2", "B", "Y", skips=4)]
sc = {t.rating_key: s for t, s in
      scoring.score_tracks("muziek", skipped, {"s1": 0.5, "s2": 0.5})}
check("weggeklikte tracks zakken", sc["s1"] > sc["s2"], f"{sc['s1']:.2f} vs {sc['s2']:.2f}")

liked = [mk("l1", "A", "X"), mk("l2", "B", "Y", liked=True)]
sc = {t.rating_key: s for t, s in
      scoring.score_tracks("muziek", liked, {"l1": 0.5, "l2": 0.5})}
check("gelikete tracks stijgen", sc["l2"] > sc["l1"], f"{sc['l2']:.2f} vs {sc['l1']:.2f}")

# A track with no vector is unknown, not neutral — it must not appear.
res = scoring.score_tracks("muziek", tracks, {"t1": 0.5})
check("track zonder vector doet niet mee", len(res) == 1, len(res))

print("\n── Jaartal is een harde filter ──")
old = [mk("y1", "A", "X", year=1985), mk("y2", "B", "Y", year=2020),
       mk("y3", "C", "Z")]           # geen jaar bekend
keys = {t.rating_key for t, _ in
        scoring.score_tracks("jaren 80", old, {"y1": .5, "y2": .5, "y3": .5})}
check("1985 blijft", "y1" in keys)
check("2020 valt weg", "y2" not in keys)
check("onbekend jaar valt weg", "y3" not in keys)

print("\n── Geen match geeft niets terug, geen bagger ──")
# select() publiceerde eerder bij een mislukking gewoon de top-20, ongeacht
# score. Dat presenteert een mislukking als resultaat.
niets = [(mk(f"n{i}", "X", "Y"), 0.05) for i in range(30)]
check("kansloze scores → lege lijst", scoring.select(niets) == [], len(scoring.select(niets)))
bijna = [(mk(f"b{i}", f"Art {i}", "Y"), scoring.SCORING["MIN_SCORE"] - 0.04)
         for i in range(30)]
check("net onder de drempel mag wel door", len(scoring.select(bijna)) > 0)

print("\n── Uitsluiting matcht geen woorddelen ──")
# "no" zonder woordgrens matchte op "pia-no", waardoor het woord erna als
# uitsluiting gold: "rustige piano muziek" gooide alles met "muziek" weg.
check("piano triggert geen uitsluiting",
      scoring.exclusions("rustige piano muziek om te slapen") == [],
      scoring.exclusions("rustige piano muziek om te slapen"))
check("casino triggert geen uitsluiting",
      scoring.exclusions("casino royale soundtrack") == [])
check("techno triggert geen uitsluiting",
      scoring.exclusions("techno muziek") == [])
check("echte uitsluiting werkt nog", scoring.exclusions("feest zonder metal") == ["metal"])
check("geen-uitsluiting werkt nog", scoring.exclusions("geen hardstyle") == ["hardstyle"])

# Een uitsluiting wordt als substring op de metadata gelegd, dus een kort
# alledaags woord sloopt de halve bibliotheek: "niet bij na hoef te denken"
# gooide alles met de letters b-i-j eruit.
check("korte functiewoorden sluiten niets uit",
      scoring.exclusions("iets waar ik niet bij na hoef te denken") == [],
      scoring.exclusions("iets waar ik niet bij na hoef te denken"))
check("'niet meer' is grammatica, geen filter",
      scoring.exclusions("nummers die ik al jaren niet meer hoor") == [])
check("'geen muziek' sluit niet het woord muziek uit",
      scoring.exclusions("geen muziek van vroeger") == [])

print("\n── Een ontkenning zet de context niet aan ──")
# "niks te hard" gaf hardstyle terug: 'hard' zette een context aan die 130-200
# BPM wil, en niemand keek naar de ontkenning ervoor.
check("'niks te hard' vraagt niet om harde muziek",
      "hard" not in scoring.active_contexts("niks te hard, mijn moeder is er"),
      scoring.active_contexts("niks te hard, mijn moeder is er"))
check("'geen hardstyle' zet de hard-context ook uit",
      "hard" not in scoring.active_contexts("feest maar geen hardstyle"))
check("zonder ontkenning gewoon aan",
      "hard" in scoring.active_contexts("lekker hard"))
check("een andere context blijft staan",
      "feest" in scoring.active_contexts("feest maar geen hardstyle"))

print("\n── Verzamelalbums ──")
comp = mk("c1", "Various Artists", "hartenbreker", orig_artist="Doe Maar",
          album="Radio Piepschuim")
check("echte artiest wordt gebruikt", comp.real_artist == "Doe Maar", comp.real_artist)
check("verzamel-albumnaam blijft uit de tekst",
      "Piepschuim" not in comp.text, comp.text)
check("embedding-tekst is betekenisvol",
      comp.text == "Doe Maar - hartenbreker", comp.text)
normal = mk("c2", "Adele", "Hello", album="25", genre="pop")
check("gewone track houdt zijn album",
      normal.text == "Adele - Hello - pop - 25", normal.text)
junk = mk("c3", "The Weeknd", "1 - The Weeknd - Blinding Lights")
check("index en dubbele artiest uit de titel",
      junk.clean_title == "Blinding Lights", junk.clean_title)

print("\n── Variatie zit in de selectie, niet in de score ──")
# Twee derde van de bibliotheek staat op verzamelalbums met artiest
# "Various Artists". Toen de straf in de score zat en op dat veld keek, werd
# de beste treffer 3711 plaatsen omlaag geduwd door onverwante tracks.
comp = [mk(f"v{i}", "Various Artists", f"Nummer {i}",
           orig_artist=f"Artiest {i}") for i in range(30)]
comp.append(mk("target", "Various Artists", "Piano for Sleep",
               orig_artist="Sleep Fruits Music"))
sims = {t.rating_key: 0.5 for t in comp}
sims["target"] = 0.85
ranked = scoring.score_tracks("rustige piano om te slapen", comp, sims)
check("beste treffer op verzamelalbum staat bovenaan",
      ranked[0][0].rating_key == "target",
      f"positie {[t.rating_key for t, _ in ranked].index('target')}")

# Eén artiest mag de lijst niet vullen — dat regelt select().
flood = [mk(f"a{i}", "Adele", f"Song {i}") for i in range(10)]
flood += [mk(f"b{i}", f"Ander {i}", f"Track {i}") for i in range(10)]
sc = scoring.score_tracks("muziek", flood, {t.rating_key: 0.7 for t in flood})
picked = scoring.select(sc, limit=10, per_artist=3)
adele = sum(1 for t in picked if t.real_artist == "Adele")
check("hooguit 3 nummers van dezelfde artiest", adele <= 3, f"{adele} stuks")
check("lijst wordt wel volgemaakt", len(picked) == 10, len(picked))

# Bij te weinig verschillende artiesten liever aanvullen dan te kort teruggeven.
few = [mk(f"c{i}", "Adele", f"Song {i}") for i in range(8)]
picked = scoring.select(
    scoring.score_tracks("muziek", few, {t.rating_key: 0.7 for t in few}),
    limit=6, per_artist=3)
check("vult aan als er te weinig artiesten zijn", len(picked) == 6, len(picked))

print("\n── Een gewoon woord is geen artiestnaam ──")
check("'focus' in een zin is niet de band Focus",
      not scoring.names_artist("instrumentale focus muziek", "Focus"))
check("'van focus' is wel de band", scoring.names_artist("iets van focus", "Focus"))
check("lange naam matcht gewoon",
      scoring.names_artist("de leukste van metallica", "Metallica"))
check("korte prompt telt als naam", scoring.names_artist("adele", "Adele"))

print("\n── Likes en skips lekken niet tussen gebruikers ──")
# De Track-objecten zijn gedeeld: library.snapshot() geeft iedereen dezelfde
# lijst. Eerder schreef de engine de likes van de beller op die objecten, dus
# scoorde de ene gebruiker met de geschiedenis van de andere — en wie zonder
# token vroeg, erfde die van de laatste beller.
shared = [mk("u1", "A", "X"), mk("u2", "B", "Y")]
sims_u = {"u1": 0.5, "u2": 0.5}
ann = scoring.Signals(likes={"u1"}, skips={})
bob = scoring.Signals(likes={"u2"}, skips={"u1": 4})

sc_ann = {t.rating_key: s for t, s in
          scoring.score_tracks("muziek", shared, sims_u, signals=ann)}
sc_bob = {t.rating_key: s for t, s in
          scoring.score_tracks("muziek", shared, sims_u, signals=bob)}
check("de like van A telt voor A", sc_ann["u1"] > sc_ann["u2"],
      f"{sc_ann['u1']:.3f} vs {sc_ann['u2']:.3f}")
check("de like van B telt voor B", sc_bob["u2"] > sc_bob["u1"],
      f"{sc_bob['u2']:.3f} vs {sc_bob['u1']:.3f}")
check("scoren muteert de gedeelde tracks niet",
      all(t.liked is False and t.skips == 0 for t in shared),
      [(t.rating_key, t.liked, t.skips) for t in shared])

# Zonder signalen gelden de velden van de track zelf, zoals de library ze zette.
own = [mk("o1", "A", "X"), mk("o2", "B", "Y", liked=True)]
sc_own = {t.rating_key: s for t, s in
          scoring.score_tracks("muziek", own, {"o1": 0.5, "o2": 0.5})}
check("zonder beller telt de library-geschiedenis", sc_own["o2"] > sc_own["o1"])

print("\n── Aanvullen levert geen dubbel nummer op ──")
# De opvulling uit de overflow sloeg de dubbel-check over. Twee exemplaren van
# hetzelfde nummer — één op het album, één op een verzamelaar — komen allebei
# in de overflow terecht zodra de artiestlimiet vol is, want die tracks worden
# nooit aan `songs` toegevoegd. De aanvulling pakte ze daarna allebei.
dup = [mk("d1", "Adele", "Hello"), mk("d2", "Adele", "Someone Like You"),
       mk("d3", "Adele", "Rolling In The Deep"),
       mk("d4", "Adele", "Skyfall"),
       mk("d5", "Various Artists", "Skyfall", orig_artist="Adele")]
picked = scoring.select([(t, 0.9) for t in dup], limit=5, per_artist=3)
titles = [t.clean_title.lower() for t in picked]
check("geen dubbel nummer na aanvullen", len(titles) == len(set(titles)), titles)
check("wel aangevuld tot wat er is", len(picked) == 4, len(picked))

print("\n── Trefwoorden overstemmen semantiek niet ──")
# 'Focus' in de titel mag een écht beter passend nummer niet verslaan.
kw = [mk("k1", "Ariana Grande", "Focus"), mk("k2", "Brian Eno", "Ambient 1")]
sc = {t.rating_key: s for t, s in
      scoring.score_tracks("instrumentale focus muziek", kw,
                           {"k1": 0.45, "k2": 0.62})}
check("hogere semantische score wint van woordmatch",
      sc["k2"] > sc["k1"], f"k2={sc['k2']:.3f} k1={sc['k1']:.3f}")

print("\n── Een afbouw loopt niet halverwege weer omhoog ──")
# De echte pool is tweetoppig: iemands favorieten zijn opzwepend, hun
# slaapmuziek is traag, en daartussen zit vrijwel niets. Zodra het trage
# materiaal op was, koos "dichtst bij het doeltempo" weer snelle nummers en
# klom de afbouw van 27 terug naar 117 BPM.
def tempo(key, bpm, artist="A"):
    return mk(key, f"{artist}{bpm}", f"Song {key}", features={"bpm": bpm})


bimodal = [(tempo(f"s{i}", 120 - i % 5), 120 - i % 5) for i in range(40)]
bimodal += [(tempo(f"t{i}", 50 - i), 50 - i) for i in range(6)]
curve = scoring.descending_tempo_curve(bimodal, 50)
speeds = [t.features["bpm"] for t in curve]
check("het tempo loopt alleen omlaag",
      all(speeds[i + 1] <= speeds[i] + 2 for i in range(len(speeds) - 1)), speeds)
check("liever korter dan verkeerd om", len(curve) < 50, len(curve))

# Met genoeg materiaal blijft het gewoon een volle, gladde glijbaan. De pool
# moet ruim zijn: de curve begint op het 75e percentiel, dus alles daarboven
# doet per definitie niet mee — met precies 60 nummers vallen er 14 af en haalt
# een eerlijke curve de 50 niet.
dense = [(tempo(f"d{i}", 140 - i, f"Art{i}"), 140 - i) for i in range(80)]
full = scoring.descending_tempo_curve(dense, 50)
fast = [t.features["bpm"] for t in full]
check("volle curve bij genoeg materiaal", len(full) == 50, len(full))
check("en die daalt van hoog naar laag", fast[0] > fast[-1], f"{fast[0]} → {fast[-1]}")

print("\n── Tempocurves in andere vormen ──")
material = [(tempo(f"m{i}", 60 + i, f"Art{i}"), 60 + i) for i in range(90)]

up = scoring.tempo_arc(material, 30, "up")
ups = [t.features["bpm"] for t in up]
check("opbouw loopt alleen omhoog",
      all(ups[i + 1] >= ups[i] - 2 for i in range(len(ups) - 1)), ups)
check("opbouw begint traag en eindigt snel", ups[-1] > ups[0], f"{ups[0]} → {ups[-1]}")

peak = scoring.tempo_arc(material, 30, "peak")
peaks = [t.features["bpm"] for t in peak]
top = peaks.index(max(peaks))
check("de boog piekt in het midden", 8 < top < 22, f"piek op {top} van {len(peaks)}")
check("en komt daarna terug omlaag", peaks[-1] < max(peaks) - 10,
      f"{max(peaks)} → {peaks[-1]}")
check("onbekende vorm geeft niets", scoring.tempo_arc(material, 10, "zigzag") == [])

print("\n── Intervaltraining ──")
check("standaard als de prompt niets zegt",
      scoring.interval_spec("intervaltraining") == (1800, 240, 120),
      scoring.interval_spec("intervaltraining"))
check("totale duur uit de prompt",
      scoring.interval_spec("intervaltraining van 40 minuten")[0] == 2400)
check("blokken uit de prompt, ongeacht volgorde",
      scoring.interval_spec("3 minuten hard en 1 minuut rustig, 40 minuten")
      == (2400, 180, 60),
      scoring.interval_spec("3 minuten hard en 1 minuut rustig, 40 minuten"))

# Elk nummer duurt 200 s in mk(), dus het plan is precies uit te rekenen:
# warmdraaien (1 nummer), dan blokken van 2 hard en 1 rustig tot 30 minuten.
ivl = [(tempo(f"i{i}", 60 + i * 2, f"Art{i}"), 60 + i * 2) for i in range(60)]
plan = scoring.interval_blocks(ivl, 1800, 240, 120)
phases = [p for _, p in plan]
check("begint rustig, als warming-up", phases[0] == "rustig", phases[:3])
check("wisselt af tussen hard en rustig", "hard" in phases and "rustig" in phases)
check("vult de gevraagde tijd",
      sum(t.duration_ms for t, _ in plan) / 1000 >= 1800,
      sum(t.duration_ms for t, _ in plan) / 1000)
hard_bpm = [t.features["bpm"] for t, p in plan if p == "hard"]
easy_bpm = [t.features["bpm"] for t, p in plan if p == "rustig"]
check("de harde blokken zijn echt sneller",
      sum(hard_bpm) / len(hard_bpm) > sum(easy_bpm) / len(easy_bpm),
      f"{sum(hard_bpm) / len(hard_bpm):.0f} vs {sum(easy_bpm) / len(easy_bpm):.0f}")

# Zonder speelduur valt een blok niet te vullen; die tracks doen niet mee.
timeless = [mk("z1", "A", "X", features={"bpm": 120})]
timeless[0].duration_ms = 0
check("nummers zonder speelduur vallen af",
      scoring.interval_blocks([(timeless[0], 120)], 600, 240, 120) == [])

print("\n── De engine routeert niet op een toevallige woordtreffer ──")
# _find_artist besliste met een kale substring-test welke strategie een prompt
# kreeg. Dat is precies waar names_artist tegen geschreven is: "instrumentale
# focus muziek" werd zo een best-of van de band Focus. Een misplaatste boost is
# hinderlijk; een hele playlist van de verkeerde artiest is dat veel meer.
from types import SimpleNamespace                              # noqa: E402

from app.engine import Engine                                  # noqa: E402

eng = Engine.__new__(Engine)          # geen netwerk, geen __init__
eng._artists, eng._artists_at = {}, 0.0
eng.library = SimpleNamespace(loaded_at=1.0)

lib = [mk("f1", "Focus", "Hocus Pocus"), mk("f2", "Adele", "Hello"),
       mk("f3", "Metallica", "One"),
       mk("f4", "Various Artists", "Hartenbreker", orig_artist="Doe Maar")]

check("gewoon woord routeert niet naar een artiest",
      eng._find_artist("instrumentale focus muziek om bij te werken", lib) is None,
      eng._find_artist("instrumentale focus muziek om bij te werken", lib))
check("met aanwijzer is het wel de band",
      eng._find_artist("de leukste van focus", lib) == "Focus")
check("lange naam wordt gewoon gevonden",
      eng._find_artist("iets zoals metallica", lib) == "Metallica")
check("artiest op een verzamelalbum wordt gevonden",
      eng._find_artist("de beste van doe maar", lib) == "Doe Maar",
      eng._find_artist("de beste van doe maar", lib))
check("verzamel-plaatshouder is geen artiest",
      "various artists" not in eng._artist_index(lib))
check("index wordt hergebruikt tot de library ververst",
      eng._artist_index(lib) is eng._artist_index(lib))

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'❌ ' + str(len(fails)) + ' gefaald: ' + ', '.join(fails) if fails else '✅ alles geslaagd'}")
sys.exit(1 if fails else 0)
