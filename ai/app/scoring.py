"""Turning a prompt into a ranked selection of tracks.

Structure follows the old engine — semantic similarity first, then boosts and
penalties — with three changes:

* Similarity is one matrix multiply over the whole library instead of a Python
  loop over a 2000-track random sample, so every track gets considered.
* Skips come from real listening data (a track abandoned in the first third),
  not from "the user pressed pause".
* Audio features (BPM/energy) steer context prompts. "Gym" wants fast and loud;
  no amount of text similarity on "artist - title" can know that, but the
  analyser measured it.
"""
from __future__ import annotations

import random
import re

import numpy as np

from .config import SCORING
from .jltamp_client import Track

# Dutch/English mood and activity words → the vocabulary a track's metadata
# might use for the same idea.
CONTEXT_MAPPER: dict[str, list[str]] = {
    "gym": ["workout", "power", "energy", "fitness", "beast mode", "lifting", "training", "crossfit"],
    "hardloop": ["running", "pace", "cardio", "fast", "marathon", "jogging", "sprint"],
    "focus": ["study", "werk", "concentratie", "lofi", "deep work", "instrumental", "minimal", "coding", "lezen"],
    "relax": ["chill", "mellow", "ontspannen", "sofa", "rustig", "lounging", "acoustic", "bad"],
    "feest": ["party", "dance", "viering", "uitgaan", "club", "disco", "gezelligheid", "verjaardag"],
    "schoonmaak": ["cleaning", "upbeat", "motivation", "sing-along", "energy", "huishouden", "opruimen"],
    "koken": ["dinner", "cooking", "jazz", "bossa nova", "italiano", "kitchen", "bakken", "tafelen"],
    "slapen": ["sleep", "insomnia", "calm", "meditation", "ambient", "night", "rust", "sluimer"],
    "auto": ["driving", "roadtrip", "onderweg", "sing-along", "travel"],
    "vrolijk": ["happy", "sunny", "feel good", "joy", "upbeat", "lachen", "positief", "blij"],
    "verdrietig": ["sad", "heartbreak", "melancholy", "tranen", "emo", "huilen"],
    "boos": ["angry", "rage", "aggressive", "metal", "hardcore", "punk", "furious", "agressief"],
    "romantisch": ["love", "romantic", "sexy", "ballad", "slow jam", "date night", "valentijn", "verliefd"],
    "nostalgie": ["throwback", "herinneringen", "old school", "classic", "vroeger", "jeugd"],
    "zomer": ["summer", "beach", "tropical", "sun", "warm", "reggae", "cocktails", "terras"],
    "winter": ["cold", "cozy", "kerst", "christmas", "snow", "fireplace", "winter", "knus"],
    "regen": ["rain", "gloomy", "storm", "indoors", "cozy", "herfst"],
    "akoestisch": ["acoustic", "unplugged", "guitar", "piano", "live", "puur"],
    "instrumentaal": ["no vocals", "instrumental", "orchestral", "beat", "karaoke", "geen zang"],
    "hard": ["loud", "heavy", "bass boost", "distorted", "extreme", "beuken", "hardstyle"],
}

# What the measured audio should look like for a context: (bpm range, min energy).
# `None` means "don't care". Energy is the mean RMS the analyser records.
AUDIO_TARGETS: dict[str, tuple[tuple[int, int] | None, float | None]] = {
    "gym": ((120, 180), 0.10),
    "hardloop": ((150, 190), 0.09),
    "feest": ((115, 140), 0.09),
    "hard": ((130, 200), 0.12),
    "schoonmaak": ((100, 140), 0.08),
    "focus": ((60, 105), None),
    "relax": ((55, 100), None),
    "slapen": ((40, 90), None),
    "romantisch": ((50, 100), None),
}

KIDS_WORDS = ["peppa", "kinder", "kids", "nursery", "lullaby", "nijntje",
              "juf roos", "zandkasteel", "koekeloere", "bumba", "kabouter"]
KIDS_ASKED = ["kids", "kind", "baby", "peuter", "kleuter"] + KIDS_WORDS

_DECADES = {
    r"\b60s\b|\bjaren 60\b": 1960, r"\b70s\b|\bjaren 70\b": 1970,
    r"\b80s\b|\bjaren 80\b": 1980, r"\b90s\b|\bjaren 90\b": 1990,
    r"\b00s\b|\bjaren 00\b": 2000, r"\b10s\b|\bjaren 10\b": 2010,
    r"\b20s\b|\bjaren 20\b": 2020,
}

_NAME_NOISE = re.compile(
    r"\b(maak|zoek|geef|een|de|het|playlist|lijst|lijstje|voor|ik|wil|graag|"
    r"met|alleen|zonder|geen|muziek|nummers|songs|mij|om|te|van|op|in|iets|"
    r"wat|make|me|some|a|the|for|with|music|songs?|playlist|bij|nog|even)\b",
    re.IGNORECASE)


# How far below the score floor a result may still be published when nothing
# clears it outright. Wide enough to keep a usable near miss, narrow enough
# that a prompt with no real match returns nothing and says so.
NEAR_MISS = 0.08

# Words that say "this is music" and nothing else. In a music library they
# match everything, so they must not earn a keyword boost and must not be
# added as an anchor to a prompt that already contains one.
GENERIC_MUSIC_WORDS = {
    "muziek", "music", "nummer", "nummers", "song", "songs", "lied", "liedje",
    "liedjes", "track", "tracks", "playlist", "lijst", "mix", "sound", "sounds",
    "genre", "audio",
}


def expand_query(prompt: str) -> str:
    """Give a very short prompt enough context to embed meaningfully.

    "hiphop" on its own returned Habiba, Herinnering, Halo and Hawái — the
    model had so little to work with that surface similarity (words starting
    with an H) outweighed meaning. Phrasing it as a sentence about music gives
    the embedding something to anchor to, and matches how the track texts
    themselves read.

    Only applied to short prompts; a descriptive one already carries context
    and rewriting it would blur what the user actually asked for.

    IMPORTANT: the expansion must let the PROMPT dominate the embedded text.
    The old wording — "{prompt} muziek nummers in het genre {prompt}" — buried a
    1-word prompt under a 4-word constant phrase, so EVERY short prompt embedded
    to almost the same generic point and the single track nearest that point
    (a sparsely-tagged Dido track) led every playlist → every AI playlist got
    the same Dido cover. Here the prompt appears 3× against only 2 light anchor
    words, so different prompts diverge again while still reading as "music".
    """
    words = prompt.split()
    if len(words) > 3:
        return prompt
    # Do not add "muziek" to a prompt that already has it. "klassieke muziek"
    # expanded to a phrase containing it four times, so the least informative
    # word in a music library became the strongest term — the top results were
    # Dutch songs with "muziek" in the title, while the actual classical piano
    # tracks the embedding had found (0.745+) dropped out entirely.
    if any(w.strip(",.").lower() in GENERIC_MUSIC_WORDS for w in words):
        return f"{prompt}, genre {prompt}"
    return f"{prompt}, {prompt} muziek, genre {prompt}"


def extract_year(prompt: str) -> int | None:
    """A decade name, or an explicit four-digit year."""
    for pattern, year in _DECADES.items():
        if re.search(pattern, prompt):
            return year
    m = re.search(r"\b(19|20)\d{2}\b", prompt)
    return int(m.group(0)) if m else None


def active_contexts(prompt: str) -> list[str]:
    return [key for key in CONTEXT_MAPPER if key in prompt]


def context_tags(contexts: list[str]) -> set[str]:
    tags: set[str] = set()
    for key in contexts:
        tags.update(CONTEXT_MAPPER[key])
        tags.add(key)
    return tags


def exclusions(prompt: str) -> list[str]:
    """Words the user asked to leave out: "feest zonder metal" -> ["metal"].

    The leading \\b is load-bearing. Without it "no" matched the last two
    letters of "piano", so "rustige piano muziek …" was read as excluding
    "muziek" — and every track whose metadata mentioned it was dropped before
    scoring. That silently removed the correct answers from any prompt
    containing a word ending in "no", "geen" or "niet".
    """
    return re.findall(
        r"\b(?:zonder|geen|niet|no|exclude|behalve)\s+(\w+)", prompt)


def audio_fit(track: Track, contexts: list[str]) -> float:
    """0..1 — how well the measured audio matches what the contexts want.

    Returns 0 when we have no measurement, so tracks that were never analysed
    are simply not boosted rather than being pushed down.
    """
    feats = track.features or {}
    bpm = feats.get("bpm")
    energy = feats.get("energy")
    if bpm is None and energy is None:
        return 0.0

    hits, checks = 0.0, 0
    for key in contexts:
        target = AUDIO_TARGETS.get(key)
        if not target:
            continue
        bpm_range, min_energy = target
        if bpm_range and bpm:
            checks += 1
            low, high = bpm_range
            if low <= bpm <= high:
                hits += 1
            else:
                # Partial credit near the edges: 118 BPM is not "wrong" for gym.
                margin = min(abs(bpm - low), abs(bpm - high))
                hits += max(0.0, 1.0 - margin / 30.0)
        if min_energy is not None and energy is not None:
            checks += 1
            hits += 1.0 if energy >= min_energy else max(0.0, energy / min_energy)
    return hits / checks if checks else 0.0


# Words that introduce an artist: "iets van Adele", "zoals Focus".
_ARTIST_CUE = re.compile(
    r"\b(?:van|zoals|door|met|artiest|artist|lijkt op|in de stijl van)\s+$")


def names_artist(prompt: str, artist: str) -> bool:
    """Did the user actually name this artist?

    A plain substring test cannot tell "instrumentale focus muziek" from a
    request for the band Focus, and the artist boost is large enough that
    getting it wrong reorders the whole playlist — that prompt returned three
    copies of "Hocus Pocus".

    So: match on word boundaries, and for a short one-word name — the kind
    that collides with ordinary vocabulary — additionally require either a
    cue word before it or a prompt short enough to be nothing but a name.
    """
    name = (artist or "").strip().lower()
    if len(name) < 4:
        return False
    m = re.search(rf"\b{re.escape(name)}\b", prompt)
    if not m:
        return False
    ambiguous = len(name) < 7 and " " not in name
    if not ambiguous:
        return True
    if _ARTIST_CUE.search(prompt[:m.start()]):
        return True
    # No cue: only when the prompt is barely more than the name itself
    # ("adele", "metallica nummers"). At three words "instrumentale focus
    # muziek" already slips through, and it is a description, not a request
    # for the band Focus.
    return len(prompt.split()) <= 2


def playlist_name(prompt: str) -> str:
    """A short, readable title. The app shows this, so it should read like
    something a person wrote."""
    # Drop "zonder metal" entirely — naming a playlist after what it excludes
    # ("Feest Metal") says the opposite of what the user asked for.
    clean = re.sub(r"(?:zonder|geen|niet|no|exclude|behalve)\s+\w+", "", prompt)
    clean = _NAME_NOISE.sub("", clean)
    clean = re.sub(r"[^\w\s]", "", clean).strip()
    words = [w for w in clean.split() if w]
    short = " ".join(words[:4])[:32].strip()
    return f"🤖 {short.title()}" if short else "🤖 AI Mix"


def named_genres(prompt: str, tracks: list[Track]) -> set[str]:
    """Genre tags the prompt actually names.

    Asking for "hardcore" when the library has a `hardcore` tag is a much
    stronger statement than the embedding can express: it puts the best
    hardcore tracks first but scatters the rest around rank 56, so a 50-track
    playlist filled up with hardstyle — a neighbouring genre, but not what was
    asked for.

    Matched on word boundaries against the prompt, so "hard" does not claim
    every genre containing it, and only for tags of a useful length.
    """
    # Exact match only. Anything looser reintroduces the same failure by
    # another route: allowing a two-word prompt let "nederlandse pop" boost
    # thousands of tracks tagged `pop` — precisely what that prompt is trying to
    # avoid — and it collapsed the results. "hardcore" is a request for a tag;
    # "nederlandse pop" is a description that happens to contain one.
    prompt_l = prompt.lower().strip()
    return {t.genre.strip().lower() for t in tracks if t.genre.strip()} & {prompt_l}


def score_tracks(prompt: str, tracks: list[Track], similarity: dict[str, float],
                 rng: random.Random | None = None) -> list[tuple[Track, float]]:
    """Rank `tracks` for `prompt`. `similarity` is {rating_key: cosine}.

    Tracks without an embedding are dropped: a missing vector is unknown, not
    neutral, and scoring it as 0 would let boosts alone float it to the top.
    """
    rng = rng or random.Random()
    prompt = prompt.lower()
    contexts = active_contexts(prompt)
    tags = context_tags(contexts)
    year_target = extract_year(prompt)
    excludes = exclusions(prompt)
    kids_asked = any(k in prompt for k in KIDS_ASKED)
    # "muziek" and friends match half the library, so a track earning points
    # for containing one is being rewarded for nothing.
    prompt_words = [w for w in prompt.split()
                    if len(w) > 3 and w not in GENERIC_MUSIC_WORDS]
    asked_genres = named_genres(prompt, tracks)

    scored: list[tuple[Track, float]] = []

    for t in tracks:
        sim = similarity.get(t.rating_key)
        if sim is None:
            continue

        hay = t.haystack
        if any(x in hay for x in excludes):
            continue
        has_kids = any(k in hay for k in KIDS_WORDS)
        if has_kids and not kids_asked:
            continue

        # Asking for a decade is a constraint, not a preference: a 2020 track
        # in a "jaren 80" playlist is wrong however well it scores otherwise.
        # Tracks with no year at all are unknown, so they are excluded too.
        if year_target:
            if not t.year or not (year_target <= t.year <= year_target + 9):
                continue

        score = float(sim)

        if tags:
            matched = sum(1 for tag in tags if tag in hay)
            score += min(matched, 3) * SCORING["BOOST_CONTEXT_MATCH"]

        fit = audio_fit(t, contexts)
        if fit:
            score += fit * SCORING["BOOST_AUDIO_FEATURE"]

        if kids_asked and has_kids:
            score += SCORING["BOOST_KIDS_MATCH"]

        # Carrying the tag the user named beats merely sounding like it.
        if asked_genres:
            tag = t.genre.strip().lower()
            # Exact tag, not a substring: `g in tag` let "pop" claim every
            # "nederpop", "synthpop" and "pop rock" track in the library.
            if tag and tag in asked_genres:
                score += SCORING["BOOST_NAMED_GENRE"]

        # The performer, not the album's grouping artist. A large share of a
        # library sits on compilations where that
        # field reads "Various Artists"; treating it as an artist name makes
        # every one of them the same artist.
        artist = t.real_artist
        targeted = names_artist(prompt, artist)
        if targeted:
            score += SCORING["BOOST_EXPLICIT_ARTIST"]

        # Capped: three literal word matches should not outrank a genuinely
        # closer track. This boost broke the first version of the ranking.
        hits = sum(1 for w in prompt_words if w in hay)
        score += min(hits, 3) * SCORING["BOOST_KEYWORD_MATCH"]

        if t.liked:
            score += SCORING["BOOST_LIKED"]
        if t.skips:
            score -= min(t.skips, 4) * SCORING["PENALTY_SKIPPED"]
        if t.play_count == 0:
            score += SCORING["BOOST_UNPLAYED"]

        score += rng.uniform(0, 0.005)  # break ties differently each run
        scored.append((t, score))

    scored.sort(key=lambda p: p[1], reverse=True)
    return scored


def select(scored: list[tuple[Track, float]], limit: int | None = None,
           relaxed: bool = False, per_artist: int = 3) -> list[Track]:
    """Take the best tracks above the score floor, keeping the list varied.

    Variety is enforced here rather than in the score, because the question is
    "how often does this artist already appear *in this playlist*" — a property
    of the selection, not of the track. Scoring it per track meant penalising
    an artist for how much of the *library* they occupy: with tens of thousands of tracks
    credited to "Various Artists", the best match in the library came out far down the list.
    

    `relaxed` lowers the floor for prompts that carry their own strong filters
    (a decade, a context, a kids request) — there the filter already did the
    narrowing, so demanding high semantic similarity on top over-restricts.
    """
    floor = SCORING["MIN_SCORE"] - (0.1 if relaxed else 0.0)
    cap = limit or SCORING["MAX_TRACKS"]

    picked: list[Track] = []
    overflow: list[Track] = []
    seen: dict[str, int] = {}
    songs: set[tuple[str, str]] = set()
    for track, score in scored:
        if score <= floor:
            break                      # sorted, so everything after is worse
        # The same song often exists twice — on an album and on a compilation.
        # They are different tracks with different ids, so nothing upstream
        # catches it, and a playlist listing it twice just looks broken.
        song = (track.real_artist.lower(), track.clean_title.lower())
        if song in songs:
            continue
        # A placeholder credit is not an artist. Where neither the album nor
        # the track names a performer, every such track reads as "Various
        # Artists" and the cap treats hundreds of *different* acts as one:
        # carnaval had 32 of its 64 tracks inside the top 36 by similarity and
        # still got 8 slots; "muziek van jan de seuter" got 3 out of 237.
        # The rule exists to stop one artist filling the list — it should not
        # fire when we simply do not know who is playing.
        artist = track.real_artist
        known = artist.strip().lower() not in Track._COMPILATION
        if known and seen.get(artist, 0) >= per_artist:
            overflow.append(track)     # keep as filler rather than discarding
            continue
        picked.append(track)
        songs.add(song)
        if known:
            seen[artist] = seen.get(artist, 0) + 1
        if len(picked) >= cap:
            break

    # A narrow prompt may not yield `cap` distinct artists. Top up from what
    # the per-artist cap held back — those still cleared the score floor.
    if len(picked) < cap:
        picked.extend(overflow[:cap - len(picked)])

    if picked:
        return picked

    # Nothing cleared the floor. Publishing the best of a bad set anyway — the
    # previous behaviour — dresses a failure as a result: the caller cannot
    # tell "here are 20 tracks that fit" from "here are 20 tracks that don't".
    # Allow a near miss through, since the floor is a judgement call and a
    # slightly-under match is still usable, but stop well short of returning
    # whatever happened to sort first.
    if scored and scored[0][1] > floor - NEAR_MISS:
        return [t for t, s in scored[:20] if s > floor - NEAR_MISS]
    return []


def similarity_map(store, query_vec: np.ndarray, tracks: list[Track]) -> dict[str, float]:
    """Cosine of the prompt against every track we have a vector for."""
    return store.search(query_vec, [t.rating_key for t in tracks])
