# JLTamp AI 🧠🎵

Semantische playlist-generatie voor JLTamp. Je typt "chill avond muziek" of
"iets zoals Adele" in de app, en er verschijnt een playlist in je bibliotheek.

Draait op de **Jetson Xavier** (voor de GPU) en praat met de **JLTamp-server**
over het netwerk. De app vindt deze dienst zelf op poort 5000 — dat doet
`frontend/src/services/aiService.ts` al.

```
   JLTamp-app  ──►  ai/ (Jetson :5000)  ──►  JLTamp-server (:8090)
                          │                        └─ playlist verschijnt hier
                          └──►  llama.cpp (:3100, GPU)
```

## Waarom deze module bestaat

De voorganger (`~/ai_music`) las uit **Plex** en schreef playlists ook terug
naar Plex. JLTamp heeft zijn eigen database en kijkt niet naar Plex, dus de app
kreeg netjes "gelukt" te horen terwijl de playlist in een ander systeem belandde.
Deze versie praat uitsluitend met JLTamp.

## Installeren

```bash
cd ai
pip3 install -r requirements.txt
cp .env.example .env      # en invullen
./scripts/fetch_model.sh  # embedding-model, ~610 MB, eenmalig
```

Vul in `.env` minimaal in:

| Variabele | Waarvoor |
|---|---|
| `JLTAMP_URL` | Adres van je server, standaard `http://192.168.1.10:8090` |
| `JLTAMP_EMAIL` + `JLTAMP_PASSWORD` | Account waaronder de playlists worden aangemaakt |
| `AI_API_KEY` | Gedeeld geheim. Genereer met `openssl rand -hex 24` |

## Draaien

```bash
./scripts/start_embed_server.sh
```

Dat start llama.cpp op `:3100` en daarna de AI-dienst op `:5000`. Bij de eerste
start indexeert hij je hele bibliotheek — de dienst is meteen bruikbaar en vult
de rest op de achtergrond aan.

Gemeten op een Jetson Xavier met een bibliotheek van tienduizenden tracks:

| | |
|---|---|
| Bibliotheek inlezen uit JLTamp | ~2 min |
| Volledig indexeren | ~14 min (82 tracks/s) |
| Opslag | 273 MB |
| Een prompt beantwoorden | ~1 s |

Controleren of het loopt:

```bash
curl -s localhost:5000/health | python3 -m json.tool
```

In `/health` telt `features` de tracks die werkelijk een gemeten tempo hebben,
en `feature_entries` de regels in het cachebestand. Die twee lopen uiteen met
alles wat het bestand nog weet over muziek die niet meer in de bibliotheek
staat; die twee getallen lopen daardoor uiteen.

**Logs staan in de journal**, niet in `logs/engine.log`. Dat bestand wordt
alleen gevuld door `scripts/engine.sh`; onder systemd schrijft de dienst naar
journald en blijft het oude bestand ongewijzigd staan, wat bij het zoeken naar
een fout dagenlang de verkeerde kant op wijst:

```bash
sudo journalctl -u jltamp-ai -f
```

## Hoe een prompt wordt beantwoord

1. **Best of** — "de leukste van Adele" → haar eigen nummers, gesorteerd op wat
   je echt draait en waardeert.
2. **Discovery** — "iets zoals Adele" → het gemiddelde van haar nummers als
   richtpunt, dan de dichtstbijzijnde tracks van *andere* artiesten.
3. **Tempocurve** — "afbouw playlist", "opbouw om wakker te worden",
   "feestboog", "intervaltraining van 40 minuten". Geen selectie maar een
   *volgorde*: het gemeten tempo bepaalt waar elk nummer staat.
4. **Semantisch** — al het andere. De prompt wordt een vector; die wordt tegen
   de hele bibliotheek gelegd, waarna context, jaartal, genre en luistergedrag
   het eindresultaat bijsturen.

### Tempocurves

Tempo is gemeten, geen gok, dus een playlist die erop is gebouwd klinkt ook
echt als een curve. Vier vormen uit dezelfde machinerie:

| | vorm | voorbeeld op deze bibliotheek |
|---|---|---|
| 🌙 Afbouw | dalend | 123 → 27 BPM over 37 nummers |
| 🌅 Opbouw | stijgend | 51 → 161 BPM over 50 nummers |
| 🎉 Energieboog | piek in het midden | 99 → 144 → 99 BPM |
| 🏃 Intervaltraining | blokken | 112 / 152 / 112 / 152 … |

Een curve die één kant op hoort te gaan, mág niet omkeren: bij de afbouw was
dat eerst niet afgedwongen, en zodra het trage materiaal op was koos "dichtst
bij het doeltempo" weer snelle nummers — de lijst klom van 27 terug naar 117
BPM. Nu stopt hij liever vroeg. Korter is een eerlijk antwoord, de verkeerde
kant op niet.

Bij een intervaltraining is het hersteltempo een **verhouding** van het
werktempo (~72%), geen percentiel van de pool. Met een percentiel bepaalt het
toeval wat "rustig" betekent: een pool vol workout-muziek zette de rustige
blokken op 136 BPM tegen 152 — geen contrast — en een pool met een paar
ballades erin op 59, wat geen dribbelpas is maar een slow. Blokken lopen door
tot het einde van een nummer, want een playlist kan geen nummer doormidden
knippen; de melding zegt wat de blokken werkelijk werden.

De scoring gebruikt signalen die JLTamp bijhoudt en Plex niet gaf:

- **Echte skips** uit `/stats/history` — een nummer dat je in het eerste derde
  deel wegklikte. De oude engine telde elke *pauze* als afkeur.
- **Likes** en **per-gebruiker afspeelstatistieken**.
- **Audio-kenmerken** (BPM, energie) als je `scripts/analyze_audio.py` hebt
  gedraaid. Dat is wat "gym" écht onderscheidt van "slapen"; aan de titel van
  een nummer valt het tempo niet af te lezen.

## Audio-analyse (optioneel)

Meet BPM en energie per nummer. Dat is wat prompts als "gym" of "slapen" echt
onderscheidt — aan een titel valt het tempo niet af te lezen.

Draai dit **op `your-server`**, waar de NAS gemount staat, in een container:

```bash
docker build -f deploy/analyzer.Dockerfile -t jltamp-analyzer .

for i in 0 1 2; do
  docker run -d --name jltamp-analyzer-$i --rm \
    -v /path/to/your/music:/music/mp3:ro \
    -v /path/to/your/flac:/music/flac:ro \
    -v $HOME/jltamp-ai:/out \
    -e MUSIC_PATH_MAP=/music/mp3:/music/mp3,/music/flac:/music/flac \
    -e AI_FEATURES_FILE=/out/track_features.json \
    -e JLTAMP_URL=... -e JLTAMP_EMAIL=... -e JLTAMP_PASSWORD=... \
    jltamp-analyzer --workers 2 --shard $i/3
done

python3 scripts/merge_features.py    # als ze klaar zijn
```

Waarom zo omslachtig:

- **In een container**, omdat librosa's numba op de Python 3.13 van de host
  reproduceerbaar segfault in `beat_track`. Python 3.11 doet dat niet.
- **Twee workers, niet meer.** Gemeten: 2 haalt elke track, 3 sloopt de pool.
  Meer parallellisme komt van meerdere containers — losse processen kunnen
  elkaar niet meesleuren.
- **Shards schrijven elk hun eigen bestand**, anders overschrijven ze elkaars
  resultaten. `merge_features.py` voegt ze samen.

Het script **leest alleen**. De NAS blijft read-only, zoals
`server/CLAUDE.md` voorschrijft, en de mounts staan expliciet op `:ro`.
Onderbreken is veilig: voortgang wordt doorlopend weggeschreven.

### Bijhouden

De analyse slaat alles over wat al gemeten is, dus hem opnieuw draaien kost
alleen de nieuwe nummers. `scripts/refresh_features.sh` is de geplande versie
daarvan — met een lock, zodat een lange nacht niet de volgende inhaalt:

```bash
# op de Jetson: de NAS staat hier niet gemount, dus via de API
0 3 * * * AUDIO_ALLOW_STREAM=1 /home/USER/jltamp/ai/scripts/refresh_features.sh >> ~/jltamp-ai/features.log 2>&1

# op your-server, waar de mounts wél zijn: sneller, zonder die vlag
0 3 * * * /pad/naar/ai/scripts/refresh_features.sh >> ~/jltamp-ai/features.log 2>&1
```

Streamen kost ~2,9 s per nummer tegen ~0,6 s lokaal, dus de eerste nacht loopt
de achterstand in een nacht weg, en daarna is er per keer nog
maar een handjevol nieuwe tracks te meten. Het script weigert te draaien als
het de muziek nergens kan bereiken — een lege mount ziet het als ontbrekend,
want anders lijkt "niets te doen" op succes.

Zonder zo'n schema groeit het gat vanzelf: een deel van je bibliotheek
houdt dan geen gemeten tempo. Die nummers kunnen niet op een
afbouw-curve staan (geen tempo, geen curve) en krijgen geen audio-boost bij
"gym" of "slapen". Het featurebestand telt er meestal meer, maar een deel
daarvan hoort bij tracks die niet meer in de bibliotheek zitten — `tests/
benchmark_context.py` drukt het werkelijke aantal af.

Herstarten hoeft niet — de engine leest het bestand opnieuw zodra het
verandert.

## Opslag

Vectoren staan in `data/vectors.npy` als een memory-mapped float32-array, met
`data/index.json` als sleutelregister. Toevoegen raakt alleen de nieuwe rijen.

De vorige opzet bewaarde alles als JSON en herschreef dat hele bestand na elke
batch — bij 4096 dimensies groeide dat uit tot 4,1 GB per keer opnieuw wegschrijven,
wat minutenlang duurde en een half geschreven `.tmp` achterliet bij onderbreking.

Zoeken vermenigvuldigt één keer over de hele memmap en indexeert daarna de
*scores*. De voor de hand liggende volgorde — eerst de rijen ophalen die je
nodig hebt, dan vermenigvuldigen — kopieert 307 MB per aanroep, en een prompt
vraagt toch naar de hele bibliotheek. In een los proces kost die kopie 1,25 s
tegen 0,02 s voor de vermenigvuldiging zelf; in de draaiende dienst is het
verschil veel kleiner (radio ging van ~0,30 s naar ~0,26 s), dus de winst zit
vooral in de 307 MB die niet meer per verzoek wordt gealloceerd. De uitkomst is
aantoonbaar identiek: dezelfde sleutels, dezelfde volgorde, verschil exact nul.

Bij een modelwissel weigert de dienst te starten in plaats van stilletjes
vectoren uit twee verschillende modellen te vergelijken — dat gaf voorheen
onzinresultaten zonder enige foutmelding. Verwijder `data/` om opnieuw te
indexeren.

## Endpoints

Vastgelegd door `aiService.ts` in een uitgeleverde Android-build; niet wijzigen.

| Endpoint | |
|---|---|
| `GET /health` | Status, aantal tracks/vectoren, model |
| `GET /whoami` | Tailscale-IP, voor het vinden van deze dienst |
| `POST /ai/playlist` | `{"prompt": "..."}` → `job_id` |
| `GET /ai/status?job_id=` | Voortgang, en het resultaat als hij klaar is |
| `GET /ai/suggest` | Voorbeeldprompts |

`/health` en `/whoami` zijn open — de app moet de dienst kunnen vinden vóór hij
kan inloggen. De rest vereist `X-AI-Key` zodra `AI_API_KEY` is gezet.

## Beveiliging

- Geen secrets in de code. Alles komt uit `.env`, die gitignored is.
- Zonder `AI_API_KEY` is de dienst onbeveiligd; hij waarschuwt daarvoor bij het
  starten. Alleen acceptabel op een afgeschermd tailnet.
- CORS staat dicht tenzij je expliciet origins opgeeft. Een wildcard zou de
  API-key voor browsers waardeloos maken.
