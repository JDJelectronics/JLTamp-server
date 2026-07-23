# JLTamp AI — uitrollen

Twee services op de **Jetson Xavier**: llama.cpp die embeddings maakt op de GPU,
en de AI-dienst die de app bedient. Ze praten met de JLTamp-server op
`192.168.1.10:8090`.

De Jetson is de plek voor deze dienst omdat de GPU daar zit. `your-server` draait
de bibliotheek en de app; de AI rekent.

## Eenmalig

```bash
cd /home/USER/jltamp/ai
pip3 install -r requirements.txt
./scripts/fetch_model.sh                 # bge-m3, ~610 MB
cp .env.example .env && chmod 600 .env   # en invullen
```

In `.env` minimaal `JLTAMP_EMAIL` + `JLTAMP_PASSWORD` (of `JLTAMP_TOKEN`) en een
`AI_API_KEY` uit `openssl rand -hex 24`.

Controleer dat het werkt vóór je er een service van maakt:

```bash
./scripts/start_embed_server.sh          # Ctrl-C stopt beide
curl -s localhost:5000/health | python3 -m json.tool
```

`ai_ready: true` en een `tracks`-aantal dat klopt: goed.

## Als service

```bash
sudo cp deploy/jltamp-embed.service deploy/jltamp-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jltamp-embed jltamp-ai
```

`jltamp-ai` heeft `Requires=jltamp-embed`, dus die tweede start de eerste mee en
herstart hem als llama.cpp omvalt.

```bash
systemctl status jltamp-ai
journalctl -u jltamp-ai -f
```

## Wat waar staat

| | |
|---|---|
| `:3100` | llama.cpp, **alleen localhost** — een open embedding-endpoint is gratis GPU-tijd voor iedereen die het vindt |
| `:5000` | De AI-dienst. De app zoekt hem hier |
| `data/` | Vectoren + index. Herbouwbaar, dus niet in de backup nodig |
| `.env` | Secrets, mode 600, niet in git |

## Bereikbaar maken van buiten

De app vindt de dienst via Tailscale (`aiService.ts` probeert eerst het
tailnet). De Jetson zit al in je tailnet — `/whoami` geeft `100.x.y.z`
terug. Er hoeft dus niets opengezet te worden op je router.

Zet `AI_API_KEY`, ook op het tailnet. Zonder sleutel kan alles wat in je tailnet
zit de engine aansturen.

## Bijwerken

```bash
cd /home/USER/jltamp && git pull
sudo systemctl restart jltamp-ai
```

Alleen bij een **ander embeddingmodel** moet `data/` weg: vectoren uit twee
modellen zijn niet vergelijkbaar. De dienst weigert dan te starten met een
duidelijke melding in plaats van stilletjes onzin terug te geven.

```bash
sudo systemctl stop jltamp-ai && rm -rf data/ && sudo systemctl start jltamp-ai
```

Opnieuw indexeren duurt minuten, niet uren: ~114 tracks/s op de Xavier.

## Als er iets niet werkt

| Symptoom | Waar te kijken |
|---|---|
| `state: jltamp-auth-failed` | `.env` — e-mail/wachtwoord of verlopen token |
| `state: embedder-offline` | `journalctl -u jltamp-embed`; draait llama.cpp? |
| `state: vector-store-mismatch` | Model gewisseld — `data/` verwijderen |
| App vindt de dienst niet | `curl http://<jetson>:5000/health` vanaf het toestel |
| `ai_ready: true`, geen resultaat | `embeddings` in `/health`: nog aan het indexeren? |
