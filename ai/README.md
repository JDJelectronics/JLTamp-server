# JLTamp AI

**Optional semantic playlist engine for a JLTamp server (this repo).**

Describe a vibe — *“a chill autumn evening with soft piano”* — and this service
builds a playlist from **your own** library and saves it to your account. It's a
completely **optional** add‑on: the JLTamp app and server work fully without it.
Run it only if you want the “AI DJ” feature.

Nothing leaves your machines: your library is embedded and matched **locally**.

---

## How it works

1. It reads your library from your JLTamp server (using your account's token).
2. It turns each track into a vector with a local embedding model
   (**bge‑m3**, served by [llama.cpp](https://github.com/ggml-org/llama.cpp)).
3. When you send a text prompt, it embeds the prompt, finds the closest tracks,
   and creates a playlist in your library via the JLTamp API.

Two small processes: an **embedding server** (llama.cpp) and this **AI service**
(a tiny Flask app) that the JLTamp app talks to.

---

## Requirements

- Python 3.10+
- A machine that can run the bge‑m3 embedding model through llama.cpp. A GPU box
  (e.g. an NVIDIA Jetson, or any CUDA machine) is recommended; CPU works but is
  slower. It does **not** have to be the same machine as your JLTamp server.
- Your JLTamp server reachable over the network.

---

## Quick start

```bash
git clone https://github.com/JDJelectronics/JLTamp-server.git
cd JLTamp-server/ai
pip install -r requirements.txt
```

1. **Embedding model + server.** Fetch the model and start the local embedding
   server (llama.cpp):

   ```bash
   scripts/fetch_model.sh            # downloads bge-m3 (~600 MB)
   scripts/start_embed_server.sh     # serves embeddings on :3100
   ```
   (Or point `EMBED_URL` at any OpenAI‑compatible embeddings endpoint.)

2. **Configure.** Copy `.env.example` to `.env` and fill in your server + login:

   ```ini
   JLTAMP_URL=http://192.168.1.10:8090     # your JLTamp server
   JLTAMP_EMAIL=you@example.com            # (or set JLTAMP_TOKEN)
   JLTAMP_PASSWORD=your-password
   EMBED_URL=http://127.0.0.1:3100
   AI_PORT=5000
   AI_API_KEY=                             # optional; see below
   ```

3. **Run it:**

   ```bash
   python -m app.main
   ```
   The service listens on `:5000` and exposes `/ai/health`, `/ai/suggest`,
   `/ai/playlist`, `/ai/status`. On first run it embeds your library (this takes
   a while); after that it's incremental.

For a permanent setup, `deploy/` contains systemd units
(`jltamp-embed.service`, `jltamp-ai.service`) — edit the paths/user for your box.

---

## Connecting the JLTamp app

The app finds the AI service in one of two ways:

- **Behind your reverse proxy (recommended for remote):** proxy `/ai/*` on the
  same domain as your JLTamp server to this service. Then the app reaches it at
  `https://your-domain.example/ai/…` and the AI bar appears automatically. Example
  (Caddy):

  ```
  your-domain.example {
      handle /ai/* {
          reverse_proxy 127.0.0.1:5000
      }
      reverse_proxy 127.0.0.1:8090     # the JLTamp server
  }
  ```

- **On your LAN / Tailscale:** run it on port `5000`; the app scans your local
  network / tailnet for it. No proxy needed at home.

The app sends your JLTamp session token with each request, so the AI acts as
**you** and writes playlists to **your** library — no separate credential.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `JLTAMP_URL` | `http://192.168.1.10:8090` | Your JLTamp server |
| `JLTAMP_TOKEN` *or* `JLTAMP_EMAIL`+`JLTAMP_PASSWORD` | — | How the engine authenticates to your server |
| `EMBED_URL` | `http://127.0.0.1:3100` | The embedding server |
| `AI_PORT` | `5000` | Port this service listens on |
| `AI_API_KEY` | *(blank)* | Optional shared key the app must send. Leave blank only on a private tailnet; generate one with `openssl rand -hex 24` |
| `AI_CORS_ORIGINS` | *(blank)* | Browser origins allowed to call it (e.g. your web URL). Blank = native app only |
| `AI_DATA_DIR` | *(data dir)* | Where vectors and the index are stored |

See [`.env.example`](.env.example) for the full list.

---

## Endpoints (the contract)

If you'd rather build your own compatible service, the app expects:

- `GET /ai/health` → `{ "ai_ready": true, … }`
- `GET /ai/suggest` → `{ "suggestions": ["…", …] }`
- `POST /ai/playlist` `{ "prompt": "…" }` (with the caller's JLTamp token) →
  `{ "status": "processing", "job_id": "…", "poll_interval": 1000 }`
- `GET /ai/status?job_id=…` → job progress / result

## License

MIT — see [LICENSE](LICENSE).
