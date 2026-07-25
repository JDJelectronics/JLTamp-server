# JLTamp Server

**Self‑hosted music server — the backend for the JLTamp app.**
Stream your own music library to the JLTamp mobile & web app, from a server you
run yourself. Think *Plex / Jellyfin, but focused on music*. No ads, no
tracking, no subscription — your library and your listening data stay on your
own machine.

JLTamp does not provide, host, or distribute any music. It only plays media from
**your** server, using **your own** files.

---

## Features

- 🎵 Stream your own library (MP3, FLAC, ALAC, WAV, AAC, OGG, Opus, …)
- 👥 Multi‑user, invite‑based — everyone gets their own account
- ❤️ Per‑user liked songs, playlists and play history
- 🗂️ Plex‑style libraries with a **folder browser** + on‑demand scanning
- 🖼️ Automatic album art & artist images (fetched from public music databases)
- 📥 Import **Spotify / YouTube** playlists (and Plex playlists/likes) into your library
- 🚀 **Set up in the browser** — first‑run wizard creates your admin account
- 📱 Works with the **JLTamp app** (Android + web)
- 🔒 Passwords hashed, per‑user tokens, music mounted **read‑only**

---

## Quick start (Docker)

You need [Docker](https://docs.docker.com/get-docker/) with Compose.

```bash
git clone https://github.com/JDJelectronics/JLTamp-server.git
cd JLTamp-server
```

1. Open `docker-compose.yml`, **mount your own music folder**, and set a
   username/password. The server ships with **no music of its own** — you point
   it at a library you already have on the server (a local disk, an external
   drive, or a NAS share mounted on the host). It is mounted **read‑only**, so
   the server can never modify or delete your files:

   ```yaml
   environment:
     SERVER_NAME: "My Music"
     # No password here — you create your admin account in the browser on first
     # run (step 3). Optionally set JLTAMP_PASSWORD for a fixed backup admin.
   volumes:
     - /path/to/your/music:/music:ro     # ← EDIT: your own music folder (read-only)
     - jltamp-data:/data                 # server DB + artwork cache
   ```

   Replace `/path/to/your/music` with the real path on your machine, e.g.
   `/home/you/Music`, `/mnt/media/music`, or a mounted network share. The `:ro`
   suffix keeps it read‑only — leave it in place.

2. Start it:

   ```bash
   docker compose up -d --build
   ```

3. Open **http://localhost:32400** in a browser. The **JLTamp web app is bundled
   with the server** and loads straight away — no separate frontend to install,
   no address to type. It talks to whatever origin served it, so `localhost`
   (and your LAN IP, or your own domain behind a reverse proxy) just works.
   **On first run it shows a quick setup wizard** — create your admin account
   (email + password) right there and you're in. (No wizard if you set a
   `JLTAMP_PASSWORD` — then just log in with that.)

4. In the web UI go to **Settings → Libraries** and **add a library**: a
   Plex‑style **folder browser** lets you click through the folders under your
   music mount (▸ to go into a folder, tap to select it), pick one or more, and
   run a **scan**. Your music then appears in the web app and the mobile app.
   From the same screen you can also **import Spotify / YouTube playlists** and,
   if you run Plex, import your Plex playlists and likes.

Your music is mounted **read‑only** — the server never modifies your files. All
writable data (database, cached artwork, users, playlists, likes) lives in the
`jltamp-data` volume, so it survives rebuilds.

> **No hosted account, no domain required.** This is *your* server: you sign in
> against it directly (email + password). There is no external sign‑in and
> nothing phones home.

### Using a NAS

Mount your NAS share on the **host** (SMB/NFS/CIFS) and pass the mount point as
the `:/music:ro` volume — the server browses it like any other folder. The
folder browser is deliberately sandboxed to the mounted music path, so it can
never expose the rest of your filesystem.

There's a **“Discover NAS on my network”** button under *Settings → Libraries*
that lists NAS/file‑share devices found via mDNS — hints to help you find the
address to mount. It can’t mount shares for you (a read‑only container has no
business doing that), and on a default Docker **bridge** network mDNS is usually
blocked, so run the container with `network_mode: host` if you want discovery to
work. Mounting on the host is the reliable path either way.

---

## Connecting the app

Download the **JLTamp** app, choose *“Connect your own server”*, and enter your
server’s address (e.g. `http://192.168.1.10:32400` on your LAN, or your public
HTTPS URL if you expose it). Sign in with your account.

> The JLTamp app can also connect to a **Plex** server, if you already run one.

---

## Optional: AI DJ playlists

JLTamp has an optional **AI DJ**: describe a vibe (“a calm sunday morning with
coffee”) and get a playlist built from **your own** library. It runs as a
**separate, optional** service — the app and server work fully without it.

To enable it, run the AI engine included in this repo:

👉 See the **[`ai/`](ai/)** folder — a small semantic playlist engine that embeds
your library **locally** (nothing goes to the cloud). Full setup is in
[`ai/README.md`](ai/README.md).

In short: run the AI service, then either put it behind your reverse proxy under
`/ai/*` on the same domain as your server, or run it on your LAN / Tailscale on
port `5000`. The AI bar then appears in the app automatically.

---

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `MUSIC_DIR` | `/music` | Path inside the container to your library (read‑only) |
| `DATA_DIR` | `/data` | Where the DB + artwork cache live |
| `JLTAMP_USERNAME` / `JLTAMP_PASSWORD` | `admin` / `changeme` | The seeded admin login — **change it** |
| `JLTAMP_ADMIN_EMAIL` | `admin@example.com` | Admin email (login + owner) |
| `SERVER_NAME` | `JLTamp` | Friendly name shown in the app |
| `JLTAMP_OPEN_REGISTRATION` | `false` | `true` = anyone can register; `false` = invite‑only |
| `PORT` | `32400` | HTTP port the server listens on |
| `LOCAL_URL` | *(blank)* | Optional LAN/Tailscale address(es) for faster local streaming |
| `SMTP_*` | *(blank)* | Optional SMTP for invites / password‑reset / welcome mail |
| `RESCAN_INTERVAL_MIN` | `0` | Auto‑rescan interval in minutes (`0` = manual only) |

### Optional: email (invites & password reset)

Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` (and
`SMTP_SSL=true` for port 465) to enable invitations, password resets and the
welcome mail. Without SMTP, invites fall back to a link you copy by hand.

---

## Remote access

Simplest is to keep it on your LAN and reach it over
[Tailscale](https://tailscale.com/). To expose it publicly, put a reverse proxy
(e.g. Caddy / nginx) with HTTPS in front of the container — don’t expose the raw
port to the internet.

---

## Privacy

The server serves a template privacy policy at `/privacy`
(`app/static/privacy.html`) — edit the contact address before publishing.

## License

MIT — see [LICENSE](LICENSE).
