"""Outgoing mail (invites).

SMTP details come from the environment — never from the code — so the mailbox
password stays out of git. When SMTP is not configured the server still works:
invites just fall back to "copy the link yourself", which is what it did before.
Mail is sent on a background thread: a slow or dead mail server must never hang
the admin's request.
"""
from __future__ import annotations

import logging
import os
import html as _html
import smtplib
import threading
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger("jltamp.mail")

# The brand logo, embedded inline (cid:logo) in the welcome mail. A small PNG
# copied from the app icon; ships with the server so no external image fetch is
# needed (mail clients block those by default).
_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip() or SMTP_USER
SMTP_SSL = os.environ.get("SMTP_SSL", "").strip().lower() in ("1", "true", "yes", "on")

# The address the invite link points at — the public URL, not the LAN IP, or the
# invited person cannot open it from outside the house.
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")


def configured() -> bool:
    return bool(SMTP_HOST and SMTP_FROM)


def _send(msg: EmailMessage) -> None:
    try:
        if SMTP_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as s:
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
                s.starttls()
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        log.info("Mail sent to %s — %s", msg["To"], msg["Subject"])
    except Exception as e:  # noqa: BLE001 — mail must never break the request
        log.error("Could not send mail to %s: %s", msg["To"], e)


def send_async(msg: EmailMessage) -> None:
    threading.Thread(target=_send, args=(msg,), daemon=True).start()


def send_reset(to_email: str, reset_token: str, server_name: str,
               valid_minutes: int) -> bool:
    """Mail a password-reset link. Returns False when SMTP is not configured."""
    if not configured():
        return False

    link = f"{PUBLIC_URL.rstrip('/')}/reset/{reset_token}"
    msg = EmailMessage()
    msg["Subject"] = f"Nieuw wachtwoord voor {server_name}"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(
        f"Hoi,\n\n"
        f"Er is een nieuw wachtwoord aangevraagd voor je account op {server_name}.\n\n"
        f"Open deze link om er een in te stellen:\n{link}\n\n"
        f"De link is {valid_minutes} minuten geldig en werkt één keer.\n"
        f"Heb je dit niet aangevraagd? Dan hoef je niets te doen — je huidige "
        f"wachtwoord blijft gewoon werken.\n"
    )
    msg.add_alternative(
        f"""\
<html><body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
                   background:#0e0e11;color:#e8e8ea;padding:32px">
  <div style="max-width:520px;margin:0 auto;background:#17171c;border-radius:16px;padding:32px">
    <h1 style="margin:0 0 8px;font-size:22px;color:#fff">Nieuw wachtwoord</h1>
    <p style="margin:0 0 24px;color:#a0a0aa">
      Er is een nieuw wachtwoord aangevraagd voor je account op {server_name}.
    </p>
    <a href="{link}"
       style="display:inline-block;background:#00d4aa;color:#04120f;font-weight:700;
              text-decoration:none;padding:13px 22px;border-radius:10px">
      Stel een nieuw wachtwoord in
    </a>
    <p style="margin:24px 0 0;color:#71717a;font-size:13px">
      Werkt de knop niet? Plak deze link in je browser:<br>
      <span style="color:#a0a0aa">{link}</span>
    </p>
    <p style="margin:16px 0 0;color:#71717a;font-size:13px">
      De link is {valid_minutes} minuten geldig en werkt één keer. Heb je dit niet
      aangevraagd? Dan hoef je niets te doen — je huidige wachtwoord blijft werken.
    </p>
  </div>
</body></html>""",
        subtype="html",
    )
    send_async(msg)
    return True


def send_invite(to_email: str, invite_token: str, inviter: str,
                server_name: str, base_url: str | None = None) -> bool:
    """Mail someone their invite link. Returns False when SMTP is not set up, so
    the caller can tell the admin to pass the link on by hand instead."""
    if not configured():
        return False

    root = (PUBLIC_URL or base_url or "").rstrip("/")
    link = f"{root}/invite/{invite_token}"

    msg = EmailMessage()
    msg["Subject"] = f"Je bent uitgenodigd voor {server_name}"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(
        f"Hoi,\n\n"
        f"{inviter} heeft je uitgenodigd voor {server_name} — een privé muziekserver.\n\n"
        f"Open deze link om een wachtwoord te kiezen en meteen in te loggen:\n"
        f"{link}\n\n"
        f"De link werkt één keer. Verwacht je dit niet? Dan kun je deze mail negeren.\n"
    )
    msg.add_alternative(
        f"""\
<html><body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
                   background:#0e0e11;color:#e8e8ea;padding:32px">
  <div style="max-width:520px;margin:0 auto;background:#17171c;border-radius:16px;padding:32px">
    <h1 style="margin:0 0 8px;font-size:22px;color:#fff">{server_name}</h1>
    <p style="margin:0 0 24px;color:#a0a0aa">
      {inviter} heeft je uitgenodigd voor een privé muziekserver.
    </p>
    <a href="{link}"
       style="display:inline-block;background:#00d4aa;color:#04120f;font-weight:700;
              text-decoration:none;padding:13px 22px;border-radius:10px">
      Kies een wachtwoord en log in
    </a>
    <p style="margin:24px 0 0;color:#71717a;font-size:13px">
      Werkt de knop niet? Plak deze link in je browser:<br>
      <span style="color:#a0a0aa">{link}</span>
    </p>
    <p style="margin:16px 0 0;color:#71717a;font-size:13px">
      De link werkt één keer. Verwacht je dit niet? Negeer deze mail.
    </p>
  </div>
</body></html>""",
        subtype="html",
    )
    send_async(msg)
    return True


# Welcome-mail copy per language. {server} / {name} are filled at send time.
# Keys: subject, heading, intro, open (plain-text lead-in), button, closing.
_WELCOME = {
    "nl": {"subject": "Welkom bij {server} 🎵", "heading": "Welkom bij {server}!",
           "intro": "Hoi {name}, je account is aangemaakt en helemaal klaar. Je eigen muziek, je eigen likes en playlists.",
           "open": "Open de app en begin met luisteren:", "button": "Open {server}",
           "closing": "Veel luisterplezier! 🎧"},
    "en": {"subject": "Welcome to {server} 🎵", "heading": "Welcome to {server}!",
           "intro": "Hi {name}, your account is created and ready to go. Your own music, your own likes and playlists.",
           "open": "Open the app and start listening:", "button": "Open {server}",
           "closing": "Enjoy the music! 🎧"},
    "fy": {"subject": "Wolkom by {server} 🎵", "heading": "Wolkom by {server}!",
           "intro": "Hoi {name}, dyn akkount is oanmakke en hielendal klear. Dyn eigen muzyk, dyn eigen likes en playlists.",
           "open": "Iepenje de app en begjin mei harkjen:", "button": "Iepenje {server}",
           "closing": "In protte harkwille! 🎧"},
    "de": {"subject": "Willkommen bei {server} 🎵", "heading": "Willkommen bei {server}!",
           "intro": "Hallo {name}, dein Konto ist erstellt und startklar. Deine eigene Musik, deine Likes und Playlists.",
           "open": "Öffne die App und leg los:", "button": "{server} öffnen",
           "closing": "Viel Hörvergnügen! 🎧"},
    "es-ES": {"subject": "Bienvenido a {server} 🎵", "heading": "¡Bienvenido a {server}!",
              "intro": "Hola {name}, tu cuenta está creada y lista. Tu propia música, tus me gusta y listas.",
              "open": "Abre la app y empieza a escuchar:", "button": "Abrir {server}",
              "closing": "¡Que disfrutes la música! 🎧"},
    "pt-BR": {"subject": "Bem-vindo ao {server} 🎵", "heading": "Bem-vindo ao {server}!",
              "intro": "Olá {name}, sua conta foi criada e está pronta. Sua própria música, suas curtidas e playlists.",
              "open": "Abra o app e comece a ouvir:", "button": "Abrir {server}",
              "closing": "Aproveite a música! 🎧"},
    "pt-PT": {"subject": "Bem-vindo ao {server} 🎵", "heading": "Bem-vindo ao {server}!",
              "intro": "Olá {name}, a tua conta foi criada e está pronta. Os teus gostos e playlists.",
              "open": "Abre a app e começa a ouvir:", "button": "Abrir {server}",
              "closing": "Bom divertimento! 🎧"},
}


def _welcome_copy(lang: str | None) -> dict:
    """Pick the welcome copy for a language, tolerating case / region variants
    ('nl-NL' → 'nl', 'PT-br' → 'pt-BR'), falling back to Dutch."""
    if lang:
        l = lang.strip()
        if l in _WELCOME:
            return _WELCOME[l]
        base = l.split("-")[0].lower()
        for k in _WELCOME:
            if k.lower() == l.lower() or k.split("-")[0].lower() == base:
                return _WELCOME[k]
    return _WELCOME["nl"]


def send_welcome(to_email: str, name: str, server_name: str,
                 base_url: str | None = None, lang: str | None = None) -> bool:
    """Welcome a user who just joined (accepted an invite / registered).
    Localized by `lang`; shows the brand logo inline. Returns False when SMTP
    is not configured."""
    if not configured():
        return False

    root = (PUBLIC_URL or base_url or "").rstrip("/")
    greeting = (name or "").strip() or "daar"
    c = _welcome_copy(lang)
    S = lambda k: c[k].format(server=server_name, name=greeting)  # noqa: E731

    msg = EmailMessage()
    msg["Subject"] = S("subject")
    msg["From"] = SMTP_FROM
    msg["To"] = to_email

    open_line = f"{c['open']}\n{root}\n\n" if root else ""
    msg.set_content(f"{S('intro')}\n\n{open_line}{c['closing']}\n")

    logo_ok = _LOGO_PATH.exists()
    logo_html = ('<img src="cid:logo" width="76" height="76" alt="" '
                 'style="display:block;margin:0 auto 22px;border-radius:20px">'
                 if logo_ok else "")
    button = (f'<a href="{root}" style="display:inline-block;background:#00d4aa;'
              f'color:#04120f;font-weight:700;text-decoration:none;padding:13px 22px;'
              f'border-radius:10px">{_html.escape(S("button"))}</a>'
              if root else "")

    msg.add_alternative(
        f"""\
<html><body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
                   background:#0e0e11;color:#e8e8ea;padding:32px">
  <div style="max-width:520px;margin:0 auto;background:#17171c;border-radius:16px;padding:36px 32px;text-align:center">
    {logo_html}
    <h1 style="margin:0 0 8px;font-size:24px;color:#fff">{_html.escape(S("heading"))}</h1>
    <p style="margin:0 0 26px;color:#a0a0aa">{_html.escape(S("intro"))}</p>
    {button}
    <p style="margin:26px 0 0;color:#71717a;font-size:13px">{_html.escape(c["closing"])}</p>
  </div>
</body></html>""",
        subtype="html",
    )

    # Attach the logo as an inline (cid:logo) image on the HTML part.
    if logo_ok:
        try:
            html_part = msg.get_payload()[1]
            html_part.add_related(_LOGO_PATH.read_bytes(), maintype="image",
                                  subtype="png", cid="<logo>")
        except Exception as e:  # noqa: BLE001 — never let the logo break the mail
            log.warning("welcome mail: could not embed logo: %s", e)

    send_async(msg)
    return True


def send_new_music(to_email: str, server_name: str, album_count: int,
                   track_count: int, albums: list[dict],
                   base_url: str | None = None) -> bool:
    """No-reply digest of the music a scan just added. `albums` is a sample list
    of {artist, title, year} (already filtered to what this user may see).
    Returns False when SMTP is not configured."""
    if not configured():
        return False

    root = (PUBLIC_URL or base_url or "").rstrip("/")
    shown = albums[:30]
    more = max(0, album_count - len(shown))

    # Plain-text part
    lines = [f"Er is nieuwe muziek toegevoegd aan {server_name}.", "",
             f"{track_count} nieuwe nummers in {album_count} albums.", ""]
    for a in shown:
        yr = f" ({a.get('year')})" if a.get("year") else ""
        lines.append(f"• {a.get('title','')}{yr} — {a.get('artist','')}")
    if more:
        lines.append(f"…en nog {more} albums.")
    if root:
        lines += ["", f"Open JLTamp: {root}"]
    lines += ["", "Geen zin meer in deze mails? Zet ze uit in JLTamp → Instellingen → Algemeen.",
              "Dit is een automatisch bericht — niet beantwoorden."]

    msg = EmailMessage()
    msg["Subject"] = f"{track_count} nieuwe nummers op {server_name}"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Reply-To"] = SMTP_FROM or "no-reply@localhost"
    msg.set_content("\n".join(lines))

    rows = "".join(
        f'<tr><td style="padding:7px 0;border-bottom:1px solid #24242c">'
        f'<span style="color:#fff;font-weight:600">{_html.escape(str(a.get("title","")))}</span>'
        f'{(" · " + str(a.get("year"))) if a.get("year") else ""}'
        f'<br><span style="color:#8b8b95;font-size:13px">{_html.escape(str(a.get("artist","")))}</span>'
        f'</td></tr>'
        for a in shown
    )
    more_html = (f'<p style="margin:14px 0 0;color:#71717a;font-size:13px">…en nog {more} albums.</p>'
                 if more else "")
    button = (f'<a href="{root}" style="display:inline-block;margin-top:20px;background:#00d4aa;'
              f'color:#04120f;font-weight:700;text-decoration:none;padding:12px 20px;border-radius:10px">'
              f'Open in JLTamp</a>' if root else "")

    msg.add_alternative(
        f"""\
<html><body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
                   background:#0e0e11;color:#e8e8ea;padding:32px">
  <div style="max-width:520px;margin:0 auto;background:#17171c;border-radius:16px;padding:32px">
    <h1 style="margin:0 0 4px;font-size:22px;color:#fff">🎵 Nieuwe muziek</h1>
    <p style="margin:0 0 20px;color:#a0a0aa">
      <b style="color:#00d4aa">{track_count}</b> nieuwe nummers in
      <b style="color:#00d4aa">{album_count}</b> albums op {_html.escape(server_name)}.
    </p>
    <table style="width:100%;border-collapse:collapse">{rows}</table>
    {more_html}
    {button}
    <p style="margin:24px 0 0;color:#71717a;font-size:12px">
      Geen zin meer in deze mails? Zet ze uit in JLTamp → Instellingen → Algemeen.<br>
      Dit is een automatisch bericht — niet beantwoorden.
    </p>
  </div>
</body></html>""",
        subtype="html",
    )
    send_async(msg)
    return True
