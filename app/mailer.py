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

# The brand logo, embedded inline (cid:logo) in every mail. A round 192 px PNG
# cut from the app icon ("nieuw logo/gegenereerd/vierkant-1024.png"); ships with
# the server so no external image fetch is needed (mail clients block those).
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


# ── Huisstijl ────────────────────────────────────────────────────────────
# Dezelfde kleuren als de app (frontend/src/theme/index.ts): de paarse
# achtergrondverloop #1a1a2e → #121212, kaarten in #252535 met een rand in
# #33334A, de teal-accent als pil-knop met zwarte tekst. Alles inline en in
# tabellen, want mailprogramma's gooien <style> en flexbox weg.
_BG = "#0a0a0a"
_CARD_TOP = "#1a1a2e"
_CARD = "#121212"
_BORDER = "#33334A"
_ACCENT = "#00d4aa"
_TEXT = "#FFFFFF"
_TEXT_2 = "#B3B3B3"
_TEXT_3 = "#727272"
_FONT = "Roboto,system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"


def _button(href: str, label: str) -> str:
    return (f'<a href="{_html.escape(href, quote=True)}" style="display:inline-block;'
            f'background:{_ACCENT};color:#000000;font-weight:700;font-size:15px;'
            f'text-decoration:none;padding:14px 28px;border-radius:999px">'
            f'{_html.escape(label)}</a>')


def _link_fallback(href: str) -> str:
    return (f'<p style="margin:22px 0 0;color:{_TEXT_3};font-size:13px;line-height:19px">'
            f'Werkt de knop niet? Plak deze link in je browser:<br>'
            f'<span style="color:{_TEXT_2};word-break:break-all">{_html.escape(href)}</span></p>')


def _shell(heading: str, body: str, footer: str = "") -> str:
    """Eén lijst om elke mail heen: logo, kop, inhoud, voetregel. `heading` wordt
    hier ge-escaped; `body` en `footer` zijn al HTML."""
    logo = ('<img src="cid:logo" width="72" height="72" alt="JLTamp" '
            'style="display:block;margin:0 auto 20px;border:0;border-radius:36px">'
            if _LOGO_PATH.exists() else "")
    foot = (f'<p style="margin:24px 0 0;color:{_TEXT_3};font-size:12px;line-height:18px">'
            f'{footer}</p>' if footer else "")
    return f"""\
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><meta name="supported-color-schemes" content="dark"></head>
<body style="margin:0;padding:0;background:{_BG};font-family:{_FONT};color:{_TEXT}">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_BG}">
<tr><td align="center" style="padding:32px 16px">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="max-width:520px;background:{_CARD};background-image:linear-gradient(180deg,{_CARD_TOP} 0%,{_CARD} 70%);
                border:1px solid {_BORDER};border-radius:20px">
  <tr><td align="center" style="padding:36px 32px;text-align:center">
    {logo}
    <h1 style="margin:0 0 10px;font-size:24px;line-height:30px;font-weight:700;letter-spacing:-0.3px;color:{_TEXT}">{_html.escape(heading)}</h1>
    {body}
    {foot}
  </td></tr>
  </table>
  <p style="margin:18px 0 0;color:{_TEXT_3};font-size:11px;letter-spacing:0.5px;text-transform:uppercase">JLTamp</p>
</td></tr>
</table>
</body></html>"""


def _attach_logo(msg: EmailMessage) -> None:
    """Hang het logo als inline (cid:logo) plaatje aan het HTML-deel."""
    if not _LOGO_PATH.exists():
        return
    try:
        msg.get_payload()[1].add_related(_LOGO_PATH.read_bytes(), maintype="image",
                                         subtype="png", cid="<logo>")
    except Exception as e:  # noqa: BLE001 — never let the logo break the mail
        log.warning("kon logo niet aan mail hangen: %s", e)


def _cover_thumb(path: str | None) -> bytes | None:
    """Een albumhoes als klein vierkant jpeg-je (112 px, 2× voor 56 op het
    scherm). Een folder-cover.jpg is soms megabytes; dertig daarvan zou de mail
    onverstuurbaar maken. Geen hoes of onleesbaar → None, dan komt er een vakje."""
    if not path:
        return None
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (112, 112),
                              Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, "JPEG", quality=82, optimize=True)
            return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.debug("geen hoes voor %s: %s", path, e)
        return None


def _p(text_html: str, margin: str = "0 0 26px") -> str:
    return (f'<p style="margin:{margin};color:{_TEXT_2};font-size:15px;line-height:22px">'
            f'{text_html}</p>')


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
    body = (_p(f"Er is een nieuw wachtwoord aangevraagd voor je account op "
               f"{_html.escape(server_name)}.")
            + _button(link, "Stel een nieuw wachtwoord in")
            + _link_fallback(link))
    footer = (f"De link is {valid_minutes} minuten geldig en werkt één keer. Heb je dit "
              f"niet aangevraagd? Dan hoef je niets te doen — je huidige wachtwoord blijft werken.")
    msg.add_alternative(_shell("Nieuw wachtwoord", body, footer), subtype="html")
    _attach_logo(msg)
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
    body = (_p(f"{_html.escape(inviter)} heeft je uitgenodigd voor een privé muziekserver.")
            + _button(link, "Kies een wachtwoord en log in")
            + _link_fallback(link))
    msg.add_alternative(
        _shell(server_name, body, "De link werkt één keer. Verwacht je dit niet? Negeer deze mail."),
        subtype="html",
    )
    _attach_logo(msg)
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

    body = _p(_html.escape(S("intro"))) + (_button(root, S("button")) if root else "")
    msg.add_alternative(_shell(S("heading"), body, _html.escape(c["closing"])),
                        subtype="html")
    _attach_logo(msg)
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
    msg["Reply-To"] = "no-reply@your-domain.example"
    msg.set_content("\n".join(lines))

    # De albumlijst als de lijsten in de app: hoes links, titel wit, artiest
    # grijs, een dunne rand ertussen — links uitgelijnd binnen de gekozen kaart.
    covers: list[bytes] = []
    rows = ""
    for a in shown:
        thumb = _cover_thumb(a.get("art_path"))
        if thumb:
            covers.append(thumb)
            cover = (f'<img src="cid:art{len(covers) - 1}" width="56" height="56" alt="" '
                     f'style="display:block;border:0;border-radius:8px">')
        else:
            cover = (f'<div style="width:56px;height:56px;line-height:56px;border-radius:8px;'
                     f'background:#252535;color:{_TEXT_3};font-size:22px;text-align:center">♪</div>')
        year = (f'<span style="color:{_TEXT_3}"> · {_html.escape(str(a.get("year")))}</span>'
                if a.get("year") else "")
        rows += (f'<tr><td width="56" style="padding:10px 0;border-top:1px solid {_BORDER};width:56px">{cover}</td>'
                 f'<td style="padding:10px 4px 10px 14px;border-top:1px solid {_BORDER};text-align:left">'
                 f'<span style="color:{_TEXT};font-size:15px;font-weight:600">'
                 f'{_html.escape(str(a.get("title", "")))}</span>{year}'
                 f'<br><span style="color:{_TEXT_2};font-size:13px">'
                 f'{_html.escape(str(a.get("artist", "")))}</span></td></tr>')
    more_html = (f'<p style="margin:14px 0 0;color:{_TEXT_3};font-size:13px">…en nog {more} albums.</p>'
                 if more else "")
    button = (f'<div style="margin-top:26px">{_button(root, "Open in JLTamp")}</div>'
              if root else "")
    body = (_p(f'<b style="color:{_ACCENT}">{track_count}</b> nieuwe nummers in '
               f'<b style="color:{_ACCENT}">{album_count}</b> albums op {_html.escape(server_name)}.',
               "0 0 22px")
            + f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
              f'style="border-collapse:collapse">{rows}</table>'
            + more_html + button)
    footer = ("Geen zin meer in deze mails? Zet ze uit in JLTamp → Instellingen → Algemeen.<br>"
              "Dit is een automatisch bericht — niet beantwoorden.")
    msg.add_alternative(_shell("Nieuwe muziek", body, footer), subtype="html")
    _attach_logo(msg)
    for i, data in enumerate(covers):
        try:
            msg.get_payload()[1].add_related(data, maintype="image", subtype="jpeg",
                                             cid=f"<art{i}>")
        except Exception as e:  # noqa: BLE001 — a cover must never break the mail
            log.warning("kon hoes niet aan mail hangen: %s", e)
    send_async(msg)
    return True
