"""De encoder: één ffmpeg die ruwe PCM op stdin krijgt en er HLS van maakt.

AAC op 128 kbps, stukken van zes seconden, een rollende playlist van tien
stukken (een minuut) waar de oude stukken vanzelf uit verdwijnen. Dat is wat
de app, de webspeler (hls.js), Safari, Chromecast en Home Assistant allemaal
afspelen.

Twee keuzes die ertoe doen:

  - `hls_start_number_source epoch`: de volgnummers beginnen bij de klok. Na
    een herstart van de encoder liggen ze dus altijd hoger dan die van
    daarvoor, en een speler die de lijst opnieuw ophaalt raakt niet in de war
    van een teller die terug naar nul springt. `discont_start` zegt er bij dat
    er een naad zit.
  - `temp_file`: de playlist wordt eerst onder een andere naam geschreven en
    dan omgedoopt. Een speler leest nooit een halve lijst.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

log = logging.getLogger("radio")

PLAYLIST = "live.m3u8"


def opdracht(ffmpeg: str, map_: Path, bitrate_k: int = 128, segment_s: int = 6,
             lijst: int = 10, rate: int = 44100, kanalen: int = 2) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(rate), "-ac", str(kanalen), "-i", "pipe:0",
        "-c:a", "aac", "-b:a", f"{bitrate_k}k", "-ar", str(rate), "-ac", str(kanalen),
        "-f", "hls",
        "-hls_time", str(segment_s),
        "-hls_list_size", str(lijst),
        "-hls_flags", "delete_segments+omit_endlist+discont_start+temp_file+independent_segments",
        "-hls_start_number_source", "epoch",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", str(map_ / "seg%d.ts"),
        str(map_ / PLAYLIST),
    ]


class Encoder:
    """Het ffmpeg-proces. `schrijf` gooit OSError als het proces weg is; de
    zender vangt dat en start een nieuwe."""

    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Wat ffmpeg op stderr zegt naar het log. Moet gelezen worden: een
        # volle pijp laat ffmpeg anders stilstaan.
        threading.Thread(target=self._log_lus, args=(self.proc,), name="radio-encoder-log",
                         daemon=True).start()

    @staticmethod
    def _log_lus(proc: subprocess.Popen) -> None:
        try:
            for regel in proc.stderr:
                tekst = regel.decode("utf-8", "replace").strip()
                if tekst:
                    log.warning("radio-encoder: %s", tekst[:300])
        except Exception:
            pass

    def levend(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def schrijf(self, data: bytes) -> None:
        if not self.proc or not self.proc.stdin:
            raise OSError("encoder draait niet")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()            # netjes: ffmpeg schrijft zijn laatste stuk
        except Exception:
            pass
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=2)
            except Exception:
                pass
