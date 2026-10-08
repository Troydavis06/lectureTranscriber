"""Small shared helpers: slugs, timestamps, WAV length, logging."""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

# Trailing noise Chrome titles collect, stripped so the note heading reads
# like a lecture name and not a browser tab.
_TITLE_NOISE = re.compile(
    r"\s*[-|–—]\s*(YouTube|Panopto|Echo360|Kaltura|Zoom|Google Drive|"
    r"Canvas|Blackboard|Moodle|Microsoft Stream|Vimeo)\s*$",
    re.IGNORECASE,
)
# Chrome prefixes the title with an unread/notification count on some sites.
_TITLE_COUNT = re.compile(r"^\(\d+\)\s*")


def clean_title(title: str) -> str:
    """Turn a raw Chrome tab title into a lecture heading."""
    out = _TITLE_COUNT.sub("", (title or "").strip())
    # Some players append the platform twice ("... - Panopto - YouTube").
    for _ in range(3):
        new = _TITLE_NOISE.sub("", out).strip()
        if new == out:
            break
        out = new
    return out or "Untitled lecture"


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug for use in file names."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", ascii_only).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return (slug[:max_len].rstrip("-")) or "lecture"


def stamped_slug(title: str, when: datetime | None = None) -> str:
    """`2026-09-16_1430_lecture-7-dynamic-programming` — sorts chronologically."""
    when = when or datetime.now()
    return f"{when:%Y-%m-%d_%H%M}_{slugify(title)}"


def slug_recorded_at(slug: str) -> datetime | None:
    """The `2026-09-16_1430` stamp a slug opens with, if it has one.

    Preferred over the WAV's mtime, which is when recording *finished* and so
    reads minutes or hours after the lecture actually started.
    """
    parts = slug.split("_")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H%M")
    except ValueError:
        return None


def slug_title(slug: str) -> str:
    """Best-effort title from a stamped slug, for when none was supplied.

    `2026-09-16_1430_lecture-7-dynamic-programming` -> `Lecture 7 Dynamic
    Programming`. Anything that is not a stamped slug is returned unchanged, so
    a hand-named WAV keeps its own name.
    """
    parts = slug.split("_", 2)
    if len(parts) != 3 or not parts[2]:
        return slug
    return parts[2].replace("-", " ").title()


def fmt_timestamp(seconds: float) -> str:
    """Seconds -> `MM:SS`, or `H:MM:SS` past an hour."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_duration(seconds: float) -> str:
    """Human duration for note headers: `52 min`, `1 h 47 min`, `48 s`."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min"


def wav_duration(path: Path) -> float:
    """Length of a WAV in seconds. Lives here so the transcriber can read it
    without importing the capture stack."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        return wf.getnframes() / rate if rate else 0.0


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # These are chatty at DEBUG and never useful here.
    for noisy in ("websocket", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
