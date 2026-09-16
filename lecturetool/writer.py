"""Append finished notes to the master notes.txt."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .util import fmt_duration

log = logging.getLogger("writer")

_RULE = "=" * 72


@dataclass
class LectureMeta:
    title: str
    url: str
    duration_sec: float
    recorded_at: datetime


def format_entry(meta: LectureMeta, body: str) -> str:
    header = f"{meta.recorded_at:%Y-%m-%d %H:%M} | {fmt_duration(meta.duration_sec)}"
    if meta.url:
        header = f"{header} | {meta.url}"
    return (
        f"\n{_RULE}\n{meta.title}\n{header}\n{_RULE}\n\n{body.strip()}\n"
    )


def already_written(notes_file: Path, meta: LectureMeta, tail_bytes: int = 8192) -> bool:
    """Has this exact lecture just been appended?

    Guards against a double START (a player that re-fires play, or a reload)
    producing the same notes twice. Only the tail is inspected, so the check
    stays cheap as notes.txt grows.
    """
    if not notes_file.is_file():
        return False
    try:
        size = notes_file.stat().st_size
        with notes_file.open("rb") as fh:
            fh.seek(max(0, size - tail_bytes))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False

    stamp = f"{meta.recorded_at:%Y-%m-%d %H:%M}"
    return meta.title in tail and stamp in tail


def append_notes(notes_file: Path, meta: LectureMeta, body: str) -> bool:
    """Append one lecture's notes. Returns False if skipped as a duplicate."""
    if already_written(notes_file, meta):
        log.warning("Notes for %r already in %s; skipping", meta.title, notes_file.name)
        return False

    notes_file.parent.mkdir(parents=True, exist_ok=True)
    entry = format_entry(meta, body)
    # fsync so a crash right after a lecture cannot lose the notes we just
    # spent minutes generating.
    with notes_file.open("a", encoding="utf-8") as fh:
        fh.write(entry)
        fh.flush()
        os.fsync(fh.fileno())
    log.info("Appended %d chars to %s", len(entry), notes_file)
    return True
