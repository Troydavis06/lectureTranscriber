"""Write a finished transcript to disk, as readable text and as JSON.

Each lecture owns its own pair of files, keyed by slug. Re-running a recording
therefore overwrites that lecture's transcript rather than accumulating copies,
which is what you want when you re-transcribe after changing the model.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .util import fmt_duration

if TYPE_CHECKING:  # importing transcribe for real would pull in faster_whisper
    from .transcribe import Transcript

log = logging.getLogger("writer")

_RULE = "=" * 72


@dataclass
class LectureMeta:
    title: str
    url: str
    duration_sec: float
    recorded_at: datetime


def format_transcript(meta: LectureMeta, transcript: "Transcript") -> str:
    """Header block followed by one `[MM:SS] text` line per segment."""
    header = f"{meta.recorded_at:%Y-%m-%d %H:%M} | {fmt_duration(meta.duration_sec)}"
    if meta.url:
        header = f"{header} | {meta.url}"
    return f"{_RULE}\n{meta.title}\n{header}\n{_RULE}\n\n{transcript.to_timestamped_text()}\n"


def transcript_json(meta: LectureMeta, transcript: "Transcript") -> dict:
    """The JSON form, which carries the lecture metadata the text header shows."""
    return {
        "title": meta.title,
        "url": meta.url,
        "recorded_at": meta.recorded_at.isoformat(timespec="seconds"),
        "duration_sec": round(meta.duration_sec, 2),
        **transcript.to_json(),
    }


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and fsync.

    A transcript costs minutes of GPU time, so a crash mid-write must not be
    able to leave a half-written file where a readable one used to be.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def write_transcript(
    outdir: Path, slug: str, meta: LectureMeta, transcript: "Transcript"
) -> tuple[Path, Path]:
    """Write `<slug>.txt` and `<slug>.json`; returns both paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    text_path = outdir / f"{slug}.txt"
    json_path = outdir / f"{slug}.json"

    _write_atomic(text_path, format_transcript(meta, transcript))
    _write_atomic(
        json_path,
        json.dumps(transcript_json(meta, transcript), indent=2, ensure_ascii=False),
    )
    log.info(
        "Wrote %s (%d segments, %d words)",
        text_path.name,
        len(transcript.segments),
        transcript.word_count,
    )
    return text_path, json_path
