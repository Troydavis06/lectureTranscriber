"""Transcribe a lecture recording with faster-whisper on the GPU.

Run as a subprocess (`python -m lecturetool.transcribe in.wav outdir`). That is
deliberate: CTranslate2 does not reliably hand VRAM back to the OS within a
live process, so a long-lived daemon that imported this would hold a model's
worth of VRAM between lectures. Process exit is the only guarantee, and it also
means a crash inside the CUDA stack cannot take the daemon down with it.

This is the stage that writes the finished transcript, so the files look the
same whether a lecture was captured automatically or a WAV was handed over by
hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import cuda_paths

# Registers the pip-installed cuDNN/cuBLAS DLL directories. Must happen before
# faster_whisper imports ctranslate2, or the model load fails to find them.
cuda_paths.apply()

from faster_whisper import WhisperModel  # noqa: E402 - must follow cuda_paths

from .util import fmt_timestamp, setup_logging  # noqa: E402

log = logging.getLogger("transcribe")


@dataclass
class Transcript:
    """A transcribed lecture: plain text plus timestamped segments."""

    source: str
    duration: float
    language: str
    segments: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s["text"].strip() for s in self.segments).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def shift(self, seconds: float) -> None:
        """Move every segment later by `seconds`.

        Used to line timestamps up with playback position when recording began
        part-way into a video.
        """
        if seconds <= 0:
            return
        for segment in self.segments:
            segment["start"] = round(segment["start"] + seconds, 2)
            segment["end"] = round(segment["end"] + seconds, 2)

    def to_timestamped_text(self) -> str:
        return "\n".join(
            f"[{fmt_timestamp(s['start'])}] {s['text'].strip()}" for s in self.segments
        )

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "duration_sec": round(self.duration, 2),
            "language": self.language,
            "segments": self.segments,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        return cls(
            source=data.get("source", ""),
            duration=float(data.get("duration_sec", 0.0)),
            language=data.get("language", "en"),
            segments=list(data.get("segments", [])),
        )

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))


def transcribe_file(wav: Path, cfg) -> Transcript:
    whisper = cfg.section("whisper")
    model_name = whisper.get("model", "large-v3-turbo")
    device = whisper.get("device", "cuda")
    compute_type = whisper.get("compute_type", "float16")

    log.info("Loading Whisper %r on %s (%s)", model_name, device, compute_type)
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 - surface a usable message, then fall back
        if device != "cuda":
            raise
        log.error("GPU load failed (%s); falling back to CPU int8", exc)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

    log.info("Transcribing %s", wav.name)
    segments_iter, info = model.transcribe(
        str(wav),
        language=whisper.get("language") or None,
        beam_size=int(whisper.get("beam_size", 5)),
        vad_filter=bool(whisper.get("vad_filter", True)),
        # Whisper's biggest long-audio failure mode is latching onto its own
        # previous output and repeating a phrase for minutes. Not conditioning
        # on previous text costs a little coherence and avoids that entirely.
        condition_on_previous_text=False,
    )

    transcript = Transcript(
        source=str(wav),
        duration=float(getattr(info, "duration", 0.0)),
        language=getattr(info, "language", "en") or "en",
    )

    # segments_iter is lazy: decoding happens while we walk it.
    last_logged = 0.0
    for segment in segments_iter:
        transcript.segments.append(
            {
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text.strip(),
            }
        )
        if segment.end - last_logged >= 60:
            last_logged = segment.end
            log.info("  ... %s / %s", fmt_timestamp(segment.end), fmt_timestamp(transcript.duration))

    log.info(
        "Transcribed %s: %d segments, %d words",
        wav.name,
        len(transcript.segments),
        transcript.word_count,
    )
    return transcript


def main() -> int:
    from . import config as config_mod
    from .util import slug_recorded_at, slug_title, wav_duration
    from .writer import LectureMeta, write_transcript

    parser = argparse.ArgumentParser(description="Transcribe a recording")
    parser.add_argument("wav", type=Path)
    parser.add_argument("outdir", type=Path, nargs="?")
    parser.add_argument("--slug", help="output basename (defaults to the wav stem)")
    parser.add_argument("--title", help="lecture title for the transcript header")
    parser.add_argument("--url", default="", help="source URL for the header")
    parser.add_argument("--recorded-at", help="ISO timestamp (defaults to the wav mtime)")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="seconds into the video that the recording began",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    cfg = config_mod.load()
    if not args.wav.is_file():
        log.error("No such file: %s", args.wav)
        return 1

    outdir = args.outdir or cfg.transcripts_dir
    slug = args.slug or args.wav.stem

    transcript = transcribe_file(args.wav, cfg)
    if not transcript.segments:
        log.error("No speech found in %s", args.wav.name)
        return 2

    if args.offset >= 1.0:
        log.info("Shifting timestamps by +%.0fs to match video position", args.offset)
        transcript.shift(args.offset)

    recorded_at = (
        datetime.fromisoformat(args.recorded_at)
        if args.recorded_at
        else slug_recorded_at(slug) or datetime.fromtimestamp(args.wav.stat().st_mtime)
    )
    meta = LectureMeta(
        title=args.title or slug_title(slug),
        url=args.url,
        duration_sec=wav_duration(args.wav),
        recorded_at=recorded_at,
    )
    text_path, _ = write_transcript(outdir, slug, meta, transcript)
    # The daemon reads this line to locate the transcript without guessing.
    print(text_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
