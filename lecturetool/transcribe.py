"""Transcribe a lecture recording with faster-whisper on the GPU.

Run as a subprocess (`python -m lecturetool.transcribe in.wav outdir`). That is
deliberate: CTranslate2 does not reliably hand VRAM back to the OS within a
live process, and the note-generation LLM needs that VRAM next. Process exit is
the only guarantee, so the daemon always shells out rather than importing this.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
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

    def to_timestamped_text(self) -> str:
        return "\n".join(
            f"[{fmt_timestamp(s['start'])}] {s['text'].strip()}" for s in self.segments
        )

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "duration": self.duration,
            "language": self.language,
            "segments": self.segments,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        return cls(
            source=data.get("source", ""),
            duration=float(data.get("duration", 0.0)),
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


def write_outputs(transcript: Transcript, outdir: Path, slug: str) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{slug}.json"
    text_path = outdir / f"{slug}.txt"
    json_path.write_text(
        json.dumps(transcript.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text_path.write_text(transcript.to_timestamped_text(), encoding="utf-8")
    return json_path, text_path


def main() -> int:
    from . import config as config_mod

    parser = argparse.ArgumentParser(description="Transcribe a recording")
    parser.add_argument("wav", type=Path)
    parser.add_argument("outdir", type=Path, nargs="?")
    parser.add_argument("--slug", help="output basename (defaults to the wav stem)")
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

    json_path, text_path = write_outputs(transcript, outdir, slug)
    log.info("Wrote %s and %s", json_path.name, text_path.name)
    # The daemon reads this line to locate the transcript without guessing.
    print(json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
