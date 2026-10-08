"""The daemon: watch Chrome, record what plays, transcribe it.

Recording and transcription are deliberately decoupled. Chrome events drive the
recorder on the main thread; finished recordings go onto a queue that a single
worker thread drains. That way a lecture starting while the previous one is
still being transcribed is still captured -- back-to-back lectures do not drop
audio, they just queue up.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import config as config_mod
from .audio_capture import CaptureError, LoopbackRecorder
from .chrome_watch import Event, LectureWatcher, VideoState
from .util import (
    clean_title,
    fmt_duration,
    fmt_timestamp,
    setup_logging,
    slug_recorded_at,
    slug_title,
    stamped_slug,
    wav_duration,
)
from .writer import LectureMeta

log = logging.getLogger("daemon")


@dataclass
class Job:
    wav: Path
    slug: str
    meta: LectureMeta
    # Playback position when recording began. Detection needs a couple of polls
    # to debounce, so recording starts several seconds into the video; without
    # this every transcript timestamp points slightly earlier than the words it
    # labels, which defeats using them to jump back into the lecture.
    offset_sec: float = 0.0


class Pipeline:
    """Turns recordings into transcripts, one at a time."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="pipeline", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, job: Job) -> None:
        self.queue.put(job)
        log.info("Queued %s for transcription (%d waiting)", job.slug, self.queue.qsize())

    def shutdown(self, timeout: float = 5.0) -> int:
        """Signal the worker to stop; returns how many jobs were left unprocessed."""
        remaining = self.queue.qsize()
        self.queue.put(None)
        self.thread.join(timeout=timeout)
        return remaining

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            try:
                self.process(job)
            except Exception:  # noqa: BLE001 - one bad lecture must not kill the worker
                log.exception("Failed to process %s", job.slug)
            finally:
                self.queue.task_done()

    def process(self, job: Job) -> Path | None:
        """Transcribe one recording; returns the transcript path, or None.

        Shells out so the transcriber's VRAM is handed back on process exit --
        see the module docstring in transcribe.py.
        """
        outdir = self.cfg.transcripts_dir
        cmd = [
            sys.executable,
            "-m",
            "lecturetool.transcribe",
            str(job.wav),
            str(outdir),
            "--slug",
            job.slug,
            "--title",
            job.meta.title,
            "--url",
            job.meta.url,
            "--recorded-at",
            job.meta.recorded_at.isoformat(timespec="seconds"),
            "--offset",
            f"{job.offset_sec:.2f}",
        ]
        log.info("Transcribing %s", job.slug)
        started = time.monotonic()
        # Stream the child's stderr instead of capturing it wholesale: a long
        # lecture takes minutes and silence looks like a hang. The tail is kept
        # so a failure still reports something useful.
        tail: list[str] = []
        process = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stderr is not None
        for line in process.stderr:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            del tail[:-40]
            if " INFO " in line or " ERROR " in line or " WARNING " in line:
                log.info("  [whisper] %s", line.split("  ", 1)[-1].strip())
        returncode = process.wait()

        if returncode != 0:
            log.error(
                "Transcription failed (exit %d):\n%s", returncode, "\n".join(tail[-20:])
            )
            log.error("The recording is kept at %s", job.wav)
            return None

        text_path = outdir / f"{job.slug}.txt"
        if not text_path.is_file():
            log.error("Transcriber reported success but %s is missing", text_path)
            return None

        log.info(
            "Done in %s: %r -> %s",
            fmt_duration(time.monotonic() - started),
            job.meta.title,
            text_path,
        )
        return text_path


class Daemon:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.min_lecture_sec = float(cfg.get("audio", "min_lecture_sec", 180))
        self.pipeline = Pipeline(cfg)
        self.watcher = LectureWatcher(cfg, self.on_event)
        self.recorder: LoopbackRecorder | None = None
        self.current: LectureMeta | None = None
        self.current_slug: str | None = None
        self.current_offset = 0.0
        self._stop = threading.Event()

    # --- event handling ------------------------------------------------
    def on_event(self, event: Event, state: VideoState) -> None:
        if event is Event.START:
            self._start_recording(state)
        elif event is Event.PAUSE and self.recorder is not None:
            self.recorder.pause()
        elif event is Event.RESUME and self.recorder is not None:
            self.recorder.resume()
        elif event is Event.END:
            self._finish_recording(state)

    def _start_recording(self, state: VideoState) -> None:
        if self.recorder is not None:
            log.warning("Already recording; ignoring start of %r", state.title)
            return

        title = clean_title(state.title)
        now = datetime.now()
        slug = stamped_slug(title, now)
        wav = self.cfg.recordings_dir / f"{slug}.wav"

        recorder = LoopbackRecorder(wav)
        try:
            recorder.start()
        except CaptureError as exc:
            log.error("Cannot record: %s", exc)
            return

        self.recorder = recorder
        self.current_slug = slug
        self.current_offset = max(0.0, state.position)
        self.current = LectureMeta(
            title=title, url=state.url, duration_sec=0.0, recorded_at=now
        )
        log.info(
            "Recording lecture %r from %s in", title, fmt_timestamp(self.current_offset)
        )

    def _finish_recording(self, state: VideoState) -> None:
        recorder, meta, slug = self.recorder, self.current, self.current_slug
        offset = self.current_offset
        self.recorder = None
        self.current = None
        self.current_slug = None
        self.current_offset = 0.0
        if recorder is None or meta is None or slug is None:
            return

        duration = recorder.stop()
        meta.duration_sec = duration

        if duration < self.min_lecture_sec:
            log.info(
                "Discarding %s: %s is under the %s minimum",
                slug,
                fmt_duration(duration),
                fmt_duration(self.min_lecture_sec),
            )
            recorder.path.unlink(missing_ok=True)
            return
        if recorder.looks_silent():
            log.warning(
                "Discarding %s: captured audio is silent. Is Chrome playing "
                "through the default output device?",
                slug,
            )
            recorder.path.unlink(missing_ok=True)
            return

        self.pipeline.submit(
            Job(wav=recorder.path, slug=slug, meta=meta, offset_sec=offset)
        )

    # --- housekeeping ---------------------------------------------------
    def prune_recordings(self) -> None:
        days = int(self.cfg.get("audio", "keep_recordings_days", 7))
        if days <= 0:
            return
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        for wav in self.cfg.recordings_dir.glob("*.wav"):
            try:
                if datetime.fromtimestamp(wav.stat().st_mtime) < cutoff:
                    wav.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            log.info("Pruned %d recording(s) older than %d days", removed, days)

    # --- lifecycle ------------------------------------------------------
    def stop(self, *_args) -> None:
        if not self._stop.is_set():
            log.info("Shutting down...")
            self._stop.set()

    def run(self) -> int:
        self.cfg.ensure_dirs()
        self.prune_recordings()
        self.pipeline.start()

        # Only the main thread may install handlers. Running the daemon from a
        # worker thread is legitimate (tests, embedding), and there Ctrl+C is
        # the host's business rather than ours.
        try:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        except ValueError:
            log.debug("Not on the main thread; skipping signal handlers")

        log.info("Watching Chrome on port %d. Transcripts go to %s",
                 self.watcher.client.port, self.cfg.transcripts_dir)
        log.info("Start Chrome with launch_chrome.cmd, then just play a lecture.")

        try:
            self.watcher.run(stop_check=self._stop.is_set)
        except KeyboardInterrupt:
            self.stop()

        # A lecture still recording at shutdown is worth keeping.
        if self.recorder is not None and self.current is not None:
            log.info("Finalising the in-progress recording before exit")
            self._finish_recording(
                VideoState(
                    target_id="",
                    title=self.current.title,
                    url=self.current.url,
                    playing=False,
                    position=0.0,
                    duration=0.0,
                    ended=True,
                    muted=False,
                )
            )

        pending = self.pipeline.queue.qsize()
        if pending:
            log.warning(
                "%d recording(s) still queued. Transcribe them with: "
                "run.cmd --process data\\recordings\\<file>.wav",
                pending,
            )
        self.pipeline.shutdown()
        self.watcher.close()
        log.info("Stopped.")
        return 0


def process_one(cfg, wav: Path, title: str | None) -> int:
    """Transcribe a saved recording, e.g. one the daemon left queued."""
    if not wav.is_file():
        log.error("No such recording: %s", wav)
        return 1

    cfg.ensure_dirs()
    # Recover the original title and time from the stamped slug when possible.
    meta = LectureMeta(
        title=title or slug_title(wav.stem),
        url="",
        duration_sec=wav_duration(wav),
        recorded_at=slug_recorded_at(wav.stem)
        or datetime.fromtimestamp(wav.stat().st_mtime),
    )
    result = Pipeline(cfg).process(Job(wav=wav, slug=wav.stem, meta=meta))
    return 0 if result is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lecturetool",
        description="Watch Chrome and transcribe the lectures that play.",
    )
    parser.add_argument(
        "--process",
        type=Path,
        metavar="WAV",
        help="transcribe a saved recording instead of running the daemon",
    )
    parser.add_argument("--title", help="lecture title to use with --process")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = config_mod.load()
    if args.process:
        return process_one(cfg, args.process, args.title)
    return Daemon(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
