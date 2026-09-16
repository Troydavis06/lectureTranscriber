"""WASAPI loopback capture: record exactly what the speakers are playing.

Loopback capture taps the render endpoint, so it picks up Chrome's audio with
headphones plugged in and needs no "Stereo Mix" device. We write the device's
native format (typically 48 kHz stereo float or int16) straight to a WAV;
faster-whisper's PyAV decode handles the downmix and 16 kHz resample later, so
there is no resampling here and no ffmpeg dependency.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

log = logging.getLogger("audio")

# Loopback can deliver silent frames, so a "did we capture anything" check has
# to look at amplitude, not byte count.
SILENCE_RMS_THRESHOLD = 1e-4
_CHUNK_FRAMES = 1024
# A WASAPI loopback endpoint produces NO frames while no application is
# rendering audio -- it does not produce silence, it produces nothing. Left
# alone, that makes the WAV shorter than the wall-clock recording and slides
# every later timestamp earlier than the moment it happened in the video. We
# detect a shortfall this large and pad it with real silence.
_DRIFT_PAD_THRESHOLD_SEC = 0.25


class CaptureError(RuntimeError):
    pass


def _loopback_device(pa: pyaudio.PyAudio) -> dict:
    """The loopback endpoint matching the current default output device."""
    try:
        return pa.get_default_wasapi_loopback()
    except (OSError, LookupError) as exc:
        raise CaptureError(
            "No WASAPI loopback device found. Check that an audio output "
            "device is enabled and set as default."
        ) from exc


class LoopbackRecorder:
    """Records system audio to a WAV file, with pause/resume.

    Paused time is simply not written, so a lecture paused for five minutes
    produces a transcript with no five-minute silent gap. That means WAV
    position is "audio time", not wall-clock time — which is what we want the
    note timestamps to reflect.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._wav: wave.Wave_write | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()

        self._sample_rate = 0
        self._channels = 0
        self._sample_width = 0
        self._frames_written = 0
        self._peak_rms = 0.0
        self._is_float = False
        # Wall-clock seconds spent unpaused, i.e. how long the WAV *should* be.
        self._active_sec = 0.0
        self._active_since: float | None = None
        self._padded_sec = 0.0

    # --- lifecycle -----------------------------------------------------
    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        device = _loopback_device(self._pa)
        self._open_device(device)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(self._channels)
        self._wav.setsampwidth(self._sample_width)
        self._wav.setframerate(self._sample_rate)

        log.info(
            "Recording %s (%s, %d Hz, %d ch)",
            self.path.name,
            device["name"],
            self._sample_rate,
            self._channels,
        )
        self._active_since = time.monotonic()
        self._thread = threading.Thread(target=self._pump, name="loopback", daemon=True)
        self._thread.start()

    def _open_device(self, device: dict) -> None:
        """Open a loopback input stream, remembering its format."""
        assert self._pa is not None
        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"]) or 2
        # int16 keeps the WAV half the size of float32 at no cost in quality
        # that survives Whisper's 16 kHz mono downmix anyway.
        fmt = pyaudio.paInt16
        try:
            self._stream = self._pa.open(
                format=fmt,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=_CHUNK_FRAMES,
            )
        except OSError as exc:
            raise CaptureError(
                f"Could not open loopback stream on {device['name']!r}: {exc}"
            ) from exc

        self._sample_rate = rate
        self._channels = channels
        self._sample_width = self._pa.get_sample_size(fmt)
        self._is_float = fmt == pyaudio.paFloat32

    def _reopen(self) -> bool:
        """Recover from the default output device changing mid-lecture.

        Only succeeds if the new device has the same format as the WAV we are
        already appending to; a format change mid-file would corrupt it.
        """
        log.warning("Loopback stream failed; attempting to reopen")
        old = (self._sample_rate, self._channels, self._sample_width)
        try:
            if self._stream is not None:
                try:
                    self._stream.close()
                except OSError:
                    pass
            assert self._pa is not None
            self._open_device(_loopback_device(self._pa))
        except CaptureError as exc:
            log.error("Reopen failed: %s", exc)
            return False

        if (self._sample_rate, self._channels, self._sample_width) != old:
            log.error(
                "New audio device format %s differs from %s; cannot keep "
                "appending to the same WAV. Recording stops here.",
                (self._sample_rate, self._channels, self._sample_width),
                old,
            )
            return False
        log.info("Loopback stream reopened")
        return True

    def _pump(self) -> None:
        consecutive_errors = 0
        while not self._stop.is_set():
            try:
                data = self._stream.read(_CHUNK_FRAMES, exception_on_overflow=False)
                consecutive_errors = 0
            except OSError as exc:
                consecutive_errors += 1
                log.debug("Read error (%d): %s", consecutive_errors, exc)
                if consecutive_errors >= 3 and not self._reopen():
                    break
                time.sleep(0.1)
                continue

            if self._paused.is_set():
                continue

            self._track_level(data)
            with self._lock:
                if self._wav is None:
                    break
                self._pad_drift_locked()
                self._wav.writeframes(data)
                self._frames_written += len(data) // (self._sample_width * self._channels)

    def _elapsed_active_sec(self) -> float:
        """Wall-clock seconds the recording has been unpaused."""
        extra = 0.0 if self._active_since is None else time.monotonic() - self._active_since
        return self._active_sec + extra

    def _pad_drift_locked(self) -> None:
        """Insert silence for time that elapsed but produced no frames.

        Keeps WAV position equal to elapsed playback time, so a `[MM:SS]` in
        the notes points at the same moment in the video. Caller holds the lock.
        """
        deficit = self._elapsed_active_sec() - (self._frames_written / self._sample_rate)
        if deficit < _DRIFT_PAD_THRESHOLD_SEC:
            return
        pad_frames = int(deficit * self._sample_rate)
        self._wav.writeframes(b"\x00" * (pad_frames * self._sample_width * self._channels))
        self._frames_written += pad_frames
        self._padded_sec += deficit
        log.debug("Padded %.2fs of silence to stay aligned with playback", deficit)

    def _track_level(self, data: bytes) -> None:
        """Keep a running peak RMS so we can reject all-silence recordings."""
        if not data:
            return
        if self._is_float:
            samples = np.frombuffer(data, dtype=np.float32)
        else:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size:
            rms = float(np.sqrt(np.mean(np.square(samples))))
            self._peak_rms = max(self._peak_rms, rms)

    def pause(self) -> None:
        if self._paused.is_set():
            return
        with self._lock:
            # Settle the alignment debt before freezing the clock, otherwise a
            # gap that opened just before the pause is silently forgiven.
            if self._wav is not None:
                self._pad_drift_locked()
            if self._active_since is not None:
                self._active_sec += time.monotonic() - self._active_since
                self._active_since = None
        self._paused.set()
        log.info("Recording paused at %.0fs", self.duration_sec)

    def resume(self) -> None:
        if not self._paused.is_set():
            return
        with self._lock:
            self._active_since = time.monotonic()
        self._paused.clear()
        log.info("Recording resumed")

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def duration_sec(self) -> float:
        if not self._sample_rate:
            return 0.0
        return self._frames_written / self._sample_rate

    @property
    def peak_rms(self) -> float:
        return self._peak_rms

    @property
    def padded_sec(self) -> float:
        """Silence inserted to keep the WAV aligned with playback time."""
        return self._padded_sec

    def stop(self) -> float:
        """Close everything and return the recorded duration in seconds."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            if self._wav is not None:
                if not self._paused.is_set():
                    self._pad_drift_locked()
                self._wav.close()
                self._wav = None
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        log.info("Stopped recording %s (%.0fs)", self.path.name, self.duration_sec)
        return self.duration_sec

    def looks_silent(self) -> bool:
        return self._peak_rms < SILENCE_RMS_THRESHOLD


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        return wf.getnframes() / rate if rate else 0.0


def _test(seconds: int, out: Path) -> int:
    """Record a short clip and report whether real audio was captured."""
    rec = LoopbackRecorder(out)
    rec.start()
    print(f"Recording {seconds}s of system audio -> {out}")
    print("Play something now (a YouTube video, anything).")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining:>3}s  level={rec.peak_rms:.5f}", end="\r", flush=True)
        time.sleep(1)
    duration = rec.stop()
    print()

    if not out.exists() or out.stat().st_size <= 44:  # 44 = bare WAV header
        print("FAIL: no audio data was written.")
        return 1
    print(
        f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {duration:.1f}s, "
        f"{rec.padded_sec:.1f}s of that padded silence)"
    )
    drift = abs(duration - seconds)
    if drift > 1.0:
        print(f"WARNING: WAV is {drift:.1f}s off the {seconds}s requested window.")
    if rec.looks_silent():
        print(
            f"WARNING: captured audio is silent (peak RMS {rec.peak_rms:.6f}).\n"
            "  Either nothing was playing, or the default output device is not "
            "the one you are listening through."
        )
        return 1
    print(f"OK: real audio captured (peak RMS {rec.peak_rms:.4f}).")
    return 0


def _list_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        print("WASAPI loopback endpoints:")
        for info in pa.get_loopback_device_info_generator():
            print(
                f"  [{info['index']:>2}] {info['name']}  "
                f"{int(info['defaultSampleRate'])} Hz  "
                f"{info['maxInputChannels']} ch"
            )
        default = pa.get_default_wasapi_loopback()
        print(f"\nDefault loopback: [{default['index']}] {default['name']}")
    finally:
        pa.terminate()


def main() -> int:
    from . import config as config_mod
    from .util import setup_logging

    parser = argparse.ArgumentParser(description="System audio loopback capture")
    parser.add_argument("--test", type=int, metavar="SECONDS", help="record a test clip")
    parser.add_argument("--list", action="store_true", help="list loopback devices")
    parser.add_argument("--out", type=Path, help="output WAV for --test")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list:
        _list_devices()
        return 0
    if args.test:
        cfg = config_mod.load()
        cfg.ensure_dirs()
        out = args.out or (cfg.recordings_dir / "_capture_test.wav")
        return _test(args.test, out)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
