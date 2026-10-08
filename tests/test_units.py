"""Checks for the pure-function pieces: titles, timestamps, transcript output.

Run with:  .venv\\Scripts\\python.exe tests\\test_units.py
No test framework needed, and nothing here touches the GPU or Chrome.
"""
import json
import sys
import tempfile
import wave
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lecturetool.util import (
    clean_title,
    fmt_duration,
    fmt_timestamp,
    slug_title,
    slugify,
    stamped_slug,
    wav_duration,
)
from lecturetool.writer import LectureMeta, format_transcript, transcript_json, write_transcript

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


print("clean_title")
check("youtube suffix", clean_title("Lecture 7 - Dynamic Programming - YouTube"),
      "Lecture 7 - Dynamic Programming")
check("panopto suffix", clean_title("Lecture 7 - Dynamic Programming - Panopto"),
      "Lecture 7 - Dynamic Programming")
check("unread count", clean_title("(3) Week 4 Lecture - Canvas"), "Week 4 Lecture")
check("empty", clean_title(""), "Untitled lecture")
check("no noise", clean_title("CS341 Week 2"), "CS341 Week 2")

print("timestamps and durations")
check("mm:ss", fmt_timestamp(75), "01:15")
check("h:mm:ss", fmt_timestamp(3725), "1:02:05")
check("zero", fmt_timestamp(0), "00:00")
check("dur min", fmt_duration(3120), "52 min")
check("dur hour", fmt_duration(6420), "1 h 47 min")
check("dur sec", fmt_duration(48), "48 s")

print("slugs")
check("stamped", stamped_slug("Lecture 7: Dynamic Programming!", datetime(2026, 9, 16, 14, 30)),
      "2026-09-16_1430_lecture-7-dynamic-programming")
check("non-ascii stripped", slugify("Café Lecture — Intro"), "cafe-lecture-intro")
check("empty falls back", slugify(""), "lecture")

print("slug_title")
check("round trip", slug_title("2026-09-16_1430_lecture-7-dynamic-programming"),
      "Lecture 7 Dynamic Programming")
check("hand-named kept", slug_title("my-own-recording"), "my-own-recording")
check("stamp only kept", slug_title("2026-09-16_1430"), "2026-09-16_1430")

print("transcript shift")
from lecturetool.transcribe import Transcript  # noqa: E402 - keeps cuda import late


def sample() -> Transcript:
    return Transcript(
        source="x.wav",
        duration=10.0,
        language="en",
        segments=[{"start": 0.0, "end": 2.0, "text": "first line"},
                  {"start": 2.0, "end": 4.0, "text": "second line"}],
    )


t = sample()
t.shift(7.5)
check("first start shifted", t.segments[0]["start"], 7.5)
check("first end shifted", t.segments[0]["end"], 9.5)
check("second start shifted", t.segments[1]["start"], 9.5)
check("rendered timestamp", t.to_timestamped_text().splitlines()[0], "[00:07] first line")
t.shift(0)
check("zero shift is a no-op", t.segments[0]["start"], 7.5)
t.shift(-5)
check("negative shift is a no-op", t.segments[0]["start"], 7.5)

print("transcript text and json round trip")
t = sample()
check("text joins segments", t.text, "first line second line")
check("word count", t.word_count, 4)
check("reloaded segments", Transcript.from_json(t.to_json()).segments, t.segments)

print("format_transcript")
meta = LectureMeta(
    title="Lecture 7 - Dynamic Programming",
    url="https://example.edu/lec7",
    duration_sec=3120,
    recorded_at=datetime(2026, 9, 16, 14, 30),
)
entry = format_transcript(meta, sample())
check("header has title", "Lecture 7 - Dynamic Programming" in entry, True)
check("header has duration", "52 min" in entry, True)
check("header has url", "https://example.edu/lec7" in entry, True)
check("header has date", "2026-09-16 14:30" in entry, True)
check("body has segments", "[00:00] first line" in entry, True)

no_url = LectureMeta(title="Lecture 8", url="", duration_sec=60,
                     recorded_at=datetime(2026, 9, 18, 9, 5))
check("no trailing separator without a url",
      format_transcript(no_url, sample()).splitlines()[2], "2026-09-18 09:05 | 1 min")

print("transcript_json carries metadata")
payload = transcript_json(meta, sample())
check("title", payload["title"], "Lecture 7 - Dynamic Programming")
check("recorded_at iso", payload["recorded_at"], "2026-09-16T14:30:00")
check("duration", payload["duration_sec"], 3120)
check("segments kept", len(payload["segments"]), 2)

print("write_transcript")
with tempfile.TemporaryDirectory() as tmp:
    outdir = Path(tmp)
    text_path, json_path = write_transcript(outdir, "lec7", meta, sample())
    check("txt written", text_path.name, "lec7.txt")
    check("json written", json_path.name, "lec7.json")
    check("txt readable", "[00:02] second line" in text_path.read_text(encoding="utf-8"), True)
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    check("json parses", reloaded["title"], "Lecture 7 - Dynamic Programming")
    check("json segments", len(reloaded["segments"]), 2)

    # Re-transcribing a lecture replaces its transcript rather than piling up.
    longer = sample()
    longer.segments.append({"start": 4.0, "end": 6.0, "text": "third line"})
    write_transcript(outdir, "lec7", meta, longer)
    check("one txt per slug", len(list(outdir.glob("*.txt"))), 1)
    check("no temp files left", list(outdir.glob("*.tmp")), [])
    check("overwritten in place",
          len(json.loads(json_path.read_text(encoding="utf-8"))["segments"]), 3)

print("wav_duration")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "quiet.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(b"\x00" * (48000 * 2 * 2 * 3))  # 3 seconds
    check("three seconds", round(wav_duration(path), 3), 3.0)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    sys.exit(1)
print("all unit checks passed")
