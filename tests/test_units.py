"""Checks for the pure-function pieces: block dedupe, titles, writer output.

Run with:  .venv\\Scripts\\python.exe tests\\test_units.py
No test framework needed, and nothing here touches the GPU, Chrome or Ollama.
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lecturetool.notes import _count_blocks, dedupe_blocks
from lecturetool.util import clean_title, fmt_duration, fmt_timestamp, stamped_slug
from lecturetool.writer import LectureMeta, append_notes, format_entry

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

print("slug")
check("stamped", stamped_slug("Lecture 7: Dynamic Programming!", datetime(2026, 9, 16, 14, 30)),
      "2026-09-16_1430_lecture-7-dynamic-programming")

print("dedupe_blocks: ordering")
unordered = """\
[02:00] Later topic
  - b

[00:30] Earlier topic
  - a
"""
check("sorted by timestamp",
      dedupe_blocks(unordered),
      "[00:30] Earlier topic\n  - a\n\n[02:00] Later topic\n  - b")

print("dedupe_blocks: duplicates")
dupes = """\
[00:30] Memoization
  - store each answer the first time

[00:35] Memoization
  - Store each answer the first time.

[01:00] Bottom up
  - fills array upward
"""
check("duplicate collapsed", _count_blocks(dedupe_blocks(dupes)), 2)

print("dedupe_blocks: contained block dropped")
contained = """\
[00:30] Recurrence
  - OPT(i) = max(OPT(i-1), v_i + OPT(p(i)))
  - two cases: skip or take

[00:40] Recurrence
  - OPT(i) = max(OPT(i-1), v_i + OPT(p(i)))
"""
check("contained dropped", _count_blocks(dedupe_blocks(contained)), 1)

print("dedupe_blocks: hour-format timestamps")
hourly = """\
[1:05:00] Late topic
  - z

[00:10] Early topic
  - a
"""
check("hour parsed and ordered",
      dedupe_blocks(hourly).splitlines()[0], "[00:10] Early topic")

print("dedupe_blocks: passthrough when no blocks")
check("no blocks", dedupe_blocks("just some prose"), "just some prose")

print("transcript shift")
from lecturetool.transcribe import Transcript  # noqa: E402 - keeps cuda import late

t = Transcript(source="x", duration=10.0, language="en",
               segments=[{"start": 0.0, "end": 2.0, "text": "a"},
                         {"start": 2.0, "end": 4.0, "text": "b"}])
t.shift(7.5)
check("first start shifted", t.segments[0]["start"], 7.5)
check("first end shifted", t.segments[0]["end"], 9.5)
check("second start shifted", t.segments[1]["start"], 9.5)
check("rendered timestamp", t.to_timestamped_text().splitlines()[0], "[00:07] a")
t.shift(0)
check("zero shift is a no-op", t.segments[0]["start"], 7.5)
t.shift(-5)
check("negative shift is a no-op", t.segments[0]["start"], 7.5)

print("writer")
meta = LectureMeta(
    title="Lecture 7 - Dynamic Programming",
    url="https://example.edu/lec7",
    duration_sec=3120,
    recorded_at=datetime(2026, 9, 16, 14, 30),
)
entry = format_entry(meta, "[00:00] Topic\n  - point")
check("header has title", "Lecture 7 - Dynamic Programming" in entry, True)
check("header has duration", "52 min" in entry, True)
check("header has url", "https://example.edu/lec7" in entry, True)
check("header has date", "2026-09-16 14:30" in entry, True)

with tempfile.TemporaryDirectory() as tmp:
    notes_file = Path(tmp) / "notes.txt"
    check("first append", append_notes(notes_file, meta, "[00:00] A\n  - x"), True)
    check("duplicate refused", append_notes(notes_file, meta, "[00:00] A\n  - x"), False)
    other = LectureMeta(title="Lecture 8 - Flows", url="", duration_sec=2400,
                        recorded_at=datetime(2026, 9, 18, 14, 30))
    check("different lecture appends", append_notes(notes_file, other, "[00:00] B\n  - y"), True)
    content = notes_file.read_text(encoding="utf-8")
    check("both present", content.count("=" * 72), 4)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    sys.exit(1)
print("all unit checks passed")
