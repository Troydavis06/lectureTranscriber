# Lecture Summary Tool

Watches Chrome for a lecture video playing, records what you hear, transcribes it
on the GPU, and appends condensed quiz-oriented notes to a single `data/notes.txt`.

Runs entirely on this machine. No cloud APIs, nothing uploaded.

## Using it

1. Start the daemon (leave the window open):

   ```
   run.cmd
   ```

2. Start Chrome with `launch_chrome.cmd`.

3. Open a lecture and press play. That's it.

When the video finishes, the tool transcribes the recording and appends a block
to `data/notes.txt`. Pausing to think is fine — the recording pauses with the
video and resumes with it. Only a pause longer than `pause_grace_sec`
(default 2 minutes) ends the lecture.

### First-time setup

`launch_chrome.cmd` uses a **dedicated Chrome profile**, because Chrome 136+
refuses `--remote-debugging-port` on the default profile. The first time you use
it you'll need to sign in to your course site in that window. Your normal Chrome
is untouched.

Install the dependencies and pull the notes model once:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:14b
```

## What comes out

```
========================================================================
Lecture 7 - Dynamic Programming
2026-09-16 14:30 | 52 min | https://...
========================================================================

[00:11] Why greedy fails
  - Greedy by finish time takes 2 short intervals worth 20
  - Optimal takes 1 long interval worth 100
  - KEY: greedy commits early, cannot reconsider an earlier choice

[01:02] Weighted interval scheduling recurrence
  - OPT(i) = max(OPT(i-1), v_i + OPT(p(i)))
  - KEY: p(i) is the largest j < i whose interval does not overlap i
```

Timestamps point back into the video, so you can jump to anything you want to
rewatch. Full transcripts are kept in `data/transcripts/` as well.

## Models

| Stage | Model | VRAM | Notes |
|---|---|---|---|
| Transcription | `faster-whisper large-v3-turbo` (float16) | ~1.6 GB | ~58x realtime on a 4080 |
| Notes | `qwen3:14b` (q4_K_M) via Ollama | ~9 GB | 32k context, thinking disabled |

The two never load at once. Transcription runs as a subprocess so its VRAM is
released on exit, and Ollama is asked to unload with `keep_alive=0` afterwards,
which is what keeps this inside 16 GB.

Both are one-line changes in `config.toml`. Swap `large-v3-turbo` for `large-v3`
if a lecturer's accent gives trouble.

## Configuration

See `config.toml` — every value has a default in `lecturetool/config.py`, so you
only need to list what you want to change. The ones most worth knowing:

| Key | Default | Why you'd change it |
|---|---|---|
| `chrome.pause_grace_sec` | 120 | How long a pause can last before the lecture is considered over |
| `chrome.min_video_duration_sec` | 120 | Ignore short clips and embedded promos |
| `chrome.url_denylist` | music/streaming hosts | Stop a non-lecture tab starting a recording |
| `audio.min_lecture_sec` | 180 | Discard recordings shorter than this |
| `audio.keep_recordings_days` | 7 | WAV retention; `0` keeps them forever |
| `notes.model` | `qwen3:14b` | Swap the note-writing model |

## Running pieces on their own

Each stage is independently runnable, which is how to diagnose anything:

```bat
rem Is system audio being captured at all? Play something, then:
.venv\Scripts\python.exe -m lecturetool.audio_capture --test 10
.venv\Scripts\python.exe -m lecturetool.audio_capture --list

rem What does the watcher see? Play/pause a video and watch the events.
.venv\Scripts\python.exe -m lecturetool.chrome_watch --watch

rem Transcribe and generate notes from files directly.
.venv\Scripts\python.exe -m lecturetool.transcribe data\recordings\x.wav
.venv\Scripts\python.exe -m lecturetool.notes data\transcripts\x.json --stdout

rem Unit checks (no GPU, Chrome or Ollama needed).
.venv\Scripts\python.exe tests\test_units.py
```

Reprocess a saved recording — useful after changing the note prompt, since it
regenerates notes without rewatching anything:

```
run.cmd --process data\recordings\2026-09-16_1430_lecture-7.wav
```

## If something doesn't work

**Notes never appear.** Check the daemon window. The most common causes are the
recording being under `audio.min_lecture_sec`, or captured audio being silent
because Chrome is playing through a device that isn't the Windows default
output. `--test 10` above settles the second one.

**"Could not locate cudnn_ops64_9.dll".** The pip NVIDIA packages are missing.
`pip install -r requirements.txt` again, then check what got registered:

```
.venv\Scripts\python.exe -m lecturetool.cuda_paths
```

**"Chrome is not reachable on port 9222".** Chrome was started normally rather
than via `launch_chrome.cmd`. The daemon can be started before Chrome; it waits.

**Nothing starts recording.** Run `chrome_watch --watch`. If your video shows
`skip`, it's being filtered — most likely shorter than
`chrome.min_video_duration_sec`, or its host is in `url_denylist`.
