"""Turn a transcript into condensed, quiz-oriented notes with a local LLM.

Format is a chronological walk through the lecture: one block per topic, in the
order the lecturer covered it, condensed to what would actually be asked. Long
lectures go through a map-reduce so length never matters -- each chunk is
summarised independently, then the blocks are merged to repair topics that
straddled a chunk boundary.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import ollama

from .transcribe import Transcript
from .util import fmt_timestamp, setup_logging

log = logging.getLogger("notes")

# Rough words-per-token for English prose. Used only to size chunks, so being
# approximate is fine; the LLM's own context limit is the real backstop.
_WORDS_PER_TOKEN = 0.75

_SHARED_RULES = """\
Rules:
- Walk the material in the order it was taught. Do not reorder or regroup.
- One block per topic. Start each block with its [MM:SS] timestamp, then a short topic name.
- Under each block, terse bullets. Fragments, not sentences. No filler verbs.
- Copy formulas, definitions, numbers and names EXACTLY as stated. Never paraphrase a formula.
- Mark the single most testable fact in a block with "KEY:" at the start of that bullet.
- If the lecturer flags something as important, examinable, a common mistake, or homework, keep it.
- Omit greetings, admin chatter, tangents and repetition.
- Output ONLY the blocks. No preamble, no closing summary, no markdown headers, no bold.
"""

_MAP_PROMPT = """\
You are writing revision notes from part of a lecture transcript.

{rules}
Format exactly like this:

[03:20] Why greedy fails
  - Greedy by finish time takes 2 short intervals worth 20
  - Optimal takes 1 long interval worth 100
  - KEY: greedy commits early, cannot reconsider an earlier choice

Transcript part {index} of {total}:
---
{chunk}
---
"""

_REDUCE_PROMPT = """\
Below are notes generated from consecutive parts of ONE lecture. Merge them into
a single clean set of notes.

{rules}
Additionally:
- Merge blocks covering the same topic, keeping the EARLIEST timestamp.
- Drop bullets repeated across parts.
- Keep every distinct formula, definition, example result and warning.
- Do not invent anything that is not in the input.

Notes to merge:
---
{blocks}
---
"""


@dataclass
class NotesResult:
    body: str
    chunks: int
    model: str


class NotesError(RuntimeError):
    pass


def _chunk_transcript(transcript: Transcript, chunk_tokens: int, overlap_words: int):
    """Split timestamped segments into windows of roughly chunk_tokens.

    Splits only on segment boundaries so a sentence is never cut in half, and
    overlaps consecutive windows so a topic spanning a boundary is visible in
    both.
    """
    target_words = max(200, int(chunk_tokens * _WORDS_PER_TOKEN))
    chunks: list[str] = []
    current: list[dict] = []
    current_words = 0

    def render(segments: list[dict]) -> str:
        return "\n".join(
            f"[{fmt_timestamp(s['start'])}] {s['text'].strip()}" for s in segments
        )

    for segment in transcript.segments:
        words = len(segment["text"].split())
        if current and current_words + words > target_words:
            chunks.append(render(current))
            # Carry back enough trailing segments to cover overlap_words.
            carried: list[dict] = []
            carried_words = 0
            for prev in reversed(current):
                if carried_words >= overlap_words:
                    break
                carried.insert(0, prev)
                carried_words += len(prev["text"].split())
            current = carried
            current_words = carried_words
        current.append(segment)
        current_words += words

    if current:
        chunks.append(render(current))
    return chunks


def _strip_model_noise(text: str) -> str:
    """Remove reasoning blocks and chatty framing some models still emit."""
    # Belt and braces: think=False should prevent these, but a GGUF with a
    # baked-in template can still leak them.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    lines = text.strip().splitlines()
    # Drop a leading "Here are the notes:" style line.
    while lines and not lines[0].lstrip().startswith("["):
        first = lines[0].strip().lower()
        if first and (first.endswith(":") or first.startswith(("here", "sure", "okay", "below"))):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


class NoteGenerator:
    def __init__(self, cfg) -> None:
        notes = cfg.section("notes")
        self.model = notes.get("model", "qwen3:14b")
        self.num_ctx = int(notes.get("num_ctx", 16384))
        self.temperature = float(notes.get("temperature", 0.2))
        self.chunk_tokens = int(notes.get("chunk_tokens", 3500))
        self.overlap_words = int(notes.get("chunk_overlap_words", 200))
        self.client = ollama.Client(host=notes.get("host", "http://127.0.0.1:11434"))

    def _generate(self, prompt: str, keep_alive: str | int = "5m") -> str:
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                think=False,
                keep_alive=keep_alive,
                options={
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                    # Repetition at this size usually means the model is padding;
                    # a mild penalty keeps bullets from duplicating.
                    "repeat_penalty": 1.05,
                },
            )
        except ollama.ResponseError as exc:
            if "not found" in str(exc).lower():
                raise NotesError(
                    f"Model {self.model!r} is not available. Run: ollama pull {self.model}"
                ) from exc
            raise NotesError(f"Ollama error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - connection refused etc.
            raise NotesError(
                f"Could not reach Ollama at {self.client._client.base_url}: {exc}"
            ) from exc
        return response.get("response", "")

    def unload(self) -> None:
        """Free the model's VRAM so the next transcription has the GPU to itself."""
        try:
            self.client.generate(model=self.model, prompt="", keep_alive=0)
            log.debug("Asked Ollama to unload %s", self.model)
        except Exception as exc:  # noqa: BLE001 - best effort only
            log.debug("Unload request failed: %s", exc)

    def generate(self, transcript: Transcript) -> NotesResult:
        if not transcript.segments:
            raise NotesError("Transcript has no segments")

        chunks = _chunk_transcript(transcript, self.chunk_tokens, self.overlap_words)
        log.info(
            "Generating notes with %s: %d words in %d chunk(s)",
            self.model,
            transcript.word_count,
            len(chunks),
        )

        try:
            blocks: list[str] = []
            for index, chunk in enumerate(chunks, start=1):
                log.info("  map %d/%d", index, len(chunks))
                raw = self._generate(
                    _MAP_PROMPT.format(
                        rules=_SHARED_RULES, index=index, total=len(chunks), chunk=chunk
                    )
                )
                cleaned = _strip_model_noise(raw)
                if cleaned:
                    blocks.append(cleaned)

            if not blocks:
                raise NotesError("Model returned nothing usable")

            if len(blocks) == 1:
                body = blocks[0]
            else:
                log.info("  reduce %d blocks", len(blocks))
                merged = self._generate(
                    _REDUCE_PROMPT.format(
                        rules=_SHARED_RULES, blocks="\n\n".join(blocks)
                    )
                )
                body = _strip_model_noise(merged)
                # A reduce that collapses everything is worse than no reduce;
                # keep the map output if the merge clearly lost content.
                if len(body) < 0.3 * len("\n\n".join(blocks)):
                    log.warning("Reduce pass lost too much content; keeping map output")
                    body = "\n\n".join(blocks)
        finally:
            self.unload()

        return NotesResult(body=body.strip(), chunks=len(chunks), model=self.model)


def main() -> int:
    from . import config as config_mod

    parser = argparse.ArgumentParser(description="Generate notes from a transcript")
    parser.add_argument("transcript", type=Path, help="a .json transcript")
    parser.add_argument("--stdout", action="store_true", help="print instead of saving")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    cfg = config_mod.load()
    if not args.transcript.is_file():
        log.error("No such transcript: %s", args.transcript)
        return 1

    transcript = Transcript.load(args.transcript)
    try:
        result = NoteGenerator(cfg).generate(transcript)
    except NotesError as exc:
        log.error("%s", exc)
        return 1

    if args.stdout:
        print(result.body)
    else:
        out = args.transcript.with_suffix(".notes.txt")
        out.write_text(result.body, encoding="utf-8")
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
