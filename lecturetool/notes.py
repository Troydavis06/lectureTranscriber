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
- Keep worked examples: the setup numbers and the final answer.
- Mark the single most testable fact in a block with "KEY:" at the start of that bullet.
- If the lecturer flags something as important, examinable, a common mistake, or homework, keep it.
- Omit greetings, admin chatter, tangents and repetition.
- Output ONLY the blocks. No preamble, no closing summary, no markdown headers, no bold.
"""

# The worked example below is deliberately from an unrelated subject. An earlier
# version used the same domain as the material under test, and the model simply
# echoed the example back as if it were a real topic block.
_MAP_PROMPT = """\
You are writing revision notes from part of a lecture transcript.

{rules}
Use exactly this shape. It is a FORMAT SAMPLE from an unrelated subject --
never copy its wording or topics into your answer:

[04:10] Enzyme saturation
  - Rate rises with substrate until active sites are full
  - V_max reached when all enzyme is bound
  - KEY: K_m is the substrate concentration at half V_max

Cover the whole excerpt, from its first timestamp through to its last. Do not
stop early.

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
    # Overlap has to stay a small fraction of the window. If it approaches the
    # window size, consecutive chunks are nearly identical and the map stage
    # emits the same topic over and over.
    overlap_words = min(overlap_words, int(target_words * 0.15))
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


_BLOCK_START = re.compile(r"^\s*\[(\d+):(\d{2})(?::(\d{2}))?\]")


def _parse_timestamp(line: str) -> float | None:
    """Leading [MM:SS] or [H:MM:SS] of a block header, in seconds."""
    match = _BLOCK_START.match(line)
    if not match:
        return None
    a, b, c = match.group(1), match.group(2), match.group(3)
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def _split_blocks(text: str) -> list[tuple[float, str]]:
    """Split note text into (timestamp, block) pairs on block headers."""
    blocks: list[tuple[float, str]] = []
    current: list[str] = []
    current_ts: float | None = None

    def flush() -> None:
        if current and current_ts is not None:
            body = "\n".join(current).rstrip()
            if body:
                blocks.append((current_ts, body))

    for line in text.splitlines():
        ts = _parse_timestamp(line)
        if ts is not None:
            flush()
            current = [line.strip()]
            current_ts = ts
        elif current:
            current.append(line.rstrip())
    flush()
    return blocks


def _block_fingerprint(block: str) -> str:
    """Identity of a block ignoring timestamp, case and punctuation.

    Used to collapse the near-duplicates that chunk overlap produces.
    """
    without_header = _BLOCK_START.sub("", block, count=1)
    words = re.findall(r"[a-z0-9]+", without_header.lower())
    return " ".join(words)


def dedupe_blocks(text: str) -> str:
    """Order blocks chronologically and drop repeats.

    A deterministic safety net: the model is asked to do this in the reduce
    pass, but when that pass is skipped or rejected this keeps the output
    readable instead of leaving the same topic repeated per chunk.
    """
    blocks = _split_blocks(text)
    if not blocks:
        return text.strip()

    blocks.sort(key=lambda pair: pair[0])
    seen: dict[str, int] = {}
    kept: list[tuple[float, str]] = []
    for ts, block in blocks:
        fingerprint = _block_fingerprint(block)
        if not fingerprint:
            continue
        if fingerprint in seen:
            continue
        # A later block whose content is contained in one already kept adds
        # nothing; overlap makes these common.
        if any(fingerprint in _block_fingerprint(k) for _, k in kept):
            continue
        seen[fingerprint] = ts
        kept.append((ts, block))
    return "\n\n".join(block for _, block in kept)


def _count_blocks(text: str) -> int:
    return len(_split_blocks(text))


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
        self.host = notes.get("host", "http://127.0.0.1:11434")
        self.client = ollama.Client(host=self.host)

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
                f"Could not reach Ollama at {self.host}: {exc}. Is `ollama serve` running?"
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
                body = dedupe_blocks(blocks[0])
            else:
                # Collapse overlap duplicates first: it shortens the reduce
                # prompt and puts the topics in order before the model sees them.
                combined = dedupe_blocks("\n\n".join(blocks))
                body = self._reduce(combined)
        finally:
            self.unload()

        return NotesResult(body=body.strip(), chunks=len(chunks), model=self.model)

    def _reduce(self, combined: str) -> str:
        """Merge map-stage blocks, batching so the prompt always fits num_ctx."""
        fallback = combined
        # Leave room for the rules, the instructions and the model's own output.
        budget_words = int(self.num_ctx * _WORDS_PER_TOKEN * 0.5)

        blocks = _split_blocks(combined)
        groups: list[list[str]] = [[]]
        group_words = 0
        for _, block in blocks:
            words = len(block.split())
            if groups[-1] and group_words + words > budget_words:
                groups.append([])
                group_words = 0
            groups[-1].append(block)
            group_words += words

        merged_parts: list[str] = []
        for index, group in enumerate(groups, start=1):
            if not group:
                continue
            if len(groups) > 1:
                log.info("  reduce batch %d/%d", index, len(groups))
            else:
                log.info("  reduce %d blocks", len(blocks))
            raw = self._generate(
                _REDUCE_PROMPT.format(rules=_SHARED_RULES, blocks="\n\n".join(group))
            )
            cleaned = _strip_model_noise(raw)
            # Reject only a degenerate merge. A good reduce over overlapping
            # chunks is *expected* to shrink the text a lot, so length is the
            # wrong signal -- an earlier version used it and kept throwing away
            # correct merges in favour of repetitive map output.
            if _count_blocks(cleaned) < 2:
                log.warning(
                    "Reduce batch %d returned %d block(s); keeping its map output",
                    index,
                    _count_blocks(cleaned),
                )
                merged_parts.append("\n\n".join(group))
            else:
                merged_parts.append(cleaned)

        result = dedupe_blocks("\n\n".join(merged_parts))
        if _count_blocks(result) < 2:
            log.warning("Reduce produced nothing usable; keeping map output")
            return fallback
        return result


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
