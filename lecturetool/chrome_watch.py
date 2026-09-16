"""Watch Chrome over the DevTools Protocol for a lecture video playing.

Chrome is polled through its debugging port: the HTTP endpoint lists page
targets, and a websocket per target runs a small `Runtime.evaluate` snippet that
reports the state of the most substantial <video> on the page. From that we
derive four events -- START, PAUSE, RESUME, END -- which drive the recorder.

Chrome 136+ refuses --remote-debugging-port on the default user profile, so
launch_chrome.cmd points at a dedicated profile directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator
from urllib.parse import urlparse

import requests
import websocket

log = logging.getLogger("chrome")

# Reports the longest video on the page -- on a lecture page the real content
# is the longest media element, which sidesteps autoplay promos and pre-roll.
_VIDEO_STATE_JS = """
(() => {
  const vs = [...document.querySelectorAll('video')]
    .filter(v => Number.isFinite(v.duration) && v.duration > 0)
    .sort((a, b) => b.duration - a.duration);
  const v = vs[0];
  if (!v) return null;
  return JSON.stringify({
    playing: !v.paused && !v.ended && v.readyState >= 2,
    t: v.currentTime,
    dur: v.duration,
    ended: v.ended,
    muted: v.muted || v.volume === 0,
  });
})()
"""


class Event(Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    END = "end"


@dataclass
class VideoState:
    target_id: str
    title: str
    url: str
    playing: bool
    position: float
    duration: float
    ended: bool
    muted: bool


class ChromeNotRunning(RuntimeError):
    pass


class CDPClient:
    """Minimal CDP client: list page targets and evaluate JS in them."""

    def __init__(self, port: int = 9222, timeout: float = 3.0) -> None:
        self.port = port
        self.timeout = timeout
        self._sockets: dict[str, websocket.WebSocket] = {}
        self._msg_id = 0

    def close(self) -> None:
        for sock in self._sockets.values():
            try:
                sock.close()
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass
        self._sockets.clear()

    def targets(self) -> list[dict]:
        """Page targets, excluding devtools/extension internals."""
        try:
            resp = requests.get(
                f"http://127.0.0.1:{self.port}/json/list", timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ChromeNotRunning(
                f"Chrome is not reachable on port {self.port}. "
                "Start it with launch_chrome.cmd."
            ) from exc

        out = []
        for target in resp.json():
            if target.get("type") != "page":
                continue
            url = target.get("url", "")
            if url.startswith(("devtools://", "chrome://", "chrome-extension://")):
                continue
            out.append(target)
        return out

    def _socket(self, target: dict) -> websocket.WebSocket | None:
        target_id = target["id"]
        sock = self._sockets.get(target_id)
        if sock is not None and sock.connected:
            return sock
        ws_url = target.get("webSocketDebuggerUrl")
        if not ws_url:
            return None
        try:
            sock = websocket.create_connection(
                ws_url, timeout=self.timeout, suppress_origin=True
            )
        except Exception as exc:  # noqa: BLE001 - websocket raises broadly
            log.debug("Cannot attach to %s: %s", target.get("url", "?")[:60], exc)
            return None
        self._sockets[target_id] = sock
        return sock

    def drop_socket(self, target_id: str) -> None:
        sock = self._sockets.pop(target_id, None)
        if sock is not None:
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass

    def evaluate(self, target: dict, expression: str) -> str | None:
        """Run JS in a target and return its string result (None on failure)."""
        sock = self._socket(target)
        if sock is None:
            return None
        self._msg_id += 1
        msg_id = self._msg_id
        payload = {
            "id": msg_id,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True, "timeout": 2000},
        }
        try:
            sock.send(json.dumps(payload))
            # CDP interleaves unsolicited events with replies; skip until ours.
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                message = json.loads(sock.recv())
                if message.get("id") != msg_id:
                    continue
                result = message.get("result", {}).get("result", {})
                value = result.get("value")
                return value if isinstance(value, str) else None
        except Exception as exc:  # noqa: BLE001 - tab closed / navigated / timed out
            log.debug("Evaluate failed on %s: %s", target.get("url", "?")[:60], exc)
            self.drop_socket(target["id"])
        return None

    def video_states(self) -> Iterator[VideoState]:
        for target in self.targets():
            raw = self.evaluate(target, _VIDEO_STATE_JS)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            yield VideoState(
                target_id=target["id"],
                title=target.get("title", ""),
                url=target.get("url", ""),
                playing=bool(data.get("playing")),
                position=float(data.get("t") or 0.0),
                duration=float(data.get("dur") or 0.0),
                ended=bool(data.get("ended")),
                muted=bool(data.get("muted")),
            )


def _host_matches(url: str, patterns: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(p.lower() in host for p in patterns if p)


class LectureWatcher:
    """State machine turning polled video state into lecture events.

    Only one lecture is tracked at a time; whichever qualifying video starts
    playing first owns the recorder until it finishes.
    """

    def __init__(self, cfg, on_event: Callable[[Event, VideoState], None]) -> None:
        chrome = cfg.section("chrome")
        self.client = CDPClient(port=int(chrome.get("debug_port", 9222)))
        self.poll_sec = float(chrome.get("poll_sec", 2.0))
        self.start_debounce = int(chrome.get("start_debounce_polls", 2))
        self.pause_grace_sec = float(chrome.get("pause_grace_sec", 120))
        self.end_slack_sec = float(chrome.get("end_slack_sec", 5.0))
        self.min_duration = float(chrome.get("min_video_duration_sec", 120))
        self.allowlist = list(chrome.get("url_allowlist", []) or [])
        self.denylist = list(chrome.get("url_denylist", []) or [])
        self.on_event = on_event

        self._active_target: str | None = None
        self._active_state: VideoState | None = None
        self._playing_streak = 0
        self._paused_since: float | None = None

    # --- eligibility ---------------------------------------------------
    def _eligible(self, state: VideoState) -> bool:
        if state.duration < self.min_duration:
            return False
        if self.denylist and _host_matches(state.url, self.denylist):
            return False
        if self.allowlist and not _host_matches(state.url, self.allowlist):
            return False
        return True

    # --- main loop -----------------------------------------------------
    def poll_once(self) -> None:
        try:
            states = {s.target_id: s for s in self.client.video_states()}
        except ChromeNotRunning:
            # Chrome closing mid-lecture is a legitimate end-of-lecture signal.
            if self._active_target is not None:
                self._finish("Chrome closed")
            raise

        if self._active_target is None:
            self._look_for_start(states)
        else:
            self._track_active(states)

    def _look_for_start(self, states: dict[str, VideoState]) -> None:
        candidate = next(
            (s for s in states.values() if s.playing and self._eligible(s)), None
        )
        if candidate is None:
            self._playing_streak = 0
            return

        self._playing_streak += 1
        if self._playing_streak < self.start_debounce:
            log.debug(
                "Candidate %r playing (%d/%d)",
                candidate.title[:50],
                self._playing_streak,
                self.start_debounce,
            )
            return

        self._active_target = candidate.target_id
        self._active_state = candidate
        self._paused_since = None
        self._playing_streak = 0
        log.info("Lecture started: %r (%.0f min)", candidate.title, candidate.duration / 60)
        self.on_event(Event.START, candidate)

    def _track_active(self, states: dict[str, VideoState]) -> None:
        state = states.get(self._active_target)
        if state is None:
            self._finish("tab closed or navigated away")
            return

        self._active_state = state

        if state.ended or (state.duration > 0 and
                           (state.duration - state.position) <= self.end_slack_sec
                           and not state.playing):
            self._finish("video finished")
            return

        if state.playing:
            if self._paused_since is not None:
                self._paused_since = None
                self.on_event(Event.RESUME, state)
            return

        # Paused: tolerate a break, but give up after the grace period.
        if self._paused_since is None:
            self._paused_since = time.monotonic()
            self.on_event(Event.PAUSE, state)
        elif time.monotonic() - self._paused_since > self.pause_grace_sec:
            self._finish(f"paused for over {self.pause_grace_sec:.0f}s")

    def _finish(self, reason: str) -> None:
        state = self._active_state
        target_id = self._active_target
        self._active_target = None
        self._active_state = None
        self._paused_since = None
        self._playing_streak = 0
        if target_id is not None:
            self.client.drop_socket(target_id)
        if state is not None:
            log.info("Lecture ended (%s): %r", reason, state.title)
            self.on_event(Event.END, state)

    def run(self, stop_check: Callable[[], bool] | None = None) -> None:
        """Poll until stop_check returns True, surviving Chrome not running."""
        warned = False
        while not (stop_check and stop_check()):
            try:
                self.poll_once()
                warned = False
            except ChromeNotRunning as exc:
                if not warned:
                    log.warning("%s", exc)
                    warned = True
            except Exception:  # noqa: BLE001 - a poll must never kill the daemon
                log.exception("Unexpected error while polling Chrome")
            time.sleep(self.poll_sec)

    def close(self) -> None:
        self.client.close()


def _watch(cfg, verbose: bool) -> int:
    """Debug mode: print detected state and events live."""

    def on_event(event: Event, state: VideoState) -> None:
        print(f"  >> {event.name:<7} {state.title[:60]!r}")

    watcher = LectureWatcher(cfg, on_event)
    print(f"Polling Chrome on port {watcher.client.port} every {watcher.poll_sec}s.")
    print("Open a lecture and press play. Ctrl+C to stop.\n")
    try:
        while True:
            try:
                states = list(watcher.client.video_states())
                if not states:
                    print("  (no page with a video)                    ", end="\r")
                for state in states:
                    flags = "playing" if state.playing else "paused "
                    if state.ended:
                        flags = "ended  "
                    eligible = "ok " if watcher._eligible(state) else "skip"
                    print(
                        f"  [{eligible}] {flags} {state.position:7.1f}/"
                        f"{state.duration:7.1f}s  {state.title[:45]}"
                    )
                watcher.poll_once()
            except ChromeNotRunning as exc:
                print(f"  {exc}", end="\r")
            time.sleep(watcher.poll_sec)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        watcher.close()
    return 0


def main() -> int:
    from . import config as config_mod
    from .util import setup_logging

    parser = argparse.ArgumentParser(description="Chrome lecture video watcher")
    parser.add_argument("--watch", action="store_true", help="print live video state")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    cfg = config_mod.load()

    if args.watch:
        return _watch(cfg, args.verbose)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
