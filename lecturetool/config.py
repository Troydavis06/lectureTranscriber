"""Configuration loading.

Defaults live here so the tool runs with no config.toml at all; config.toml
only needs to contain the keys you want to change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "paths": {
        "data_dir": "./data",
    },
    "chrome": {
        "debug_port": 9222,
        "poll_sec": 2.0,
        "start_debounce_polls": 2,
        "pause_grace_sec": 120,
        "end_slack_sec": 5.0,
        "min_video_duration_sec": 120,
        "url_allowlist": [],
        "url_denylist": [],
    },
    "audio": {
        "min_lecture_sec": 180,
        "keep_recordings_days": 7,
    },
    "whisper": {
        "model": "large-v3-turbo",
        "compute_type": "float16",
        "device": "cuda",
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
    },
}


class Config:
    """Dotted-section access to merged config values."""

    def __init__(self, data: dict[str, dict[str, Any]], root: Path) -> None:
        self._data = data
        self.root = root

    def section(self, name: str) -> dict[str, Any]:
        return self._data.get(name, {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    # --- derived paths -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        raw = Path(self.get("paths", "data_dir", "./data"))
        return raw if raw.is_absolute() else (self.root / raw).resolve()

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.recordings_dir, self.transcripts_dir):
            d.mkdir(parents=True, exist_ok=True)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key].update(val)
        else:
            out[key] = val
    return out


def load(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    data = DEFAULTS
    if path.is_file():
        with path.open("rb") as fh:
            data = _merge(DEFAULTS, tomllib.load(fh))
    return Config(data, PROJECT_ROOT)
