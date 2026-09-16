"""Make pip-installed NVIDIA CUDA libraries loadable on Windows.

CTranslate2 (the engine behind faster-whisper) links against cuDNN 9 and
cuBLAS 12. When those come from pip (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`)
the DLLs land in `site-packages/nvidia/*/bin`, which the Windows loader does
not search. Without registering those directories you get a bare
"Could not locate cudnn_ops64_9.dll" / "cublas64_12.dll" failure at the moment
the model loads.

Import this module BEFORE `faster_whisper`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_applied = False


def _nvidia_dll_dirs() -> list[Path]:
    """Every existing NVIDIA DLL directory inside the active environment."""
    dirs: list[Path] = []
    for site_dir in {Path(p) for p in sys.path if p}:
        nvidia_root = site_dir / "nvidia"
        if not nvidia_root.is_dir():
            continue
        # Layout is nvidia/<component>/bin on Windows, nvidia/<component>/lib
        # on Linux. Check both so the module is not Windows-only.
        for component in sorted(nvidia_root.iterdir()):
            if not component.is_dir():
                continue
            for leaf in ("bin", "lib"):
                candidate = component / leaf
                if candidate.is_dir() and candidate not in dirs:
                    dirs.append(candidate)
    return dirs


def apply() -> list[Path]:
    """Register NVIDIA DLL directories with the loader. Idempotent."""
    global _applied
    found = _nvidia_dll_dirs()
    if _applied:
        return found

    for directory in found:
        if hasattr(os, "add_dll_directory"):  # Windows only
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
        # ctranslate2 also consults PATH in some builds; belt and braces.
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")

    _applied = True
    return found


if __name__ == "__main__":
    dirs = apply()
    if not dirs:
        print("No pip-installed NVIDIA DLL directories found.")
    else:
        print("Registered NVIDIA DLL directories:")
        for d in dirs:
            print(f"  {d}")
