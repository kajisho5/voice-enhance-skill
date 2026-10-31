"""Shared test helpers: paths, the CLI runner, a fake `deep-filter` executable and a WAV writer. Nothing here is
part of the package; the fake binary is never registered anywhere but a test's environment."""
from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
import wave
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_enhance_skill import cli  # noqa: E402


def _dir_from_env(key: str, sibling: str, marker: str) -> Optional[Path]:
    for c in ([Path(os.environ[key])] if os.environ.get(key) else []) + [ROOT.parent / sibling]:
        if (c / marker).is_file():
            return c
    return None


OS_DIR = _dir_from_env("AI_VIDEO_PRODUCTION_OS_DIR", "AI-video-production-OS", "registry/conformance.py")
FFMPEG_SKILL_DIR = _dir_from_env("VOICE_ENHANCE_FFMPEG_SKILL_DIR", "ffmpeg-skill", "scripts/_contract.py")


def run_cli(*argv: str) -> Tuple[int, Dict[str, Any]]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, json.loads(buf.getvalue())


@contextmanager
def env(**values: Optional[str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fake_deep_filter(directory: str, version: str = "0.5.6") -> str:
    """A POSIX executable that answers `--version` like the real binary ("deep-filter 0.5.6")."""
    path = os.path.join(directory, "deep-filter")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'deep-filter {version}'; exit 0; fi\nexit 9\n")
    os.chmod(path, 0o755)
    return path


def write_tone_wav(path: str, seconds: float = 1.0, rate: int = 48000, freq: float = 440.0) -> str:
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)))
    return path
