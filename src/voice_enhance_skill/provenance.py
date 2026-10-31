"""Provenance block every op result carries (ADR-001): what ran, on which input, with which licences."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

from . import SKILL_ID, __version__


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_provenance(*, op: str, engine: Dict[str, Any], input_path: str, output_path: Optional[str],
                     parameters: Dict[str, Any], ffmpeg_skill: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`engine` is an engine report (Engine.report()); only publishable fields are copied (no local paths)."""
    return {
        "skill": {"id": SKILL_ID, "version": __version__},
        "op": op,
        "engine": {k: engine.get(k) for k in ("id", "version", "model", "code_license", "weights_license", "attribution",
                                              "execution_mode", "deterministic")},
        "input": {"sha256": sha256_file(input_path)},
        "output": {"sha256": sha256_file(output_path)} if output_path else None,
        "parameters": dict(parameters),
        "ffmpeg_skill": None if ffmpeg_skill is None else {k: ffmpeg_skill.get(k) for k in ("version", "contract_version")},
        "network": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
