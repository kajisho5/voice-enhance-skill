"""nara_wpe: Weighted Prediction Error dereverberation (ADR-001). Signal processing on NumPy, no weights, so
nothing to download, ever. Looked up with importlib (never imported) when inspecting, so doctor stays fast."""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional

from .base import INCOMPATIBLE, MISSING, OK, Engine, parse_version

SUPPORTED = ((0, 0, 11), (0, 1, 0))   # >=0.0.11, <0.1


def _dist_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _found(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class NaraWpeEngine(Engine):
    id = "nara_wpe"
    ops = ("dereverb",)
    description = "WPE dereverberation (nara_wpe, offline iterative), single- or multi-channel; no model weights"
    kind = "python"
    code_license = "MIT"
    weights_license = "none (signal processing, no weights)"
    attribution = "nara_wpe by the Paderborn University Communications Engineering group (https://github.com/fgnt/nara_wpe), MIT"
    version_range = ">=0.0.11 <0.1.0"
    install = "pip install 'voice-enhance-skill[wpe]'  (nara_wpe + numpy)"
    located_by = ["the Python environment of this skill"]
    roadmap_step = 4

    def inspect(self) -> Dict[str, Any]:
        if not _found("nara_wpe"):
            return {"status": MISSING, "version": None, "missing": "package", "detail": "nara_wpe is not installed", "model": None}
        if not _found("numpy"):
            return {"status": MISSING, "version": None, "missing": "package", "detail": "numpy is not installed", "model": None}
        vstr = _dist_version("nara_wpe") or _dist_version("nara-wpe")
        version = parse_version(vstr)
        if version is None:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": "nara_wpe version unknown", "model": None}
        version = (version + (0, 0, 0))[:3]
        if not SUPPORTED[0] <= version < SUPPORTED[1]:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"nara_wpe {vstr} is outside {self.version_range}", "model": None}
        return {"status": OK, "version": vstr, "detail": f"nara_wpe {vstr}", "model": None}
