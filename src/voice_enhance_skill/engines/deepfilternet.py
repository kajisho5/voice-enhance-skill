"""DeepFilterNet3 through the standalone `deep-filter` binary (ADR-001): Rust + tract, no Python dependencies,
no PyTorch. The DeepFilterNet3 ONNX model tarball is passed explicitly (`-m`), so the binary's built-in default
(DeepFilterNet2) is never used silently. The binary accepts 48 kHz WAV only."""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, Optional

from ..procs import run_process_group
from .base import INCOMPATIBLE, MISSING, OK, Engine, parse_version

ENV_BINARY = "VOICE_ENHANCE_DEEP_FILTER"
ENV_MODEL = "VOICE_ENHANCE_DFN_MODEL"
MODEL_FILE = "DeepFilterNet3_onnx.tar.gz"
SUPPORTED = ((0, 5), (0, 6))        # >=0.5, <0.6
RELEASES = "https://github.com/Rikorose/DeepFilterNet/releases"


def find_binary() -> Optional[str]:
    explicit = os.environ.get(ENV_BINARY)
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(explicit, os.X_OK) else None
    return shutil.which("deep-filter")


class DeepFilterNetEngine(Engine):
    id = "deepfilternet"
    ops = ("isolate",)
    description = "DeepFilterNet3 noise suppression (full band, 48 kHz) via the standalone deep-filter binary"
    kind = "binary"
    code_license = "MIT OR Apache-2.0"
    weights_license = "MIT OR Apache-2.0 (checkpoints are files of the same repository)"
    attribution = "DeepFilterNet by Hendrik Schröter (https://github.com/Rikorose/DeepFilterNet), MIT OR Apache-2.0"
    version_range = ">=0.5.0 <0.6.0"
    install = (f"download the deep-filter binary for your platform and {MODEL_FILE} from {RELEASES} (v0.5.x), then put "
               f"deep-filter on PATH (or set {ENV_BINARY}) and set {ENV_MODEL} to the model tarball")
    located_by = [f"${ENV_BINARY}", "PATH (deep-filter)", f"${ENV_MODEL}"]
    roadmap_step = 3

    def inspect(self) -> Dict[str, Any]:
        binary = find_binary()
        if binary is None:
            where = f"{ENV_BINARY}={os.environ[ENV_BINARY]} is not an executable file" if os.environ.get(ENV_BINARY) \
                else "deep-filter is not on PATH"
            return {"status": MISSING, "version": None, "missing": "binary", "detail": where, "model": None}
        code, out, err = run_process_group([binary, "--version"], 15.0)
        version_text = (out or err).strip().splitlines()[0] if (out or err).strip() else ""
        version = parse_version(version_text)
        vstr = ".".join(map(str, version)) if version else None
        if code != 0 or version is None:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"`deep-filter --version` failed or printed no version "
                    f"(exit {code}): {version_text or err.strip()[-200:]}", "model": None}
        if not SUPPORTED[0] <= version[:2] < SUPPORTED[1]:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"deep-filter {vstr} is outside {self.version_range}", "model": None}
        model = os.environ.get(ENV_MODEL)
        if not model or not os.path.isfile(model):
            return {"status": MISSING, "version": vstr, "missing": "model",
                    "detail": f"{ENV_MODEL} is not set" if not model else f"{ENV_MODEL} does not name a file", "model": None}
        if not model.endswith(".tar.gz"):
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"{ENV_MODEL} must name the ONNX model tarball ({MODEL_FILE})",
                    "model": os.path.basename(model)}
        return {"status": OK, "version": vstr, "detail": f"deep-filter {vstr}, model {os.path.basename(model)}",
                "model": os.path.basename(model), "model_bytes": os.path.getsize(model),
                "platform": sys.platform}
