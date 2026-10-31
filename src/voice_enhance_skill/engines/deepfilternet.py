"""DeepFilterNet3 through the standalone `deep-filter` binary (ADR-001): Rust + tract, no Python dependencies,
no PyTorch. The DeepFilterNet3 ONNX model tarball is passed explicitly (`-m`), so the binary's built-in default
(DeepFilterNet2) is never used silently. The binary accepts 48 kHz WAV only; the service prepares a 48 kHz mono WAV
through ffmpeg-skill before the worker calls process() (ADR-003).

argv (fixed; only the two paths and the optional attenuation limit vary):
  deep-filter -D -m <DeepFilterNet3_onnx.tar.gz> -o <out_dir> [--atten-lim-db N] <input.wav>
-D compensates the STFT and lookahead delay, so the stem stays sample-aligned with the source (step 6 puts it back
under the picture). The binary writes <out_dir>/<input file name>."""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from ..errors import VoiceEnhanceError
from ..procs import TIMEOUT_EXIT, run_process_group
from ..provenance import sha256_file
from .base import INCOMPATIBLE, MISSING, OK, Engine, EngineRequest, EngineResult, parse_version

ENV_BINARY = "VOICE_ENHANCE_DEEP_FILTER"
ENV_MODEL = "VOICE_ENHANCE_DFN_MODEL"
MODEL_FILE = "DeepFilterNet3_onnx.tar.gz"
SUPPORTED = ((0, 5), (0, 6))        # >=0.5, <0.6
RELEASES = "https://github.com/Rikorose/DeepFilterNet/releases"
# The model tarball is not a release asset: it is a file of the repository's models/ directory (ADR-007).
MODEL_URL = "https://github.com/Rikorose/DeepFilterNet/raw/v0.5.6/models/DeepFilterNet3_onnx.tar.gz"
SAMPLE_RATE = 48000
ATTEN_LIM_RANGE = (0.0, 100.0)     # dB; the binary's default, 100, means no limit
ENGINE_TIMEOUT = 3600.0


def build_argv(binary: str, model: str, out_dir: str, input_path: str, atten_lim_db: Optional[float] = None) -> List[str]:
    argv = [binary, "-D", "-m", model, "-o", out_dir]
    if atten_lim_db is not None:
        argv += ["--atten-lim-db", f"{float(atten_lim_db):g}"]
    return argv + [input_path]


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
    install = (f"download the deep-filter binary for your platform from {RELEASES} (v0.5.x, for example "
               f"deep-filter-0.5.6-x86_64-unknown-linux-musl) and {MODEL_FILE} from {MODEL_URL}, then put deep-filter on "
               f"PATH (or set {ENV_BINARY}) and set {ENV_MODEL} to the model tarball")
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

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker. `req.input_path` is a 48 kHz mono WAV; the stem is written to `req.output_path`."""
        info = self.require()
        if req.op not in self.ops:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"engine {self.id!r} does not do {req.op}", {"engine": self.id, "op": req.op})
        if int(req.sample_rate) != SAMPLE_RATE:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"deep-filter needs {SAMPLE_RATE} Hz input, the request says {req.sample_rate}",
                                    {"engine": self.id})
        binary, model = find_binary(), os.environ[ENV_MODEL]
        atten = req.params.get("atten_lim_db")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(req.output_path)), "deep_filter_out")
        os.makedirs(out_dir, exist_ok=True)
        argv = build_argv(str(binary), model, out_dir, os.path.abspath(req.input_path), atten)
        timeout = float(req.params.get("timeout") or ENGINE_TIMEOUT)
        # Same process group as the worker: the service's timeout kills the worker's group, and the binary with it.
        code, out, err = run_process_group(argv, timeout, new_group=False)
        if code == TIMEOUT_EXIT:
            raise VoiceEnhanceError("ENGINE_TIMEOUT", f"deep-filter exceeded {timeout:g}s", {"engine": self.id, "timeout": timeout})
        if code != 0:
            raise VoiceEnhanceError("ENGINE_FAILED", f"deep-filter exited {code}", {"engine": self.id, "exit_code": code,
                                    "stderr_tail": (err or out).strip().splitlines()[-6:]})
        produced = os.path.join(out_dir, os.path.basename(req.input_path))
        if not os.path.isfile(produced) or os.path.getsize(produced) <= 44:
            raise VoiceEnhanceError("INVALID_RESULT", "deep-filter exited 0 but wrote no audio", {"engine": self.id, "expected": "<out_dir>/"
                                    + os.path.basename(req.input_path)})
        os.replace(produced, req.output_path)
        shown = build_argv("deep-filter", "<model>", "<out_dir>", "<input.wav>", atten)
        return EngineResult(output_path=req.output_path, engine_version=str(info["version"]), model=os.path.basename(model),
                            extra={"model_sha256": sha256_file(model), "argv": shown, "compensate_delay": True,
                                   "atten_lim_db": None if atten is None else float(atten)})
