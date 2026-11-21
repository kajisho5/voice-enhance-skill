"""DPDFNet, an opt-in second ISOLATE engine (ADR-015): the 48 kHz ONNX build `dpdfnet2_48khz_hr` (or
`dpdfnet8_48khz_hr`) run on CPU by the `dpdfnet` package (onnxruntime). DeepFilterNet3 stays the default; this engine
runs only with `--engine dpdfnet`. Looked up with importlib (never imported) when inspecting, so doctor stays fast;
numpy and dpdfnet are imported only inside process(), which runs in the worker.

The model file is passed explicitly (`onnx_path`, from $VOICE_ENHANCE_DPDFNET_MODEL), so the package's own model
cache and its download from Hugging Face are never used: nothing is downloaded at run time.

process(), step by step:
  read      the prepared 48 kHz mono WAV (ADR-003), streamed in blocks of STREAM_BLOCK frames; never the whole file
  enhance   dpdfnet.StreamEnhancer at the model's own rate: causal STFT (960-sample Vorbis window, hop 480), one ONNX
            call per hop, the recurrent state carried across blocks
  align     the stream comes out DELAY_SAMPLES (1920, 40 ms: one window of buffering and the model's lookahead) late;
            those first samples are dropped and DELAY_SAMPLES + one hop of zeros are fed at the end, so the stem is
            sample-aligned with the input and exactly as long
  limit     with --atten-lim-db L (< 100), out = a * input + (1 - a) * enhanced with a = 10^(-L/20): the noise is
            attenuated by at most L dB, the same mix the package applies offline
  write     32-bit float WAV, 48 kHz, mono, same length as the input (nothing is clipped or requantised)"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Tuple

from ..errors import VoiceEnhanceError
from ..provenance import sha256_file
from .base import INCOMPATIBLE, MISSING, OK, Engine, EngineRequest, EngineResult, parse_version
from .wavio import FloatWavWriter, WavReader

ENV_MODEL = "VOICE_ENHANCE_DPDFNET_MODEL"
SUPPORTED = ((0, 6, 0), (0, 7, 0))      # dpdfnet >=0.6.0, <0.7.0
MIN_PYTHON = (3, 11)                    # the dpdfnet package's own Requires-Python
# The 48 kHz ONNX builds of the Hugging Face repository Ceva-IP/DPDFNet (onnx/), revision dd6818d0 (2026-07-05):
# file name -> (model name in the package, size in bytes, SHA-256). The file name says which model it is; the
# SHA-256 goes into the result (engine.model_sha256), so a run names the exact weights it used.
MODELS: Dict[str, Tuple[str, int, str]] = {
    "dpdfnet2_48khz_hr.onnx": ("dpdfnet2_48khz_hr", 10493337,
                               "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"),
    "dpdfnet8_48khz_hr.onnx": ("dpdfnet8_48khz_hr", 14857107,
                               "7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631"),
}
RECOMMENDED_MODEL = "dpdfnet2_48khz_hr.onnx"
MODEL_URL = f"https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/{RECOMMENDED_MODEL}"
SAMPLE_RATE = 48000
HOP_SAMPLES = 480                       # 10 ms
DELAY_SAMPLES = 1920                    # 40 ms, measured on the real model (ADR-015)
STREAM_BLOCK = SAMPLE_RATE * 10         # frames read and fed per call
NO_LIMIT_DB = 100.0                     # as for deep-filter: 100 means no limit
OUTPUT_FORMAT = "pcm_f32le"


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


def mix_ratio(atten_lim_db: Optional[float]) -> float:
    """The share of the input mixed back in for an attenuation limit: 0 for no limit (None or >= 100 dB)."""
    if atten_lim_db is None or float(atten_lim_db) >= NO_LIMIT_DB:
        return 0.0
    return float(10.0 ** (-float(atten_lim_db) / 20.0))


def enhance_file(src: str, dst: str, model_name: str, onnx_path: str, atten_lim_db: Optional[float], np: Any,
                 enhancer_class: Any) -> Dict[str, Any]:
    """The prepared mono WAV at `src` through `enhancer_class` (dpdfnet.StreamEnhancer's interface: process(chunk,
    sample_rate) and flush()), written to `dst` as 32-bit float, sample-aligned and as long as the input."""
    reader = WavReader(src)
    try:
        if reader.rate != SAMPLE_RATE or reader.channels != 1:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"dpdfnet needs {SAMPLE_RATE} Hz mono input, the prepared file is "
                                    f"{reader.rate} Hz with {reader.channels} channels", {"engine": "dpdfnet"})
        n = reader.frames
        enhancer = enhancer_class(model=model_name, onnx_path=onnx_path)
        alpha = mix_ratio(atten_lim_db)
        writer = FloatWavWriter(dst, SAMPLE_RATE, 1)
        pending = np.zeros(0, dtype=np.float32)     # enhanced samples not yet written
        dropped = 0                                  # of the DELAY_SAMPLES at the start of the stream
        written = 0

        def emit(enhanced: Any) -> None:
            nonlocal pending, dropped, written
            enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
            if dropped < DELAY_SAMPLES:
                skip = min(DELAY_SAMPLES - dropped, len(enhanced))
                dropped += skip
                enhanced = enhanced[skip:]
            pending = np.concatenate([pending, enhanced])
            take = min(len(pending), n - written)
            if take <= 0:
                pending = pending[:0]
                return
            out = pending[:take].astype(np.float64)
            if alpha:
                out = alpha * reader.read(np, written, take)[:, 0] + (1.0 - alpha) * out
            writer.write(np, out.reshape(-1, 1))
            written += take
            pending = pending[take:]

        try:
            for start in range(0, n, STREAM_BLOCK):
                block = reader.read(np, start, min(STREAM_BLOCK, n - start))[:, 0].astype(np.float32)
                emit(enhancer.process(block, sample_rate=SAMPLE_RATE))
            emit(enhancer.process(np.zeros(DELAY_SAMPLES + HOP_SAMPLES, dtype=np.float32), sample_rate=SAMPLE_RATE))
            emit(enhancer.flush())
            if written < n:
                raise VoiceEnhanceError("INVALID_RESULT", f"dpdfnet returned {written} of {n} samples", {"engine": "dpdfnet"})
        finally:
            writer.close()
        return {"frames": n, "delay_samples": DELAY_SAMPLES, "mix_ratio": alpha}
    finally:
        reader.close()


class DpdfNetEngine(Engine):
    id = "dpdfnet"
    ops = ("isolate",)
    description = ("DPDFNet noise suppression (full band, 48 kHz ONNX build dpdfnet2_48khz_hr) via the dpdfnet package on "
                   "onnxruntime; opt-in, not the default")
    kind = "python"
    code_license = "Apache-2.0"
    weights_license = "Apache-2.0"
    attribution = "DPDFNet by CEVA (https://github.com/ceva-ip/DPDFNet, https://huggingface.co/Ceva-IP/DPDFNet), Apache-2.0"
    version_range = ">=0.6.0 <0.7.0"
    install = (f"pip install 'voice-enhance-skill[dpdfnet]'  (dpdfnet + onnxruntime; Python >= 3.11), then download "
               f"{RECOMMENDED_MODEL} (10.5 MB) from {MODEL_URL} and set {ENV_MODEL} to it")
    located_by = ["the Python environment of this skill", f"${ENV_MODEL}"]
    roadmap_step = 15

    def inspect(self) -> Dict[str, Any]:
        if sys.version_info[:2] < MIN_PYTHON:
            return {"status": MISSING, "version": None, "missing": "package", "model": None,
                    "detail": f"dpdfnet needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}, this is {sys.version_info[0]}.{sys.version_info[1]}"}
        for module in ("dpdfnet", "onnxruntime", "numpy"):
            if not _found(module):
                return {"status": MISSING, "version": None, "missing": "package", "detail": f"{module} is not installed", "model": None}
        vstr = _dist_version("dpdfnet")
        version = parse_version(vstr)
        if version is None:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": "dpdfnet version unknown", "model": None}
        version = (version + (0, 0, 0))[:3]
        if not SUPPORTED[0] <= version < SUPPORTED[1]:
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"dpdfnet {vstr} is outside {self.version_range}", "model": None}
        model = os.environ.get(ENV_MODEL)
        if not model or not os.path.isfile(model):
            return {"status": MISSING, "version": vstr, "missing": "model",
                    "detail": f"{ENV_MODEL} is not set" if not model else f"{ENV_MODEL} does not name a file", "model": None}
        name = os.path.basename(model)
        if name not in MODELS:
            return {"status": INCOMPATIBLE, "version": vstr, "model": name,
                    "detail": f"{ENV_MODEL} must name a 48 kHz DPDFNet ONNX build ({', '.join(sorted(MODELS))})"}
        return {"status": OK, "version": vstr, "detail": f"dpdfnet {vstr}, model {name}", "model": name,
                "model_bytes": os.path.getsize(model), "onnxruntime": _dist_version("onnxruntime"), "numpy": _dist_version("numpy")}

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker. `req.input_path` is a 48 kHz mono WAV; the stem (32-bit float, same length) is written to
        `req.output_path`."""
        info = self.require()
        if req.op not in self.ops:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"engine {self.id!r} does not do {req.op}", {"engine": self.id, "op": req.op})
        if int(req.sample_rate) != SAMPLE_RATE:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"dpdfnet needs {SAMPLE_RATE} Hz input, the request says {req.sample_rate}",
                                    {"engine": self.id})
        model = os.environ[ENV_MODEL]
        name = os.path.basename(model)
        atten = req.params.get("atten_lim_db")
        import numpy as np
        from dpdfnet import StreamEnhancer
        facts = enhance_file(req.input_path, req.output_path, MODELS[name][0], model, atten, np, StreamEnhancer)
        digest = sha256_file(model)
        return EngineResult(output_path=req.output_path, engine_version=str(info["version"]), model=name,
                            extra={"model_sha256": digest, "model_known": digest == MODELS[name][2],
                                   "atten_lim_db": None if atten is None else float(atten), "compensate_delay": True,
                                   "delay_samples": facts["delay_samples"],
                                   "output_format": OUTPUT_FORMAT, "onnxruntime": info.get("onnxruntime")})
