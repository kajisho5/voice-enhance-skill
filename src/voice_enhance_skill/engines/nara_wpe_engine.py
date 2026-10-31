"""nara_wpe: Weighted Prediction Error dereverberation (ADR-001, ADR-004). Signal processing on NumPy, no weights, so
nothing to download, ever. Looked up with importlib (never imported) when inspecting, so doctor stays fast; numpy and
nara_wpe are imported only inside process(), which runs in the worker.

process(), step by step (ADR-004):
  read      the prepared PCM WAV (any rate, any channel count), streamed in blocks; never the whole file at once
  stft      Hann window of 32 ms, hop of 8 ms (window = 4 hops) at the file's own rate, padded so every sample is
            covered by all four overlapping frames
  wpe       nara_wpe.wpe.wpe_v8 per block of BLOCK_SECONDS, multi-channel (all channels of the stream predict each
            other), with the taps + delay - 1 frames before the block as context. Channels that are linear copies of
            each other (dual mono, a panned mono source) are first reduced to their independent components, because
            wpe_v8 has no regularisation and diverges on a singular correlation matrix (ADR-007)
  istft     weighted overlap-add; with no filtering the input comes back sample for sample (no delay)
  write     32-bit float WAV, same rate, same channels, same length as the input (nothing is clipped or requantised)
The WAV reader and writer, the STFT and the overlap-add live in wavio, shared with the crosstalk engine (ADR-005)."""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional, Tuple

from ..errors import VoiceEnhanceError
from . import wavio
from .base import INCOMPATIBLE, MISSING, OK, Engine, EngineRequest, EngineResult, parse_version
from .wavio import FloatWavWriter, WavReader

SUPPORTED = ((0, 0, 11), (0, 1, 0))   # >=0.0.11, <0.1
DEFAULTS = {"taps": 10, "delay": 3, "iterations": 3}   # nara_wpe's own defaults
LIMITS = {"taps": (1, 50), "delay": (1, 20), "iterations": (1, 20)}
HOP_SECONDS = 0.008                  # STFT hop; the window is 4 hops (32 ms), as in nara_wpe's 16 kHz examples (512 / 128)
BLOCK_SECONDS = 30.0                 # WPE statistics per block, so memory does not grow with the file's length
WPE_FUNCTION = "nara_wpe.wpe.wpe_v8"
# A channel direction whose energy is this far below the strongest one is treated as absent (-60 dB): WPE runs on the
# independent components only, and what lies outside them passes through unchanged (ADR-007).
RANK_TOLERANCE = 1e-6
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


def stft_sizes(rate: int) -> Tuple[int, int]:
    """(window, hop) in samples at `rate`: a hop of 8 ms and a window of four hops."""
    return wavio.stft_sizes(rate, HOP_SECONDS)


def independent_basis(np: Any, y: Any, tolerance: float = RANK_TOLERANCE) -> Any:
    """Orthonormal (D, r) basis of the channel directions that carry energy in the (F, D, T) STFT `y`, or None when all
    D are independent (nothing to reduce). r is 0 for a silent block. From the eigenvectors of the broadband channel
    covariance sum_f,t y y^H whose eigenvalue is above `tolerance` times the largest."""
    d = y.shape[1]
    if d < 2:
        return None
    cov = np.einsum("fdt,fet->de", y, y.conj())
    values, vectors = np.linalg.eigh((cov + cov.conj().T) / 2)
    top = float(values[-1].real)
    if top <= 0.0:
        return vectors[:, :0]
    keep = values.real > tolerance * top
    return None if bool(keep.all()) else vectors[:, keep]


def dereverb_file(src: str, dst: str, taps: int, delay: int, iterations: int, np: Any, wpe: Any,
                  block_seconds: float = BLOCK_SECONDS) -> Dict[str, Any]:
    """WPE on the WAV at `src`, written to `dst` as 32-bit float with the same rate, channels and length. `np` is
    numpy and `wpe` a function with nara_wpe's wpe_v8 signature (Y: (F, D, T) complex -> same shape). The STFT and
    the overlap-add are wavio.stft_process; each block gets the taps + delay - 1 frames before it as context."""
    reader = WavReader(src)
    try:
        rate, channels, n = reader.rate, reader.channels, reader.frames
        size, hop = stft_sizes(rate)
        block = max(1, int(round(block_seconds * rate / hop)))

        ranks = []

        def run_wpe(y: Any) -> Any:
            z = wpe(y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
            if getattr(z, "shape", None) != y.shape:
                raise VoiceEnhanceError("INVALID_RESULT", f"{WPE_FUNCTION} returned shape {getattr(z, 'shape', None)}, "
                                        f"expected {y.shape}", {"engine": "nara_wpe"})
            return np.asarray(z)

        def process(spec: Any, skip: int) -> Any:
            y = np.ascontiguousarray(spec.transpose(1, 2, 0))                             # (F, D, T)
            basis = independent_basis(np, y)
            if basis is None:
                ranks.append(y.shape[1])
                z = run_wpe(y)
            else:
                # x = y + U (wpe(U^H y) - U^H y): WPE on the r independent components, the rest of y untouched
                ranks.append(basis.shape[1])
                z = y
                if basis.shape[1]:
                    c = np.ascontiguousarray(np.einsum("dr,fdt->frt", basis.conj(), y))
                    z = y + np.einsum("dr,frt->fdt", basis, run_wpe(c) - c)
            return z.transpose(2, 0, 1)[skip:]

        writer = FloatWavWriter(dst, rate, channels)
        try:
            wavio.stft_process(reader, writer, np, size, hop, block, taps + delay - 1, process)
        finally:
            writer.close()
        if writer.frames != n:
            raise VoiceEnhanceError("INVALID_RESULT", f"wrote {writer.frames} frames, expected {n}", {"engine": "nara_wpe"})
        return {"sample_rate": rate, "channels": channels, "frames": n, "stft": {"window": "hann", "size": size, "hop": hop},
                "block_seconds": float(block_seconds), "independent_channels": min(ranks) if ranks else channels}
    finally:
        reader.close()


def check_params(params: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for key, default in DEFAULTS.items():
        value = params.get(key)
        value = default if value is None else value
        lo, hi = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"--{key} must be an integer from {lo} to {hi}", {key: value})
        out[key] = value
    return out


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

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker. `req.input_path` is a PCM WAV at any rate with any number of channels; the
        dereverberated WAV (32-bit float, same rate, channels and length) is written to `req.output_path`."""
        info = self.require()
        if req.op not in self.ops:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"engine {self.id!r} does not do {req.op}", {"engine": self.id, "op": req.op})
        params = check_params(req.params)
        import numpy as np
        from nara_wpe.wpe import wpe_v8
        facts = dereverb_file(req.input_path, req.output_path, params["taps"], params["delay"], params["iterations"], np, wpe_v8)
        return EngineResult(output_path=req.output_path, engine_version=str(info["version"]), model=None,
                            extra=dict(params, method=WPE_FUNCTION, statistics_mode="full", stft=facts["stft"],
                                       block_seconds=facts["block_seconds"], output_format=OUTPUT_FORMAT,
                                       independent_channels=facts["independent_channels"]))
