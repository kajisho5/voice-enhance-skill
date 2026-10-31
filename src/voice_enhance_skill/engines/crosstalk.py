"""crosstalk: microphone bleed (cross-talk) reduction for time-aligned per-microphone tracks (ADR-005). Deterministic
signal processing on NumPy, written in this repository: no model, no weights, nothing to download, ever (ADR-001 found
no AI engine for this job). NumPy is looked up with importlib when inspecting and imported only inside process(),
which runs in the worker.

Model. Each microphone i hears its own talker S_i plus the other talkers, attenuated and delayed:
    X_i(t, f) = S_i(t, f) + sum_{j != i} H_ij(f) S_j(t, f)          (per STFT bin, one complex coupling per pair)

process(), step by step:
  read      N mono PCM WAVs at one sample rate, streamed in blocks as the channels of one signal
  stft      Hann window of 64 ms, hop of 16 ms (window = 4 hops) at the tracks' own rate
  estimate  pass 1. For every ordered pair (i, j): the least-squares coupling H_ij(f) = sum X_i conj(X_j) / sum |X_j|^2,
            summed only over the bins where track j is louder than track i by more than --dominance-db, so those
            bins hold talker j (and its bleed into i), not talker i
  demix     pass 2. Per bin, S = (I + H)^-1 X: each track minus the other talkers' bleed, without colouring its own
            talker. A bin where I + H is badly conditioned (condition number > MAX_CONDITION) falls back to the
            first-order X_i - sum H_ij X_j. Linear and time-invariant: nothing is gated, nothing is generated
  istft     weighted overlap-add (wavio.stft_process); sample-aligned with the input, no delay added
  write     one 32-bit float WAV per track, same rate, mono, same length as that track"""
from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict, List, Optional, Sequence

from .. import __version__
from ..errors import VoiceEnhanceError
from . import wavio
from .base import INCOMPATIBLE, MISSING, OK, Engine, EngineRequest, EngineResult, parse_version

NUMPY_MIN = (1, 21)
DEFAULTS = {"dominance_db": 3.0}
LIMITS = {"dominance_db": (0.0, 30.0)}
MIN_TRACKS, MAX_TRACKS = 2, 8
HOP_SECONDS = 0.016                  # STFT hop; the window is 4 hops (64 ms): room for the acoustic delay between mics
BLOCK_SECONDS = 30.0                 # frames per block in both passes, so memory does not grow with the file's length
MAX_CONDITION = 10.0                 # above this, a bin is demixed to first order instead of by the inverse
METHOD = "dominance-gated least-squares coupling, per-bin linear demixing"
OUTPUT_FORMAT = "pcm_f32le"


def _found(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _numpy_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version("numpy")
    except PackageNotFoundError:
        return None


def check_params(params: Dict[str, Any]) -> Dict[str, float]:
    value = params.get("dominance_db")
    value = DEFAULTS["dominance_db"] if value is None else value
    lo, hi = LIMITS["dominance_db"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"--dominance-db must be a number from {lo:g} to {hi:g}", {"dominance_db": value})
    return {"dominance_db": float(value)}


def coupling(np: Any, num: Any, den: Any) -> Any:
    """H (F, N, N) from the gated sums; 0 where a pair had no usable bins, so nothing is subtracted there."""
    floor = 1e-12 * max(float(np.max(den)), 1e-30)
    h = np.where(den > floor, num / np.maximum(den, floor), 0.0)
    for i in range(h.shape[1]):
        h[:, i, i] = 0.0
    return h


def demixing_matrices(np: Any, h: Any) -> Any:
    """W (F, N, N) with S = W X: (I + H)^-1 per bin, or I - H where I + H is badly conditioned."""
    n = h.shape[1]
    m = np.eye(n)[None] + h
    first_order = np.eye(n)[None] - h
    cond = np.linalg.cond(m)
    ok = np.isfinite(cond) & (cond <= MAX_CONDITION)
    w = first_order.copy()
    if np.any(ok):
        w[ok] = np.linalg.inv(m[ok])
    return w, int(np.count_nonzero(~ok))


def reduce_files(srcs: Sequence[str], dsts: Sequence[str], dominance_db: float, np: Any,
                 block_seconds: float = BLOCK_SECONDS) -> Dict[str, Any]:
    """Cross-talk reduction of the mono WAVs `srcs` (one sample rate), written to `dsts` as 32-bit float WAVs with
    each track's own length. Two streaming passes: estimate the couplings, then demix."""
    reader = wavio.TrackReader(srcs)
    try:
        rate, n_tracks, n = reader.rate, reader.channels, reader.frames
        size, hop = wavio.stft_sizes(rate, HOP_SECONDS)
        block = max(1, int(round(block_seconds * rate / hop)))
        window = wavio.stft_window(np, size)
        total = wavio.frame_count(n, size, hop)
        bins = size // 2 + 1
        rho = 10.0 ** (dominance_db / 10.0)
        num = np.zeros((bins, n_tracks, n_tracks), dtype=complex)
        den = np.zeros((bins, n_tracks, n_tracks))
        used = np.zeros((n_tracks, n_tracks), dtype=np.int64)
        for a in range(0, total, block):                                  # pass 1: estimate
            spec = wavio.stft_frames(np, reader, a, min(a + block, total), size, hop, window)      # (T, F, N)
            power = np.abs(spec) ** 2
            for i in range(n_tracks):
                for j in range(n_tracks):
                    if i == j:
                        continue
                    gate = power[:, :, j] > rho * power[:, :, i]
                    num[:, i, j] += np.sum(np.where(gate, spec[:, :, i] * np.conj(spec[:, :, j]), 0.0), axis=0)
                    den[:, i, j] += np.sum(np.where(gate, power[:, :, j], 0.0), axis=0)
                    used[i, j] += int(np.count_nonzero(gate))
        h = coupling(np, num, den)
        w, fallback = demixing_matrices(np, h)
        writer = wavio.TrackWriter(dsts, rate, reader.lengths)               # pass 2: demix
        try:
            wavio.stft_process(reader, writer, np, size, hop, block, 0,
                               lambda spec, skip: np.einsum("fij,tfj->tfi", w, spec[skip:]))
        finally:
            writer.close()
        if writer.frames != n:
            raise VoiceEnhanceError("INVALID_RESULT", f"wrote {writer.frames} frames, expected {n}", {"engine": "crosstalk"})
        # Energy-weighted broadband coupling of track j into track i, in dB (None: no bin where j dominated i).
        db: List[List[Optional[float]]] = []
        for i in range(n_tracks):
            row: List[Optional[float]] = []
            for j in range(n_tracks):
                total_den = float(np.sum(den[:, i, j]))
                if i == j or total_den <= 0.0 or used[i, j] == 0:
                    row.append(None)
                    continue
                gain = float(np.sum(np.abs(h[:, i, j]) ** 2 * den[:, i, j])) / total_den
                row.append(round(10.0 * math.log10(max(gain, 1e-12)), 2))
            db.append(row)
        return {"sample_rate": rate, "tracks": n_tracks, "frames": list(reader.lengths),
                "stft": {"window": "hann", "size": size, "hop": hop}, "block_seconds": float(block_seconds),
                "coupling_db": db, "estimation_bins": used.tolist(), "first_order_bins": fallback}
    finally:
        reader.close()


class CrosstalkEngine(Engine):
    id = "crosstalk"
    ops = ("bleed_reduce",)
    description = ("Microphone bleed reduction for time-aligned per-mic tracks: dominance-gated least-squares coupling and "
                   "per-bin linear demixing (NumPy, no model weights)")
    kind = "python"
    code_license = "MIT (this repository)"
    weights_license = "none (signal processing, no weights)"
    attribution = "voice-enhance-skill crosstalk engine (ADR-005), MIT; computes with NumPy (BSD-3-Clause)"
    version_range = "numpy >=1.21"
    install = "pip install 'voice-enhance-skill[bleed]'  (numpy)"
    located_by = ["the Python environment of this skill"]
    roadmap_step = 5

    def inspect(self) -> Dict[str, Any]:
        if not _found("numpy"):
            return {"status": MISSING, "version": None, "missing": "package", "detail": "numpy is not installed", "model": None}
        nv = _numpy_version()
        parsed = parse_version(nv)
        if parsed is None or parsed[:2] < NUMPY_MIN:
            return {"status": INCOMPATIBLE, "version": __version__, "detail": f"numpy {nv} is outside {self.version_range}",
                    "model": None, "numpy": nv}
        return {"status": OK, "version": __version__, "detail": f"crosstalk {__version__} on numpy {nv}", "model": None, "numpy": nv}

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker. `req.input_paths` are the prepared mono WAVs (one sample rate, time-aligned); one
        32-bit float WAV per track is written to the matching `req.output_paths`."""
        info = self.require()
        if req.op not in self.ops:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"engine {self.id!r} does not do {req.op}", {"engine": self.id, "op": req.op})
        srcs, dsts = list(req.input_paths), list(req.output_paths)
        if not MIN_TRACKS <= len(srcs) <= MAX_TRACKS or len(dsts) != len(srcs):
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{self.id} needs {MIN_TRACKS} to {MAX_TRACKS} tracks and one output per track",
                                    {"engine": self.id, "inputs": len(srcs), "outputs": len(dsts)})
        params = check_params(req.params)
        import numpy as np
        facts = reduce_files(srcs, dsts, params["dominance_db"], np)
        return EngineResult(output_path=dsts[0], engine_version=str(info["version"]), model=None, output_paths=dsts,
                            extra=dict(params, method=METHOD, numpy=info.get("numpy"), stft=facts["stft"],
                                       block_seconds=facts["block_seconds"], coupling_db=facts["coupling_db"],
                                       estimation_bins=facts["estimation_bins"], first_order_bins=facts["first_order_bins"],
                                       output_format=OUTPUT_FORMAT))
