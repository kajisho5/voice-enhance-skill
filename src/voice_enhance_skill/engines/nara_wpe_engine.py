"""nara_wpe: Weighted Prediction Error dereverberation (ADR-001, ADR-004). Signal processing on NumPy, no weights, so
nothing to download, ever. Looked up with importlib (never imported) when inspecting, so doctor stays fast; numpy and
nara_wpe are imported only inside process(), which runs in the worker.

process(), step by step (ADR-004):
  read      the prepared PCM WAV (any rate, any channel count), streamed in blocks; never the whole file at once
  stft      Hann window of 32 ms, hop of 8 ms (window = 4 hops) at the file's own rate, padded so every sample is
            covered by all four overlapping frames
  wpe       nara_wpe.wpe.wpe_v8 per block of BLOCK_SECONDS, multi-channel (all channels of the stream predict each
            other), with the taps + delay - 1 frames before the block as context
  istft     weighted overlap-add; with no filtering the input comes back sample for sample (no delay)
  write     32-bit float WAV, same rate, same channels, same length as the input (nothing is clipped or requantised)"""
from __future__ import annotations

import importlib.util
import math
import os
import struct
from typing import Any, Dict, Optional, Tuple

from ..errors import VoiceEnhanceError
from .base import INCOMPATIBLE, MISSING, OK, Engine, EngineRequest, EngineResult, parse_version

SUPPORTED = ((0, 0, 11), (0, 1, 0))   # >=0.0.11, <0.1
DEFAULTS = {"taps": 10, "delay": 3, "iterations": 3}   # nara_wpe's own defaults
LIMITS = {"taps": (1, 50), "delay": (1, 20), "iterations": (1, 20)}
HOP_SECONDS = 0.008                  # STFT hop; the window is 4 hops (32 ms), as in nara_wpe's 16 kHz examples (512 / 128)
BLOCK_SECONDS = 30.0                 # WPE statistics per block, so memory does not grow with the file's length
WPE_FUNCTION = "nara_wpe.wpe.wpe_v8"
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
    hop = max(1, int(round(rate * HOP_SECONDS)))
    return 4 * hop, hop


class WavReader:
    """A RIFF/WAVE file read in frames on demand: PCM 16/24/32-bit integer or IEEE float 32/64-bit, plain or
    WAVE_FORMAT_EXTENSIBLE (the stdlib wave module reads neither 24-bit extensible nor float on Python 3.9)."""

    def __init__(self, path: str) -> None:
        self.fh = open(path, "rb")
        try:
            self._parse(path)
        except BaseException:
            self.fh.close()
            raise

    def _parse(self, path: str) -> None:
        head = self.fh.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{os.path.basename(path)} is not a RIFF/WAVE file")
        fmt = None
        self.data_offset = self.data_bytes = None
        while True:
            chunk = self.fh.read(8)
            if len(chunk) < 8:
                break
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                body = self.fh.read(size)
                fmt = struct.unpack("<HHIIHH", body[:16])
                if fmt[0] == 0xFFFE and len(body) >= 26:
                    fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
                if size & 1:
                    self.fh.seek(1, os.SEEK_CUR)
            elif cid == b"data":
                self.data_offset, self.data_bytes = self.fh.tell(), size
                break
            else:
                self.fh.seek(size + (size & 1), os.SEEK_CUR)
        if fmt is None or self.data_offset is None:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{os.path.basename(path)} has no fmt or data chunk")
        self.tag, self.channels, self.rate, _, self.block_align, self.bits = fmt
        if (self.tag, self.bits) not in ((1, 16), (1, 24), (1, 32), (3, 32), (3, 64)) or self.channels < 1:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{os.path.basename(path)}: unsupported WAV format {self.tag}/{self.bits} bit")
        self.width = self.bits // 8
        size_on_disk = os.fstat(self.fh.fileno()).st_size - self.data_offset
        self.frames = min(self.data_bytes, size_on_disk) // (self.width * self.channels)

    def read(self, np: Any, start: int, count: int) -> Any:
        """Frames [start, start + count) as float64 (count, channels); zeros outside the file."""
        out = np.zeros((max(count, 0), self.channels))
        lo, hi = max(start, 0), min(start + count, self.frames)
        if hi <= lo:
            return out
        self.fh.seek(self.data_offset + lo * self.width * self.channels)
        raw = self.fh.read((hi - lo) * self.width * self.channels)
        if self.tag == 3:
            vals = np.frombuffer(raw, dtype="<f4" if self.bits == 32 else "<f8").astype(np.float64)
        elif self.bits == 16:
            vals = np.frombuffer(raw, dtype="<i2") / 32768.0
        elif self.bits == 32:
            vals = np.frombuffer(raw, dtype="<i4") / 2147483648.0
        else:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            vals = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0
        out[lo - start:hi - start] = vals.reshape(-1, self.channels)
        return out

    def close(self) -> None:
        self.fh.close()


class FloatWavWriter:
    """32-bit IEEE float WAV, written front to back; the sizes are filled in on close."""

    def __init__(self, path: str, rate: int, channels: int) -> None:
        self.fh = open(path, "wb")
        self.rate, self.channels, self.frames = rate, channels, 0
        self.fh.write(self._header(0))

    def _header(self, frames: int) -> bytes:
        data = frames * 4 * self.channels
        return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + b"fmt " +
                struct.pack("<IHHIIHH", 16, 3, self.channels, self.rate, self.rate * 4 * self.channels, 4 * self.channels, 32) +
                b"data" + struct.pack("<I", data))

    def write(self, np: Any, block: Any) -> None:
        self.fh.write(np.ascontiguousarray(block, dtype="<f4").tobytes())
        self.frames += len(block)

    def close(self) -> None:
        self.fh.seek(0)
        self.fh.write(self._header(self.frames))
        self.fh.close()


def dereverb_file(src: str, dst: str, taps: int, delay: int, iterations: int, np: Any, wpe: Any,
                  block_seconds: float = BLOCK_SECONDS) -> Dict[str, Any]:
    """WPE on the WAV at `src`, written to `dst` as 32-bit float with the same rate, channels and length. `np` is
    numpy and `wpe` a function with nara_wpe's wpe_v8 signature (Y: (F, D, T) complex -> same shape)."""
    reader = WavReader(src)
    try:
        rate, channels, n = reader.rate, reader.channels, reader.frames
        size, hop = stft_sizes(rate)
        pad = size - hop                                   # xp (the padded signal) = pad zeros + the file + zeros
        window = np.hanning(size + 1)[:-1]                 # periodic Hann
        norm = np.zeros(hop)
        for k in range(size // hop):
            norm += window[k * hop:(k + 1) * hop] ** 2     # overlap-added window^2, the same at every hop offset
        total_frames = int(math.ceil((n + pad) / hop))     # frames whose start lies before the last real sample
        block = max(1, int(round(block_seconds * rate / hop)))
        context = taps + delay - 1
        writer = FloatWavWriter(dst, rate, channels)
        try:
            acc = np.zeros((size, channels))               # overlap-add tail, starting at xp index `emitted`
            emitted = 0
            for a in range(0, total_frames, block):
                b = min(a + block, total_frames)
                a0 = max(0, a - context)
                x = reader.read(np, a0 * hop - pad, (b - a0 - 1) * hop + size)
                frames = np.stack([x[j * hop:j * hop + size] for j in range(b - a0)])          # (T, size, D)
                spec = np.fft.rfft(frames * window[None, :, None], axis=1)                    # (T, F, D)
                y = np.ascontiguousarray(spec.transpose(1, 2, 0))                             # (F, D, T)
                z = wpe(y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
                if getattr(z, "shape", None) != y.shape:
                    raise VoiceEnhanceError("INVALID_RESULT", f"{WPE_FUNCTION} returned shape {getattr(z, 'shape', None)}, "
                                            f"expected {y.shape}", {"engine": "nara_wpe"})
                out = np.fft.irfft(np.asarray(z).transpose(2, 0, 1)[a - a0:], n=size, axis=1) * window[None, :, None]
                need = (b - a) * hop + size - (a * hop - emitted)
                if len(acc) < need:
                    acc = np.concatenate([acc, np.zeros((need - len(acc), channels))])
                base = a * hop - emitted
                for j in range(b - a):
                    acc[base + j * hop:base + j * hop + size] += out[j]
                final = b * hop if b < total_frames else pad + n          # frames >= b start at b * hop
                done = final - emitted
                chunk = acc[:done] / np.tile(norm, done // hop + 2)[(emitted % hop):(emitted % hop) + done, None]
                lo, hi = max(emitted, pad), min(final, pad + n)
                if hi > lo:
                    writer.write(np, chunk[lo - emitted:hi - emitted])
                acc = acc[done:]
                emitted = final
        finally:
            writer.close()
        if writer.frames != n:
            raise VoiceEnhanceError("INVALID_RESULT", f"wrote {writer.frames} frames, expected {n}", {"engine": "nara_wpe"})
        return {"sample_rate": rate, "channels": channels, "frames": n, "stft": {"window": "hann", "size": size, "hop": hop},
                "block_seconds": float(block_seconds)}
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
                                       block_seconds=facts["block_seconds"], output_format=OUTPUT_FORMAT))
