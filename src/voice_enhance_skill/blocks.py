"""Long inputs in blocks, resumable (roadmap step 11, ADR-009). Standard library only: ISOLATE runs without NumPy.

A prepared input longer than one block is cut into overlapping blocks. The engine runs on each block in its own
worker call, and each block's output is cached under a key that hashes the block's input samples together with the
engine's identity and parameters. A run that stops (an engine failure, a timeout, a kill) leaves the finished blocks in
the cache; the next run with the same input and settings finds them, skips them, and resumes at the first missing
block. The block outputs are then joined into one file with a raised-cosine crossfade at each seam.

  plan        blocks of at most `block_seconds`, all about the same length, each overlapping the next by
              OVERLAP_SECONDS; an input no longer than one block is one block (the engine sees the whole file, as
              before this step, and nothing is cached)
  key         sha256 of the cache schema, the skill version, the op, the engine id, version and model, the engine
              parameters (not the timeout), and every track's WAV format and raw block bytes
  cache       <cache dir>/<key>/ holds out<i>.wav per output and block.json (engine version, model, engine seconds,
              the engine's extra facts). It is written into a temporary name and renamed into place, so an entry is
              whole or absent
  join        from each overlap [a, b) between block k and block k + 1, the first quarter comes from block k, the last
              quarter from block k + 1, and the middle half is crossfaded (raised cosine, the weights sum to 1). The rest
              of each block is copied byte for byte. The joined file keeps the blocks' own sample format, and turns into
              RF64 when it passes 4 GiB
  clean       after the op has placed its output, the run's cache entries are deleted (and the cache directory, when it
              is empty)

The realtime factor reported is the engine's wall time (worker start included) divided by the audio it processed,
over every block, also the blocks that came from the cache (their time is kept in block.json)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .engines.base import EngineRequest, EngineResult
from .engines.wavio import WavReader
from .errors import VoiceEnhanceError

BLOCK_SECONDS = 600.0               # default block length: 10 minutes
BLOCK_LIMITS = (15.0, 3600.0)       # --block-seconds range (at least 3 overlaps, so only neighbours overlap)
OVERLAP_SECONDS = 4.0               # each block overlaps the next by this much; the crossfade is its middle half
CACHE_NAME = ".voice-enhance-blocks"
CACHE_SCHEMA = "voice-enhance/block@1"
BLOCK_JSON = "block.json"
LENGTH_TOLERANCE = 0.1              # seconds (or 1 %) a block's output may differ from its input, as validate_output allows
CHUNK_FRAMES = 1 << 18
WAV_LIMIT = 0xFFFFFFFF
FORMATS = {(1, 16): "h", (1, 32): "i", (3, 32): "f", (3, 64): "d"}      # (tag, bits) -> array typecode; 24-bit by hand
SCALE = {(1, 16): 32768.0, (1, 24): 8388608.0, (1, 32): 2147483648.0}


def check_block_seconds(value: Any) -> float:
    lo, hi = BLOCK_LIMITS
    if value is None:
        return BLOCK_SECONDS
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"--block-seconds must be a number from {lo:g} to {hi:g}", {"block_seconds": value})
    return float(value)


def plan(frames: int, rate: int, block_seconds: float = BLOCK_SECONDS, overlap_seconds: float = OVERLAP_SECONDS) -> List[Tuple[int, int]]:
    """[(start, end)] in frames. Consecutive blocks overlap by exactly round(overlap_seconds * rate) frames."""
    size, overlap = int(round(block_seconds * rate)), int(round(overlap_seconds * rate))
    if frames <= size or size < 3 * overlap:
        return [(0, frames)]
    count = int(math.ceil((frames - overlap) / (size - overlap)))
    step = int(math.ceil((frames - overlap) / count))
    return [(k * step, min(frames, (k + 1) * step + overlap)) for k in range(count)]


# ---------------------------------------------------------------------------------------------------------- WAV I/O

def _fmt(reader: WavReader) -> Tuple[int, int, int, int]:
    return reader.tag, reader.bits, reader.channels, reader.rate


def _read_raw(reader: WavReader, start: int, end: int) -> bytes:
    """Raw frames [start, end) of the data chunk, clipped to the file."""
    lo, hi = max(0, start), min(end, reader.frames)
    if hi <= lo:
        return b""
    align = reader.width * reader.channels
    reader.fh.seek(reader.data_offset + lo * align)
    return reader.fh.read((hi - lo) * align)


class RawWavWriter:
    """PCM / float WAV of a given format, written front to back. With `rf64` a JUNK chunk reserves room for a ds64
    chunk, and a file whose data passes 4 GiB is turned into RF64 on close (EBU Tech 3306, as ffmpeg -rf64 auto)."""

    def __init__(self, path: str, tag: int, bits: int, channels: int, rate: int, rf64: bool = False) -> None:
        self.path, self.tag, self.bits, self.channels, self.rate, self.rf64 = path, tag, bits, channels, rate, rf64
        self.align = channels * bits // 8
        self.frames = 0
        self.fh = open(path, "wb")
        self.fh.write(self._header(0))

    def _header(self, data: int) -> bytes:
        junk = b"JUNK" + struct.pack("<I", 28) + bytes(28) if self.rf64 else b""
        fmt = b"fmt " + struct.pack("<IHHIIHH", 16, self.tag, self.channels, self.rate, self.rate * self.align, self.align, self.bits)
        riff = 4 + len(junk) + len(fmt) + 8 + data
        if riff <= WAV_LIMIT:
            return b"RIFF" + struct.pack("<I", riff) + b"WAVE" + junk + fmt + b"data" + struct.pack("<I", data)
        if not self.rf64:
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{os.path.basename(self.path)} would pass the 4 GiB WAV limit; use a shorter "
                                    "--block-seconds", {"bytes": data})
        ds64 = b"ds64" + struct.pack("<IQQQI", 28, riff, data, self.frames, 0)
        return b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)

    def write_raw(self, raw: bytes) -> None:
        self.fh.write(raw)
        self.frames += len(raw) // self.align

    def close(self) -> None:
        try:
            header = self._header(self.frames * self.align)
            self.fh.seek(0)
            self.fh.write(header)
        finally:
            self.fh.close()


def decode(raw: bytes, tag: int, bits: int) -> List[float]:
    """Interleaved samples as floats (integer PCM scaled to [-1, 1))."""
    if (tag, bits) == (1, 24):
        return [int.from_bytes(raw[i:i + 3], "little", signed=True) / SCALE[(1, 24)] for i in range(0, len(raw) - 2, 3)]
    a = array(FORMATS[(tag, bits)])
    a.frombytes(raw[:len(raw) - len(raw) % a.itemsize])
    if sys.byteorder == "big":
        a.byteswap()
    scale = SCALE.get((tag, bits))
    return [v / scale for v in a] if scale else list(a)


def encode(values: Sequence[float], tag: int, bits: int) -> bytes:
    if tag == 3:
        a = array(FORMATS[(tag, bits)], values)
    else:
        scale = SCALE[(tag, bits)]
        lo, hi = -int(scale), int(scale) - 1
        ints = [max(lo, min(hi, int(round(v * scale)))) for v in values]
        if bits == 24:
            return b"".join(v.to_bytes(3, "little", signed=True) for v in ints)
        a = array(FORMATS[(tag, bits)], ints)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def join(parts: List[Tuple[int, str]], dst: str, engine_id: str) -> int:
    """Joins block outputs [(start frame, path)] into `dst`, crossfading the middle half of each overlap. Returns the
    number of frames written."""
    readers = [WavReader(p) for _, p in parts]
    try:
        fmt = _fmt(readers[0])
        if any(_fmt(r) != fmt for r in readers):
            raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r} wrote its blocks in different formats: "
                                    f"{sorted({_fmt(r) for r in readers})}", {"engine": engine_id})
        tag, bits, channels, rate = fmt
        writer = RawWavWriter(dst, tag, bits, channels, rate, rf64=True)
        try:
            cursor = 0
            for k, ((start, _), r) in enumerate(zip(parts, readers)):
                end = start + r.frames
                if k + 1 == len(parts):
                    _copy(writer, r, cursor - start, end - start)
                    break
                nstart, nr = parts[k + 1][0], readers[k + 1]
                a, b = nstart, min(end, nstart + nr.frames)
                if b < a or cursor > a:
                    raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r}: block {k} ends at frame {end}, before block "
                                            f"{k + 1} starts at frame {nstart}; the blocks cannot be joined", {"engine": engine_id})
                g = (b - a) // 4
                _copy(writer, r, cursor - start, a + g - start)
                _crossfade(writer, r, nr, a + g - start, a + g - nstart, (b - g) - (a + g))
                cursor = b - g
            frames = writer.frames
        finally:
            writer.close()
        return frames
    finally:
        for r in readers:
            r.close()


def _copy(writer: RawWavWriter, reader: WavReader, lo: int, hi: int) -> None:
    for a in range(lo, hi, CHUNK_FRAMES):
        writer.write_raw(_read_raw(reader, a, min(hi, a + CHUNK_FRAMES)))


def _crossfade(writer: RawWavWriter, prev: WavReader, nxt: WavReader, at_prev: int, at_next: int, length: int) -> None:
    """`length` frames: prev from frame at_prev fading out, nxt from frame at_next fading in (raised cosine)."""
    if length <= 0:
        return
    tag, bits, ch = writer.tag, writer.bits, writer.channels
    x = decode(_read_raw(prev, at_prev, at_prev + length), tag, bits)
    y = decode(_read_raw(nxt, at_next, at_next + length), tag, bits)
    out = []
    for i in range(length):
        w = 0.5 - 0.5 * math.cos(math.pi * (i + 0.5) / length)
        for c in range(ch):
            j = i * ch + c
            out.append((1.0 - w) * x[j] + w * y[j])
    writer.write_raw(encode(out, tag, bits))


# ---------------------------------------------------------------------------------------------------------- the run

@dataclass
class BlockedRun:
    """The engine's result (output paths are the joined files), the `blocks` and `timing` result sections, and the
    cache entries to delete once the op has placed its output."""
    result: EngineResult
    blocks: Dict[str, Any]
    timing: Dict[str, Any]
    cache_dir: Optional[str] = None
    keys: List[str] = field(default_factory=list)

    def discard_cache(self) -> None:
        if not self.cache_dir:
            return
        for key in self.keys:
            shutil.rmtree(os.path.join(self.cache_dir, key), ignore_errors=True)
        _rmdir_if_empty(self.cache_dir)


def _rmdir_if_empty(directory: str) -> None:
    try:
        os.rmdir(directory)
    except OSError:
        pass


def block_key(identity: Dict[str, Any], readers: Sequence[WavReader], start: int, end: int) -> str:
    h = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for r in readers:
        h.update(struct.pack("<IIII", *_fmt(r)))
        lo, hi = max(0, start), min(end, r.frames)
        h.update(struct.pack("<QQ", lo, max(lo, hi)))
        for a in range(lo, hi, CHUNK_FRAMES):
            h.update(_read_raw(r, a, min(hi, a + CHUNK_FRAMES)))
    return h.hexdigest()


def _load_entry(cache_dir: str, key: str, outputs: int) -> Optional[Dict[str, Any]]:
    entry = os.path.join(cache_dir, key)
    try:
        with open(os.path.join(entry, BLOCK_JSON), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    names = meta.get("outputs") if isinstance(meta, dict) else None
    if meta.get("schema") != CACHE_SCHEMA or meta.get("key") != key or not isinstance(names, list) or len(names) != outputs \
            or not all(isinstance(n, str) and os.path.isfile(os.path.join(entry, n)) for n in names):
        shutil.rmtree(entry, ignore_errors=True)          # a damaged entry is recomputed
        return None
    return meta


def _write_block_inputs(readers: Sequence[WavReader], start: int, end: int, directory: str) -> List[str]:
    paths = []
    for i, r in enumerate(readers):
        path = os.path.join(directory, f"in{i}.wav")
        w = RawWavWriter(path, r.tag, r.bits, r.channels, r.rate)
        try:
            _copy(w, r, start, min(end, r.frames))
        finally:
            w.close()
        paths.append(path)
    return paths


def _check_length(engine_id: str, index: int, path: str, expected: int, rate: int) -> int:
    r = WavReader(path)
    try:
        frames = r.frames
    finally:
        r.close()
    tolerance = max(LENGTH_TOLERANCE * rate, 0.01 * expected)
    if abs(frames - expected) > tolerance:
        raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r} wrote {frames} frames for block {index}, expected {expected} "
                                f"(+/- {int(tolerance)})", {"engine": engine_id, "block": index})
    return frames


RunEngine = Callable[[str, EngineRequest, float, str], EngineResult]


def run_blocked(run_engine: RunEngine, engine_id: str, op: str, inputs: List[str], outputs: List[str], rate: int,
                params: Dict[str, Any], timeout: float, run_dir: str, identity: Dict[str, Any], cache_dir: str,
                block_seconds: float = BLOCK_SECONDS, merge: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None) -> BlockedRun:
    """Runs `engine_id` on the prepared `inputs` (one per track; several only for BLEED_REDUCE) block by block and
    writes one joined file per entry of `outputs`. `identity` names the engine build (version, model) for the cache
    key; `merge` combines the engines' per-block `extra` facts (default: the first block's)."""
    multi = op == "bleed_reduce"
    readers = [WavReader(p) for p in inputs]
    try:
        frames = max(r.frames for r in readers)
        blocks = plan(frames, rate, block_seconds)
        audio_seconds = frames / float(rate) if rate else 0.0
        if len(blocks) == 1:          # the whole file in one worker call, nothing cached: exactly what ran before step 11
            for r in readers:
                r.close()
            readers = []
            req = EngineRequest(op=op, input_path=inputs[0], output_path=outputs[0], sample_rate=rate, params=params,
                                input_paths=list(inputs) if multi else [], output_paths=list(outputs) if multi else [])
            t0 = time.monotonic()
            result = run_engine(engine_id, req, timeout, run_dir)
            seconds = time.monotonic() - t0
            return BlockedRun(result=result, blocks=_blocks_doc(blocks, rate, block_seconds, [0], [], None, None),
                              timing=_timing(engine_id, seconds, audio_seconds, seconds))
        key_params = {k: v for k, v in sorted(params.items()) if k != "timeout"}
        ident = dict(identity, schema=CACHE_SCHEMA, skill_version=__version__, op=op, engine=engine_id, params=key_params, rate=rate)
        os.makedirs(cache_dir, exist_ok=True)
        keys, metas, processed, cached = [], [], [], []
        this_run = 0.0
        for k, (start, end) in enumerate(blocks):
            key = block_key(ident, readers, start, end)
            keys.append(key)
            meta = _load_entry(cache_dir, key, len(outputs))
            if meta is not None:
                cached.append(k)
                metas.append(meta)
                continue
            work = os.path.join(run_dir, f"block{k}")
            os.mkdir(work)
            block_in = _write_block_inputs(readers, start, end, work)
            names = [f"out{i}.wav" for i in range(len(outputs))]
            block_out = [os.path.join(work, n) for n in names]
            req = EngineRequest(op=op, input_path=block_in[0], output_path=block_out[0], sample_rate=rate, params=params,
                                input_paths=block_in if multi else [], output_paths=block_out if multi else [])
            t0 = time.monotonic()
            result = run_engine(engine_id, req, timeout, work)
            seconds = time.monotonic() - t0
            this_run += seconds
            for path, r in zip(block_out, readers if multi else readers[:1]):
                _check_length(engine_id, k, path, max(0, min(end, r.frames) - start), rate)
            meta = {"schema": CACHE_SCHEMA, "key": key, "op": op, "engine": engine_id, "engine_version": result.engine_version,
                    "model": result.model, "index": k, "start": start, "frames": end - start, "engine_seconds": round(seconds, 3),
                    "outputs": names, "extra": result.extra}
            tmp = os.path.join(cache_dir, f"{key}.tmp-{os.getpid()}")
            shutil.rmtree(tmp, ignore_errors=True)
            os.mkdir(tmp)
            for path, name in zip(block_out, names):
                shutil.move(path, os.path.join(tmp, name))
            with open(os.path.join(tmp, BLOCK_JSON), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False)
            try:
                os.rename(tmp, os.path.join(cache_dir, key))
            except OSError:           # the same block finished elsewhere meanwhile (identical input, identical output)
                shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
            processed.append(k)
            metas.append(meta)
        lengths = [r.frames for r in readers]
    finally:
        for r in readers:
            r.close()
    for i, dst in enumerate(outputs):
        length = lengths[i] if multi else lengths[0]
        parts = [(start, os.path.join(cache_dir, key, metas[k]["outputs"][i]))
                 for k, ((start, _), key) in enumerate(zip(blocks, keys)) if start < length or k == 0]
        join(parts, dst, engine_id)
    extras = [m.get("extra") or {} for m in metas]
    extra = merge(extras) if merge else dict(extras[0])
    result = EngineResult(output_path=outputs[0], engine_version=str(metas[0].get("engine_version")), model=metas[0].get("model"),
                          extra=extra, output_paths=list(outputs) if multi else [])
    resumed = next((k for k in range(len(blocks)) if k not in cached), None) if cached else None
    total = sum(float(m.get("engine_seconds") or 0.0) for m in metas)
    block_audio = sum(e - s for s, e in blocks) / float(rate)
    return BlockedRun(result=result, blocks=_blocks_doc(blocks, rate, block_seconds, processed, cached, resumed, cache_dir),
                      timing=_timing(engine_id, total, block_audio, this_run), cache_dir=cache_dir, keys=keys)


def _blocks_doc(blocks: List[Tuple[int, int]], rate: int, block_seconds: float, processed: List[int], cached: List[int],
                resumed: Optional[int], cache_dir: Optional[str]) -> Dict[str, Any]:
    one = len(blocks) == 1
    return {"count": len(blocks), "block_seconds": block_seconds, "overlap_seconds": 0.0 if one else OVERLAP_SECONDS,
            "crossfade": None if one else "raised cosine over the middle half of each overlap",
            "starts": [round(s / float(rate), 6) for s, _ in blocks] if rate else [0.0],
            "processed": processed, "cached": cached, "resumed_from": resumed, "cache_dir": cache_dir}


def _timing(engine_id: str, engine_seconds: float, audio_seconds: float, this_run: float) -> Dict[str, Any]:
    return {"engine": engine_id, "engine_seconds": round(engine_seconds, 3), "audio_seconds": round(audio_seconds, 3),
            "realtime_factor": round(engine_seconds / audio_seconds, 4) if audio_seconds > 0 else None,
            "engine_seconds_this_run": round(this_run, 3)}
