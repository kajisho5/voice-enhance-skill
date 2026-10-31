"""WAV I/O and the streaming STFT shared by the signal-processing engines (nara_wpe for DEREVERB, ADR-004; crosstalk
for BLEED_REDUCE, ADR-005). NumPy is passed in as `np` by the caller, never imported here, so this module imports with
the stdlib only and doctor stays fast.

  WavReader       a RIFF/WAVE file read in frames on demand (PCM 16/24/32-bit, float 32/64-bit, plain or extensible)
  FloatWavWriter  32-bit float WAV written front to back
  TrackReader     several mono WAVs read as the channels of one signal (zeros past a shorter file's end)
  TrackWriter     the channels of one signal written as several float WAVs, each cut to its own length
  stft_frames     the STFT of a run of frames (Hann window, hop = window / 4)
  stft_process    STFT -> a per-block function -> weighted overlap-add, block by block; with an identity function the
                  input comes back sample for sample (no delay)"""
from __future__ import annotations

import math
import os
import struct
from typing import Any, Callable, List, Sequence, Tuple

from ..errors import VoiceEnhanceError

HOP_SECONDS = 0.008                  # default STFT hop; the window is 4 hops


def stft_sizes(rate: int, hop_seconds: float = HOP_SECONDS) -> Tuple[int, int]:
    """(window, hop) in samples at `rate`: a hop of `hop_seconds` and a window of four hops."""
    hop = max(1, int(round(rate * hop_seconds)))
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



class TrackReader:
    """Mono WAV files at one sample rate, read as the channels of one signal: read() gives (count, len(paths)) float64,
    with zeros past the end of a shorter file. `frames` is the longest file's length, `lengths` each file's own."""

    def __init__(self, paths: Sequence[str]) -> None:
        self.readers: List[WavReader] = []
        try:
            for p in paths:
                self.readers.append(WavReader(p))
            rates = {r.rate for r in self.readers}
            if len(rates) != 1:
                raise VoiceEnhanceError("INVALID_ARGUMENT", f"the tracks have different sample rates {sorted(rates)}")
            bad = [os.path.basename(p) for p, r in zip(paths, self.readers) if r.channels != 1]
            if bad:
                raise VoiceEnhanceError("INVALID_ARGUMENT", f"every track must be mono: {bad}")
        except BaseException:
            self.close()
            raise
        self.rate, self.channels = self.readers[0].rate, len(self.readers)
        self.lengths = [r.frames for r in self.readers]
        self.frames = max(self.lengths)

    def read(self, np: Any, start: int, count: int) -> Any:
        return np.concatenate([r.read(np, start, count) for r in self.readers], axis=1)

    def close(self) -> None:
        for r in self.readers:
            r.close()


class TrackWriter:
    """One 32-bit float WAV per channel; frames past a track's own length (`lengths`) are dropped, so every output is
    exactly as long as its input."""

    def __init__(self, paths: Sequence[str], rate: int, lengths: Sequence[int]) -> None:
        self.writers: List[FloatWavWriter] = []
        self.lengths = list(lengths)
        self.position = 0
        try:
            for p in paths:
                self.writers.append(FloatWavWriter(p, rate, 1))
        except BaseException:
            self.close()
            raise

    @property
    def frames(self) -> int:
        return self.position

    def write(self, np: Any, block: Any) -> None:
        for k, w in enumerate(self.writers):
            keep = max(0, min(len(block), self.lengths[k] - self.position))
            if keep:
                w.write(np, block[:keep, k:k + 1])
        self.position += len(block)

    def close(self) -> None:
        for w in self.writers:
            w.close()


def stft_window(np: Any, size: int) -> Any:
    return np.hanning(size + 1)[:-1]                       # periodic Hann


def frame_count(n: int, size: int, hop: int) -> int:
    """Frames whose start lies before the last real sample of an `n`-sample signal padded with size - hop zeros."""
    return int(math.ceil((n + size - hop) / hop))


def stft_frames(np: Any, reader: Any, a: int, b: int, size: int, hop: int, window: Any) -> Any:
    """The STFT of frames [a, b) of `reader`'s signal, (b - a, size // 2 + 1, channels) complex. Frame k starts at
    sample k * hop - (size - hop), so every sample lies under all four overlapping frames."""
    x = reader.read(np, a * hop - (size - hop), (b - a - 1) * hop + size)
    frames = np.stack([x[j * hop:j * hop + size] for j in range(b - a)])           # (T, size, D)
    return np.fft.rfft(frames * window[None, :, None], axis=1)                     # (T, F, D)


def stft_process(reader: Any, writer: Any, np: Any, size: int, hop: int, block: int, context: int,
                 process: Callable[[Any, int], Any]) -> None:
    """Streams `reader` through STFT -> `process` -> weighted overlap-add into `writer`, `block` frames at a time.
    `process(spec, skip)` gets the (T, F, D) STFT of a block plus up to `context` frames before it, and returns the
    processed (T - skip, F, D) frames of the block itself. Exactly reader.frames samples are written."""
    n, channels = reader.frames, reader.channels
    pad = size - hop
    window = stft_window(np, size)
    norm = np.zeros(hop)
    for k in range(size // hop):
        norm += window[k * hop:(k + 1) * hop] ** 2     # overlap-added window^2, the same at every hop offset
    total_frames = frame_count(n, size, hop)
    acc = np.zeros((size, channels))                   # overlap-add tail, starting at padded-signal index `emitted`
    emitted = 0
    for a in range(0, total_frames, block):
        b = min(a + block, total_frames)
        a0 = max(0, a - context)
        out = np.fft.irfft(process(stft_frames(np, reader, a0, b, size, hop, window), a - a0), n=size, axis=1) * window[None, :, None]
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
