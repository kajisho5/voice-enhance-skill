"""Shared test helpers: paths, the CLI runner, a fake `deep-filter` executable, a fake `nara_wpe` package, WAV writers
and readers, the synthetic fixtures (voice+noise for ISOLATE, reverberant for DEREVERB, two-microphone cross-talk for
BLEED_REDUCE) and their measures. Nothing here is part of the package; the fake binary is never registered
anywhere but a test's environment."""
from __future__ import annotations

import io
import json
import math
import os
import random
import struct
import sys
import wave
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_enhance_skill import cli  # noqa: E402


def _dir_from_env(key: str, sibling: str, marker: str) -> Optional[Path]:
    for c in ([Path(os.environ[key])] if os.environ.get(key) else []) + [ROOT.parent / sibling]:
        if (c / marker).is_file():
            return c
    return None


OS_DIR = _dir_from_env("AI_VIDEO_PRODUCTION_OS_DIR", "AI-video-production-OS", "registry/conformance.py")
FFMPEG_SKILL_DIR = _dir_from_env("VOICE_ENHANCE_FFMPEG_SKILL_DIR", "ffmpeg-skill", "scripts/_contract.py")


def run_cli(*argv: str) -> Tuple[int, Dict[str, Any]]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, json.loads(buf.getvalue())


@contextmanager
def env(**values: Optional[str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# The program behind the fake `deep-filter` (modes other than "fail"). It records its argv to argv.json next to
# itself, checks the input like the real binary (48 kHz WAV), and then, per mode:
#   lowpass    a zero-phase low-pass (2nd-order Butterworth at 2 kHz, run forward and backward): it removes most of
#              a white noise and keeps the fixture's voice band, so the SNR against the clean voice rises
#   sleep      writes its pid to pid and sleeps (timeout / process-group tests)
#   nooutput   exits 0 without writing anything
#   wrongrate  writes a 16 kHz file
FAKE_PROGRAM = r"""
import json, math, os, struct, sys, time, wave
mode, args = sys.argv[1], sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "argv.json"), "w") as fh:
    json.dump(args, fh)
opts, files, i = {}, [], 0
while i < len(args):
    if args[i] in ("-m", "-o", "--atten-lim-db"):
        opts[args[i]] = args[i + 1]; i += 2
    elif args[i].startswith("-"):
        opts[args[i]] = True; i += 1
    else:
        files.append(args[i]); i += 1
if mode == "sleep":
    with open(os.path.join(here, "pid"), "w") as fh:
        fh.write(str(os.getpid()))
    time.sleep(60)
    sys.exit(0)
if "-m" not in opts or len(files) != 1:
    sys.stderr.write("usage: deep-filter -m MODEL -o DIR FILE\n"); sys.exit(2)
with wave.open(files[0], "rb") as w:
    rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n)
if rate != 48000:
    sys.stderr.write("Error: input sample rate %d does not match the model (48000)\n" % rate); sys.exit(1)
if mode == "nooutput":
    sys.exit(0)
x = [v / 32768.0 for v in struct.unpack("<%dh" % (n * ch), raw)][::ch]
def biquad(sig, fc=2000.0, fs=48000.0, q=0.7071067811865476):
    w0 = 2 * math.pi * fc / fs; al = math.sin(w0) / (2 * q); c = math.cos(w0); a0 = 1 + al
    b0, b1, b2, a1, a2 = (1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, -2 * c / a0, (1 - al) / a0
    y, x1, x2, y1, y2 = [], 0.0, 0.0, 0.0, 0.0
    for v in sig:
        o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, v, y1, o
        y.append(o)
    return y
y = biquad(biquad(x)[::-1])[::-1]
out_rate = 16000 if mode == "wrongrate" else 48000
os.makedirs(opts.get("-o", "out"), exist_ok=True)
with wave.open(os.path.join(opts.get("-o", "out"), os.path.basename(files[0])), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(out_rate)
    w.writeframes(struct.pack("<%dh" % len(y), *[max(-32768, min(32767, int(round(v * 32767)))) for v in y]))
"""


def fake_deep_filter(directory: str, version: str = "0.5.6", mode: str = "fail") -> str:
    """A POSIX executable that answers `--version` like the real binary ("deep-filter 0.5.6"). mode "fail" exits 9
    on anything else; the other modes run FAKE_PROGRAM (see above)."""
    path = os.path.join(directory, "deep-filter")
    if mode == "fail":
        tail = "exit 9\n"
    else:
        program = os.path.join(directory, "fake_deep_filter.py")
        with open(program, "w", encoding="utf-8") as fh:
            fh.write(FAKE_PROGRAM)
        tail = f"exec '{sys.executable}' '{program}' {mode} \"$@\"\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'deep-filter {version}'; exit 0; fi\n{tail}")
    os.chmod(path, 0o755)
    return path


def fake_model(directory: str) -> str:
    model = os.path.join(directory, "DeepFilterNet3_onnx.tar.gz")
    with open(model, "wb") as fh:
        fh.write(b"\x1f\x8b fake model")
    return model


def write_wav(path: str, samples: List[float], rate: int = 48000, channels: int = 1) -> str:
    """16-bit PCM; `samples` is mono in [-1, 1], duplicated to every channel."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, int(round(v * 32767))))) * channels for v in samples))
    return path


def voice_noise_fixture(path: str, seconds: float = 1.5, rate: int = 48000, snr_db: float = 0.0, channels: int = 1,
                        seed: int = 7) -> List[float]:
    """The synthetic voice+noise fixture: a voiced "speech" signal (140 Hz fundamental, harmonics to 840 Hz, a 4 Hz
    syllable envelope) plus white noise at `snr_db`. Writes the noisy mix and returns the clean voice at the same
    scale, so snr_db(clean, read_wav(x)) measures any processed version of it."""
    rng = random.Random(seed)
    n = int(seconds * rate)
    voice = [(0.5 - 0.5 * math.cos(2 * math.pi * 4 * i / rate)) *
             sum(math.sin(2 * math.pi * 140 * k * i / rate) / k for k in range(1, 7)) for i in range(n)]
    noise = [rng.gauss(0.0, 1.0) for _ in range(n)]
    pv, pn = sum(v * v for v in voice) / n, sum(v * v for v in noise) / n
    g = math.sqrt(pv / (pn * 10 ** (snr_db / 10)))
    mix = [v + g * e for v, e in zip(voice, noise)]
    scale = 0.5 / max(abs(v) for v in mix)
    write_wav(path, [v * scale for v in mix], rate, channels)
    return [v * scale for v in voice]


def read_wav(path: str) -> Tuple[int, int, List[float]]:
    """(sample rate, channels, mono samples in [-1, 1]) for PCM 16/24/32-bit and IEEE float WAV (the real deep-filter's
    output format is not pinned down here, and the stdlib wave module reads integer PCM only)."""
    rate, ch, vals = _wav_interleaved(path)
    frames = len(vals) // ch
    return rate, ch, [sum(vals[f * ch:(f + 1) * ch]) / ch for f in range(frames)]


def read_float_wav_channels(path: str) -> Tuple[int, List[List[float]]]:
    """(sample rate, one list per channel) for any WAV read_wav reads, float included."""
    rate, ch, vals = _wav_interleaved(path)
    return rate, [vals[k::ch] for k in range(ch)]


def _wav_interleaved(path: str) -> Tuple[int, int, List[float]]:
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    pos, fmt, payload = 12, None, b""
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE:
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None:
        raise ValueError(f"{path} has no fmt chunk")
    tag, ch, rate, _, _, bits = fmt
    width = bits // 8
    count = len(payload) // width
    if tag == 3 and bits == 32:
        vals = list(struct.unpack(f"<{count}f", payload[:count * 4]))
    elif tag == 1 and bits == 16:
        vals = [v / 32768.0 for v in struct.unpack(f"<{count}h", payload[:count * 2])]
    elif tag == 1 and bits == 32:
        vals = [v / 2147483648.0 for v in struct.unpack(f"<{count}i", payload[:count * 4])]
    elif tag == 1 and bits == 24:
        vals = [int.from_bytes(payload[i:i + 3], "little", signed=True) / 8388608.0 for i in range(0, count * 3, 3)]
    else:
        raise ValueError(f"{path}: unsupported WAV format {tag}/{bits} bit")
    return rate, ch, vals


def snr_db(clean: List[float], test: List[float]) -> float:
    """SNR of `test` against the clean reference, in dB, over their common length."""
    n = min(len(clean), len(test))
    signal = sum(c * c for c in clean[:n])
    error = sum((t - c) ** 2 for c, t in zip(clean[:n], test[:n]))
    return 10 * math.log10(signal / max(error, 1e-20))


def best_lag(reference: List[float], test: List[float], start: int, length: int, max_lag: int) -> int:
    """The lag in samples (positive: `test` is late) at which `test` best matches `reference`, by the
    cross-correlation of reference[start:start + length] with test shifted by -max_lag..max_lag. Pure Python."""
    ref = reference[start:start + length]
    def corr(lag: int) -> float:
        return sum(r * test[start + lag + i] for i, r in enumerate(ref))
    return max(range(-max_lag, max_lag + 1), key=corr)


def write_tone_wav(path: str, seconds: float = 1.0, rate: int = 48000, freq: float = 440.0) -> str:
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)))
    return path


# The fake `nara_wpe` package (DEREVERB, ADR-004): nara_wpe/wpe.py with wpe_v8, plus a dist-info so the engine's
# version check reads 0.0.11. It records each call (shape and parameters) to calls.json next to the package, then,
# per mode:
#   wpe        a minimal WPE on NumPy (per frequency: delayed-frame stack, power-weighted least squares, subtract the
#              prediction), so the reverberant fixture really gets drier
#   identity   returns Y unchanged (the STFT round trip must give the input back, sample-aligned)
#   fail       raises RuntimeError
#   badshape   returns an array of the wrong shape
FAKE_WPE_PROGRAM = r'''
import json, os
import numpy as np
MODE = __MODE__
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wpe_v8(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode="full", inplace=False):
    path = os.path.join(HERE, "calls.json")
    calls = json.load(open(path)) if os.path.exists(path) else []
    calls.append({"shape": list(Y.shape), "dtype": str(Y.dtype), "taps": taps, "delay": delay, "iterations": iterations,
                  "statistics_mode": statistics_mode})
    json.dump(calls, open(path, "w"))
    if MODE == "fail":
        raise RuntimeError("fake nara_wpe failed")
    if MODE == "badshape":
        return Y[..., :-1]
    if MODE == "identity":
        return Y.copy()
    F, D, T = Y.shape
    X = np.empty_like(Y)
    for f in range(F):
        y = Y[f]
        yt = np.zeros((taps * D, T), dtype=Y.dtype)
        for k in range(taps):
            s = delay + k
            if s < T:
                yt[k * D:(k + 1) * D, s:] = y[:, :T - s]
        x = y
        for _ in range(iterations):
            w = yt / np.maximum(np.mean(np.abs(x) ** 2, axis=0), 1e-10)
            r = w @ yt.conj().T
            r += (1e-6 * np.trace(r).real / len(r) + 1e-12) * np.eye(len(r))   # identical channels are singular
            g = np.linalg.solve(r, w @ y.conj().T)
            x = y - g.conj().T @ yt
        X[f] = x
    return X
'''


def fake_nara_wpe(directory: str, mode: str = "wpe", version: str = "0.0.11") -> str:
    """A directory to put on sys.path / PYTHONPATH that holds the fake `nara_wpe` package; returns it."""
    root = os.path.join(directory, "fake_site")
    pkg = os.path.join(root, "nara_wpe")
    os.makedirs(pkg, exist_ok=True)
    Path(pkg, "__init__.py").write_text("", encoding="utf-8")
    Path(pkg, "wpe.py").write_text(FAKE_WPE_PROGRAM.replace("__MODE__", repr(mode)), encoding="utf-8")
    dist = os.path.join(root, f"nara_wpe-{version}.dist-info")
    os.makedirs(dist, exist_ok=True)
    Path(dist, "METADATA").write_text(f"Metadata-Version: 2.1\nName: nara_wpe\nVersion: {version}\n", encoding="utf-8")
    return root


def fake_wpe_calls(site: str) -> List[Dict[str, Any]]:
    path = Path(site, "calls.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@contextmanager
def on_python_path(directory: str) -> Iterator[None]:
    """`directory` first on sys.path (this process) and on PYTHONPATH (the worker), with import caches reset on the
    way in and out, so a fake package is found by the engine's checks and by the worker, and forgotten afterwards."""
    import importlib
    def forget() -> None:
        for name in [m for m in sys.modules if m == "nara_wpe" or m.startswith("nara_wpe.")]:
            del sys.modules[name]
        importlib.invalidate_caches()
    sys.path.insert(0, directory)
    forget()
    try:
        with env(PYTHONPATH=os.pathsep.join([directory] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])):
            yield
    finally:
        sys.path.remove(directory)
        forget()


def have_numpy() -> bool:
    import importlib.util
    return importlib.util.find_spec("numpy") is not None


def syllables(seconds: float, rate: int, seed: int = 5, breath: float = 0.3) -> List[float]:
    """Speech-like dry signal for DEREVERB: syllables of 80-250 ms, each with its own fundamental (100-240 Hz, gliding
    by up to 30 %), harmonics to the 6th, a little breath noise and a sine envelope, separated by 30-150 ms gaps. It
    has to be non-stationary: WPE (correctly) treats a steady tone as predictable, i.e. as reverberation, so
    voice_noise_fixture's steady 140 Hz voice would be removed along with the reverb."""
    rng = random.Random(seed)
    n = int(seconds * rate)
    out = [0.0] * n
    i = int(0.02 * rate)
    while i < n:
        length, f0, glide, phase = int(rng.uniform(0.08, 0.25) * rate), rng.uniform(100, 240), rng.uniform(-0.3, 0.3), 0.0
        for k in range(min(length, n - i)):
            phase += 2 * math.pi * f0 * (1 + glide * k / length) / rate
            out[i + k] = math.sin(math.pi * k / length) * (sum(math.sin(h * phase) / h for h in range(1, 7)) + breath * rng.gauss(0, 1))
        i += length + int(rng.uniform(0.03, 0.15) * rate)
    return out


REVERB_COMBS = ((0.037, 0.6), (0.059, 0.5))     # (delay in s, feedback gain): two feedback comb filters in series
REVERB_COMBS_B = ((0.043, 0.55), (0.067, 0.45))  # the second microphone's room in a distinct multi-channel fixture


def reverb_fixture(path: str, seconds: float = 1.0, rate: int = 16000, channels: int = 1, distinct: bool = False) -> List[float]:
    """The synthetic reverberant fixture: syllables() through two feedback comb filters in series (37 ms / 0.6 and
    59 ms / 0.5, a decaying echo tail whose inverse is a short prediction filter, the model WPE assumes). Writes the
    reverberant signal and returns the dry signal at the same scale, so srr_db(dry, read_wav(x)) is the
    signal-to-reverberation ratio of any processed version of it. O(n) pure Python. With several channels the same
    signal goes to every channel (dual mono), unless `distinct`: then every other channel hears its own room
    (REVERB_COMBS_B), as a second microphone would, and the channels are linearly independent."""
    dry = syllables(seconds, rate)
    rooms = [REVERB_COMBS_B if distinct and k % 2 else REVERB_COMBS for k in range(channels if distinct else 1)]
    tracks = []
    for combs in rooms:
        wet = list(dry)
        for delay, gain in combs:
            d = int(delay * rate)
            for i in range(d, len(wet)):
                wet[i] += gain * wet[i - d]
        tracks.append(wet)
    scale = 0.5 / max(abs(v) for wet in tracks for v in wet)
    if distinct:
        write_wav_tracks(path, [[v * scale for v in wet] for wet in tracks], rate)
    else:
        write_wav(path, [v * scale for v in tracks[0]], rate, channels)
    return [v * scale for v in dry]


srr_db = snr_db     # the reverberation is the "noise": SRR is the SNR of a signal against its direct path


def write_wav_tracks(path: str, tracks: List[List[float]], rate: int = 48000) -> str:
    """16-bit PCM with one channel per track (each in [-1, 1]); a shorter track is padded with zeros."""
    n = max(len(t) for t in tracks)
    with wave.open(path, "wb") as w:
        w.setnchannels(len(tracks))
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, int(round((t[i] if i < len(t) else 0.0) * 32767)))))
                               for i in range(n) for t in tracks))
    return path


def read_wav_channels(path: str) -> Tuple[int, List[List[float]]]:
    """(sample rate, one list per channel) for 16-bit PCM WAV."""
    with wave.open(path, "rb") as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        vals = [v / 32768.0 for v in struct.unpack(f"<{n * ch}h", w.readframes(n))]
    return rate, [vals[c::ch] for c in range(ch)]


CROSSTALK_LEAK_DB = -8.0        # how loud each talker is in the other microphone, relative to its own
CROSSTALK_DELAY_MS = 4.0        # and how much later it arrives there (about 1.4 m further away)


def crosstalk_fixture(seconds: float = 2.0, rate: int = 16000, leak_db: float = CROSSTALK_LEAK_DB,
                      delay_ms: float = CROSSTALK_DELAY_MS, seed: int = 3) -> Dict[str, Any]:
    """The synthetic two-microphone fixture for BLEED_REDUCE. Talker 0 speaks in the first 60 % of the time and talker
    1 in the last 60 % (they overlap in the middle fifth). Each microphone hears its own talker plus the other one at
    `leak_db`, `delay_ms` later, plus a faint independent noise floor (-60 dB). Returns {"mics": [mic0, mic1] scaled
    to a 0.5 peak, "rate", "own": [region of track 0's own talker alone, region of track 1's], "other": [region of
    the other talker alone in track 0, in track 1]}, regions as (start, end) in seconds. Cross-talk energy is a
    track's energy where only the other talker speaks; its own speech is its energy where only its own talker does."""
    rng = random.Random(seed)
    n, d, g = int(seconds * rate), int(delay_ms * rate / 1000), 10 ** (leak_db / 20)
    lo1, hi0 = 0.4 * seconds, 0.6 * seconds
    t0 = [v if i < hi0 * rate else 0.0 for i, v in enumerate(syllables(seconds, rate, seed=5))]
    t1 = [v if i >= lo1 * rate else 0.0 for i, v in enumerate(syllables(seconds, rate, seed=11))]
    mic0 = [t0[i] + g * (t1[i - d] if i >= d else 0.0) + 0.001 * rng.gauss(0, 1) for i in range(n)]
    mic1 = [t1[i] + g * (t0[i - d] if i >= d else 0.0) + 0.001 * rng.gauss(0, 1) for i in range(n)]
    scale = 0.5 / max(max(abs(v) for v in mic0), max(abs(v) for v in mic1))
    # the regions keep 50 ms away from where a talker starts or stops, so the delayed bleed has arrived / faded
    edge = 0.05
    return {"mics": [[v * scale for v in mic0], [v * scale for v in mic1]], "rate": rate,
            "own": [(0.0, lo1 - edge), (hi0 + edge, seconds)], "other": [(hi0 + edge, seconds), (0.0, lo1 - edge)]}


def energy_db(samples: List[float], rate: int, region: Tuple[float, float]) -> float:
    """Energy of `samples` within `region` (seconds), in dB."""
    part = samples[int(region[0] * rate):int(region[1] * rate)]
    return 10 * math.log10(max(sum(v * v for v in part), 1e-20))
