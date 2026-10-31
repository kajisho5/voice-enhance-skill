"""Measure the real engines on the synthetic fixtures and print a Markdown table (CI's engines job appends it to the
job summary; ADR-007 records the numbers). Not a test: the assertions are in test_integration.py. Needs the real
deep-filter + DeepFilterNet3 model, nara_wpe, NumPy and an ffmpeg-skill checkout (VOICE_ENHANCE_FFMPEG_SKILL_DIR).

  python tests/measure_engines.py"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from helpers import (FFMPEG_SKILL_DIR, read_float_wav_channels, read_wav, read_wav_channels, reverb_fixture, snr_db, srr_db,  # noqa: E402
                     voice_noise_fixture, write_wav_tracks)

from voice_enhance_skill.engines.deepfilternet import ENV_MODEL, find_binary  # noqa: E402
from voice_enhance_skill.ffmpeg_skill import FfmpegSkill  # noqa: E402
from voice_enhance_skill.ops import DereverbRequest, IsolateRequest, dereverb, isolate  # noqa: E402


def lag(reference, test, max_lag):
    """Lag of `test` against `reference` (positive: late) at the cross-correlation peak within +-max_lag."""
    a, b = np.asarray(reference), np.asarray(test)
    n = min(len(a), len(b)) - 2 * max_lag
    seg = a[max_lag:max_lag + n]
    return max(range(-max_lag, max_lag + 1), key=lambda k: float(np.dot(seg, b[max_lag + k:max_lag + k + n])))


def main() -> int:
    skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for snr_in, atten in ((0.0, None), (0.0, 12.0), (10.0, None)):
            src = os.path.join(tmp, f"noisy_{snr_in:g}_{atten}.wav")
            clean = voice_noise_fixture(src, seconds=3.0, snr_db=snr_in)
            out = os.path.join(tmp, f"voice_{snr_in:g}_{atten}.wav")
            t0 = time.monotonic()
            doc = isolate(IsolateRequest(src, out, atten_lim_db=atten), skill=skill)
            secs = time.monotonic() - t0
            rate, ch, stem = read_wav(out)
            rows.append(("isolate", f"voice+white noise {snr_in:g} dB, 3 s, 48 kHz mono" + (f", --atten-lim-db {atten:g}" if atten else ""),
                         f"SNR {snr_db(clean, read_wav(src)[2]):.1f} -> {snr_db(clean, stem):.1f} dB",
                         f"{rate} Hz / {ch} ch, {len(clean) - len(stem)} samples shorter, lag {lag(clean, stem, 160)}",
                         f"{secs:.2f} s", doc["engine"]["version"]))
        # The binary itself on a 16 kHz file (the skill never sends one: ADR-003 converts or refuses).
        src16 = os.path.join(tmp, "noisy16.wav")
        voice_noise_fixture(src16, seconds=2.0, rate=16000, snr_db=0.0)
        outdir = os.path.join(tmp, "df16")
        res = subprocess.run([find_binary(), "-D", "-m", os.environ[ENV_MODEL], "-o", outdir, src16], capture_output=True, text=True,
                             timeout=600, stdin=subprocess.DEVNULL)
        produced = os.path.join(outdir, "noisy16.wav")
        if res.returncode == 0 and os.path.isfile(produced):
            r16, c16, x16 = read_wav(produced)
            facts = f"exit 0, wrote {r16} Hz / {c16} ch, {len(x16)} of 32000 samples"
        else:
            facts = f"exit {res.returncode}: {(res.stderr or res.stdout).strip()[-200:]}"
        rows.append(("deep-filter (direct)", "voice+white noise 0 dB, 2 s, 16 kHz mono", "-", facts, "-", "-"))
        cases = (("1 ch", 16000, 1, None), ("1 ch", 48000, 1, None), ("2 ch dual mono", 48000, 2, None),
                 ("2 ch, R = L + noise at -40 dB", 48000, 2, -40.0), ("2 ch, a room per channel", 48000, 2, "distinct"))
        for label, rate, channels, variant in cases:
            src = os.path.join(tmp, f"room_{rate}_{channels}_{variant}.wav")
            dry = reverb_fixture(src, seconds=2.0, rate=rate, channels=channels, distinct=variant == "distinct")
            if isinstance(variant, float):
                _, (left, _) = read_wav_channels(src)
                rng = np.random.default_rng(1)
                right = np.asarray(left) + rng.standard_normal(len(left)) * np.std(left) * 10 ** (variant / 20)
                write_wav_tracks(src, [left, list(right)], rate)
            out = os.path.join(tmp, f"dry_{rate}_{channels}_{variant}.wav")
            t0 = time.monotonic()
            doc = dereverb(DereverbRequest(src, out), skill=skill)
            secs = time.monotonic() - t0
            wet, cleaned = read_wav_channels(src)[1], read_float_wav_channels(out)[1]
            srr = ", ".join(f"{srr_db(dry, a):.1f} -> {srr_db(dry, b):.1f}" for a, b in zip(wet, cleaned))
            rows.append(("dereverb", f"speech-like + two combs, 2 s, {rate // 1000} kHz, {label}", f"SRR per channel {srr} dB",
                         f"{len(cleaned)} ch, {len(dry) - len(cleaned[0])} samples shorter, lag {lag(dry, cleaned[0], 40)}, "
                         f"independent channels {doc['parameters']['independent_channels']}", f"{secs:.2f} s", doc["engine"]["version"]))
        # The cause ADR-007 works around: wpe_v8 itself on two identical channels (no regularisation in nara_wpe).
        from nara_wpe.wpe import wpe_v8
        rng = np.random.default_rng(2)
        a = rng.standard_normal((4, 1, 3000)) + 1j * rng.standard_normal((4, 1, 3000))
        for label, y in (("one channel", a), ("two identical channels", np.concatenate([a, a], axis=1))):
            with np.errstate(all="ignore"):
                z = wpe_v8(y, taps=10, delay=3, iterations=3, statistics_mode="full")
            ratio = float(np.nanmax(np.abs(z)) / np.max(np.abs(y))) if np.isfinite(z).any() else float("nan")
            rows.append(("nara_wpe.wpe_v8 (direct)", f"white complex STFT, 4 bins x 3000 frames, {label}", "-",
                         f"finite {bool(np.isfinite(z).all())}, output/input peak {ratio:.3g}", "-", "-"))
    print("| Op | Fixture | Measure | Output | Wall time | Engine |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
