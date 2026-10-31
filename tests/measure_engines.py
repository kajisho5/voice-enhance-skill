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
from helpers import FFMPEG_SKILL_DIR, read_wav, reverb_fixture, snr_db, srr_db, voice_noise_fixture  # noqa: E402

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
        for rate, channels in ((16000, 1), (48000, 1), (48000, 2)):
            src = os.path.join(tmp, f"room_{rate}_{channels}.wav")
            dry = reverb_fixture(src, seconds=2.0, rate=rate, channels=channels)
            out = os.path.join(tmp, f"dry_{rate}_{channels}.wav")
            t0 = time.monotonic()
            doc = dereverb(DereverbRequest(src, out), skill=skill)
            secs = time.monotonic() - t0
            r, ch, x = read_wav(out)
            rows.append(("dereverb", f"speech-like + two combs, 2 s, {rate // 1000} kHz, {channels} ch",
                         f"SRR {srr_db(dry, read_wav(src)[2]):.1f} -> {srr_db(dry, x):.1f} dB",
                         f"{r} Hz / {ch} ch, {len(dry) - len(x)} samples shorter, lag {lag(dry, x, 40)}",
                         f"{secs:.2f} s", doc["engine"]["version"]))
    print("| Op | Fixture | Measure | Output | Wall time | Engine |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
