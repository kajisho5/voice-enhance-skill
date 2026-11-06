"""Long inputs in blocks, resumable (roadmap step 11, ADR-009): the block plan, the RF64-capable WAV writer, the join
with its crossfade, ISOLATE in blocks with a fake deep-filter that stops halfway, and the "Done when" of the step: a
1 h DEREVERB fixture interrupted halfway resumes, and its output equals an uninterrupted run within -90 dB (fake
nara_wpe on NumPy; the integration and engines jobs run it, the unit jobs skip it without NumPy)."""
from __future__ import annotations

import math
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

from helpers import (crosstalk_fixture, energy_db, env, fake_deep_filter, fake_model, fake_nara_wpe, have_numpy, on_python_path, read_wav,
                     run_cli, write_wav)

from test_unit import StubSkill

from voice_enhance_skill import blocks
from voice_enhance_skill.contract import contract
from voice_enhance_skill.engines.wavio import WavReader
from voice_enhance_skill.errors import VoiceEnhanceError
from voice_enhance_skill.ops import BleedReduceRequest, DereverbRequest, IsolateRequest, bleed_reduce, dereverb, isolate

POSIX = os.name == "posix"
NUMPY = have_numpy()
NEEDS_NUMPY = "needs numpy: the fake nara_wpe computes on it (the integration job installs it)"


def write_s16(path: str, values, rate: int, channels: int = 1) -> str:
    """16-bit PCM from integers, fast (array, not struct per sample)."""
    a = array("h", values)
    if sys.byteorder == "big":
        a.byteswap()
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(a.tobytes())
    return path


def read_s16(path: str):
    r = WavReader(path)
    try:
        raw = blocks._read_raw(r, 0, r.frames)
    finally:
        r.close()
    a = array("h")
    a.frombytes(raw)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def signal(n: int, seed: int = 1):
    """A deterministic, non-repeating 16-bit test signal: two drifting tones plus a linear-congruential noise."""
    x, out = seed, []
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        v = 6000 * math.sin(0.0021 * i + 1e-9 * i * i) + 3000 * math.sin(0.031 * i) + (x % 2001 - 1000)
        out.append(int(v))
    return out


class PlanTests(unittest.TestCase):
    def test_short_input_is_one_block(self):
        self.assertEqual(blocks.plan(1000, 100, 600.0), [(0, 1000)])
        self.assertEqual(blocks.plan(60000, 100, 600.0), [(0, 60000)])

    def test_blocks_overlap_their_neighbours_only(self):
        for frames, rate, size in ((28_800_000, 8000, 600.0), (1_920_000, 48000, 15.0), (100_001, 100, 20.0), (8 * 3600 * 48000, 48000, 600.0)):
            plan = blocks.plan(frames, rate, size)
            overlap = int(round(blocks.OVERLAP_SECONDS * rate))
            self.assertGreater(len(plan), 1)
            self.assertEqual((plan[0][0], plan[-1][1]), (0, frames))
            for (a0, b0), (a1, b1) in zip(plan, plan[1:]):
                self.assertEqual(b0 - a1, overlap)                    # exactly one overlap between neighbours
            for (a0, b0), (a2, b2) in zip(plan, plan[2:]):
                self.assertLessEqual(b0, a2)                          # never three blocks at one point
            self.assertTrue(all(b - a <= int(round(size * rate)) for a, b in plan))
            self.assertGreater(plan[-1][1] - plan[-1][0], overlap)

    def test_eight_hours_in_ten_minute_blocks(self):
        self.assertEqual(len(blocks.plan(8 * 3600 * 48000, 48000)), 49)

    def test_block_seconds_limits(self):
        self.assertEqual(blocks.check_block_seconds(None), blocks.BLOCK_SECONDS)
        for bad in (14.9, 3601, True, "60", 0):
            with self.assertRaises(VoiceEnhanceError) as e:
                blocks.check_block_seconds(bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")


class WavTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_rf64_past_the_limit(self):
        """Above 4 GiB the joined file turns into RF64; WavReader reads the 64-bit size from ds64 (the limit is lowered
        here so the test writes 400 bytes, not 4 GiB)."""
        path = os.path.join(self.tmp, "big.wav")
        with mock.patch.object(blocks, "WAV_LIMIT", 300):
            w = blocks.RawWavWriter(path, 1, 16, 1, 8000, rf64=True)
            w.write_raw(struct.pack("<200h", *range(200)))
            w.close()
        head = Path(path).read_bytes()[:16]
        self.assertEqual((head[:4], head[12:16]), (b"RF64", b"ds64"))
        self.assertEqual(list(read_s16(path)), list(range(200)))
        small = os.path.join(self.tmp, "small.wav")
        w = blocks.RawWavWriter(small, 1, 16, 1, 8000, rf64=True)
        w.write_raw(struct.pack("<3h", 1, 2, 3))
        w.close()
        self.assertEqual(Path(small).read_bytes()[12:16], b"JUNK")    # the room for ds64, as ffmpeg -rf64 auto
        self.assertEqual(read_wav(small)[2], [1 / 32768, 2 / 32768, 3 / 32768])

    def test_oversized_plain_wav_is_refused(self):
        with mock.patch.object(blocks, "WAV_LIMIT", 100):
            w = blocks.RawWavWriter(os.path.join(self.tmp, "x.wav"), 1, 16, 1, 8000)
            w.write_raw(bytes(200))
            with self.assertRaises(VoiceEnhanceError):
                w.close()

    def test_codec_round_trip(self):
        vals = [0.0, 0.5, -0.25, -1.0, 0.999]
        for tag, bits in ((1, 16), (1, 24), (1, 32), (3, 32), (3, 64)):
            got = blocks.decode(blocks.encode(vals, tag, bits), tag, bits)
            for g, v in zip(got, vals):
                self.assertAlmostEqual(g, v, delta=1e-4, msg=(tag, bits))

    def test_join_of_exact_slices_is_the_signal(self):
        """Blocks that are exact slices of one signal (16-bit, stereo) join back to it sample for sample, also where a
        block ends short (deep-filter -D ends 1440 samples early)."""
        rate, n = 1000, 20000
        left, right = signal(n, 1), signal(n, 2)
        inter = [v for pair in zip(left, right) for v in pair]
        plan = blocks.plan(n, rate, 15.0)
        parts = []
        for k, (a, b) in enumerate(plan):
            b = b - 50 if k + 1 < len(plan) else b                   # a short block end is fine inside the overlap
            parts.append((a, write_s16(os.path.join(self.tmp, f"b{k}.wav"), inter[2 * a:2 * b], rate, 2)))
        out = os.path.join(self.tmp, "joined.wav")
        self.assertEqual(blocks.join(parts, out, "test"), n)
        self.assertEqual(list(read_s16(out)), inter)

    def test_crossfade_weights_sum_to_one(self):
        """Block 0 is all +1000, block 1 all -1000: the joined file is +1000 until the crossfade, then falls monotonically
        to -1000 over the middle half of the overlap."""
        rate = 1000
        plan = blocks.plan(40000, rate, 20.0)
        self.assertEqual(len(plan), 3)
        parts = [(a, write_s16(os.path.join(self.tmp, f"c{k}.wav"), [1000 if k % 2 == 0 else -1000] * (b - a), rate))
                 for k, (a, b) in enumerate(plan)]
        out = os.path.join(self.tmp, "joined.wav")
        blocks.join(parts, out, "test")
        y = list(read_s16(out))
        a, overlap = plan[1][0], int(blocks.OVERLAP_SECONDS * rate)
        g = overlap // 4
        self.assertTrue(all(v == 1000 for v in y[:a + g]))
        fade = y[a + g:a + overlap - g]
        self.assertTrue(all(p >= q for p, q in zip(fade, fade[1:])))
        self.assertTrue(all(v == -1000 for v in y[a + overlap - g:plan[2][0] + g]))
        self.assertLess(abs(fade[len(fade) // 2]), 10)

    def test_gap_between_blocks_is_invalid_result(self):
        rate = 1000
        p0 = write_s16(os.path.join(self.tmp, "g0.wav"), [1] * 5000, rate)
        p1 = write_s16(os.path.join(self.tmp, "g1.wav"), [1] * 5000, rate)
        with self.assertRaises(VoiceEnhanceError) as e:
            blocks.join([(0, p0), (6000, p1)], os.path.join(self.tmp, "j.wav"), "test")
        self.assertEqual(e.exception.code, "INVALID_RESULT")


@unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
class IsolateBlocksTests(unittest.TestCase):
    """ISOLATE on 40 s at 48 kHz in 15 s blocks (4 blocks) with the fake deep-filter in `gain` mode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.engine_dir = os.path.join(self.tmp, "engine")
        os.mkdir(self.engine_dir)
        self.binary = fake_deep_filter(self.engine_dir, mode="gain")
        self.model = fake_model(self.engine_dir)
        self.src = write_s16(os.path.join(self.tmp, "talk.wav"), signal(40 * 48000), 48000)

    def tearDown(self):
        self._tmp.cleanup()

    def run_isolate(self, out, **kw):
        with env(VOICE_ENHANCE_DEEP_FILTER=self.binary, VOICE_ENHANCE_DFN_MODEL=self.model):
            return isolate(IsolateRequest(self.src, out, block_seconds=15.0, **kw), skill=StubSkill())

    def test_blocks_join_to_the_whole_file_result(self):
        out = os.path.join(self.tmp, "voice.wav")
        doc = self.run_isolate(out)
        b = doc["blocks"]
        self.assertEqual((b["count"], b["processed"], b["cached"], b["resumed_from"]), (4, [0, 1, 2, 3], [], None))
        self.assertEqual((b["overlap_seconds"], b["block_seconds"]), (4.0, 15.0))
        x, y = read_s16(self.src), read_s16(out)
        self.assertEqual(len(y), len(x) - 1440)                      # the last block ends early, like the whole file would
        self.assertEqual(list(y), [v // 2 for v in x[:len(y)]])       # every seam is invisible: the blocks agree there
        t = doc["timing"]
        self.assertEqual((t["engine"], t["audio_seconds"]), ("deepfilternet", round(sum(e - s for s, e in blocks.plan(len(x), 48000, 15.0)) / 48000, 3)))
        self.assertGreater(t["realtime_factor"], 0)
        self.assertEqual(t["engine_seconds"], t["engine_seconds_this_run"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, blocks.CACHE_NAME)))          # emptied on success
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".voice-enhance-")], [])

    def test_a_stopped_run_resumes_at_the_first_missing_block(self):
        ref = os.path.join(self.tmp, "ref.wav")
        self.run_isolate(ref)
        Path(self.engine_dir, "stop_after").write_text("2", encoding="utf-8")
        out = os.path.join(self.tmp, "voice.wav")
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_isolate(out)
        self.assertEqual(e.exception.code, "ENGINE_FAILED")
        self.assertFalse(os.path.exists(out))
        cache = os.path.join(self.tmp, blocks.CACHE_NAME)
        self.assertEqual(len(os.listdir(cache)), 2)                   # blocks 0 and 1 survived the stop
        os.remove(os.path.join(self.engine_dir, "stop_after"))
        doc = self.run_isolate(out)
        b = doc["blocks"]
        self.assertEqual((b["cached"], b["processed"], b["resumed_from"]), ([0, 1], [2, 3], 2))
        self.assertEqual(Path(out).read_bytes(), Path(ref).read_bytes())
        self.assertGreater(doc["timing"]["engine_seconds"], doc["timing"]["engine_seconds_this_run"])
        self.assertFalse(os.path.exists(cache))

    def test_other_settings_do_not_reuse_the_cache(self):
        Path(self.engine_dir, "stop_after").write_text("1", encoding="utf-8")
        cache = os.path.join(self.tmp, "my-cache")
        with self.assertRaises(VoiceEnhanceError):
            self.run_isolate(os.path.join(self.tmp, "a.wav"), cache_dir=cache)
        self.assertEqual(len(os.listdir(cache)), 1)
        os.remove(os.path.join(self.engine_dir, "stop_after"))
        doc = self.run_isolate(os.path.join(self.tmp, "a.wav"), cache_dir=cache, atten_lim_db=12.0)
        self.assertEqual(doc["blocks"]["cached"], [])                 # another parameter, another key
        self.assertEqual(len(os.listdir(cache)), 1)                   # the earlier run's block is left alone

    def test_damaged_cache_entry_is_recomputed(self):
        Path(self.engine_dir, "stop_after").write_text("1", encoding="utf-8")
        with self.assertRaises(VoiceEnhanceError):
            self.run_isolate(os.path.join(self.tmp, "a.wav"))
        cache = os.path.join(self.tmp, blocks.CACHE_NAME)
        (entry,) = os.listdir(cache)
        os.remove(os.path.join(cache, entry, "out0.wav"))
        os.remove(os.path.join(self.engine_dir, "stop_after"))
        doc = self.run_isolate(os.path.join(self.tmp, "a.wav"))
        self.assertEqual((doc["blocks"]["cached"], doc["blocks"]["resumed_from"]), ([], None))

    def test_cli_flags(self):
        out = os.path.join(self.tmp, "voice.wav")
        with env(VOICE_ENHANCE_DEEP_FILTER=self.binary, VOICE_ENHANCE_DFN_MODEL=self.model), \
                mock.patch("voice_enhance_skill.ops.FfmpegSkill.locate", return_value=StubSkill()):
            rc, doc = run_cli("isolate", self.src, "-o", out, "--block-seconds", "20", "--json")
            rc2, err = run_cli("isolate", self.src, "-o", out, "--overwrite", "--block-seconds", "5", "--json")
        self.assertEqual((rc, doc["blocks"]["count"]), (0, 3))
        self.assertEqual((rc2, err["error"]["code"]), (2, "INVALID_ARGUMENT"))

    def test_short_input_is_one_block_as_before(self):
        src = write_wav(os.path.join(self.tmp, "short.wav"), [0.1 * math.sin(i / 7) for i in range(48000)])
        out = os.path.join(self.tmp, "voice.wav")
        with env(VOICE_ENHANCE_DEEP_FILTER=self.binary, VOICE_ENHANCE_DFN_MODEL=self.model):
            doc = isolate(IsolateRequest(src, out), skill=StubSkill())
        self.assertEqual(doc["blocks"], {"count": 1, "block_seconds": 600.0, "overlap_seconds": 0.0, "crossfade": None,
                                         "starts": [0.0], "processed": [0], "cached": [], "resumed_from": None, "cache_dir": None})
        self.assertEqual(Path(out).read_bytes()[:4], b"RIFF")
        self.assertEqual(Path(out).read_bytes()[12:16], b"fmt ")       # the engine's own file, not re-joined


@unittest.skipUnless(NUMPY, "needs numpy: the crosstalk engine computes on it (the integration job installs it)")
class BleedReduceBlocksTests(unittest.TestCase):
    """BLEED_REDUCE in blocks: every track is cut at the same frames, each block measures its own coupling, and every
    output keeps its own track's length."""

    def test_two_tracks_of_different_lengths_in_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = crosstalk_fixture(seconds=40.0, rate=8000)
            mic0, mic1 = fx["mics"][0], fx["mics"][1][:37 * 8000]          # the second recorder stopped 3 s early
            srcs = [write_wav(os.path.join(tmp, "a.wav"), mic0, 8000), write_wav(os.path.join(tmp, "b.wav"), mic1, 8000)]
            out = os.path.join(tmp, "clean")
            doc = bleed_reduce(BleedReduceRequest(srcs, out, block_seconds=15.0), skill=StubSkill())
            self.assertEqual(doc["blocks"]["count"], 4)
            coupling = doc["parameters"]["coupling_db"]
            self.assertTrue(all(isinstance(coupling[i][j], float) for i, j in ((0, 1), (1, 0))), coupling)
            for k, (x, name) in enumerate(((mic0, "a_a0.wav"), (mic1, "b_a0.wav"))):
                rate, ch, y = read_wav(os.path.join(out, name))
                self.assertEqual((rate, ch, len(y)), (8000, 1, len(x)))
                region = (fx["other"][k][0], min(fx["other"][k][1], 36.0))
                self.assertLess(energy_db(y, rate, region), energy_db(x, rate, region) - 10.0)
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith(".voice-enhance-")], [])


class ContractBlocksTests(unittest.TestCase):
    def test_flags_and_result_keys(self):
        c = contract()
        for tool in c["tools"][:3]:
            self.assertIn("[--block-seconds 600]", tool["argv"])
            self.assertIn("[--cache-dir <dir>]", tool["argv"])
            self.assertTrue({"blocks", "timing"} <= set(tool["result_keys"]))
        self.assertEqual(c["long_inputs"]["block_seconds_range"], [15.0, 3600.0])


class FastWavStub(StubSkill):
    """StubSkill for long files: probe reads the header only and audio_to_wav copies a mono file as it is."""

    def probe(self, path):
        try:
            r = WavReader(path)
        except VoiceEnhanceError as e:
            raise VoiceEnhanceError("UNSUPPORTED_MEDIA", e.message)
        try:
            a = {"codec": "pcm_s16le" if r.tag == 1 else "pcm_f32le", "sample_rate": r.rate, "channels": r.channels}
            return {"format": "wav", "duration": r.frames / r.rate, "video": None, "audio": a, "audio_streams": [dict(a, index=0)]}
        finally:
            r.close()

    def audio_to_wav(self, src, dst, audio_stream=None, mono=False, sample_rate=None, timeout=None):
        self.calls.append({"src": os.path.basename(src), "audio_stream": audio_stream, "mono": mono, "sample_rate": sample_rate})
        assert sample_rate is None and self.probe(src)["audio"]["channels"] == 1
        shutil.copyfile(src, dst)
        return {"status": "completed", "output": dst, "probe": self.probe(dst)}


@unittest.skipUnless(NUMPY, NEEDS_NUMPY)
class OneHourResumeTests(unittest.TestCase):
    """Roadmap step 11, Done when: a 1 h fixture interrupted halfway resumes, and its output equals an uninterrupted
    run within -90 dB. DEREVERB at 8 kHz mono (57.6 MB of input), 600 s blocks (7 blocks), the fake nara_wpe really
    filtering; the first run stops after 3 blocks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def fixture(self, path, rate=8000, seconds=3600):
        """Speech-like syllables through two feedback combs (the reverberant fixture of ADR-004, built on NumPy here),
        with a slow level drift so no two blocks are alike."""
        import numpy as np
        rng = np.random.default_rng(11)
        n = rate * seconds
        t = np.arange(n) / rate
        syll = np.maximum(0.0, np.sin(2 * np.pi * 3.1 * t + 2 * np.sin(2 * np.pi * 0.07 * t)))
        f0 = 140 + 60 * np.sin(2 * np.pi * 0.013 * t)
        phase = 2 * np.pi * np.cumsum(f0) / rate
        dry = syll * sum(np.sin(h * phase) / h for h in range(1, 6)) + 0.05 * rng.standard_normal(n)
        wet = dry.copy()
        for delay, gain in ((0.037, 0.6), (0.059, 0.5)):           # feedback comb: run as a block recursion on NumPy
            d = int(delay * rate)
            for a in range(d, n, d):
                wet[a:a + d] += gain * wet[a - d:a - d + min(d, n - a)]
        wet *= (0.6 + 0.3 * np.sin(2 * np.pi * t / 1700.0)) / np.max(np.abs(wet))
        pcm = np.clip(np.round(wet * 32767), -32768, 32767).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())

    def read(self, path):
        import numpy as np
        r = WavReader(path)
        try:
            return r.read(np, 0, r.frames)[:, 0]
        finally:
            r.close()

    def test_interrupted_halfway_resumes_and_equals_an_uninterrupted_run(self):
        import numpy as np
        src = os.path.join(self.tmp, "day.wav")
        self.fixture(src)
        site = fake_nara_wpe(os.path.join(self.tmp, "engine"))
        ref, out = os.path.join(self.tmp, "ref.wav"), os.path.join(self.tmp, "out.wav")
        with on_python_path(site):
            t0 = time.monotonic()
            whole = dereverb(DereverbRequest(src, ref), skill=FastWavStub())
            elapsed = time.monotonic() - t0
            self.assertEqual((whole["blocks"]["count"], whole["blocks"]["cached"]), (7, []))
            Path(site, "stop_after").write_text("3", encoding="utf-8")
            with self.assertRaises(VoiceEnhanceError) as e:
                dereverb(DereverbRequest(src, out), skill=FastWavStub())
            self.assertEqual(e.exception.code, "ENGINE_FAILED")
            self.assertFalse(os.path.exists(out))
            self.assertEqual(len(os.listdir(os.path.join(self.tmp, blocks.CACHE_NAME))), 3)
            os.remove(os.path.join(site, "stop_after"))
            resumed = dereverb(DereverbRequest(src, out), skill=FastWavStub())
        b = resumed["blocks"]
        self.assertEqual((b["cached"], b["processed"], b["resumed_from"]), ([0, 1, 2], [3, 4, 5, 6], 3))
        x, y, z = self.read(src), self.read(ref), self.read(out)
        self.assertEqual((len(y), len(z)), (len(x), len(x)))
        residual = 10 * math.log10(max(float(np.sum((z - y) ** 2)), 1e-30) / float(np.sum(y ** 2)))
        self.assertLess(residual, -90.0, residual)
        self.assertGreater(float(np.sum((y - x) ** 2)) / float(np.sum(x ** 2)), 1e-3)     # the engine did change the audio
        t = resumed["timing"]
        self.assertEqual(t["engine"], "nara_wpe")
        self.assertTrue(0 < t["realtime_factor"] < 1, t)              # far faster than real time, even the fake
        self.assertLess(t["engine_seconds_this_run"], t["engine_seconds"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, blocks.CACHE_NAME)))
        sys.stderr.write(f"\n  1 h fixture: uninterrupted run {elapsed:.1f}s, realtime factor {whole['timing']['realtime_factor']}, "
                         f"resumed-vs-uninterrupted residual {residual:.1f} dB ")


if __name__ == "__main__":
    unittest.main()
