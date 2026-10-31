"""Integration with a real ffmpeg-skill checkout and a real FFmpeg: locate it, check every tool and flag this skill
emits against its live contract, run its doctor, probe a WAV through it, and run ISOLATE, DEREVERB and BLEED_REDUCE end
to end on a video with two audio streams (fake deep-filter, fake nara_wpe on NumPy, the crosstalk engine on NumPy; the
real ISOLATE / DEREVERB engines only when installed), and REMUX a stem back into a video (Matroska / FFV1 and MP4 /
MPEG-4 Part 2). Skipped without a checkout (VOICE_ENHANCE_FFMPEG_SKILL_DIR or ../ffmpeg-skill); CI's integration job
always has one. FFmpeg is called directly here only to build the fixtures, and ffprobe only to hash the video packets
REMUX must leave byte-identical; the skill itself never calls either."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from helpers import (FFMPEG_SKILL_DIR, best_lag, crosstalk_fixture, energy_db, env, fake_deep_filter, fake_model, fake_nara_wpe,
                     fake_wpe_calls, have_numpy, on_python_path, read_float_wav_channels, read_wav, read_wav_channels, reverb_fixture,
                     run_cli, snr_db, srr_db, voice_noise_fixture, write_tone_wav, write_wav)

from voice_enhance_skill.doctor import doctor
from voice_enhance_skill.engines import get_engine
from voice_enhance_skill.engines.deepfilternet import build_argv, find_binary
from voice_enhance_skill.errors import VoiceEnhanceError
from voice_enhance_skill.ffmpeg_skill import TOOLS_USED, FfmpegSkill
from voice_enhance_skill.ops import (BleedReduceRequest, DereverbRequest, IsolateRequest, RemuxRequest, bleed_reduce, dereverb, isolate,
                                     remux)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
NO_DEEP_FILTER = {"VOICE_ENHANCE_DEEP_FILTER": os.path.join(tempfile.gettempdir(), "no-such-deep-filter")}
POSIX = os.name == "posix"
NO_REAL_DEEP_FILTER = "deep-filter and DeepFilterNet3_onnx.tar.gz are not installed (only CI's engines job installs them, ADR-007)"
# What the real engines must at least gain on the synthetic fixtures (ADR-007 records what CI measured).
REAL_ISOLATE_MIN_GAIN_DB = 0.0
REAL_DEREVERB_MIN_GAIN_DB = 0.0


def real_engine_ok() -> bool:
    return get_engine("deepfilternet").inspect()["status"] == "OK"


def real_wpe_ok() -> bool:
    return get_engine("nara_wpe").inspect()["status"] == "OK"


def two_stream_video(directory: str, voice_wav: str) -> str:
    """A small video (FFV1) whose audio stream 0 is a 48 kHz mono decoy tone and stream 1 the stereo voice+noise WAV,
    both PCM so the fixture is not altered by a lossy codec."""
    decoy = write_tone_wav(os.path.join(directory, "decoy.wav"), seconds=1.0, freq=3000.0)
    out = os.path.join(directory, "talk.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=25:d=1",
                    "-i", decoy, "-i", voice_wav, "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "ffv1",
                    "-c:a", "pcm_s16le", "-shortest", out], check=True, stdin=subprocess.DEVNULL)
    return out


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class FfmpegSkillIntegrationTests(unittest.TestCase):
    def test_live_contract_declares_every_tool_and_flag(self):
        info = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).info()
        self.assertTrue(info.supported, info.problems)
        self.assertEqual(sorted(t for t in info.tools if t in TOOLS_USED), sorted(TOOLS_USED))
        self.assertNotEqual(info.version, "unknown")

    def test_probe_through_ffmpeg_skill(self):
        skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
        with tempfile.TemporaryDirectory() as tmp:
            wav = write_tone_wav(os.path.join(tmp, "tone.wav"), seconds=1.0, rate=48000)
            facts = skill.probe(wav)
            self.assertAlmostEqual(float(facts["duration"]), 1.0, delta=0.05)
            not_media = os.path.join(tmp, "not-media.wav")
            with open(not_media, "w", encoding="utf-8") as fh:
                fh.write("not audio")
            with self.assertRaises(VoiceEnhanceError) as e:
                skill.probe(not_media)
            self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")

    def test_doctor_with_real_ffmpeg_skill_reports_missing_engines(self):
        with env(**NO_DEEP_FILTER):
            d = doctor(str(FFMPEG_SKILL_DIR))
            rc, cli_doc = run_cli("doctor", "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "OK", d["problems"])
        self.assertEqual((d["checks"]["ffmpeg"]["status"], d["checks"]["ffprobe"]["status"]), ("OK", "OK"))
        self.assertEqual(d["checks"]["engines"]["deepfilternet"]["status"], "MISSING")
        self.assertIn(d["status"], ("degraded", "ok"))
        self.assertEqual((rc, cli_doc["status"]), (0, d["status"]))


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class IsolateIntegrationTests(unittest.TestCase):
    """ISOLATE through the real ffmpeg-skill (probe, audio) and the real worker."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def tearDown(self):
        self._tmp.cleanup()

    def fake_engine(self, mode="lowpass"):
        engine_dir = os.path.join(self.tmp, "engine")
        os.makedirs(engine_dir, exist_ok=True)
        return env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(engine_dir, mode=mode), VOICE_ENHANCE_DFN_MODEL=fake_model(engine_dir))

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_second_stream_of_a_video_snr_rises(self):
        voice = os.path.join(self.tmp, "voice_noise.wav")
        clean = voice_noise_fixture(voice, seconds=1.0, channels=2)
        video = two_stream_video(self.tmp, voice)
        out = os.path.join(self.tmp, "voice.wav")
        with self.fake_engine():
            doc = isolate(IsolateRequest(video, out, audio_stream=1), skill=self.skill)
        rate, ch, stem = read_wav(out)
        self.assertEqual((rate, ch), (48000, 1))
        self.assertEqual((doc["input"]["has_video"], doc["input"]["audio_stream_count"], doc["input"]["audio"]["channels"]), (True, 2, 2))
        # stream 1 has 2 channels and stream 0 has 1: the fold-down is a second pass on the extracted WAV (ADR-003)
        self.assertEqual([st["flags"] for st in doc["steps"]], [["--audio-stream", "1"], ["--mono"]])
        self.assertEqual(doc["steps"][-1]["result"], {"sample_rate": 48000, "channels": 1})
        before, after = snr_db(clean, read_wav(voice)[2]), snr_db(clean, stem)
        self.assertGreater(after, before + 6.0, (before, after))     # and the decoy (stream 0) was not the one cleaned
        self.assertAlmostEqual(doc["output"]["duration"], 1.0, delta=0.05)
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], self.skill.info().version)

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_other_sample_rate(self):
        src = write_tone_wav(os.path.join(self.tmp, "cd.wav"), seconds=0.5, rate=44100, freq=300.0)
        out = os.path.join(self.tmp, "voice.wav")
        with self.fake_engine():
            if self.skill.info().takes_flag("audio", "sample_rate"):
                doc = isolate(IsolateRequest(src, out), skill=self.skill)
                self.assertEqual((doc["output"]["sample_rate"], doc["steps"][-1]["flags"]), (48000, ["--sample-rate", "48000"]))
            else:
                with self.assertRaises(VoiceEnhanceError) as e:
                    isolate(IsolateRequest(src, out), skill=self.skill)
                self.assertEqual((e.exception.code, e.exception.details["input_sample_rate"]), ("FFMPEG_SKILL_UNAVAILABLE", 44100))
                self.assertFalse(os.path.exists(out))

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_cli_against_real_ffmpeg_skill(self):
        src = os.path.join(self.tmp, "noisy.wav")
        voice_noise_fixture(src, seconds=0.5)
        out = os.path.join(self.tmp, "voice.wav")
        with self.fake_engine():
            rc, doc = run_cli("isolate", src, "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
            rc2, err = run_cli("isolate", src, "-o", out, "--audio-stream", "3", "--overwrite", "--ffmpeg-skill", str(FFMPEG_SKILL_DIR),
                               "--json")
        self.assertEqual((rc, doc["ok"], doc["output"]["sample_rate"]), (0, True, 48000))
        self.assertEqual((rc2, err["error"]["code"]), (2, "INVALID_ARGUMENT"))

    def test_audio_tool_refuses_non_media(self):
        bad = os.path.join(self.tmp, "not-media.wav")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("not audio")
        with self.assertRaises(VoiceEnhanceError) as e:
            self.skill.audio_to_wav(bad, os.path.join(self.tmp, "x.wav"))
        self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")

    @unittest.skipUnless(real_engine_ok(), NO_REAL_DEEP_FILTER)
    def test_real_deep_filter_cli_flags(self):
        # Every flag of ADR-003's argv is one the installed binary declares (ADR-007 checked them on 0.5.6). Comments,
        # not docstrings, in the real-engine tests: CI's engines job greps for their names in the verbose log.
        res = subprocess.run([find_binary(), "--help"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
        self.assertEqual(res.returncode, 0, res.stderr)
        declared = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", res.stdout))
        used = [a for a in build_argv("deep-filter", "<model>", "<out_dir>", "<input.wav>", 12.0) if a.startswith("-")]
        self.assertEqual(used, ["-D", "-m", "-o", "--atten-lim-db"])
        self.assertEqual([a for a in used if a not in declared], [], res.stdout)

    @unittest.skipUnless(real_engine_ok(), NO_REAL_DEEP_FILTER)
    def test_real_deep_filter_raises_snr(self):
        src = os.path.join(self.tmp, "noisy.wav")
        clean = voice_noise_fixture(src, seconds=3.0, snr_db=0.0)
        out = os.path.join(self.tmp, "voice.wav")
        doc = isolate(IsolateRequest(src, out), skill=self.skill)
        self.assertEqual(doc["engine"]["model"], "DeepFilterNet3_onnx.tar.gz")
        before, after = snr_db(clean, read_wav(src)[2]), snr_db(clean, read_wav(out)[2])
        self.assertGreater(after, before + REAL_ISOLATE_MIN_GAIN_DB, (before, after))

    @unittest.skipUnless(real_engine_ok(), NO_REAL_DEEP_FILTER)
    def test_real_deep_filter_is_sample_aligned(self):
        # `-D` removes the STFT and lookahead delay: the stem lines up with the clean voice at lag 0, and it ends at
        # most the delay plus one hop (1440 + 480 samples at 48 kHz) before the input. The search stays within half
        # the fixture's 140 Hz period (343 samples), where the correlation has a single peak; an uncompensated delay
        # of 1440 samples would show as a lag of 68.
        src = os.path.join(self.tmp, "noisy.wav")
        clean = voice_noise_fixture(src, seconds=2.0, snr_db=10.0)
        out = os.path.join(self.tmp, "voice.wav")
        isolate(IsolateRequest(src, out), skill=self.skill)
        rate, _, stem = read_wav(out)
        self.assertEqual(rate, 48000)
        self.assertEqual(best_lag(clean, stem, start=48000, length=12000, max_lag=160), 0)
        self.assertLessEqual(len(clean) - len(stem), 1920)
        self.assertGreaterEqual(len(clean) - len(stem), 0)


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class DereverbIntegrationTests(unittest.TestCase):
    """DEREVERB through the real ffmpeg-skill (probe, audio) and the real worker."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def tearDown(self):
        self._tmp.cleanup()

    @unittest.skipUnless(have_numpy(), "needs numpy: the fake nara_wpe computes on it")
    def test_second_stream_of_a_video_srr_rises(self):
        room = os.path.join(self.tmp, "room.wav")
        dry = reverb_fixture(room, seconds=1.0, rate=48000, channels=2, distinct=True)
        video = two_stream_video(self.tmp, room)
        out = os.path.join(self.tmp, "dry.wav")
        site = fake_nara_wpe(os.path.join(self.tmp, "engine"))
        with on_python_path(site):
            doc = dereverb(DereverbRequest(video, out, audio_stream=1), skill=self.skill)
        rate, ch, x = read_wav(out)
        self.assertEqual((rate, ch), (48000, 2))            # the stream's own layout: no fold-down for WPE (ADR-004)
        self.assertEqual((doc["output"]["codec"], doc["output"]["channels"]), ("pcm_f32le", 2))
        self.assertEqual((doc["input"]["has_video"], doc["input"]["audio_stream_count"]), (True, 2))
        self.assertEqual([st["flags"] for st in doc["steps"]], [["--audio-stream", "1"]])
        self.assertEqual(fake_wpe_calls(site)[0]["shape"][:2], [769, 2])
        self.assertEqual((doc["parameters"]["independent_channels"], doc["warnings"]), (2, []))
        for k, (wet, out_ch) in enumerate(zip(read_wav_channels(room)[1], read_float_wav_channels(out)[1])):
            before, after = srr_db(dry, wet), srr_db(dry, out_ch)
            self.assertGreater(after, before + 6.0, (k, before, after))     # and the decoy (stream 0) was not the one processed
        self.assertAlmostEqual(doc["output"]["duration"], 1.0, delta=0.05)
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], self.skill.info().version)

    @unittest.skipUnless(have_numpy(), "needs numpy: the fake nara_wpe computes on it")
    def test_other_sample_rate_needs_no_conversion(self):
        src = os.path.join(self.tmp, "cd.wav")
        reverb_fixture(src, seconds=0.5, rate=44100)
        out = os.path.join(self.tmp, "dry.wav")
        with on_python_path(fake_nara_wpe(os.path.join(self.tmp, "engine"))):
            rc, doc = run_cli("dereverb", src, "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual((rc, doc["ok"], doc["output"]["sample_rate"], doc["output"]["channels"]), (0, True, 44100, 1), doc)
        self.assertAlmostEqual(doc["output"]["duration"], 0.5, delta=0.02)

    @unittest.skipUnless(real_wpe_ok(), "nara_wpe is not installed (only CI's engines job installs it, ADR-007)")
    def test_real_nara_wpe_raises_srr(self):
        src = os.path.join(self.tmp, "room.wav")
        dry = reverb_fixture(src, seconds=2.0, rate=16000)
        out = os.path.join(self.tmp, "dry.wav")
        doc = dereverb(DereverbRequest(src, out), skill=self.skill)
        self.assertEqual(doc["engine"]["id"], "nara_wpe")
        before, after = srr_db(dry, read_wav(src)[2]), srr_db(dry, read_wav(out)[2])
        self.assertGreater(after, before + REAL_DEREVERB_MIN_GAIN_DB, (before, after))

    @unittest.skipUnless(real_wpe_ok(), "nara_wpe is not installed (only CI's engines job installs it, ADR-007)")
    def test_real_nara_wpe_dual_mono(self):
        # ADR-007: on two identical channels the real wpe_v8 diverged (SRR -0.8 -> -48 dB in CI) before the engine
        # learnt to run WPE on the independent components only.
        src = os.path.join(self.tmp, "dual.wav")
        dry = reverb_fixture(src, seconds=2.0, rate=48000, channels=2)
        out = os.path.join(self.tmp, "dry.wav")
        doc = dereverb(DereverbRequest(src, out), skill=self.skill)
        self.assertEqual((doc["parameters"]["independent_channels"], len(doc["warnings"])), (1, 1))
        _, (left, right) = read_float_wav_channels(out)
        self.assertLess(max(abs(a - b) for a, b in zip(left, right)), 1e-6)
        before, after = srr_db(dry, read_wav(src)[2]), srr_db(dry, left)
        self.assertGreater(after, before + REAL_DEREVERB_MIN_GAIN_DB, (before, after))

    @unittest.skipUnless(real_wpe_ok(), "nara_wpe is not installed (only CI's engines job installs it, ADR-007)")
    def test_real_nara_wpe_distinct_stereo(self):
        src = os.path.join(self.tmp, "pair.wav")
        dry = reverb_fixture(src, seconds=2.0, rate=48000, channels=2, distinct=True)
        out = os.path.join(self.tmp, "dry.wav")
        doc = dereverb(DereverbRequest(src, out), skill=self.skill)
        self.assertEqual((doc["parameters"]["independent_channels"], doc["warnings"]), (2, []))
        for k, (wet, out_ch) in enumerate(zip(read_wav_channels(src)[1], read_float_wav_channels(out)[1])):
            before, after = srr_db(dry, wet), srr_db(dry, out_ch)
            self.assertGreater(after, before + REAL_DEREVERB_MIN_GAIN_DB, (k, before, after))


def mic_video(directory: str, wavs: list) -> str:
    """A small video (FFV1) whose audio streams are the given WAVs in order (PCM, so the fixture is not altered)."""
    out = os.path.join(directory, "panel.mkv")
    argv = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=25:d=2"]
    for w in wavs:
        argv += ["-i", w]
    argv += ["-map", "0:v"] + [x for k in range(len(wavs)) for x in ("-map", f"{k + 1}:a")]
    subprocess.run(argv + ["-c:v", "ffv1", "-c:a", "pcm_s16le", "-shortest", out], check=True, stdin=subprocess.DEVNULL)
    return out


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class BleedReduceIntegrationTests(unittest.TestCase):
    """BLEED_REDUCE through the real ffmpeg-skill (probe, audio) and the crosstalk engine in the real worker."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
        self.out = os.path.join(self.tmp, "clean")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_cross_talk_falls(self, fx, outs):
        for k, out in enumerate(outs):
            rate, ch, y = read_wav(out)
            x = fx["mics"][k]
            self.assertEqual((rate, ch), (fx["rate"], 1))
            self.assertAlmostEqual(len(y), len(x), delta=fx["rate"] // 100)
            before, after = energy_db(x, rate, fx["other"][k]), energy_db(y, rate, fx["other"][k])
            self.assertLess(after, before - 10.0, (k, before, after))
            self.assertAlmostEqual(energy_db(y, rate, fx["own"][k]), energy_db(x, rate, fx["own"][k]), delta=1.0)

    @unittest.skipUnless(have_numpy(), "needs numpy: the crosstalk engine computes on it")
    def test_two_mic_streams_of_a_video_cross_talk_falls(self):
        fx = crosstalk_fixture(seconds=2.0, rate=48000)
        wavs = [write_wav(os.path.join(self.tmp, f"mic{k}.wav"), m, 48000) for k, m in enumerate(fx["mics"])]
        video = mic_video(self.tmp, wavs)
        doc = bleed_reduce(BleedReduceRequest([video], self.out), skill=self.skill)
        outs = [o["path"] for o in doc["outputs"]]
        self.assertEqual(outs, [os.path.join(self.out, "panel_a0.wav"), os.path.join(self.out, "panel_a1.wav")])
        self.assertEqual([(i["has_video"], i["audio_stream"], i["audio_stream_count"]) for i in doc["inputs"]], [(True, 0, 2), (True, 1, 2)])
        self.assertEqual([(st["track"], st["flags"]) for st in doc["steps"]], [(0, ["--audio-stream", "0"]), (1, ["--audio-stream", "1"])])
        self.assertEqual([o["codec"] for o in doc["outputs"]], ["pcm_f32le", "pcm_f32le"])
        self.assert_cross_talk_falls(fx, outs)
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], self.skill.info().version)

    @unittest.skipUnless(have_numpy(), "needs numpy: the crosstalk engine computes on it")
    def test_cli_two_files_with_a_stereo_track(self):
        fx = crosstalk_fixture(seconds=1.5, rate=48000)
        a = write_wav(os.path.join(self.tmp, "lav_a.wav"), fx["mics"][0], 48000)
        b = write_wav(os.path.join(self.tmp, "lav_b.wav"), fx["mics"][1], 48000, channels=2)
        rc, doc = run_cli("bleed-reduce", a, b, "-o", self.out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual((rc, doc["ok"]), (0, True), doc)
        self.assertEqual([(st["track"], st["flags"]) for st in doc["steps"]],
                         [(0, ["--audio-stream", "0"]), (1, ["--audio-stream", "0"]), (1, ["--mono"])])
        self.assert_cross_talk_falls(fx, [o["path"] for o in doc["outputs"]])

    @unittest.skipUnless(have_numpy(), "needs numpy: the crosstalk engine is checked before the rates")
    def test_different_sample_rates(self):
        a = write_tone_wav(os.path.join(self.tmp, "a.wav"), seconds=0.5, rate=48000)
        b = write_tone_wav(os.path.join(self.tmp, "b.wav"), seconds=0.5, rate=44100, freq=300.0)
        if self.skill.info().takes_flag("audio", "sample_rate"):
            doc = bleed_reduce(BleedReduceRequest([a, b], self.out), skill=self.skill)
            self.assertEqual(doc["steps"][-1]["flags"], ["--sample-rate", "48000"])
        else:
            with self.assertRaises(VoiceEnhanceError) as e:
                bleed_reduce(BleedReduceRequest([a, b], self.out), skill=self.skill)
            self.assertEqual((e.exception.code, e.exception.details["sample_rates"]), ("FFMPEG_SKILL_UNAVAILABLE", [48000, 44100]))
            self.assertFalse(os.path.exists(self.out))


def video_packets(path: str) -> list:
    """(pts, MD5 of the data) of every packet of the first video stream: equal lists mean the picture is byte-identical."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,data_hash",
                          "-show_data_hash", "MD5", "-of", "csv=p=0", path], check=True, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def stream_types(path: str) -> list:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path], check=True,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class RemuxIntegrationTests(unittest.TestCase):
    """REMUX through the real ffmpeg-skill (probe, audio --replace): roadmap step 6 is done when the video stream is
    byte-identical."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_leftovers(self):
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".voice-enhance-")], [])

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_isolate_stem_back_into_a_two_stream_video(self):
        voice = os.path.join(self.tmp, "voice_noise.wav")
        voice_noise_fixture(voice, seconds=1.0, channels=2)
        video = two_stream_video(self.tmp, voice)
        stem, out = os.path.join(self.tmp, "voice.wav"), os.path.join(self.tmp, "talk_clean.mkv")
        engine_dir = os.path.join(self.tmp, "engine")
        os.mkdir(engine_dir)
        with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(engine_dir, mode="lowpass"), VOICE_ENHANCE_DFN_MODEL=fake_model(engine_dir)):
            isolate(IsolateRequest(video, stem, audio_stream=1), skill=self.skill)
        # two audio streams: ffmpeg-skill/audio --replace cannot keep the other one, so it is refused unless dropped
        with self.assertRaises(VoiceEnhanceError) as e:
            remux(RemuxRequest(video, stem, out), skill=self.skill)
        self.assertEqual((e.exception.code, e.exception.details["audio_stream_count"]), ("FFMPEG_SKILL_UNAVAILABLE", 2))
        self.assertFalse(os.path.exists(out))
        doc = remux(RemuxRequest(video, stem, out, drop_other_audio=True), skill=self.skill)
        self.assertEqual(video_packets(out), video_packets(video))                 # the picture is byte-identical
        self.assertEqual(sorted(stream_types(out)), ["audio", "video"])
        self.assertEqual((doc["output"]["video"]["codec"], doc["output"]["video"]["stream_copied"]), ("ffv1", True))
        self.assertEqual((doc["output"]["audio"]["codec"], doc["output"]["audio"]["channels"]), ("aac", 1))
        self.assertIn("--drop-other-audio", doc["warnings"][0])
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], self.skill.info().version)
        self.assert_no_leftovers()

    def test_cli_mp4_video_byte_identical(self):
        video = os.path.join(self.tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc=s=96x64:r=25:d=2", "-f", "lavfi",
                        "-i", "sine=f=300:d=2:r=48000", "-c:v", "mpeg4", "-g", "12", "-c:a", "aac", "-shortest", video],
                       check=True, stdin=subprocess.DEVNULL)
        stem = write_tone_wav(os.path.join(self.tmp, "stem.wav"), seconds=1.5, rate=48000, freq=500.0)
        out = os.path.join(self.tmp, "clip_clean.mp4")
        rc, doc = run_cli("remux", video, stem, "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual((rc, doc["ok"]), (0, True), doc)
        self.assertEqual(video_packets(out), video_packets(video))
        self.assertEqual(sorted(stream_types(out)), ["audio", "video"])
        self.assertEqual(doc["output"]["video"]["nb_frames"], doc["input"]["video"]["nb_frames"])
        self.assertAlmostEqual(doc["output"]["audio"]["duration"], 2.0, delta=0.05)    # the 1.5 s stem padded to the video
        self.assertIn("pads it with silence", doc["warnings"][0])
        rc2, err = run_cli("remux", video, stem, "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual((rc2, err["error"]["code"]), (2, "OUTPUT_EXISTS"))
        self.assert_no_leftovers()


if __name__ == "__main__":
    unittest.main()
