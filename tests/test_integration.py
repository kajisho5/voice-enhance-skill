"""Integration with a real ffmpeg-skill checkout and a real FFmpeg: locate it, check every tool and flag this skill
emits against its live contract, run its doctor, probe a WAV through it, and run ISOLATE and DEREVERB end to end on a
video with two audio streams (fake deep-filter, fake nara_wpe on NumPy; the real engines only when installed). Skipped without a checkout
(VOICE_ENHANCE_FFMPEG_SKILL_DIR or ../ffmpeg-skill); CI's integration job always has one. FFmpeg is called directly
here only to build the fixtures; the skill itself never does."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from helpers import (FFMPEG_SKILL_DIR, env, fake_deep_filter, fake_model, fake_nara_wpe, fake_wpe_calls, have_numpy, on_python_path,
                     read_wav, reverb_fixture, run_cli, snr_db, srr_db, voice_noise_fixture, write_tone_wav)

from voice_enhance_skill.doctor import doctor
from voice_enhance_skill.engines import get_engine
from voice_enhance_skill.errors import VoiceEnhanceError
from voice_enhance_skill.ffmpeg_skill import TOOLS_USED, FfmpegSkill
from voice_enhance_skill.ops import DereverbRequest, IsolateRequest, dereverb, isolate

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
NO_DEEP_FILTER = {"VOICE_ENHANCE_DEEP_FILTER": os.path.join(tempfile.gettempdir(), "no-such-deep-filter")}
POSIX = os.name == "posix"


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

    @unittest.skipUnless(real_engine_ok(), "deep-filter and DeepFilterNet3_onnx.tar.gz are not installed (CI installs no engine, ADR-001)")
    def test_real_deep_filter_raises_snr(self):
        src = os.path.join(self.tmp, "noisy.wav")
        clean = voice_noise_fixture(src, seconds=3.0, snr_db=0.0)
        out = os.path.join(self.tmp, "voice.wav")
        doc = isolate(IsolateRequest(src, out), skill=self.skill)
        self.assertEqual(doc["engine"]["model"], "DeepFilterNet3_onnx.tar.gz")
        self.assertGreater(snr_db(clean, read_wav(out)[2]), snr_db(clean, read_wav(src)[2]))


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
        dry = reverb_fixture(room, seconds=1.0, rate=48000, channels=2)
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
        before, after = srr_db(dry, read_wav(room)[2]), srr_db(dry, x)
        self.assertGreater(after, before + 6.0, (before, after))     # and the decoy (stream 0) was not the one processed
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

    @unittest.skipUnless(real_wpe_ok(), "nara_wpe is not installed (CI installs no engine, ADR-001)")
    def test_real_nara_wpe_raises_srr(self):
        src = os.path.join(self.tmp, "room.wav")
        dry = reverb_fixture(src, seconds=2.0, rate=16000)
        out = os.path.join(self.tmp, "dry.wav")
        doc = dereverb(DereverbRequest(src, out), skill=self.skill)
        self.assertEqual(doc["engine"]["id"], "nara_wpe")
        self.assertGreater(srr_db(dry, read_wav(out)[2]), srr_db(dry, read_wav(src)[2]))


if __name__ == "__main__":
    unittest.main()
