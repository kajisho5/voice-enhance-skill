"""Integration with a real ffmpeg-skill checkout and a real FFmpeg: locate it, check every tool and flag this skill
emits against its live contract, run its doctor, and probe a WAV through it. Skipped without a checkout
(VOICE_ENHANCE_FFMPEG_SKILL_DIR or ../ffmpeg-skill); CI's integration job always has one."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from helpers import FFMPEG_SKILL_DIR, env, run_cli, write_tone_wav

from voice_enhance_skill.doctor import doctor
from voice_enhance_skill.errors import VoiceEnhanceError
from voice_enhance_skill.ffmpeg_skill import TOOLS_USED, FfmpegSkill

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
NO_DEEP_FILTER = {"VOICE_ENHANCE_DEEP_FILTER": os.path.join(tempfile.gettempdir(), "no-such-deep-filter")}


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


if __name__ == "__main__":
    unittest.main()
