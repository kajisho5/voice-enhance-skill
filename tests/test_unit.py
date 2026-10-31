"""Unit tests: contract, typed errors, engine registry and availability checks (with a fake deep-filter binary and
mocked package lookups), the worker subprocess, the ISOLATE op (fake deep-filter, a stub ffmpeg-skill that reads and
writes WAV with the stdlib), the DEREVERB op (a fake nara_wpe package on NumPy, the same stub ffmpeg-skill), the
BLEED_REDUCE op (the crosstalk engine on NumPy, the same stub plus one that reads each channel of a WAV as an audio
stream), doctor, provenance and the OS registry conformance checks. No engine, no weights and no ffmpeg are needed;
nothing is downloaded. The DEREVERB and BLEED_REDUCE engine tests need NumPy and are skipped without it."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from helpers import (OS_DIR, ROOT, crosstalk_fixture, energy_db, env, fake_deep_filter, fake_model, fake_nara_wpe, fake_wpe_calls,
                     have_numpy, on_python_path, read_wav, read_wav_channels, reverb_fixture, run_cli, snr_db, srr_db,
                     voice_noise_fixture, write_tone_wav, write_wav, write_wav_tracks)

from voice_enhance_skill import SKILL_ID, __version__
from voice_enhance_skill.contract import CAPABILITY_IDS, contract
from voice_enhance_skill.doctor import doctor
from voice_enhance_skill.engines import DEFAULT_ENGINE, engine_ids, engines_for, get_engine
from voice_enhance_skill.engines import crosstalk, deepfilternet, nara_wpe_engine, wavio
from voice_enhance_skill.engines.base import OPS, Engine, EngineRequest, EngineResult, parse_version
from voice_enhance_skill.errors import ERROR_TABLE, EXIT_CODES, VoiceEnhanceError
from voice_enhance_skill.ffmpeg_skill import FLAGS_USED, OPTIONAL_FLAGS, TOOLS_USED, FfmpegSkill, parse_contract
from voice_enhance_skill.ops import (BleedReduceRequest, DereverbRequest, IsolateRequest, bleed_reduce, check_paths, dereverb,
                                     isolate, select_engine, select_stream, track_name)
from voice_enhance_skill.procs import TIMEOUT_EXIT, run_process_group
from voice_enhance_skill.provenance import build_provenance, sha256_file
from voice_enhance_skill.service import run_engine

POSIX = os.name == "posix"
NUMPY = have_numpy()
NEEDS_NUMPY = "needs numpy: the fake nara_wpe computes on it (the integration job installs it)"
NEEDS_NUMPY_BLEED = "needs numpy: the crosstalk engine computes on it (the integration job installs it)"
NO_DEEP_FILTER = {"VOICE_ENHANCE_DEEP_FILTER": os.path.join(tempfile.gettempdir(), "no-such-deep-filter"),
                  "VOICE_ENHANCE_DFN_MODEL": None}


def fake_ffmpeg_contract(drop=None, extra=()):
    """An ffmpeg-skill contract document declaring FLAGS_USED (minus `drop`, plus `extra` (tool, flag) pairs)."""
    tools = []
    for name, flags in FLAGS_USED.items():
        props = {f: {"type": "string"} for f in flags if (name, f) != drop}
        props.update({f: {"type": "integer"} for t, f in extra if t == name})
        tools.append({"name": name, "audio_only": True, "input_schema": {"properties": props}})
    return {"skill": {"version": "2.22.4"}, "contract_version": "1.0", "tools": tools}


class StubSkill:
    """Stands in for ffmpeg-skill in unit tests: probe and audio_to_wav on WAV files with the stdlib (no FFmpeg). It
    cannot resample, so it never declares audio --sample-rate."""

    def __init__(self):
        self._info = parse_contract(fake_ffmpeg_contract(), Path("stub-ffmpeg-skill"))
        self.calls = []

    def info(self):
        return self._info

    def probe(self, path):
        try:
            rate, ch, x = read_wav(path)
        except ValueError as e:
            raise VoiceEnhanceError("UNSUPPORTED_MEDIA", str(e))
        a = {"codec": "pcm_s16le", "sample_rate": rate, "channels": ch}
        return {"format": "wav", "duration": len(x) / rate, "video": None, "audio": a, "audio_streams": [dict(a, index=0)]}

    def audio_to_wav(self, src, dst, audio_stream=None, mono=False, sample_rate=None, timeout=None):
        self.calls.append({"src": os.path.basename(src), "audio_stream": audio_stream, "mono": mono, "sample_rate": sample_rate})
        assert sample_rate is None, "the stub cannot resample"
        rate, ch, x = read_wav(src)
        write_wav(dst, x, rate, 1 if mono else ch)
        return {"status": "completed", "output": dst, "probe": self.probe(dst)}


class ErrorTests(unittest.TestCase):
    def test_table_shape_and_retryable(self):
        for code, (exit_code, retryable) in ERROR_TABLE.items():
            self.assertTrue(re.fullmatch(r"[A-Z_]+", code))
            self.assertGreaterEqual(exit_code, 2)
            self.assertIsInstance(retryable, bool)
        self.assertEqual({c for c, (_, r) in ERROR_TABLE.items() if r}, {"TOOL_ERROR", "ENGINE_TIMEOUT", "CANCELLED"})

    def test_error_document(self):
        e = VoiceEnhanceError("ENGINE_TIMEOUT", "slow", {"engine": "x"})
        self.assertEqual(e.to_dict(), {"code": "ENGINE_TIMEOUT", "message": "slow", "retryable": True, "details": {"engine": "x"}})
        self.assertEqual(e.exit_code, EXIT_CODES["ENGINE_TIMEOUT"])
        self.assertFalse(VoiceEnhanceError("TOOL_ERROR", "x", retryable=False).retryable)
        with self.assertRaises(ValueError):
            VoiceEnhanceError("NOPE", "x")
        self.assertEqual(VoiceEnhanceError.from_dict({"code": "NOPE", "message": "m"}).code, "INTERNAL_ERROR")


class ContractTests(unittest.TestCase):
    def test_os_registry_keys_and_provides(self):
        c = contract()
        for k in ("skill_id", "id", "skill", "version", "contract_version", "provides", "dependencies", "not_provided"):
            self.assertIn(k, c)
        self.assertEqual((c["skill_id"], c["version"]), (SKILL_ID, __version__))
        self.assertEqual({p["id"] for p in c["provides"]}, set(CAPABILITY_IDS.values()))
        tool_ids = {t["tool_id"] for t in c["tools"]}
        for p in c["provides"]:
            self.assertEqual(set(p), {"id", "lifecycle", "tool_id"})
            self.assertIn(p["tool_id"], tool_ids)
            # ISOLATE (step 3, ADR-003), DEREVERB (step 4, ADR-004) and BLEED_REDUCE (step 5, ADR-005) are implemented
            self.assertEqual(p["lifecycle"], "EXPERIMENTAL", p)
        for d in c["dependencies"]:
            self.assertIsInstance(d["version_range"], str)
            self.assertFalse(d["version_range"].replace(".", "").isdigit(), d)
        self.assertFalse(c["execution"]["network"])
        json.dumps(c, allow_nan=False)

    def test_errors_and_engines_mirror_the_code(self):
        c = contract()
        self.assertEqual(c["errors"], {k: {"exit_code": e, "retryable": r} for k, (e, r) in ERROR_TABLE.items()})
        self.assertEqual(sorted(c["engines"]["registered"]), engine_ids())
        self.assertEqual(c["engines"]["default"], DEFAULT_ENGINE)
        for spec in c["engines"]["registered"].values():
            self.assertTrue(spec["code_license"] and spec["weights_license"] and spec["attribution"])
            self.assertFalse(spec["network_required"])
        ff = next(d for d in c["dependencies"] if d["id"] == "ffmpeg-skill")
        self.assertEqual(ff["flags"], {t: list(f) for t, f in FLAGS_USED.items()})
        self.assertEqual(ff["optional_flags"], {t: list(f) for t, f in OPTIONAL_FLAGS.items()})
        self.assertEqual(ff["tools"], ["probe", "audio"])       # join refuses one clip, so it is not used (ADR-003)
        self.assertEqual(list(TOOLS_USED), ["probe", "audio"])

    def test_isolate_tool_entry(self):
        tool = next(t for t in contract()["tools"] if t["tool_id"] == f"{SKILL_ID}/isolate")
        self.assertTrue(tool["implemented"])
        self.assertEqual((tool["schema"], tool["output"]["sample_rate"], tool["output"]["channels"]),
                         (f"{SKILL_ID}/isolate@1", 48000, 1))
        for flag in ("--audio-stream", "--atten-lim-db", "--timeout", "--overwrite", "--ffmpeg-skill", "--offline"):
            self.assertTrue(any(a.startswith("[" + flag) for a in tool["argv"]), flag)

    def test_bleed_reduce_tool_entry(self):
        tool = next(t for t in contract()["tools"] if t["tool_id"] == f"{SKILL_ID}/bleed-reduce")
        self.assertTrue(tool["implemented"])
        self.assertEqual((tool["schema"], tool["engine"], tool["roadmap_step"], tool["output"]["codec"], tool["output"]["channels"]),
                         (f"{SKILL_ID}/bleed-reduce@1", "crosstalk", 5, "pcm_f32le", 1))
        self.assertIn("outputs", tool["result_keys"])
        for flag in ("--audio-streams", "--engine", "--dominance-db", "--timeout", "--overwrite", "--ffmpeg-skill", "--offline"):
            self.assertTrue(any(a.startswith("[" + flag) for a in tool["argv"]), flag)
        spec = contract()["engines"]["registered"]["crosstalk"]
        self.assertEqual((spec["kind"], spec["weights_license"]), ("python", "none (signal processing, no weights)"))

    def test_dereverb_tool_entry(self):
        tool = next(t for t in contract()["tools"] if t["tool_id"] == f"{SKILL_ID}/dereverb")
        self.assertTrue(tool["implemented"])
        self.assertEqual((tool["schema"], tool["engine"], tool["roadmap_step"], tool["output"]["codec"]),
                         (f"{SKILL_ID}/dereverb@1", "nara_wpe", 4, "pcm_f32le"))
        for flag in ("--audio-stream", "--engine", "--taps", "--delay", "--iterations", "--timeout", "--overwrite", "--ffmpeg-skill",
                     "--offline"):
            self.assertTrue(any(a.startswith("[" + flag) for a in tool["argv"]), flag)

    def test_version_comes_from_pyproject(self):
        want = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(__version__, want)

    def test_cli_contract_and_skill_alias(self):
        rc, doc = run_cli("contract", "--json")
        rc2, doc2 = run_cli("skill", "--json")
        self.assertEqual((rc, rc2, doc["skill_id"]), (0, 0, SKILL_ID))
        self.assertEqual(doc, doc2)

    def test_no_shell_no_direct_ffmpeg_no_downloads(self):
        for f in (ROOT / "src" / "voice_enhance_skill").rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("shell=True", text, f.name)
            self.assertNotIn('"ffmpeg"', text.replace('"ffmpeg", "ffprobe"', ""), f.name)   # media goes through ffmpeg-skill
            for banned in ("hf_hub_download", "snapshot_download", "urllib.request", "requests.", "urlopen", "torch.hub"):
                self.assertNotIn(banned, text, f"{f.name}: {banned}")


class EngineRegistryTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(engine_ids(), ["crosstalk", "deepfilternet", "nara_wpe"])
        self.assertEqual(set(DEFAULT_ENGINE), set(OPS))
        self.assertEqual(engines_for("isolate"), ["deepfilternet"])
        self.assertEqual(engines_for("dereverb"), ["nara_wpe"])
        self.assertEqual(engines_for("bleed_reduce"), ["crosstalk"])    # ADR-001: no AI engine; ADR-005: signal processing
        self.assertEqual(DEFAULT_ENGINE["bleed_reduce"], "crosstalk")
        with self.assertRaises(VoiceEnhanceError) as e:
            get_engine("demucs")
        self.assertEqual(e.exception.code, "ENGINE_UNAVAILABLE")

    def test_parse_version(self):
        self.assertEqual(parse_version("deep-filter 0.5.6"), (0, 5, 6))
        self.assertEqual(parse_version("0.0.11"), (0, 0, 11))
        self.assertIsNone(parse_version("unknown"))

    def test_request_result_round_trip(self):
        req = EngineRequest(op="isolate", input_path="a.wav", output_path="b.wav", params={"x": 1})
        self.assertEqual(EngineRequest.from_dict(req.to_dict()), req)
        res = EngineResult(output_path="b.wav", engine_version="0.5.6", model="m.tar.gz", extra={"k": 1})
        self.assertEqual(EngineResult.from_dict(res.to_dict()), res)


class DeepFilterNetAvailabilityTests(unittest.TestCase):
    def test_missing_binary_is_missing(self):
        with env(**NO_DEEP_FILTER):
            r = get_engine("deepfilternet").report()
        self.assertEqual((r["status"], r["missing"]), ("MISSING", "binary"))
        self.assertIn("releases", r["install"])

    def test_not_on_path_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp, env(VOICE_ENHANCE_DEEP_FILTER=None, PATH=tmp):
            self.assertEqual(get_engine("deepfilternet").inspect()["status"], "MISSING")

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_binary_without_model_is_missing_then_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = fake_deep_filter(tmp)
            with env(VOICE_ENHANCE_DEEP_FILTER=binary, VOICE_ENHANCE_DFN_MODEL=None):
                r = get_engine("deepfilternet").inspect()
                self.assertEqual((r["status"], r["missing"], r["version"]), ("MISSING", "model", "0.5.6"))
                with self.assertRaises(VoiceEnhanceError) as e:
                    get_engine("deepfilternet").require()
                self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
            model = os.path.join(tmp, "DeepFilterNet3_onnx.tar.gz")
            Path(model).write_bytes(b"\x1f\x8b fake")
            with env(VOICE_ENHANCE_DEEP_FILTER=binary, VOICE_ENHANCE_DFN_MODEL=model):
                r = get_engine("deepfilternet").report()
            self.assertEqual((r["status"], r["model"], r["install"]), ("OK", "DeepFilterNet3_onnx.tar.gz", None))
            with env(PATH=tmp + os.pathsep + os.environ.get("PATH", ""), VOICE_ENHANCE_DEEP_FILTER=None, VOICE_ENHANCE_DFN_MODEL=model):
                self.assertEqual(get_engine("deepfilternet").inspect()["status"], "OK")

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_wrong_version_is_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = fake_deep_filter(tmp, version="0.4.0")
            with env(VOICE_ENHANCE_DEEP_FILTER=binary):
                r = get_engine("deepfilternet").inspect()
        self.assertEqual((r["status"], r["version"]), ("INCOMPATIBLE", "0.4.0"))


class NaraWpeAvailabilityTests(unittest.TestCase):
    def test_statuses(self):
        eng = get_engine("nara_wpe")
        with mock.patch.object(nara_wpe_engine, "_found", return_value=False):
            self.assertEqual(eng.inspect()["status"], "MISSING")
        with mock.patch.object(nara_wpe_engine, "_found", return_value=True), \
                mock.patch.object(nara_wpe_engine, "_dist_version", return_value="0.0.11"):
            self.assertEqual(eng.report()["status"], "OK")
        with mock.patch.object(nara_wpe_engine, "_found", return_value=True), \
                mock.patch.object(nara_wpe_engine, "_dist_version", return_value="0.0.7"):
            self.assertEqual(eng.inspect()["status"], "INCOMPATIBLE")
        with mock.patch.object(nara_wpe_engine, "_found", side_effect=lambda m: m != "numpy"):
            self.assertIn("numpy", eng.inspect()["detail"])

    @unittest.skipIf(nara_wpe_engine._found("nara_wpe"), "nara_wpe is installed")
    def test_not_installed_here_is_missing(self):
        self.assertEqual(get_engine("nara_wpe").report()["status"], "MISSING")


class WorkerTests(unittest.TestCase):
    """The real worker subprocess path (no engine needed)."""

    def test_process_group_is_killed_on_timeout(self):
        t0 = time.time()
        code, _, _ = run_process_group([sys.executable, "-c", "import time; time.sleep(30)"], 1.0)
        self.assertEqual(code, TIMEOUT_EXIT)
        self.assertLess(time.time() - t0, 15)

    def test_argv_must_be_a_clean_list(self):
        with self.assertRaises(VoiceEnhanceError):
            run_process_group([], 1.0)
        with self.assertRaises(VoiceEnhanceError):
            run_process_group(["a\x00b"], 1.0)
        self.assertEqual(run_process_group([os.path.join(tempfile.gettempdir(), "no-such-exe")], 5.0)[0], 127)

    def test_unknown_engine_is_refused_before_a_process_starts(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(VoiceEnhanceError) as e:
            run_engine("nope", EngineRequest("isolate", "a.wav", "b.wav"), 30, tmp)
        self.assertEqual(e.exception.code, "ENGINE_UNAVAILABLE")

    def test_worker_reports_structured_errors(self):
        from voice_enhance_skill.service import worker_argv, worker_env
        with tempfile.TemporaryDirectory() as tmp:
            req, out = os.path.join(tmp, "req.json"), os.path.join(tmp, "out.json")
            Path(req).write_text(json.dumps({"engine_id": "nope", "request": {"op": "isolate", "input_path": "a", "output_path": "b"}}),
                                 encoding="utf-8")
            code, _, err = run_process_group(worker_argv(req, out), 60, env=worker_env())
            self.assertEqual(code, 1, err)
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual((payload["ok"], payload["error"]["code"], payload["error"]["retryable"]),
                             (False, "ENGINE_UNAVAILABLE", False))

    def test_missing_engine_is_reported_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp, env(**NO_DEEP_FILTER), self.assertRaises(VoiceEnhanceError) as e:
            run_engine("deepfilternet", EngineRequest("isolate", "a.wav", "b.wav"), 60, tmp)
        self.assertEqual((e.exception.code, e.exception.exit_code), ("ENGINE_UNAVAILABLE", 3))

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_installed_engine_failure_comes_back_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "input.wav"), 0.2)
            with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp), VOICE_ENHANCE_DFN_MODEL=fake_model(tmp)), \
                    self.assertRaises(VoiceEnhanceError) as e:
                run_engine("deepfilternet", EngineRequest("isolate", src, os.path.join(tmp, "b.wav")), 60, tmp)
        self.assertEqual((e.exception.code, e.exception.details["exit_code"]), ("ENGINE_FAILED", 9))   # mode "fail" exits 9

    def test_an_engine_without_process_is_not_implemented(self):
        class Declared(Engine):          # an engine declared before its op exists (as BLEED_REDUCE was until step 5)
            id, ops, roadmap_step = "declared", ("bleed_reduce",), 5

            def inspect(self):
                return {"status": "OK", "version": "1.0"}
        with self.assertRaises(VoiceEnhanceError) as e:
            Declared().process(EngineRequest("bleed_reduce", "a.wav", "b.wav"))
        self.assertEqual((e.exception.code, e.exception.details["op"]), ("NOT_IMPLEMENTED", "bleed_reduce"))

    @unittest.skipUnless(POSIX, "process groups are POSIX here")
    def test_new_group_false_keeps_the_child_in_the_callers_group(self):
        probe = [sys.executable, "-c", "import os; print(os.getpgrp())"]
        self.assertEqual(int(run_process_group(probe, 30, new_group=False)[1]), os.getpgrp())
        self.assertNotEqual(int(run_process_group(probe, 30)[1]), os.getpgrp())

    def test_engine_timeout_is_retryable(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("voice_enhance_skill.service.worker_argv", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]), \
                self.assertRaises(VoiceEnhanceError) as e:
            run_engine("nara_wpe", EngineRequest("dereverb", "a.wav", "b.wav"), 1.0, tmp)
        self.assertEqual((e.exception.code, e.exception.retryable), ("ENGINE_TIMEOUT", True))

    def test_worker_without_result_is_engine_failed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("voice_enhance_skill.service.worker_argv", return_value=[sys.executable, "-c", "raise SystemExit(3)"]), \
                self.assertRaises(VoiceEnhanceError) as e:
            run_engine("nara_wpe", EngineRequest("dereverb", "a.wav", "b.wav"), 30, tmp)
        self.assertEqual(e.exception.code, "ENGINE_FAILED")


def _pid_alive(pid):
    """Linux: a process that exists and is not a zombie (a killed child may linger as one until it is reaped)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


class DeepFilterNetProcessTests(unittest.TestCase):
    """DeepFilterNetEngine.process through the real worker subprocess, with a fake deep-filter."""

    def test_argv(self):
        self.assertEqual(deepfilternet.build_argv("deep-filter", "m.tar.gz", "/o", "/i.wav"),
                         ["deep-filter", "-D", "-m", "m.tar.gz", "-o", "/o", "/i.wav"])
        self.assertEqual(deepfilternet.build_argv("deep-filter", "m.tar.gz", "/o", "/i.wav", 12.0),
                         ["deep-filter", "-D", "-m", "m.tar.gz", "-o", "/o", "--atten-lim-db", "12", "/i.wav"])

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_stem_through_the_worker_raises_snr(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "input.wav")
            clean = voice_noise_fixture(src, seconds=1.0)
            model = fake_model(tmp)
            out = os.path.join(tmp, "stem.wav")
            with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp, mode="lowpass"), VOICE_ENHANCE_DFN_MODEL=model):
                res = run_engine("deepfilternet", EngineRequest("isolate", src, out, params={"atten_lim_db": 30.0}), 60, tmp)
            argv = json.loads(Path(tmp, "argv.json").read_text(encoding="utf-8"))
            self.assertEqual(argv, ["-D", "-m", model, "-o", os.path.join(tmp, "deep_filter_out"), "--atten-lim-db", "30", src])
            self.assertEqual((res.output_path, res.engine_version, res.model), (out, "0.5.6", "DeepFilterNet3_onnx.tar.gz"))
            self.assertEqual(res.extra["model_sha256"], sha256_file(model))
            self.assertEqual(res.extra["argv"][:2], ["deep-filter", "-D"])
            self.assertNotIn(tmp, json.dumps(res.extra))                 # no local paths in what is published
            before, after = snr_db(clean, read_wav(src)[2]), snr_db(clean, read_wav(out)[2])
            self.assertGreater(after, before + 6.0, (before, after))

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_no_output_is_invalid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "input.wav"), 0.2)
            with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp, mode="nooutput"), VOICE_ENHANCE_DFN_MODEL=fake_model(tmp)), \
                    self.assertRaises(VoiceEnhanceError) as e:
                run_engine("deepfilternet", EngineRequest("isolate", src, os.path.join(tmp, "b.wav")), 60, tmp)
        self.assertEqual(e.exception.code, "INVALID_RESULT")

    @unittest.skipUnless(sys.platform.startswith("linux"), "process state is read from /proc")
    def test_timeout_kills_the_binary_with_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "input.wav"), 0.2)
            with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp, mode="sleep"), VOICE_ENHANCE_DFN_MODEL=fake_model(tmp)), \
                    self.assertRaises(VoiceEnhanceError) as e:
                run_engine("deepfilternet", EngineRequest("isolate", src, os.path.join(tmp, "b.wav")), 3.0, tmp)
            self.assertEqual((e.exception.code, e.exception.retryable), ("ENGINE_TIMEOUT", True))
            pid = int(Path(tmp, "pid").read_text())
            deadline = time.time() + 10
            while _pid_alive(pid) and time.time() < deadline:
                time.sleep(0.1)
            self.assertFalse(_pid_alive(pid), "deep-filter outlived the worker")


class IsolatePlanningTests(unittest.TestCase):
    """The checks ISOLATE makes before any media work."""

    def test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "in.wav"), 0.1)
            out = os.path.join(tmp, "out.wav")
            self.assertEqual(check_paths(src, out, False), (os.path.abspath(src), os.path.abspath(out)))
            cases = [((os.path.join(tmp, "nope.wav"), out, False), "INPUT_NOT_FOUND"),
                     ((src, src, True), "OUTPUT_IS_INPUT"),
                     ((src, os.path.join(tmp, "out.mp3"), False), "INVALID_ARGUMENT"),
                     ((src, os.path.join(tmp, "no-dir", "out.wav"), False), "INVALID_ARGUMENT")]
            for args, code in cases:
                with self.assertRaises(VoiceEnhanceError) as e:
                    check_paths(*args)
                self.assertEqual(e.exception.code, code, args)
            Path(out).write_bytes(b"x")
            with self.assertRaises(VoiceEnhanceError) as e:
                check_paths(src, out, False)
            self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
            self.assertEqual(check_paths(src, out, True)[1], os.path.abspath(out))
            os.mkdir(os.path.join(tmp, "d.wav"))
            with self.assertRaises(VoiceEnhanceError) as e:
                check_paths(src, os.path.join(tmp, "d.wav"), True)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_engine_selection(self):
        with env(**NO_DEEP_FILTER), self.assertRaises(VoiceEnhanceError) as e:
            select_engine("isolate", None)
        self.assertEqual((e.exception.code, e.exception.details["engine"]), ("ENGINE_UNAVAILABLE", "deepfilternet"))
        self.assertIn("releases", e.exception.details["install"])
        with self.assertRaises(VoiceEnhanceError) as e:
            select_engine("isolate", "nara_wpe")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(VoiceEnhanceError) as e:
            select_engine("isolate", "demucs")
        self.assertEqual(e.exception.code, "ENGINE_UNAVAILABLE")
        with mock.patch.object(crosstalk, "_found", return_value=False), self.assertRaises(VoiceEnhanceError) as e:
            select_engine("bleed_reduce", None)
        self.assertEqual((e.exception.code, e.exception.details["engine"]), ("ENGINE_UNAVAILABLE", "crosstalk"))
        self.assertIn("[bleed]", e.exception.details["install"])
        with self.assertRaises(VoiceEnhanceError) as e:
            select_engine("bleed_reduce", "nara_wpe")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    @unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
    def test_engine_without_model_is_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmp, env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp), VOICE_ENHANCE_DFN_MODEL=None), \
                self.assertRaises(VoiceEnhanceError) as e:
            select_engine("isolate", None)
        self.assertEqual((e.exception.code, e.exception.exit_code), ("MODEL_NOT_FOUND", 3))

    def test_stream_selection(self):
        facts = {"audio": {"sample_rate": 48000}, "audio_streams": [{"index": 0, "channels": 1}, {"index": 1, "channels": 2}]}
        self.assertEqual(select_stream(facts, 1, "v.mp4"), ({"index": 1, "channels": 2}, 2))
        with self.assertRaises(VoiceEnhanceError) as e:
            select_stream(facts, 2, "v.mp4")
        self.assertEqual((e.exception.code, e.exception.details["audio_stream_count"]), ("INVALID_ARGUMENT", 2))
        with self.assertRaises(VoiceEnhanceError) as e:
            select_stream({"audio": None, "audio_streams": []}, 0, "silent.mp4")
        self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")
        self.assertEqual(select_stream({"audio": {"channels": 1}}, 0, "old-probe.wav")[1], 1)   # probe without audio_streams

    def test_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "in.wav"), 0.1)
            for kw in ({"timeout": 0}, {"audio_stream": -1}, {"atten_lim_db": 0}, {"atten_lim_db": 101}):
                with self.assertRaises(VoiceEnhanceError) as e:
                    isolate(IsolateRequest(src, os.path.join(tmp, "o.wav"), **kw), skill=StubSkill())
                self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)


@unittest.skipUnless(POSIX, "the fake deep-filter is a shell script")
class IsolateOpTests(unittest.TestCase):
    """ISOLATE end to end with the fake deep-filter and the stub ffmpeg-skill."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.engine_dir = os.path.join(self.tmp, "engine")
        os.mkdir(self.engine_dir)
        self.model = fake_model(self.engine_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def run_isolate(self, src, out, mode="lowpass", **kw):
        with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(self.engine_dir, mode=mode), VOICE_ENHANCE_DFN_MODEL=self.model):
            return isolate(IsolateRequest(src, out, **kw), skill=self.skill)

    def assert_no_leftovers(self):
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".voice-enhance-")], [])

    def test_voice_noise_fixture_snr_rises(self):
        self.skill = StubSkill()
        src, out = os.path.join(self.tmp, "noisy.wav"), os.path.join(self.tmp, "voice.wav")
        clean = voice_noise_fixture(src)
        src_sha = sha256_file(src)
        doc = self.run_isolate(src, out, atten_lim_db=40)
        self.assertEqual(sha256_file(src), src_sha)                       # the input is never modified
        rate, ch, stem = read_wav(out)
        self.assertEqual((rate, ch), (48000, 1))
        before, after = snr_db(clean, read_wav(src)[2]), snr_db(clean, stem)
        self.assertGreater(after, before + 6.0, (before, after))
        self.assertEqual((doc["schema"], doc["ok"], doc["op"], doc["capability"]), (f"{SKILL_ID}/isolate@1", True, "isolate", "voice.isolate"))
        self.assertEqual((doc["output"]["path"], doc["output"]["sample_rate"], doc["output"]["channels"]), (out, 48000, 1))
        self.assertEqual((doc["input"]["audio_stream"], doc["input"]["audio_stream_count"], doc["input"]["has_video"]), (0, 1, False))
        self.assertEqual(doc["engine"], {"id": "deepfilternet", "version": "0.5.6", "model": "DeepFilterNet3_onnx.tar.gz",
                                         "model_sha256": sha256_file(self.model)})
        self.assertEqual((doc["parameters"]["atten_lim_db"], doc["parameters"]["compensate_delay"]), (40.0, True))
        prov = doc["provenance"]
        self.assertEqual((prov["op"], prov["input"]["sha256"], prov["output"]["sha256"]), ("isolate", src_sha, sha256_file(out)))
        self.assertEqual(prov["engine"]["model_sha256"], sha256_file(self.model))
        self.assertEqual(prov["ffmpeg_skill"]["version"], "2.22.4")
        self.assertEqual(self.skill.calls, [{"src": "noisy.wav", "audio_stream": 0, "mono": False, "sample_rate": None}])
        self.assertEqual(doc["warnings"], [])
        json.dumps(doc, allow_nan=False)
        self.assert_no_leftovers()

    def test_stereo_is_folded_down_in_a_second_pass(self):
        self.skill = StubSkill()
        src, out = os.path.join(self.tmp, "stereo.wav"), os.path.join(self.tmp, "voice.wav")
        clean = voice_noise_fixture(src, seconds=1.0, channels=2)
        doc = self.run_isolate(src, out)
        self.assertEqual([(c["src"], c["audio_stream"], c["mono"]) for c in self.skill.calls],
                         [("stereo.wav", 0, False), ("extracted.wav", None, True)])
        self.assertEqual([st["flags"] for st in doc["steps"]], [["--audio-stream", "0"], ["--mono"]])
        self.assertIn("folded down to mono", doc["warnings"][0])
        self.assertGreater(snr_db(clean, read_wav(out)[2]), snr_db(clean, read_wav(src)[2]) + 6.0)

    def test_other_rate_without_audio_sample_rate_is_refused_before_any_work(self):
        self.skill = StubSkill()
        src = write_wav(os.path.join(self.tmp, "cd.wav"), [0.1] * 4410, rate=44100)
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_isolate(src, os.path.join(self.tmp, "voice.wav"))
        self.assertEqual((e.exception.code, e.exception.details["needs_flag"], e.exception.details["input_sample_rate"]),
                         ("FFMPEG_SKILL_UNAVAILABLE", "sample_rate", 44100))
        self.assertEqual(self.skill.calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.engine_dir, "argv.json")))
        self.assert_no_leftovers()

    def test_engine_output_failing_validation_writes_nothing(self):
        self.skill = StubSkill()
        src, out = os.path.join(self.tmp, "noisy.wav"), os.path.join(self.tmp, "voice.wav")
        voice_noise_fixture(src, seconds=0.5)
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_isolate(src, out, mode="wrongrate")
        self.assertEqual(e.exception.code, "INVALID_RESULT")
        self.assertIn("sample rate 16000", e.exception.message)
        self.assertFalse(os.path.exists(out))
        self.assert_no_leftovers()

    def test_overwrite(self):
        self.skill = StubSkill()
        src, out = os.path.join(self.tmp, "noisy.wav"), os.path.join(self.tmp, "voice.wav")
        voice_noise_fixture(src, seconds=0.5)
        Path(out).write_bytes(b"old")
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_isolate(src, out)
        self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
        self.assertEqual(Path(out).read_bytes(), b"old")
        self.run_isolate(src, out, overwrite=True)
        self.assertEqual(read_wav(out)[:2], (48000, 1))

    def test_cli_isolate_json(self):
        self.skill = StubSkill()
        src, out = os.path.join(self.tmp, "noisy.wav"), os.path.join(self.tmp, "voice.wav")
        voice_noise_fixture(src, seconds=0.5)
        with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(self.engine_dir, mode="lowpass"), VOICE_ENHANCE_DFN_MODEL=self.model), \
                mock.patch("voice_enhance_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("isolate", src, "-o", out, "--json")
            rc2, err = run_cli("isolate", src, "-o", out, "--json")
        self.assertEqual((rc, doc["ok"], doc["output"]["path"]), (0, True, out))
        self.assertEqual((rc2, err["ok"], err["error"]["code"]), (EXIT_CODES["OUTPUT_EXISTS"], False, "OUTPUT_EXISTS"))


class IsolateCliErrorTests(unittest.TestCase):
    def test_errors_are_one_json_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, doc = run_cli("isolate", os.path.join(tmp, "nope.mp4"), "-o", os.path.join(tmp, "o.wav"), "--json")
            self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (2, False, "INPUT_NOT_FOUND"))
            src = write_tone_wav(os.path.join(tmp, "in.wav"), 0.1)
            with env(**NO_DEEP_FILTER):
                rc, doc = run_cli("isolate", src, "-o", os.path.join(tmp, "o.wav"), "--json")
            self.assertEqual((rc, doc["error"]["code"], doc["error"]["retryable"]), (3, "ENGINE_UNAVAILABLE", False))
            self.assertFalse(os.path.exists(os.path.join(tmp, "o.wav")))


class NaraWpeParamTests(unittest.TestCase):
    """What the nara_wpe engine accepts, checked without NumPy."""

    def test_params(self):
        self.assertEqual(nara_wpe_engine.check_params({}), {"taps": 10, "delay": 3, "iterations": 3})
        self.assertEqual(nara_wpe_engine.check_params({"taps": 20, "delay": 2, "iterations": 1, "timeout": 5.0}),
                         {"taps": 20, "delay": 2, "iterations": 1})
        for bad in ({"taps": 0}, {"taps": 51}, {"delay": 0}, {"iterations": 21}, {"taps": 2.5}, {"taps": True}):
            with self.assertRaises(VoiceEnhanceError) as e:
                nara_wpe_engine.check_params(bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", bad)

    def test_stft_sizes_follow_the_rate(self):
        self.assertEqual(nara_wpe_engine.stft_sizes(16000), (512, 128))    # nara_wpe's own 16 kHz example
        self.assertEqual(nara_wpe_engine.stft_sizes(48000), (1536, 384))
        self.assertEqual(nara_wpe_engine.stft_sizes(44100), (1412, 353))


@unittest.skipUnless(NUMPY, NEEDS_NUMPY)
class NaraWpeProcessTests(unittest.TestCase):
    """NaraWpeEngine.process through the real worker subprocess, with the fake nara_wpe package."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def run_wpe(self, src, mode="wpe", **params):
        site = fake_nara_wpe(self.tmp, mode=mode)
        with on_python_path(site):
            res = run_engine("nara_wpe", EngineRequest("dereverb", src, os.path.join(self.tmp, "stem.wav"), sample_rate=16000,
                                                       params=params), 60, self.tmp)
        return res, fake_wpe_calls(site)

    def test_dereverb_through_the_worker_raises_srr(self):
        src = os.path.join(self.tmp, "input.wav")
        dry = reverb_fixture(src, seconds=1.0, rate=16000)
        res, calls = self.run_wpe(src, taps=10, delay=3, iterations=3)
        self.assertEqual((res.engine_version, res.model), ("0.0.11", None))
        self.assertEqual(res.extra["method"], "nara_wpe.wpe.wpe_v8")
        self.assertEqual((res.extra["stft"], res.extra["output_format"]), ({"window": "hann", "size": 512, "hop": 128}, "pcm_f32le"))
        self.assertEqual(len(calls), 1)                     # 1 s is one block
        self.assertEqual((calls[0]["shape"][:2], calls[0]["taps"], calls[0]["delay"], calls[0]["iterations"], calls[0]["statistics_mode"]),
                         ([257, 1], 10, 3, 3, "full"))
        rate, ch, out = read_wav(res.output_path)
        self.assertEqual((rate, ch, len(out)), (16000, 1, 16000))
        self.assertEqual(Path(res.output_path).read_bytes()[20:22], b"\x03\x00")     # WAVE_FORMAT_IEEE_FLOAT
        before, after = srr_db(dry, read_wav(src)[2]), srr_db(dry, out)
        self.assertGreater(after, before + 6.0, (before, after))

    def test_failure_in_nara_wpe_is_engine_failed(self):
        src = write_tone_wav(os.path.join(self.tmp, "input.wav"), 0.2, rate=16000)
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_wpe(src, mode="fail")
        self.assertEqual(e.exception.code, "ENGINE_FAILED")
        self.assertIn("fake nara_wpe failed", e.exception.message)

    def test_wrong_shape_from_nara_wpe_is_invalid_result(self):
        src = write_tone_wav(os.path.join(self.tmp, "input.wav"), 0.2, rate=16000)
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_wpe(src, mode="badshape")
        self.assertEqual(e.exception.code, "INVALID_RESULT")

    def test_identity_round_trip_is_sample_aligned_across_blocks(self):
        """With no filtering, STFT -> overlap-add gives the input back sample for sample, also across block borders
        (0.25 s blocks here) and per channel: the engine adds no delay and loses no sample."""
        import numpy as np
        src = os.path.join(self.tmp, "stereo.wav")
        left = [math.sin(2 * math.pi * 220 * i / 22050) * (i % 997) / 997 for i in range(int(0.9 * 22050))]
        right = [((i * 7919) % 2000 - 1000) / 1500 for i in range(len(left))]
        with wave.open(src, "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(22050)
            w.writeframes(b"".join(struct.pack("<hh", int(a * 32767), int(b * 32767)) for a, b in zip(left, right)))
        out = os.path.join(self.tmp, "out.wav")
        identity = lambda y, **kw: y.copy()  # noqa: E731
        facts = nara_wpe_engine.dereverb_file(src, out, 10, 3, 3, np, identity, block_seconds=0.25)
        self.assertEqual((facts["sample_rate"], facts["channels"], facts["frames"]), (22050, 2, len(left)))
        a, b = nara_wpe_engine.WavReader(src), nara_wpe_engine.WavReader(out)
        try:
            x, y = a.read(np, 0, a.frames), b.read(np, 0, b.frames)
        finally:
            a.close(), b.close()
        self.assertEqual(y.shape, x.shape)
        self.assertLess(float(np.max(np.abs(y - x))), 1e-6)

    def test_wav_reader_formats(self):
        import numpy as np
        vals = [0.0, 0.5, -0.25, -1.0]
        plain = write_wav(os.path.join(self.tmp, "s16.wav"), vals, 8000)
        ext24 = os.path.join(self.tmp, "s24.wav")       # WAVE_FORMAT_EXTENSIBLE, 24-bit, as ffmpeg writes it
        data = b"".join(int(v * 8388607).to_bytes(3, "little", signed=True) for v in vals)
        fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, 1, 8000, 24000, 3, 24, 22, 24, 4, 1, b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71")
        Path(ext24).write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt))
                                + fmt + b"data" + struct.pack("<I", len(data)) + data)
        for path in (plain, ext24):
            r = nara_wpe_engine.WavReader(path)
            try:
                got = r.read(np, -1, 6)[:, 0]
            finally:
                r.close()
            self.assertEqual(len(got), 6)
            self.assertEqual((got[0], got[-1]), (0.0, 0.0))           # outside the file: zeros
            for g, v in zip(got[1:5], vals):
                self.assertAlmostEqual(float(g), v, delta=1e-4)
        bad = os.path.join(self.tmp, "bad.wav")
        Path(bad).write_bytes(b"not a wav file at all")
        with self.assertRaises(VoiceEnhanceError):
            nara_wpe_engine.WavReader(bad)


class DereverbPlanningTests(unittest.TestCase):
    """The checks DEREVERB makes before any media work (no NumPy, no engine)."""

    def test_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "in.wav"), 0.1)
            for kw in ({"timeout": 0}, {"audio_stream": -1}, {"taps": 0}, {"taps": 51}, {"delay": 0}, {"iterations": 21},
                       {"taps": True}):
                with self.assertRaises(VoiceEnhanceError) as e:
                    dereverb(DereverbRequest(src, os.path.join(tmp, "o.wav"), **kw), skill=StubSkill())
                self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)

    def test_missing_engine_is_refused_before_any_media_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "in.wav"), 0.1)
            skill = StubSkill()
            with mock.patch.object(nara_wpe_engine, "_found", return_value=False), self.assertRaises(VoiceEnhanceError) as e:
                dereverb(DereverbRequest(src, os.path.join(tmp, "o.wav")), skill=skill)
            self.assertEqual((e.exception.code, e.exception.details["engine"]), ("ENGINE_UNAVAILABLE", "nara_wpe"))
            self.assertIn("[wpe]", e.exception.details["install"])
            self.assertEqual(skill.calls, [])
            with mock.patch.object(nara_wpe_engine, "_found", return_value=False):
                rc, doc = run_cli("dereverb", src, "-o", os.path.join(tmp, "o.wav"), "--json")
            self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (3, False, "ENGINE_UNAVAILABLE"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "o.wav")))
        with self.assertRaises(VoiceEnhanceError) as e:
            select_engine("dereverb", "deepfilternet")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")


@unittest.skipUnless(NUMPY, NEEDS_NUMPY)
class DereverbOpTests(unittest.TestCase):
    """DEREVERB end to end with the fake nara_wpe and the stub ffmpeg-skill."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = StubSkill()

    def tearDown(self):
        self._tmp.cleanup()

    def run_dereverb(self, src, out, mode="wpe", **kw):
        self.site = fake_nara_wpe(os.path.join(self.tmp, "engine"), mode=mode)
        with on_python_path(self.site):
            return dereverb(DereverbRequest(src, out, **kw), skill=self.skill)

    def assert_no_leftovers(self):
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".voice-enhance-")], [])

    def test_reverberant_fixture_srr_rises(self):
        src, out = os.path.join(self.tmp, "room.wav"), os.path.join(self.tmp, "dry.wav")
        dry = reverb_fixture(src, seconds=1.0, rate=16000)
        src_sha = sha256_file(src)
        doc = self.run_dereverb(src, out, taps=12)
        self.assertEqual(sha256_file(src), src_sha)                       # the input is never modified
        rate, ch, x = read_wav(out)
        self.assertEqual((rate, ch, len(x)), (16000, 1, 16000))          # the input's own rate: nothing was resampled
        before, after = srr_db(dry, read_wav(src)[2]), srr_db(dry, x)
        self.assertGreater(after, before + 6.0, (before, after))
        self.assertEqual((doc["schema"], doc["ok"], doc["op"], doc["capability"]), (f"{SKILL_ID}/dereverb@1", True, "dereverb", "voice.dereverb"))
        self.assertEqual((doc["output"]["path"], doc["output"]["sample_rate"], doc["output"]["channels"]), (out, 16000, 1))
        self.assertEqual(doc["engine"], {"id": "nara_wpe", "version": "0.0.11", "model": None})
        p = doc["parameters"]
        self.assertEqual((p["taps"], p["delay"], p["iterations"], p["statistics_mode"], p["output_format"]), (12, 3, 3, "full", "pcm_f32le"))
        self.assertEqual(p["stft"], {"window": "hann", "size": 512, "hop": 128})
        self.assertEqual([c["taps"] for c in fake_wpe_calls(self.site)], [12])
        self.assertEqual(doc["steps"], [{"tool": "ffmpeg-skill/audio", "flags": ["--audio-stream", "0"],
                                         "result": {"sample_rate": 16000, "channels": 1}}])
        self.assertEqual(self.skill.calls, [{"src": "room.wav", "audio_stream": 0, "mono": False, "sample_rate": None}])
        prov = doc["provenance"]
        self.assertEqual((prov["op"], prov["input"]["sha256"], prov["output"]["sha256"]), ("dereverb", src_sha, sha256_file(out)))
        self.assertEqual((prov["engine"]["id"], prov["engine"]["weights_license"], prov["ffmpeg_skill"]["version"]),
                         ("nara_wpe", "none (signal processing, no weights)", "2.22.4"))
        self.assertEqual(doc["warnings"], [])
        json.dumps(doc, allow_nan=False)
        self.assert_no_leftovers()

    def test_stereo_44k1_keeps_rate_and_channels(self):
        """WPE takes any rate and channel count: a 44.1 kHz stereo stream needs no audio --sample-rate (the stub has
        none) and no fold-down; all channels go to one multi-channel WPE call."""
        src, out = os.path.join(self.tmp, "stereo.wav"), os.path.join(self.tmp, "dry.wav")
        dry = reverb_fixture(src, seconds=0.5, rate=44100, channels=2)
        doc = self.run_dereverb(src, out)
        rate, ch, x = read_wav(out)
        self.assertEqual((rate, ch, doc["output"]["channels"], doc["parameters"]["sample_rate"]), (44100, 2, 2, 44100))
        self.assertEqual(fake_wpe_calls(self.site)[0]["shape"][:2], [707, 2])        # 1412-point STFT, 2 channels
        self.assertEqual(len(self.skill.calls), 1)
        self.assertGreater(srr_db(dry, x), srr_db(dry, read_wav(src)[2]) + 6.0)

    def test_engine_failure_writes_nothing(self):
        src, out = os.path.join(self.tmp, "room.wav"), os.path.join(self.tmp, "dry.wav")
        reverb_fixture(src, seconds=0.3, rate=16000)
        for mode, code in (("badshape", "INVALID_RESULT"), ("fail", "ENGINE_FAILED")):
            with self.assertRaises(VoiceEnhanceError) as e:
                self.run_dereverb(src, out, mode=mode)
            self.assertEqual(e.exception.code, code)
            self.assertFalse(os.path.exists(out))
            self.assert_no_leftovers()

    def test_output_failing_validation_writes_nothing(self):
        src, out = os.path.join(self.tmp, "room.wav"), os.path.join(self.tmp, "dry.wav")
        reverb_fixture(src, seconds=0.3, rate=16000)
        real_probe = self.skill.probe
        def probe(path):                     # ffmpeg-skill reads the stem as mono when it should be stereo
            facts = real_probe(path)
            if os.path.basename(path) == "stem.wav":
                facts["audio"] = dict(facts["audio"], channels=2)
            return facts
        self.skill.probe = probe
        with self.assertRaises(VoiceEnhanceError) as e:
            self.run_dereverb(src, out)
        self.assertEqual(e.exception.code, "INVALID_RESULT")
        self.assertIn("2 channels, expected 1", e.exception.message)
        self.assertFalse(os.path.exists(out))
        self.assert_no_leftovers()

    def test_cli_dereverb_json(self):
        src, out = os.path.join(self.tmp, "room.wav"), os.path.join(self.tmp, "dry.wav")
        reverb_fixture(src, seconds=0.3, rate=16000)
        site = fake_nara_wpe(os.path.join(self.tmp, "engine"))
        with on_python_path(site), mock.patch("voice_enhance_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("dereverb", src, "-o", out, "--taps", "8", "--json")
            rc2, err = run_cli("dereverb", src, "-o", out, "--json")
            rc3, again = run_cli("dereverb", src, "-o", out, "--overwrite", "--iterations", "1", "--json")
        self.assertEqual((rc, doc["ok"], doc["output"]["path"], doc["parameters"]["taps"]), (0, True, out, 8))
        self.assertEqual((rc2, err["ok"], err["error"]["code"]), (EXIT_CODES["OUTPUT_EXISTS"], False, "OUTPUT_EXISTS"))
        self.assertEqual((rc3, again["parameters"]["iterations"]), (0, 1))


class ChannelStreamsStub(StubSkill):
    """Like StubSkill, but every channel of a WAV is a separate mono audio stream, so one file can stand for a
    recorder that wrote one stream per microphone. audio_to_wav --audio-stream k writes channel k."""

    def probe(self, path):
        facts = super().probe(path)
        ch = facts["audio"]["channels"]
        if os.path.basename(path).startswith("multi"):
            a = dict(facts["audio"], channels=1)
            facts.update(audio=a, audio_streams=[dict(a, index=k) for k in range(ch)])
        return facts

    def audio_to_wav(self, src, dst, audio_stream=None, mono=False, sample_rate=None, timeout=None):
        if not os.path.basename(src).startswith("multi"):
            return super().audio_to_wav(src, dst, audio_stream, mono, sample_rate, timeout)
        self.calls.append({"src": os.path.basename(src), "audio_stream": audio_stream, "mono": mono, "sample_rate": sample_rate})
        rate, channels = read_wav_channels(src)
        write_wav(dst, channels[audio_stream or 0], rate, 1)
        return {"status": "completed", "output": dst, "probe": self.probe(dst)}


def write_mics(directory, fixture, names=("lav_a.wav", "lav_b.wav"), rates=None):
    return [write_wav(os.path.join(directory, name), mic, (rates or {}).get(k, fixture["rate"]))
            for k, (name, mic) in enumerate(zip(names, fixture["mics"]))]


class CrosstalkEngineTests(unittest.TestCase):
    """What the crosstalk engine accepts and reports, checked without NumPy."""

    def test_statuses(self):
        eng = get_engine("crosstalk")
        with mock.patch.object(crosstalk, "_found", return_value=False):
            r = eng.report()
        self.assertEqual((r["status"], r["missing"]), ("MISSING", "package"))
        self.assertIn("[bleed]", r["install"])
        with mock.patch.object(crosstalk, "_found", return_value=True), mock.patch.object(crosstalk, "_numpy_version", return_value="1.26.4"):
            r = eng.report()
        self.assertEqual((r["status"], r["version"], r["install"], r["numpy"]), ("OK", __version__, None, "1.26.4"))
        with mock.patch.object(crosstalk, "_found", return_value=True), mock.patch.object(crosstalk, "_numpy_version", return_value="1.19.5"):
            self.assertEqual(eng.inspect()["status"], "INCOMPATIBLE")
        self.assertEqual((eng.kind, eng.ops, eng.network_required), ("python", ("bleed_reduce",), False))

    def test_params(self):
        self.assertEqual(crosstalk.check_params({}), {"dominance_db": 3.0})
        self.assertEqual(crosstalk.check_params({"dominance_db": 10, "timeout": 5.0}), {"dominance_db": 10.0})
        for bad in ({"dominance_db": -1}, {"dominance_db": 31}, {"dominance_db": True}, {"dominance_db": "3"}):
            with self.assertRaises(VoiceEnhanceError) as e:
                crosstalk.check_params(bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", bad)

    def test_track_count_is_checked_in_the_worker_too(self):
        with mock.patch.object(crosstalk, "_found", return_value=True), mock.patch.object(crosstalk, "_numpy_version", return_value="1.26.4"):
            for ins, outs in ((["a.wav"], ["b.wav"]), (["a.wav", "b.wav"], ["c.wav"])):
                with self.assertRaises(VoiceEnhanceError) as e:
                    get_engine("crosstalk").process(EngineRequest("bleed_reduce", ins[0], outs[0], input_paths=ins, output_paths=outs))
                self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_request_carries_several_tracks(self):
        req = EngineRequest("bleed_reduce", "a.wav", "x.wav", input_paths=["a.wav", "b.wav"], output_paths=["x.wav", "y.wav"])
        self.assertEqual(EngineRequest.from_dict(json.loads(json.dumps(req.to_dict()))), req)
        res = EngineResult("x.wav", "0.1.0", None, output_paths=["x.wav", "y.wav"])
        self.assertEqual(EngineResult.from_dict(json.loads(json.dumps(res.to_dict()))), res)
        self.assertEqual(EngineResult.from_dict({"output_path": "x", "engine_version": "1"}).output_paths, [])


@unittest.skipUnless(NUMPY, NEEDS_NUMPY_BLEED)
class CrosstalkProcessTests(unittest.TestCase):
    """CrosstalkEngine.process through the real worker subprocess, and the signal-processing pieces it is made of."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_track_fixture_cross_talk_energy_falls(self):
        fx = crosstalk_fixture(seconds=2.0, rate=16000)
        srcs = write_mics(self.tmp, fx)
        outs = [os.path.join(self.tmp, "out0.wav"), os.path.join(self.tmp, "out1.wav")]
        res = run_engine("crosstalk", EngineRequest("bleed_reduce", srcs[0], outs[0], sample_rate=16000, input_paths=srcs,
                                                    output_paths=outs), 120, self.tmp)
        self.assertEqual((res.output_paths, res.engine_version, res.model), (outs, __version__, None))
        x = res.extra
        self.assertEqual((x["dominance_db"], x["output_format"], x["stft"]), (3.0, "pcm_f32le", {"window": "hann", "size": 1024, "hop": 256}))
        for i, j in ((0, 1), (1, 0)):          # the measured coupling is the fixture's -8 dB leak
            self.assertAlmostEqual(x["coupling_db"][i][j], -8.0, delta=1.5, msg=x["coupling_db"])
            self.assertGreater(x["estimation_bins"][i][j], 0)
        for k in (0, 1):
            rate, ch, y = read_wav(outs[k])
            self.assertEqual((rate, ch, len(y)), (16000, 1, len(fx["mics"][k])))
            self.assertEqual(Path(outs[k]).read_bytes()[20:22], b"\x03\x00")          # WAVE_FORMAT_IEEE_FLOAT
            x_in = read_wav(srcs[k])[2]
            before, after = energy_db(x_in, rate, fx["other"][k]), energy_db(y, rate, fx["other"][k])
            self.assertLess(after, before - 10.0, (k, before, after))                   # cross-talk energy falls by > 10 dB
            own_before, own_after = energy_db(x_in, rate, fx["own"][k]), energy_db(y, rate, fx["own"][k])
            self.assertAlmostEqual(own_after, own_before, delta=1.0, msg=(k, own_before, own_after))   # its own talker stays

    def test_identity_round_trip_keeps_each_track_and_its_length(self):
        """TrackReader -> STFT -> overlap-add -> TrackWriter with no processing gives every track back sample for sample,
        across block borders, each at its own length (the shorter one is not padded)."""
        import numpy as np
        a = write_wav(os.path.join(self.tmp, "a.wav"), [math.sin(i / 7.0) * (i % 501) / 501 for i in range(9000)], 22050)
        b = write_wav(os.path.join(self.tmp, "b.wav"), [((i * 7919) % 2000 - 1000) / 1500 for i in range(7000)], 22050)
        outs = [os.path.join(self.tmp, "oa.wav"), os.path.join(self.tmp, "ob.wav")]
        reader = wavio.TrackReader([a, b])
        try:
            size, hop = wavio.stft_sizes(22050, crosstalk.HOP_SECONDS)
            writer = wavio.TrackWriter(outs, 22050, reader.lengths)
            try:
                wavio.stft_process(reader, writer, np, size, hop, 5, 0, lambda spec, skip: spec[skip:])
            finally:
                writer.close()
        finally:
            reader.close()
        for src, out in zip((a, b), outs):
            x, y = read_wav(src)[2], read_wav(out)[2]
            self.assertEqual(len(y), len(x))
            self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-6)

    def test_track_reader_refuses_mixed_rates_and_stereo(self):
        a = write_wav(os.path.join(self.tmp, "a.wav"), [0.1] * 100, 16000)
        b = write_wav(os.path.join(self.tmp, "b.wav"), [0.1] * 100, 48000)
        c = write_wav(os.path.join(self.tmp, "c.wav"), [0.1] * 100, 16000, channels=2)
        for pair in ((a, b), (a, c)):
            with self.assertRaises(VoiceEnhanceError):
                wavio.TrackReader(pair)

    def test_demixing_is_the_inverse_unless_badly_conditioned(self):
        import numpy as np
        h = np.zeros((2, 2, 2), dtype=complex)
        h[0, 0, 1], h[0, 1, 0] = 0.3, 0.2 * np.exp(1j)            # bin 0: a moderate leak both ways
        h[1, 0, 1] = h[1, 1, 0] = 0.98                             # bin 1: I + H is nearly singular
        w, fallback = crosstalk.demixing_matrices(np, h)
        self.assertEqual(fallback, 1)
        self.assertLess(float(np.max(np.abs(w[0] @ (np.eye(2) + h[0]) - np.eye(2)))), 1e-12)
        self.assertLess(float(np.max(np.abs(w[1] - (np.eye(2) - h[1])))), 1e-12)
        num = np.ones((3, 2, 2), dtype=complex)
        den = np.array([[[0, 2], [4, 0]], [[0, 0], [0, 0]], [[0, 1], [1, 0]]], dtype=float)
        c = crosstalk.coupling(np, num, den)                     # no usable bins -> nothing is subtracted
        self.assertEqual((c[0, 0, 1], c[0, 1, 0], c[1, 0, 1], c[0, 0, 0]), (0.5, 0.25, 0, 0))


class BleedReducePlanningTests(unittest.TestCase):
    """The checks BLEED_REDUCE makes before any media work. The engine is reported OK through mocked lookups, so these
    run without NumPy; nothing here reaches the engine."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = StubSkill()
        self.a = write_tone_wav(os.path.join(self.tmp, "a.wav"), 0.1)
        self.b = write_tone_wav(os.path.join(self.tmp, "b.wav"), 0.1, freq=300.0)
        self.out = os.path.join(self.tmp, "clean")
        patches = (mock.patch.object(crosstalk, "_found", return_value=True), mock.patch.object(crosstalk, "_numpy_version", return_value="1.26.4"))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def refused(self, code, req, skill=None):
        with self.assertRaises(VoiceEnhanceError) as e:
            bleed_reduce(req, skill=skill or self.skill)
        self.assertEqual(e.exception.code, code, e.exception.message)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".voice-enhance-")], [])
        return e.exception

    def test_numbers_and_paths(self):
        a, b = self.a, self.b
        for kw in ({"timeout": 0}, {"audio_streams": [-1, 0]}, {"audio_streams": []}, {"audio_streams": [0]}, {"dominance_db": 31},
                   {"dominance_db": -0.5}):
            self.refused("INVALID_ARGUMENT", BleedReduceRequest([a, b], self.out, **kw))
        self.refused("INVALID_ARGUMENT", BleedReduceRequest([a] * 9, self.out))
        self.refused("INPUT_NOT_FOUND", BleedReduceRequest([a, os.path.join(self.tmp, "nope.wav")], self.out))
        self.refused("INVALID_ARGUMENT", BleedReduceRequest([a, b], os.path.join(self.tmp, "no-dir", "clean")))
        self.refused("INVALID_ARGUMENT", BleedReduceRequest([a, b], a))                     # -o names a directory, not a file
        self.assertEqual(self.skill.calls, [])

    def test_tracks(self):
        e = self.refused("INVALID_ARGUMENT", BleedReduceRequest([self.a], self.out))        # one file with one stream: one track
        self.assertEqual(e.details["tracks"], 1)
        self.refused("INVALID_ARGUMENT", BleedReduceRequest([self.a, self.a], self.out))    # the same stream twice
        os.mkdir(os.path.join(self.tmp, "other"))
        twin = write_tone_wav(os.path.join(self.tmp, "other", "a.wav"), 0.1)
        e = self.refused("INVALID_ARGUMENT", BleedReduceRequest([self.a, twin], self.out))  # both would be a_a0.wav
        self.assertEqual(e.details["names"], ["a_a0.wav", "a_a0.wav"])
        self.refused("INVALID_ARGUMENT", BleedReduceRequest([self.a, self.b], self.out, audio_streams=[0, 1]))   # b has one stream
        self.assertEqual(self.skill.calls, [])
        self.assertEqual(track_name("/x/talk.take2.mkv", 3), "talk.take2_a3.wav")

    def test_outputs(self):
        os.mkdir(self.out)
        Path(self.out, "a_a0.wav").write_bytes(b"old")
        with self.assertRaises(VoiceEnhanceError) as e:
            bleed_reduce(BleedReduceRequest([self.a, self.b], self.out), skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
        self.assertEqual(Path(self.out, "a_a0.wav").read_bytes(), b"old")
        self.assertEqual(self.skill.calls, [])

    @unittest.skipUnless(POSIX, "uses a symlink")
    def test_output_that_is_an_input_is_refused_even_with_overwrite(self):
        os.mkdir(self.out)
        target = write_tone_wav(os.path.join(self.out, "c_a0.wav"), 0.1)
        link = os.path.join(self.tmp, "c.wav")              # its output would be clean/c_a0.wav: the file it points to
        os.symlink(target, link)
        with self.assertRaises(VoiceEnhanceError) as e:
            bleed_reduce(BleedReduceRequest([link, self.b], self.out, overwrite=True), skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_IS_INPUT")
        self.assertEqual(self.skill.calls, [])

    def test_different_rates_without_audio_sample_rate_are_refused_before_any_work(self):
        c = write_tone_wav(os.path.join(self.tmp, "c.wav"), 0.1, rate=44100)
        e = self.refused("FFMPEG_SKILL_UNAVAILABLE", BleedReduceRequest([self.a, c], self.out))
        self.assertEqual((e.details["needs_flag"], e.details["sample_rates"], e.details["required_sample_rate"]),
                         ("sample_rate", [48000, 44100], 48000))
        self.assertEqual(self.skill.calls, [])

    def test_missing_engine_is_refused_before_any_media_work(self):
        with mock.patch.object(crosstalk, "_found", return_value=False):
            e = self.refused("ENGINE_UNAVAILABLE", BleedReduceRequest([self.a, self.b], self.out))
            rc, doc = run_cli("bleed-reduce", self.a, self.b, "-o", self.out, "--json")
        self.assertIn("[bleed]", e.details["install"])
        self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (3, False, "ENGINE_UNAVAILABLE"))
        self.assertEqual(self.skill.calls, [])

    def test_cli_argument_errors_are_one_json_document(self):
        rc, doc = run_cli("bleed-reduce", self.a, os.path.join(self.tmp, "nope.wav"), "-o", self.out, "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "INPUT_NOT_FOUND"))
        rc, doc = run_cli("bleed-reduce", self.a, self.b, "-o", self.out, "--dominance-db", "40", "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "INVALID_ARGUMENT"))
        rc, doc = run_cli("bleed-reduce", self.a, self.b, "-o", self.out, "--audio-streams", "0", "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "INVALID_ARGUMENT"))


@unittest.skipUnless(NUMPY, NEEDS_NUMPY_BLEED)
class BleedReduceOpTests(unittest.TestCase):
    """BLEED_REDUCE end to end with the crosstalk engine in the worker and the stub ffmpeg-skill."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.skill = StubSkill()
        self.out = os.path.join(self.tmp, "clean")

    def tearDown(self):
        self._tmp.cleanup()

    def assert_no_leftovers(self):
        for d in (self.tmp, self.out):
            if os.path.isdir(d):
                self.assertEqual([n for n in os.listdir(d) if n.startswith(".voice-enhance-")], [])

    def assert_cross_talk_falls(self, fx, srcs, outs, channels_of=None):
        for k, out in enumerate(outs):
            rate, ch, y = read_wav(out)
            x = fx["mics"][k]
            self.assertEqual((rate, ch, len(y)), (fx["rate"], 1, len(x)))
            before, after = energy_db(x, rate, fx["other"][k]), energy_db(y, rate, fx["other"][k])
            self.assertLess(after, before - 10.0, (k, before, after))
            self.assertAlmostEqual(energy_db(y, rate, fx["own"][k]), energy_db(x, rate, fx["own"][k]), delta=1.0)

    def test_two_track_fixture_cross_talk_energy_falls(self):
        fx = crosstalk_fixture(seconds=2.0, rate=16000)
        srcs = write_mics(self.tmp, fx)
        shas = [sha256_file(p) for p in srcs]
        doc = bleed_reduce(BleedReduceRequest(srcs, self.out), skill=self.skill)
        outs = [os.path.join(self.out, "lav_a_a0.wav"), os.path.join(self.out, "lav_b_a0.wav")]
        self.assertEqual([sha256_file(p) for p in srcs], shas)                 # the inputs are never modified
        self.assertEqual(sorted(os.listdir(self.out)), ["lav_a_a0.wav", "lav_b_a0.wav"])   # created whole, nothing else in it
        self.assert_cross_talk_falls(fx, srcs, outs)
        self.assertEqual((doc["schema"], doc["ok"], doc["op"], doc["capability"]),
                         (f"{SKILL_ID}/bleed-reduce@1", True, "bleed_reduce", "voice.bleed_reduce"))
        self.assertEqual(doc["output_dir"], os.path.abspath(self.out))
        self.assertEqual([(o["track"], o["path"], o["sample_rate"], o["channels"]) for o in doc["outputs"]],
                         [(0, outs[0], 16000, 1), (1, outs[1], 16000, 1)])
        self.assertEqual([(i["track"], i["path"], i["audio_stream"]) for i in doc["inputs"]],
                         [(0, os.path.abspath(srcs[0]), 0), (1, os.path.abspath(srcs[1]), 0)])
        self.assertEqual(doc["engine"], {"id": "crosstalk", "version": __version__, "model": None})
        p = doc["parameters"]
        self.assertEqual((p["tracks"], p["audio_streams"], p["dominance_db"], p["sample_rate"], p["output_format"]),
                         (2, [0, 0], 3.0, 16000, "pcm_f32le"))
        self.assertAlmostEqual(p["coupling_db"][0][1], -8.0, delta=1.5)
        self.assertEqual([(st["track"], st["flags"]) for st in doc["steps"]], [(0, ["--audio-stream", "0"]), (1, ["--audio-stream", "0"])])
        prov = doc["provenance"]
        self.assertEqual((prov["op"], prov["inputs"], prov["outputs"]),
                         ("bleed_reduce", [{"sha256": s} for s in shas], [{"sha256": sha256_file(o)} for o in outs]))
        self.assertNotIn("input", prov)
        self.assertEqual((prov["engine"]["id"], prov["ffmpeg_skill"]["version"]), ("crosstalk", "2.22.4"))
        self.assertEqual(doc["warnings"], [])
        json.dumps(doc, allow_nan=False)
        self.assert_no_leftovers()

    def test_streams_of_one_file_and_stereo_fold_down(self):
        """One recorder file with one stream per microphone: all its streams are the tracks. A stereo track is folded
        down to mono in a second ffmpeg-skill pass, as for ISOLATE."""
        fx = crosstalk_fixture(seconds=2.0, rate=16000)
        self.skill = ChannelStreamsStub()
        multi = write_wav_tracks(os.path.join(self.tmp, "multi.wav"), fx["mics"], 16000)
        doc = bleed_reduce(BleedReduceRequest([multi], self.out), skill=self.skill)
        self.assertEqual([o["path"] for o in doc["outputs"]], [os.path.join(self.out, "multi_a0.wav"), os.path.join(self.out, "multi_a1.wav")])
        self.assertEqual([(i["audio_stream"], i["audio_stream_count"]) for i in doc["inputs"]], [(0, 2), (1, 2)])
        self.assert_cross_talk_falls(fx, None, [o["path"] for o in doc["outputs"]])
        shutil.rmtree(self.out)
        stereo = write_wav(os.path.join(self.tmp, "stereo_b.wav"), fx["mics"][1], 16000, channels=2)
        mono = write_wav(os.path.join(self.tmp, "mono_a.wav"), fx["mics"][0], 16000)
        doc = bleed_reduce(BleedReduceRequest([mono, stereo], self.out), skill=StubSkill())
        self.assertEqual([(st["track"], st["flags"]) for st in doc["steps"]],
                         [(0, ["--audio-stream", "0"]), (1, ["--audio-stream", "0"]), (1, ["--mono"])])
        self.assertIn("track 1: audio stream 0 has 2 channels; it was folded down to mono", doc["warnings"][0])
        self.assert_cross_talk_falls(fx, None, [o["path"] for o in doc["outputs"]])

    def test_audio_streams_picks_tracks(self):
        fx = crosstalk_fixture(seconds=1.0, rate=16000)
        self.skill = ChannelStreamsStub()
        three = write_wav_tracks(os.path.join(self.tmp, "multi3.wav"), [fx["mics"][0], [0.0] * 100, fx["mics"][1]], 16000)
        doc = bleed_reduce(BleedReduceRequest([three], self.out, audio_streams=[2, 0], dominance_db=6.0), skill=self.skill)
        self.assertEqual([o["path"] for o in doc["outputs"]], [os.path.join(self.out, "multi3_a2.wav"), os.path.join(self.out, "multi3_a0.wav")])
        self.assertEqual((doc["parameters"]["audio_streams"], doc["parameters"]["dominance_db"]), ([2, 0], 6.0))
        self.assertEqual([c["audio_stream"] for c in self.skill.calls], [2, 0])

    def test_existing_directory_and_overwrite(self):
        fx = crosstalk_fixture(seconds=1.0, rate=16000)
        srcs = write_mics(self.tmp, fx)
        os.mkdir(self.out)
        Path(self.out, "keep.txt").write_text("mine", encoding="utf-8")
        bleed_reduce(BleedReduceRequest(srcs, self.out), skill=self.skill)
        self.assertEqual(sorted(os.listdir(self.out)), ["keep.txt", "lav_a_a0.wav", "lav_b_a0.wav"])
        with self.assertRaises(VoiceEnhanceError) as e:
            bleed_reduce(BleedReduceRequest(srcs, self.out), skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
        doc = bleed_reduce(BleedReduceRequest(srcs, self.out, overwrite=True), skill=self.skill)
        self.assertTrue(doc["ok"])
        self.assertEqual(Path(self.out, "keep.txt").read_text(encoding="utf-8"), "mine")
        self.assert_no_leftovers()

    def test_output_failing_validation_writes_nothing(self):
        fx = crosstalk_fixture(seconds=1.0, rate=16000)
        srcs = write_mics(self.tmp, fx)
        real_probe = self.skill.probe
        def probe(path):                     # ffmpeg-skill reads the second stem at the wrong rate
            facts = real_probe(path)
            if os.path.basename(path) == "lav_b_a0.wav" and ".voice-enhance-" in path:
                facts["audio"] = dict(facts["audio"], sample_rate=8000)
            return facts
        self.skill.probe = probe
        with self.assertRaises(VoiceEnhanceError) as e:
            bleed_reduce(BleedReduceRequest(srcs, self.out), skill=self.skill)
        self.assertEqual(e.exception.code, "INVALID_RESULT")
        self.assertIn("sample rate 8000", e.exception.message)
        self.assertFalse(os.path.exists(self.out))                          # not even the first, valid stem
        self.assert_no_leftovers()

    def test_warnings_for_unaligned_lengths_and_unmeasurable_pairs(self):
        fx = crosstalk_fixture(seconds=1.0, rate=16000)
        a = write_wav(os.path.join(self.tmp, "a.wav"), fx["mics"][0], 16000)
        quiet = write_wav(os.path.join(self.tmp, "quiet.wav"), [0.0] * 12000, 16000)     # never louder than a, and shorter
        doc = bleed_reduce(BleedReduceRequest([a, quiet], self.out), skill=self.skill)
        self.assertTrue(any("differ in length" in w for w in doc["warnings"]), doc["warnings"])
        self.assertTrue(any("track 1 was never louder than track 0" in w for w in doc["warnings"]), doc["warnings"])
        self.assertIsNone(doc["parameters"]["coupling_db"][0][1])
        self.assertEqual([o["duration"] for o in doc["outputs"]], [1.0, 0.75])                # each keeps its own length
        x, y = read_wav(a)[2], read_wav(doc["outputs"][0]["path"])[2]
        self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-4)                          # nothing removed from a

    def test_cli_bleed_reduce_json(self):
        fx = crosstalk_fixture(seconds=1.0, rate=16000)
        srcs = write_mics(self.tmp, fx)
        with mock.patch("voice_enhance_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("bleed-reduce", *srcs, "-o", self.out, "--audio-streams", "0,0", "--dominance-db", "4", "--json")
            rc2, err = run_cli("bleed-reduce", *srcs, "-o", self.out, "--json")
            rc3, again = run_cli("bleed-reduce", *srcs, "-o", self.out, "--overwrite", "--json")
        self.assertEqual((rc, doc["ok"], len(doc["outputs"]), doc["parameters"]["dominance_db"]), (0, True, 2, 4.0))
        self.assertEqual((rc2, err["ok"], err["error"]["code"]), (EXIT_CODES["OUTPUT_EXISTS"], False, "OUTPUT_EXISTS"))
        self.assertEqual((rc3, again["ok"]), (0, True))


class FfmpegSkillContractTests(unittest.TestCase):
    def doc(self, drop=None):
        return fake_ffmpeg_contract(drop)

    def test_capability_detection(self):
        info = parse_contract(self.doc(), Path("."))
        self.assertTrue(info.supported, info.problems)
        self.assertEqual(info.version, "2.22.4")
        bad = parse_contract(self.doc(drop=("audio", "replace")), Path("."))
        self.assertFalse(bad.supported)
        self.assertIn("replace", bad.problems[0])
        missing_tool = self.doc()
        missing_tool["tools"] = [t for t in missing_tool["tools"] if t["name"] != "audio"]
        self.assertIn("'audio' is missing", parse_contract(missing_tool, Path("."))
                      .problems[0])
        self.assertFalse(info.takes_flag("audio", "sample_rate"))
        self.assertEqual(info.to_dict()["optional_flags"], {"audio": {"sample_rate": False}})
        with_rate = parse_contract(fake_ffmpeg_contract(extra=[("audio", "sample_rate")]), Path("."))
        self.assertTrue(with_rate.supported and with_rate.takes_flag("audio", "sample_rate"))
        self.assertTrue(parse_contract(dict(self.doc(), contract_version="2.0"), Path(".")).warnings)

    def test_locate_explicit_dir_is_never_replaced(self):
        with self.assertRaises(VoiceEnhanceError) as e:
            FfmpegSkill.locate(os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill"))
        self.assertEqual(e.exception.code, "FFMPEG_SKILL_UNAVAILABLE")
        self.assertEqual(len(e.exception.details["tried"]), 1)


class DoctorTests(unittest.TestCase):
    def test_missing_engines_are_reported_missing(self):
        missing = os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill")
        with env(**NO_DEEP_FILTER), mock.patch.object(nara_wpe_engine, "_found", return_value=False), \
                mock.patch.object(crosstalk, "_found", return_value=False):
            d = doctor(missing)
            rc, cli_doc = run_cli("doctor", "--ffmpeg-skill", missing, "--offline", "--json")
        for eid in ("crosstalk", "deepfilternet", "nara_wpe"):
            self.assertEqual(d["checks"]["engines"][eid]["status"], "MISSING")
            self.assertTrue(d["checks"]["engines"][eid]["install"])
        self.assertEqual(d["unavailable_engines"], ["crosstalk", "deepfilternet", "nara_wpe"])
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "MISSING")
        self.assertEqual((d["status"], d["ok"]), ("fail", False))
        self.assertEqual(set(d["checks"]["operations"]), set(OPS))
        self.assertFalse(any(o["ready"] for o in d["checks"]["operations"].values()))
        self.assertEqual({op: o["implemented"] for op, o in d["checks"]["operations"].items()},
                         {"isolate": True, "dereverb": True, "bleed_reduce": True})
        self.assertEqual(d["checks"]["operations"]["bleed_reduce"]["engine"], "crosstalk")
        self.assertEqual((rc, cli_doc["status"], cli_doc["offline"]), (1, "fail", True))

    def test_engines_command(self):
        with env(**NO_DEEP_FILTER):
            rc, doc = run_cli("engines", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(doc["engines"]["deepfilternet"]["status"], "MISSING")
        self.assertTrue(doc["engines"]["nara_wpe"]["usable_offline"])


class ProvenanceTests(unittest.TestCase):
    def test_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_tone_wav(os.path.join(tmp, "a.wav"), 0.1)
            report = dict(get_engine("nara_wpe").spec(), version="0.0.11", model=None, path="/secret/local/path")
            p = build_provenance(op="dereverb", engine=report, input_path=a, output_path=a, parameters={"taps": 10})
            self.assertEqual(p["input"]["sha256"], sha256_file(a))
            self.assertEqual(p["engine"]["id"], "nara_wpe")
            self.assertNotIn("path", p["engine"])
            self.assertFalse(p["network"])
            self.assertEqual(p["skill"], {"id": SKILL_ID, "version": __version__})
            b = write_tone_wav(os.path.join(tmp, "b.wav"), 0.1, freq=300.0)
            many = build_provenance(op="bleed_reduce", engine=report, input_path=[a, b], output_path=[b, a], parameters={})
            self.assertEqual((many["inputs"], many["outputs"]), ([{"sha256": sha256_file(a)}, {"sha256": sha256_file(b)}],
                                                                 [{"sha256": sha256_file(b)}, {"sha256": sha256_file(a)}]))
            self.assertNotIn("input", many)


@unittest.skipUnless(OS_DIR, "needs an AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS)")
class ConformanceTests(unittest.TestCase):
    def test_registry_checks_pass(self):
        sys.path.insert(0, str(OS_DIR))
        try:
            from registry import conformance as conf
        finally:
            sys.path.remove(str(OS_DIR))
        results = conf.run_static_checks(contract())
        results.append(conf.check_no_unsafe_shell_out(str(ROOT / "src" / "voice_enhance_skill")))
        results.append(conf.check_doctor_status(lambda: doctor()))
        self.assertEqual([r.check for r in results], ["publishes_contract", "lifecycle_declared", "dependency_version_ranges",
                                                      "no_unsafe_shell_out", "doctor_status"])
        for r in results:
            self.assertEqual(r.status, "PASS", f"{r.check}: {r.detail}")


if __name__ == "__main__":
    unittest.main()
