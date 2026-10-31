"""Unit tests: contract, typed errors, engine registry and availability checks (with a fake deep-filter binary and
mocked package lookups), the worker subprocess, doctor, provenance and the OS registry conformance checks. No
engine, no weights and no ffmpeg are needed; nothing is downloaded."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import OS_DIR, ROOT, env, fake_deep_filter, run_cli, write_tone_wav

from voice_enhance_skill import SKILL_ID, __version__
from voice_enhance_skill.contract import CAPABILITY_IDS, contract
from voice_enhance_skill.doctor import doctor
from voice_enhance_skill.engines import DEFAULT_ENGINE, engine_ids, engines_for, get_engine
from voice_enhance_skill.engines import nara_wpe_engine
from voice_enhance_skill.engines.base import OPS, EngineRequest, EngineResult, parse_version
from voice_enhance_skill.errors import ERROR_TABLE, EXIT_CODES, VoiceEnhanceError
from voice_enhance_skill.ffmpeg_skill import FLAGS_USED, FfmpegSkill, parse_contract
from voice_enhance_skill.procs import TIMEOUT_EXIT, run_process_group
from voice_enhance_skill.provenance import build_provenance, sha256_file
from voice_enhance_skill.service import run_engine

POSIX = os.name == "posix"
NO_DEEP_FILTER = {"VOICE_ENHANCE_DEEP_FILTER": os.path.join(tempfile.gettempdir(), "no-such-deep-filter"),
                  "VOICE_ENHANCE_DFN_MODEL": None}


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
            self.assertEqual(p["lifecycle"], "PROPOSED")       # no op is implemented in this release (ADR-002)
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
        self.assertEqual(engine_ids(), ["deepfilternet", "nara_wpe"])
        self.assertEqual(set(DEFAULT_ENGINE), set(OPS))
        self.assertEqual(engines_for("isolate"), ["deepfilternet"])
        self.assertEqual(engines_for("dereverb"), ["nara_wpe"])
        self.assertEqual(engines_for("bleed_reduce"), [])      # ADR-001: no AI engine for cross-talk
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
    def test_installed_engine_reaches_process_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "DeepFilterNet3_onnx.tar.gz")
            Path(model).write_bytes(b"x")
            with env(VOICE_ENHANCE_DEEP_FILTER=fake_deep_filter(tmp), VOICE_ENHANCE_DFN_MODEL=model), \
                    self.assertRaises(VoiceEnhanceError) as e:
                run_engine("deepfilternet", EngineRequest("isolate", "a.wav", "b.wav"), 60, tmp)
        self.assertEqual(e.exception.code, "NOT_IMPLEMENTED")      # step 3 implements process()
        self.assertEqual(e.exception.details["op"], "isolate")

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


class FfmpegSkillContractTests(unittest.TestCase):
    def doc(self, drop=None):
        tools = []
        for name, flags in FLAGS_USED.items():
            props = {f: {"type": "string"} for f in flags if (name, f) != drop}
            tools.append({"name": name, "audio_only": True, "input_schema": {"properties": props}})
        return {"skill": {"version": "2.22.4"}, "contract_version": "1.0", "tools": tools}

    def test_capability_detection(self):
        info = parse_contract(self.doc(), Path("."))
        self.assertTrue(info.supported, info.problems)
        self.assertEqual(info.version, "2.22.4")
        bad = parse_contract(self.doc(drop=("audio", "replace")), Path("."))
        self.assertFalse(bad.supported)
        self.assertIn("replace", bad.problems[0])
        missing_tool = self.doc()
        missing_tool["tools"] = [t for t in missing_tool["tools"] if t["name"] != "join"]
        self.assertIn("'join' is missing", parse_contract(missing_tool, Path("."))
                      .problems[0])
        self.assertTrue(parse_contract(dict(self.doc(), contract_version="2.0"), Path(".")).warnings)

    def test_locate_explicit_dir_is_never_replaced(self):
        with self.assertRaises(VoiceEnhanceError) as e:
            FfmpegSkill.locate(os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill"))
        self.assertEqual(e.exception.code, "FFMPEG_SKILL_UNAVAILABLE")
        self.assertEqual(len(e.exception.details["tried"]), 1)


class DoctorTests(unittest.TestCase):
    def test_missing_engines_are_reported_missing(self):
        missing = os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill")
        with env(**NO_DEEP_FILTER), mock.patch.object(nara_wpe_engine, "_found", return_value=False):
            d = doctor(missing)
            rc, cli_doc = run_cli("doctor", "--ffmpeg-skill", missing, "--offline", "--json")
        for eid in ("deepfilternet", "nara_wpe"):
            self.assertEqual(d["checks"]["engines"][eid]["status"], "MISSING")
            self.assertTrue(d["checks"]["engines"][eid]["install"])
        self.assertEqual(d["unavailable_engines"], ["deepfilternet", "nara_wpe"])
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "MISSING")
        self.assertEqual((d["status"], d["ok"]), ("fail", False))
        self.assertEqual(set(d["checks"]["operations"]), set(OPS))
        self.assertFalse(any(o["ready"] for o in d["checks"]["operations"].values()))
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
