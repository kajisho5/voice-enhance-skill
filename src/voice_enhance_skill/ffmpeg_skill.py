"""ffmpeg-skill adapter. This skill runs no ffmpeg of its own: probing, format conversion for an engine and putting a
cleaned stem back into a video are ffmpeg-skill tool calls (ADR-001, ADR-002).

- Locates an ffmpeg-skill checkout (explicit dir > VOICE_ENHANCE_FFMPEG_SKILL_DIR > VIDEO_AGENT_FFMPEG_SKILL_DIR >
  ~/.claude/skills/ffmpeg-skill > ./vendor/ffmpeg-skill > ../ffmpeg-skill) and reads its contract
  (`scripts/_contract.py --json --static`). Compatibility is capability detection, never the version number: every
  tool in TOOLS_USED must be in the contract and every flag in FLAGS_USED in its input_schema (audio-production-skill
  ADR-15).
- Runs one named tool as [sys.executable, <dir>/scripts/<tool>.py, <typed argv...>] through procs.run_process_group.

Which ffmpeg-skill tools are used, and for what (ADR-003):
  probe   input facts before an op (duration, audio streams), output validation after it
  audio   `-o <x.wav> --audio-stream N`: the chosen audio stream as PCM WAV (the picture is dropped); a second pass
          `--mono` on that one-stream WAV when it has more than one channel; `--sample-rate` only when this
          checkout declares it (OPTIONAL_FLAGS, capability-detected). DEREVERB uses the first pass only: WPE takes any
          rate and channel count (ADR-004). BLEED_REDUCE prepares every track like ISOLATE, at the first
          track's rate (ADR-005). REMUX (step 6, ADR-006) is `<video> --replace <stem> -o <out>`: the stem
          becomes the only audio stream, encoded by the output's extension, padded or trimmed to the input's
          duration, the first video stream copied packet for packet (-c:v copy)
`join` is not used: it refuses a single clip, so it cannot convert one file (ADR-003). `--overwrite` is never passed:
every ffmpeg-skill output is a fresh file in a private run directory, renamed into place by this skill (ADR-006).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import VoiceEnhanceError
from .procs import TIMEOUT_EXIT, run_process_group

# contract_version values ffmpeg-skill declares: "1.0" (0.x-2.x) and "2.0" (3.x, whose bump changed only cut.py's
# precision / keyframe_snapped meaning -- a tool this skill never calls; ADR-008). Reported, not gated on; only a value
# outside these majors is a warning. SUPPORTED_CONTRACT_VERSION is the oldest.
KNOWN_CONTRACT_VERSIONS = ("1.0", "2.0")
SUPPORTED_CONTRACT_VERSION = KNOWN_CONTRACT_VERSIONS[0]
ENV_DIR_KEYS = ("VOICE_ENHANCE_FFMPEG_SKILL_DIR", "VIDEO_AGENT_FFMPEG_SKILL_DIR")
TOOLS_USED = ("probe", "audio")
FLAGS_USED: Dict[str, Tuple[str, ...]] = {
    "probe": ("inputs",),
    "audio": ("input", "output", "replace", "audio_stream", "mono", "json"),
}
# Used when the checkout declares them, never required: without audio --sample-rate an input whose audio is not at the
# engine's rate is refused with FFMPEG_SKILL_UNAVAILABLE (ADR-003).
OPTIONAL_FLAGS: Dict[str, Tuple[str, ...]] = {"audio": ("sample_rate",)}
AUDIO_ONLY_TOOLS = ("probe", "audio")


@dataclass
class SkillInfo:
    directory: Path
    version: str
    contract_version: str
    tools: Dict[str, Dict[str, Any]]
    problems: List[str]
    warnings: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.problems

    def takes_flag(self, tool: str, flag: str) -> bool:
        props = ((self.tools.get(tool) or {}).get("input_schema") or {}).get("properties") or {}
        return isinstance(props, dict) and flag in props

    def to_dict(self) -> Dict[str, Any]:
        return {"directory": str(self.directory), "version": self.version, "contract_version": self.contract_version,
                "tools_used": [t for t in TOOLS_USED if t in self.tools], "problems": list(self.problems),
                "warnings": list(self.warnings),
                "optional_flags": {t: {f: self.takes_flag(t, f) for f in fl} for t, fl in OPTIONAL_FLAGS.items()},
                "compatibility": "capability detection: tools and flags read from this checkout's contract; no version window"}


def candidates(explicit: Optional[str] = None) -> List[Path]:
    if explicit:
        return [Path(explicit)]          # an explicit directory is never silently replaced by a fallback
    out = [Path(os.environ[k]) for k in ENV_DIR_KEYS if os.environ.get(k)]
    out += [Path.home() / ".claude" / "skills" / "ffmpeg-skill", Path.cwd() / "vendor" / "ffmpeg-skill",
            Path.cwd().parent / "ffmpeg-skill"]
    return out


def parse_contract(doc: Dict[str, Any], directory: Path) -> SkillInfo:
    """What this skill needs from an ffmpeg-skill contract document. Pure: no process, so it is unit-testable."""
    version = str((doc.get("skill") or {}).get("version", "unknown")) if isinstance(doc.get("skill"), dict) else "unknown"
    contract_version = str(doc.get("contract_version", "unknown"))
    tools = {t["name"]: t for t in doc.get("tools", []) if isinstance(t, dict) and "name" in t}
    problems: List[str] = []
    warnings: List[str] = []
    if contract_version != "unknown" and contract_version.split(".")[0] not in {v.split(".")[0] for v in KNOWN_CONTRACT_VERSIONS}:
        warnings.append(f"ffmpeg-skill contract_version {contract_version} is not one of {', '.join(KNOWN_CONTRACT_VERSIONS)}; "
                        "compatibility was decided by the tool / flag check, not by this number")
    for tool, flags in FLAGS_USED.items():
        spec = tools.get(tool)
        if spec is None:
            problems.append(f"ffmpeg-skill tool {tool!r} is missing from its contract")
            continue
        props = (spec.get("input_schema") or {}).get("properties") or {}
        missing = [f for f in flags if f not in props]
        if missing:
            problems.append(f"ffmpeg-skill/{tool} lacks flag(s) {missing}")
        if tool in AUDIO_ONLY_TOOLS and spec.get("audio_only") is not True:
            problems.append(f"ffmpeg-skill/{tool} does not declare audio_only")
    return SkillInfo(directory, version, contract_version, tools, problems, warnings)


class FfmpegSkill:
    """Handle on one ffmpeg-skill checkout."""

    def __init__(self, directory: Path, timeout: float = 600.0) -> None:
        self.directory = directory
        self.timeout = timeout
        self._info: Optional[SkillInfo] = None

    @classmethod
    def locate(cls, explicit: Optional[str] = None, timeout: float = 600.0) -> "FfmpegSkill":
        tried = []
        for c in candidates(explicit):
            tried.append(str(c))
            if (c / "scripts" / "_contract.py").is_file() and all((c / "scripts" / f"{t}.py").is_file() for t in TOOLS_USED):
                return cls(c.resolve(), timeout)
        raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", "ffmpeg-skill not found (need scripts/_contract.py and "
                                f"scripts/{{{','.join(TOOLS_USED)}}}.py)", {"tried": tried})

    def script(self, name: str) -> str:
        if name not in TOOLS_USED + ("_contract",):
            raise VoiceEnhanceError("INTERNAL_ERROR", f"{name!r} is not on the adapter allowlist")
        return str(self.directory / "scripts" / f"{name}.py")

    def _run(self, args: List[str], timeout: float) -> Tuple[int, str, str]:
        code, out, err = run_process_group([sys.executable, *args], timeout, cwd=str(self.directory))
        if code == TIMEOUT_EXIT:
            raise VoiceEnhanceError("TOOL_ERROR", f"ffmpeg-skill/{Path(args[0]).stem} exceeded {timeout}s", {"reason": "timeout"})
        return code, out, err

    def info(self, timeout: float = 60.0) -> SkillInfo:
        if self._info is None:
            code, out, err = self._run([self.script("_contract"), "--json", "--static"], timeout)
            try:
                doc = json.loads(out or "{}")
            except ValueError:
                doc = {}
            if not isinstance(doc, dict) or not doc.get("tools"):
                tail = err.strip().splitlines()[-1] if err.strip() else "no output"
                self._info = SkillInfo(self.directory, "unknown", "unknown", {}, [f"cannot read ffmpeg-skill contract (exit {code}): {tail}"])
            else:
                self._info = parse_contract(doc, self.directory)
        return self._info

    def doctor(self, timeout: float = 120.0) -> Dict[str, Any]:
        """ffmpeg-skill's own doctor (`_contract.py doctor --json`): ffmpeg / ffprobe versions and filters."""
        code, out, _ = self._run([self.script("_contract"), "doctor", "--json"], timeout)
        try:
            doc = json.loads(out or "{}")
        except ValueError:
            doc = {}
        doc = doc if isinstance(doc, dict) else {}
        doc["_exit_code"] = code
        return doc

    def probe(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """ffmpeg-skill/probe prints its document directly (no status envelope). Paths are passed absolute, so they can
        never be read as an option."""
        code, out, err = self._run([self.script("probe"), os.path.abspath(path)], timeout or min(self.timeout, 120.0))
        data = _parse_json(out)
        if code != 0 or "duration" not in data:
            tail = err.strip().splitlines()[-1] if err.strip() else "no output"
            raise VoiceEnhanceError("UNSUPPORTED_MEDIA", f"ffmpeg-skill/probe could not read {os.path.basename(path)}: {tail}",
                                    {"tool": "ffmpeg-skill/probe", "exit_code": code})
        return data

    def audio_to_wav(self, src: str, dst: str, *, audio_stream: Optional[int] = None, mono: bool = False,
                     sample_rate: Optional[int] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """ffmpeg-skill/audio with a .wav output: one audio stream of `src` as PCM WAV, the picture dropped. `dst` must
        not exist (it is always a fresh file in a private run directory, so --overwrite is never passed here).
        Returns ffmpeg-skill's result document (its `probe` describes `dst`)."""
        args = [self.script("audio"), os.path.abspath(src), "-o", os.path.abspath(dst), "--json"]
        if audio_stream is not None:
            args += ["--audio-stream", str(int(audio_stream))]
        if mono:
            args.append("--mono")
        if sample_rate is not None:
            if not self.info().takes_flag("audio", "sample_rate"):
                raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", f"resampling to {int(sample_rate)} Hz needs ffmpeg-skill/audio "
                                        "--sample-rate, which this ffmpeg-skill checkout does not declare",
                                        {"tool": "ffmpeg-skill/audio", "needs_flag": "sample_rate", "directory": str(self.directory)})
            args += ["--sample-rate", str(int(sample_rate))]
        return self._audio(args, src, dst, timeout)

    def replace_audio(self, video: str, stem: str, dst: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """ffmpeg-skill/audio <video> --replace <stem> -o <dst>: `dst` holds the first video stream of `video` copied
        packet for packet (-c:v copy) and the stem's first audio stream as its only audio stream, encoded by `dst`'s
        extension and padded or trimmed to `video`'s duration; subtitle and data streams are copied when the
        container takes them. `dst` must not exist (a fresh file in a private run directory; --overwrite is never
        passed). Returns ffmpeg-skill's result document (its `probe` describes `dst`)."""
        args = [self.script("audio"), os.path.abspath(video), "--replace", os.path.abspath(stem), "-o", os.path.abspath(dst), "--json"]
        return self._audio(args, video, dst, timeout)

    def _audio(self, args: List[str], src: str, dst: str, timeout: Optional[float]) -> Dict[str, Any]:
        code, out, err = self._run(args, timeout or self.timeout)
        doc = _parse_json(out)
        if code != 0 or doc.get("status") == "failed" or not os.path.isfile(dst):
            error = doc.get("error") if isinstance(doc.get("error"), dict) else {}
            tail = err.strip().splitlines()[-1] if err.strip() else "no output"
            message = str(error.get("message") or tail)
            if error.get("kind") == "input":
                raise VoiceEnhanceError("UNSUPPORTED_MEDIA", f"ffmpeg-skill/audio refused {os.path.basename(src)}: {message}",
                                        {"tool": "ffmpeg-skill/audio", "exit_code": code, "ffmpeg_skill_error": error})
            raise VoiceEnhanceError("TOOL_ERROR", f"ffmpeg-skill/audio failed on {os.path.basename(src)}: {message}",
                                    {"tool": "ffmpeg-skill/audio", "exit_code": code, "ffmpeg_skill_error": error})
        return doc


def _parse_json(text: str) -> Dict[str, Any]:
    """ffmpeg-skill prints one JSON document on stdout; tolerate a preceding plain line."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return {}
    try:
        doc = json.loads(text[start:])
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}
