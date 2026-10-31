"""ffmpeg-skill adapter. This skill runs no ffmpeg of its own: probing, format conversion for an engine and putting a
cleaned stem back into a video are ffmpeg-skill tool calls (ADR-001, ADR-002).

- Locates an ffmpeg-skill checkout (explicit dir > VOICE_ENHANCE_FFMPEG_SKILL_DIR > VIDEO_AGENT_FFMPEG_SKILL_DIR >
  ~/.claude/skills/ffmpeg-skill > ./vendor/ffmpeg-skill > ../ffmpeg-skill) and reads its contract
  (`scripts/_contract.py --json --static`). Compatibility is capability detection, never the version number: every
  tool in TOOLS_USED must be in the contract and every flag in FLAGS_USED in its input_schema (audio-production-skill
  ADR-15).
- Runs one named tool as [sys.executable, <dir>/scripts/<tool>.py, <typed argv...>] through procs.run_process_group.

Which ffmpeg-skill tools are used, and for what:
  probe   input facts before an op, output validation after it
  join    one clip -> the engine's input format (--sample-rate, --channels; 48 kHz mono for DeepFilterNet), step 3
  audio   --replace: the cleaned stem back into the video, video stream copied (step 6); --audio-stream picks the
          stream to clean; --overwrite only when the caller asked for it
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

SUPPORTED_CONTRACT_VERSION = "1.0"   # reported, not gated on (ffmpeg-skill kept "1.0" across 0.x-2.x)
ENV_DIR_KEYS = ("VOICE_ENHANCE_FFMPEG_SKILL_DIR", "VIDEO_AGENT_FFMPEG_SKILL_DIR")
TOOLS_USED = ("probe", "join", "audio")
FLAGS_USED: Dict[str, Tuple[str, ...]] = {
    "probe": ("inputs",),
    "join": ("inputs", "output", "sample_rate", "channels", "json"),
    "audio": ("input", "output", "replace", "audio_stream", "overwrite", "json"),
}
AUDIO_ONLY_TOOLS = ("probe", "join", "audio")


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
    if contract_version != "unknown" and contract_version.split(".")[0] != SUPPORTED_CONTRACT_VERSION.split(".")[0]:
        warnings.append(f"ffmpeg-skill contract_version {contract_version} differs from {SUPPORTED_CONTRACT_VERSION}; "
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
