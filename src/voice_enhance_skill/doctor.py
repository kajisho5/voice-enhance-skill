"""`voice-enhance doctor`: what is installed, what is missing and how to get it. Never downloads, never imports an
engine (only looks it up), so it is fast and safe to run anywhere. Media capabilities are detected through
ffmpeg-skill (its contract and its own doctor); this skill runs no ffmpeg itself.

status: ok (ffmpeg-skill usable, every engine OK) | degraded (an engine MISSING / INCOMPATIBLE) | fail (Python or
ffmpeg-skill not usable). Each op reports whether it is implemented and whether its engine is ready."""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Optional

from . import SKILL_ID, __version__
from .engines import DEFAULT_ENGINE, get_engine, engine_ids
from .engines.base import OK
from .errors import VoiceEnhanceError
from .ffmpeg_skill import FfmpegSkill

DOCTOR_SCHEMA = f"{SKILL_ID}/doctor@1"
# op -> roadmap step that implements it; none is implemented in this release (ADR-002)
IMPLEMENTED_OPS: Dict[str, bool] = {"isolate": False, "dereverb": False, "bleed_reduce": False}
OP_STEP = {"isolate": 3, "dereverb": 4, "bleed_reduce": 5}


def engines_report(offline: bool = False) -> Dict[str, Any]:
    return {eid: get_engine(eid).report(offline=offline) for eid in engine_ids()}


def doctor(ffmpeg_skill_dir: Optional[str] = None, offline: bool = False) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    problems: List[str] = []
    warnings: List[str] = []
    py_ok = sys.version_info >= (3, 9)
    checks["python"] = {"status": "OK" if py_ok else "MISSING", "version": platform.python_version(), "platform": platform.system(),
                        "install": None if py_ok else "Python >= 3.9"}
    if not py_ok:
        problems.append("python: needs >= 3.9")

    try:
        skill = FfmpegSkill.locate(ffmpeg_skill_dir)
        info = skill.info()
        checks["ffmpeg_skill"] = dict(info.to_dict(), status="OK" if info.supported else "INCOMPATIBLE")
        problems += ["ffmpeg-skill: " + p for p in info.problems]
        warnings += ["ffmpeg-skill: " + w for w in info.warnings]
        if info.supported:
            ffdoc = skill.doctor()
            for tool in ("ffmpeg", "ffprobe"):
                found = ffdoc.get(tool)
                checks[tool] = {"status": "OK" if found else "MISSING", "version": found or None, "detected_by": "ffmpeg-skill doctor"}
                if not found:
                    problems.append(f"{tool}: not detected by ffmpeg-skill doctor")
        else:
            for tool in ("ffmpeg", "ffprobe"):
                checks[tool] = {"status": "UNKNOWN", "detail": "not checked: ffmpeg-skill not usable"}
    except VoiceEnhanceError as e:
        checks["ffmpeg_skill"] = {"status": "MISSING", "detail": e.message, "tried": e.details.get("tried"),
                                  "install": "git clone https://github.com/kajisho5/ffmpeg-skill ~/.claude/skills/ffmpeg-skill "
                                             "(or pass --ffmpeg-skill / set VOICE_ENHANCE_FFMPEG_SKILL_DIR)"}
        for tool in ("ffmpeg", "ffprobe"):
            checks[tool] = {"status": "UNKNOWN", "detail": "not checked: ffmpeg-skill missing"}
        problems.append("ffmpeg-skill: " + e.message)

    engines = engines_report(offline=offline)
    checks["engines"] = engines
    media_ok = checks["ffmpeg_skill"]["status"] == "OK" and all(checks[t]["status"] == "OK" for t in ("ffmpeg", "ffprobe"))
    ops: Dict[str, Any] = {}
    for op, eid in DEFAULT_ENGINE.items():
        engine_status = engines[eid]["status"] if eid else None
        ops[op] = {"implemented": IMPLEMENTED_OPS[op], "roadmap_step": OP_STEP[op], "engine": eid, "engine_status": engine_status,
                   "ready": bool(IMPLEMENTED_OPS[op] and media_ok and engine_status == OK)}
    checks["operations"] = ops
    missing_engines = sorted(eid for eid, e in engines.items() if e["status"] != OK)
    status = "fail" if problems else ("degraded" if missing_engines else "ok")
    return {
        "schema": DOCTOR_SCHEMA,
        "ok": status != "fail",
        "status": status,
        "skill": {"id": SKILL_ID, "version": __version__},
        "offline": bool(offline),
        "checks": checks,
        "unavailable_engines": missing_engines,
        "problems": problems + [f"engine {eid}: {engines[eid]['status']} ({engines[eid].get('detail')})" for eid in missing_engines],
        "warnings": warnings,
        "network": "never used at run time; no engine downloads anything",
    }
