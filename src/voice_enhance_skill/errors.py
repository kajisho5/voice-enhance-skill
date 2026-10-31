"""Structured error model. Every failure that crosses the Skill boundary is a VoiceEnhanceError with a code from
ERROR_TABLE; the code decides the CLI exit code and whether retrying the same call can succeed. The CLI prints
{"ok": false, "error": {"code", "message", "retryable", "details"}}."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# code -> (exit code, retryable)
ERROR_TABLE: Dict[str, Tuple[int, bool]] = {
    "INVALID_ARGUMENT": (2, False),          # a flag value or request field is malformed
    "INPUT_NOT_FOUND": (2, False),           # an input file does not exist or is not a regular file
    "UNSUPPORTED_MEDIA": (2, False),         # ffmpeg-skill/probe cannot read the input as media
    "OUTPUT_EXISTS": (2, False),             # the output exists and --overwrite was not given
    "OUTPUT_IS_INPUT": (2, False),           # the output path resolves to an input
    "FFMPEG_SKILL_UNAVAILABLE": (3, False),  # no usable ffmpeg-skill checkout (missing, or lacks a tool / flag)
    "ENGINE_UNAVAILABLE": (3, False),        # unknown engine id, or the engine is not installed
    "MODEL_NOT_FOUND": (3, False),           # the engine is installed but its model file is not
    "TOOL_ERROR": (4, True),                 # an ffmpeg-skill tool failed or timed out
    "ENGINE_FAILED": (4, False),             # the engine ran and failed
    "ENGINE_TIMEOUT": (4, True),             # the engine exceeded --timeout and its process group was killed
    "INVALID_RESULT": (4, False),            # the engine's output failed validation; nothing is returned
    "NOT_IMPLEMENTED": (5, False),           # declared in the contract, not implemented yet (see the roadmap step)
    "CANCELLED": (6, True),                  # interrupted by a signal
    "INTERNAL_ERROR": (7, False),            # a bug in this skill
}
ERROR_CODES = tuple(ERROR_TABLE)
EXIT_CODES = {code: ERROR_TABLE[code][0] for code in ERROR_CODES}


class VoiceEnhanceError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None, retryable: Optional[bool] = None) -> None:
        if code not in ERROR_TABLE:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.retryable = ERROR_TABLE[code][1] if retryable is None else bool(retryable)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable, "details": self.details}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VoiceEnhanceError":
        code = d.get("code") if d.get("code") in ERROR_TABLE else "INTERNAL_ERROR"
        return cls(code, str(d.get("message") or code), d.get("details") if isinstance(d.get("details"), dict) else None)
