"""Engine contract. An engine turns one prepared WAV into one cleaned WAV; probing, format conversion, validation
and provenance are the service's job (through ffmpeg-skill), so every engine yields the same kind of result.

Statuses (doctor vocabulary): OK (installed and usable), MISSING (not installed, or its model file is absent),
INCOMPATIBLE (installed, but a version outside the supported range). An engine never downloads anything."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..errors import VoiceEnhanceError

OK, MISSING, INCOMPATIBLE = "OK", "MISSING", "INCOMPATIBLE"
STATUSES = (OK, MISSING, INCOMPATIBLE)
OPS = ("isolate", "dereverb", "bleed_reduce")


@dataclass
class EngineRequest:
    op: str
    input_path: str
    output_path: str
    sample_rate: int = 48000
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EngineResult:
    output_path: str
    engine_version: str
    model: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        return cls(output_path=str(d["output_path"]), engine_version=str(d["engine_version"]),
                   model=None if d.get("model") is None else str(d["model"]), extra=dict(d.get("extra") or {}))


def parse_version(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not text:
        return None
    import re
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return tuple(int(x) for x in m.groups() if x is not None)


class Engine:
    """Subclass and register in `engines/__init__.py`. `process` may be slow and runs in the worker process."""
    id = "base"
    ops: Tuple[str, ...] = ()
    description = ""
    kind = ""                       # "binary" (external executable) | "python" (optional package)
    code_license = ""
    weights_license = ""
    attribution = ""
    execution_mode = "local"
    network_required = False        # at run time; no engine registered here needs it
    deterministic = True
    version_range = ""
    install = ""
    located_by: List[str] = []
    roadmap_step: Optional[int] = None   # the step that implements process()

    def inspect(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        """Static facts, safe to publish (no paths): what the contract lists."""
        return {"id": self.id, "ops": list(self.ops), "description": self.description, "kind": self.kind,
                "code_license": self.code_license, "weights_license": self.weights_license, "attribution": self.attribution,
                "execution_mode": self.execution_mode, "network_required": self.network_required,
                "deterministic": self.deterministic, "version_range": self.version_range, "install": self.install,
                "located_by": list(self.located_by)}

    def report(self, offline: bool = False) -> Dict[str, Any]:
        """inspect() plus the static facts; with offline=True an engine that needs the network is MISSING."""
        info = dict(self.spec(), **self.inspect())
        if offline and self.network_required and info["status"] == OK:
            info.update(status=MISSING, detail="needs the network at run time; --offline was given")
        info["usable_offline"] = not self.network_required
        if info["status"] == OK:
            info["install"] = None
        return info

    def require(self) -> Dict[str, Any]:
        info = self.inspect()
        if info["status"] != OK:
            code = "MODEL_NOT_FOUND" if info.get("missing") == "model" else "ENGINE_UNAVAILABLE"
            raise VoiceEnhanceError(code, f"engine {self.id!r} is {info['status']}: {info.get('detail')}",
                                    {"engine": self.id, "install": self.install})
        return info

    def process(self, req: EngineRequest) -> EngineResult:
        self.require()
        raise VoiceEnhanceError("NOT_IMPLEMENTED", f"engine {self.id!r} is installed, but {req.op} is not implemented yet "
                                f"(roadmap step {self.roadmap_step})", {"engine": self.id, "op": req.op})
