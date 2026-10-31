"""Engine registry. Only engines listed here can be selected or run by the worker; test engines are never
registered. One engine per op today (ADR-001); BLEED_REDUCE has none until roadmap step 5."""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from ..errors import VoiceEnhanceError
from .base import Engine
from .deepfilternet import DeepFilterNetEngine
from .nara_wpe_engine import NaraWpeEngine

ENGINES: Dict[str, Type[Engine]] = {DeepFilterNetEngine.id: DeepFilterNetEngine, NaraWpeEngine.id: NaraWpeEngine}
DEFAULT_ENGINE: Dict[str, Optional[str]] = {"isolate": DeepFilterNetEngine.id, "dereverb": NaraWpeEngine.id, "bleed_reduce": None}


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> Engine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise VoiceEnhanceError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()


def engines_for(op: str) -> List[str]:
    return [eid for eid in engine_ids() if op in ENGINES[eid].ops]
