"""The ops. ISOLATE (roadmap step 3, ADR-003) is implemented; DEREVERB and BLEED_REDUCE arrive in steps 4-5 and will
reuse the same frame: checked paths, an engine that is OK, the input prepared through ffmpeg-skill, the engine in the
worker subprocess, the result validated through ffmpeg-skill, then moved into place.

ISOLATE, step by step:
  1. paths      the input is a regular file; the output is a .wav that is not the input and does not exist (unless
                --overwrite); its directory exists
  2. engine     the selected engine does this op and reports OK (else ENGINE_UNAVAILABLE / MODEL_NOT_FOUND, before any
                media work)
  3. probe      ffmpeg-skill/probe: the input has the requested audio stream
  4. prepare    ffmpeg-skill/audio -o extracted.wav --audio-stream N (PCM WAV, picture dropped); then, only if needed,
                a second ffmpeg-skill/audio pass on that one-stream WAV with --mono and/or --sample-rate 48000
  5. engine     the worker runs the engine on the prepared 48 kHz mono WAV, killed with its process group on --timeout
  6. validate   ffmpeg-skill/probe on the stem: 48 kHz, mono, same duration as the prepared input (else INVALID_RESULT)
  7. place      the stem replaces nothing unless --overwrite; it is renamed into place from a private run directory in
                the output's own directory, so the output appears whole or not at all
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from .engines import DEFAULT_ENGINE, engines_for, get_engine
from .engines.base import OK, Engine, EngineRequest
from .errors import VoiceEnhanceError
from .ffmpeg_skill import FfmpegSkill
from .provenance import build_provenance
from .service import DEFAULT_TIMEOUT, run_engine

ISOLATE_SCHEMA = f"{SKILL_ID}/isolate@1"
ISOLATE_RATE = 48000                 # deep-filter accepts 48 kHz WAV only (ADR-001)
DURATION_TOLERANCE = 0.1             # seconds, or 1 % of the duration when that is larger
PREPARED_NAME = "input.wav"


@dataclass
class IsolateRequest:
    input_path: str
    output_path: str
    audio_stream: int = 0
    engine: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    overwrite: bool = False
    atten_lim_db: Optional[float] = None
    ffmpeg_skill_dir: Optional[str] = None
    offline: bool = False


def check_paths(input_path: str, output_path: str, overwrite: bool) -> Tuple[str, str]:
    """Returns (absolute input, absolute output). Pure checks on the file system; nothing is written."""
    src = os.path.abspath(input_path)
    dst = os.path.abspath(output_path)
    if not os.path.isfile(src):
        raise VoiceEnhanceError("INPUT_NOT_FOUND", f"{input_path} is not a file", {"input": input_path})
    if not dst.lower().endswith(".wav"):
        raise VoiceEnhanceError("INVALID_ARGUMENT", "the output must be a .wav file (the engine writes PCM WAV; it is not "
                                "re-encoded)", {"output": output_path})
    if os.path.realpath(dst) == os.path.realpath(src) or (os.path.exists(dst) and os.path.samefile(dst, src)):
        raise VoiceEnhanceError("OUTPUT_IS_INPUT", "the output path is the input; the input is never modified", {"output": output_path})
    if os.path.isdir(dst):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"{output_path} is a directory", {"output": output_path})
    if os.path.lexists(dst) and not overwrite:
        raise VoiceEnhanceError("OUTPUT_EXISTS", f"{output_path} exists; pass --overwrite to replace it", {"output": output_path})
    if not os.path.isdir(os.path.dirname(dst)):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"the output directory {os.path.dirname(dst)} does not exist", {"output": output_path})
    return src, dst


def select_engine(op: str, engine_id: Optional[str], offline: bool = False) -> Tuple[Engine, Dict[str, Any]]:
    """The engine for `op` and its report; refuses an engine that is not OK before any media work."""
    eid = engine_id or DEFAULT_ENGINE.get(op)
    if eid is None:
        raise VoiceEnhanceError("ENGINE_UNAVAILABLE", f"no engine is registered for {op}", {"op": op})
    engine = get_engine(eid)
    if op not in engine.ops:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"engine {eid!r} does not do {op}; engines for {op}: {engines_for(op)}",
                                {"engine": eid, "op": op, "engines": engines_for(op)})
    report = engine.report(offline=offline)
    if report["status"] != OK:
        code = "MODEL_NOT_FOUND" if report.get("missing") == "model" else "ENGINE_UNAVAILABLE"
        raise VoiceEnhanceError(code, f"engine {eid!r} is {report['status']}: {report.get('detail')}",
                                {"engine": eid, "status": report["status"], "install": engine.install})
    return engine, report


def audio_streams(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    streams = facts.get("audio_streams")
    if isinstance(streams, list):
        return [s for s in streams if isinstance(s, dict)]
    return [dict(facts["audio"], index=0)] if isinstance(facts.get("audio"), dict) else []


def select_stream(facts: Dict[str, Any], audio_stream: int, name: str) -> Tuple[Dict[str, Any], int]:
    """(the chosen audio stream's facts, the number of audio streams) from an ffmpeg-skill probe document."""
    streams = audio_streams(facts)
    if not streams:
        raise VoiceEnhanceError("UNSUPPORTED_MEDIA", f"{name} has no audio stream", {"input": name})
    if not 0 <= audio_stream < len(streams):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"--audio-stream {audio_stream} does not exist: {name} has {len(streams)} audio "
                                f"stream(s), 0..{len(streams) - 1}", {"audio_stream": audio_stream, "audio_stream_count": len(streams)})
    return streams[audio_stream], len(streams)


def _wav_facts(skill: FfmpegSkill, path: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """ffmpeg-skill/audio's result carries a probe of its output; probe again only if it does not."""
    probe = doc.get("probe")
    if isinstance(probe, dict) and isinstance(probe.get("audio"), dict):
        return probe
    return skill.probe(path)


def prepare_wav(skill: FfmpegSkill, src: str, run_dir: str, audio_stream: int, rate: int,
                stream: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """One audio stream of `src` as a `rate` Hz mono PCM WAV in `run_dir`, through ffmpeg-skill/audio only.
    Returns (path, its probe, the ffmpeg-skill steps taken, warnings).

    The channel fold-down is a second pass on the extracted one-stream WAV, never `--mono` together with
    `--audio-stream`: ffmpeg-skill's --mono reads the channel count of the file's first audio stream, so on a later
    stream with a different layout it would be skipped (ADR-003)."""
    if int(stream.get("sample_rate") or 0) != rate and not skill.info().takes_flag("audio", "sample_rate"):
        raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", f"audio stream {audio_stream} is {stream.get('sample_rate')} Hz; the engine "
                                f"needs {rate} Hz and this ffmpeg-skill checkout has no audio --sample-rate to convert it. Convert "
                                f"the input to {rate} Hz first, or use an ffmpeg-skill release that declares audio --sample-rate",
                                {"tool": "ffmpeg-skill/audio", "needs_flag": "sample_rate", "input_sample_rate": stream.get("sample_rate"),
                                 "required_sample_rate": rate})
    steps: List[Dict[str, Any]] = []
    warnings: List[str] = []
    extracted = os.path.join(run_dir, "extracted.wav")
    facts = _wav_facts(skill, extracted, skill.audio_to_wav(src, extracted, audio_stream=audio_stream))
    a = facts["audio"]
    steps.append({"tool": "ffmpeg-skill/audio", "flags": ["--audio-stream", str(audio_stream)],
                  "result": {"sample_rate": a.get("sample_rate"), "channels": a.get("channels")}})
    prepared = os.path.join(run_dir, PREPARED_NAME)
    mono = int(a.get("channels") or 0) != 1
    resample = int(a.get("sample_rate") or 0) != rate
    if not (mono or resample):
        os.replace(extracted, prepared)
        return prepared, facts, steps, warnings
    facts = _wav_facts(skill, prepared, skill.audio_to_wav(extracted, prepared, mono=mono, sample_rate=rate if resample else None))
    os.remove(extracted)
    b = facts["audio"]
    steps.append({"tool": "ffmpeg-skill/audio", "flags": (["--mono"] if mono else []) + (["--sample-rate", str(rate)] if resample else []),
                  "result": {"sample_rate": b.get("sample_rate"), "channels": b.get("channels")}})
    if mono:
        warnings.append(f"audio stream {audio_stream} has {a.get('channels')} channels; it was folded down to mono for the engine")
    if resample:
        warnings.append(f"audio stream {audio_stream} is {a.get('sample_rate')} Hz; it was resampled to {rate} Hz for the engine")
    if int(b.get("sample_rate") or 0) != rate or int(b.get("channels") or 0) != 1:
        raise VoiceEnhanceError("TOOL_ERROR", f"ffmpeg-skill/audio produced {b.get('sample_rate')} Hz {b.get('channels')} ch, expected "
                                f"{rate} Hz mono", {"tool": "ffmpeg-skill/audio"}, retryable=False)
    return prepared, facts, steps, warnings


def validate_output(skill: FfmpegSkill, path: str, rate: int, expected_duration: float, engine_id: str) -> Dict[str, Any]:
    """The engine's stem through ffmpeg-skill/probe: readable, `rate` Hz, mono, as long as the prepared input."""
    try:
        facts = skill.probe(path)
    except VoiceEnhanceError as e:
        raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r} wrote a file ffmpeg-skill cannot read: {e.message}",
                                {"engine": engine_id})
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    problems = []
    if int(a.get("sample_rate") or 0) != rate:
        problems.append(f"sample rate {a.get('sample_rate')}, expected {rate}")
    if int(a.get("channels") or 0) != 1:
        problems.append(f"{a.get('channels')} channels, expected 1")
    duration = float(facts.get("duration") or 0.0)
    tolerance = max(DURATION_TOLERANCE, 0.01 * expected_duration)
    if abs(duration - expected_duration) > tolerance:
        problems.append(f"duration {duration:.3f}s, expected {expected_duration:.3f}s (+/- {tolerance:.3f}s)")
    if problems:
        raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r} output failed validation: " + "; ".join(problems),
                                {"engine": engine_id, "problems": problems})
    return facts


def _place(stem: str, dst: str, overwrite: bool) -> None:
    if os.path.lexists(dst) and not overwrite:      # appeared while the engine ran
        raise VoiceEnhanceError("OUTPUT_EXISTS", f"{dst} appeared while the engine ran; pass --overwrite to replace it", {"output": dst})
    os.replace(stem, dst)


def _check_numbers(req: IsolateRequest) -> None:
    if not isinstance(req.timeout, (int, float)) or req.timeout <= 0:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--timeout must be > 0", {"timeout": req.timeout})
    if not isinstance(req.audio_stream, int) or req.audio_stream < 0:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--audio-stream must be an integer >= 0", {"audio_stream": req.audio_stream})
    if req.atten_lim_db is not None and not 0 < float(req.atten_lim_db) <= 100:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--atten-lim-db must be > 0 and <= 100 (100 = no limit)",
                                {"atten_lim_db": req.atten_lim_db})


def isolate(req: IsolateRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """ISOLATE: the voice stem of one audio stream of `req.input_path`, as a 48 kHz mono WAV at `req.output_path`.
    `skill` is the ffmpeg-skill handle (located from `req.ffmpeg_skill_dir` when None)."""
    _check_numbers(req)
    src, dst = check_paths(req.input_path, req.output_path, req.overwrite)
    engine, report = select_engine("isolate", req.engine, req.offline)
    if skill is None:
        skill = FfmpegSkill.locate(req.ffmpeg_skill_dir, timeout=float(req.timeout))
    info = skill.info()
    if not info.supported:
        raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", "ffmpeg-skill is not usable: " + "; ".join(info.problems),
                                {"directory": str(info.directory), "problems": info.problems})
    name = os.path.basename(src)
    facts = skill.probe(src)
    stream, stream_count = select_stream(facts, req.audio_stream, name)
    try:
        run_dir = tempfile.mkdtemp(prefix=".voice-enhance-", dir=os.path.dirname(dst))
    except OSError as e:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"cannot write to the output directory: {e}", {"output": req.output_path})
    try:
        prepared, prepared_facts, steps, warnings = prepare_wav(skill, src, run_dir, req.audio_stream, ISOLATE_RATE, stream)
        params: Dict[str, Any] = {"timeout": float(req.timeout)}
        if req.atten_lim_db is not None:
            params["atten_lim_db"] = float(req.atten_lim_db)
        stem = os.path.join(run_dir, "stem.wav")
        result = run_engine(engine.id, EngineRequest(op="isolate", input_path=prepared, output_path=stem, sample_rate=ISOLATE_RATE,
                                                     params=params), float(req.timeout), run_dir)
        out_facts = validate_output(skill, result.output_path, ISOLATE_RATE, float(prepared_facts.get("duration") or 0.0), engine.id)
        engine_doc = dict(report, version=result.engine_version, model=result.model, model_sha256=result.extra.get("model_sha256"))
        parameters = {"audio_stream": req.audio_stream, "atten_lim_db": result.extra.get("atten_lim_db"),
                      "compensate_delay": result.extra.get("compensate_delay"), "sample_rate": ISOLATE_RATE, "channels": 1,
                      "engine_argv": result.extra.get("argv")}
        provenance = build_provenance(op="isolate", engine=engine_doc, input_path=src, output_path=result.output_path,
                                      parameters=parameters, ffmpeg_skill=info.to_dict())
        _place(result.output_path, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    out_audio = out_facts.get("audio") or {}
    return {
        "schema": ISOLATE_SCHEMA,
        "ok": True,
        "op": "isolate",
        "capability": "voice.isolate",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": {"path": dst, "format": "wav", "codec": out_audio.get("codec"), "sample_rate": out_audio.get("sample_rate"),
                   "channels": out_audio.get("channels"), "duration": out_facts.get("duration")},
        "input": {"path": src, "duration": facts.get("duration"), "has_video": isinstance(facts.get("video"), dict),
                  "audio_stream": req.audio_stream, "audio_stream_count": stream_count,
                  "audio": {k: stream.get(k) for k in ("codec", "sample_rate", "channels", "channel_layout", "language")}},
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model", "model_sha256")},
        "parameters": parameters,
        "steps": steps,
        "provenance": provenance,
        "warnings": warnings,
    }
