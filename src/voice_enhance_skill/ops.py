"""The ops: ISOLATE (roadmap step 3, ADR-003), DEREVERB (step 4, ADR-004) and BLEED_REDUCE (step 5, ADR-005). All three
use the same frame: checked paths, an engine that is OK, the input prepared through ffmpeg-skill, the engine in the
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

DEREVERB is the same frame with two differences (ADR-004): WPE works at any sample rate and on any number of channels,
so step 4 is a single ffmpeg-skill/audio --audio-stream N pass (no fold-down, no resampling, so no audio --sample-rate
is needed), and step 6 checks the stem against the prepared input's own rate and channel count.

BLEED_REDUCE is the same frame over several tracks at once (ADR-005): two or more audio streams (of one file, or one
per file), each prepared like ISOLATE's input but at the first track's sample rate (mono fold-down in a second pass;
a different rate only with audio --sample-rate, else FFMPEG_SKILL_UNAVAILABLE before any work). The crosstalk engine
runs once on all of them and writes one float WAV per track. Every stem is validated through ffmpeg-skill/probe
before any is placed; into a new output directory they move with one rename of the whole directory.
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
from .engines.crosstalk import LIMITS as CROSSTALK_LIMITS, MAX_TRACKS, MIN_TRACKS
from .engines.nara_wpe_engine import LIMITS as WPE_LIMITS
from .errors import VoiceEnhanceError
from .ffmpeg_skill import FfmpegSkill
from .provenance import build_provenance
from .service import DEFAULT_TIMEOUT, run_engine

ISOLATE_SCHEMA = f"{SKILL_ID}/isolate@1"
DEREVERB_SCHEMA = f"{SKILL_ID}/dereverb@1"
BLEED_REDUCE_SCHEMA = f"{SKILL_ID}/bleed-reduce@1"
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


@dataclass
class DereverbRequest:
    input_path: str
    output_path: str
    audio_stream: int = 0
    engine: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    overwrite: bool = False
    taps: Optional[int] = None
    delay: Optional[int] = None
    iterations: Optional[int] = None
    ffmpeg_skill_dir: Optional[str] = None
    offline: bool = False


@dataclass
class BleedReduceRequest:
    input_paths: List[str]
    output_dir: str
    audio_streams: Optional[List[int]] = None
    engine: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    overwrite: bool = False
    dominance_db: Optional[float] = None
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


def extract_stream(skill: FfmpegSkill, src: str, dst: str, audio_stream: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ffmpeg-skill/audio <src> -o <dst.wav> --audio-stream N: that one stream as PCM WAV at its own rate and channel
    count, the picture dropped. Returns (the WAV's probe, the step record)."""
    facts = _wav_facts(skill, dst, skill.audio_to_wav(src, dst, audio_stream=audio_stream))
    a = facts["audio"]
    return facts, {"tool": "ffmpeg-skill/audio", "flags": ["--audio-stream", str(audio_stream)],
                   "result": {"sample_rate": a.get("sample_rate"), "channels": a.get("channels")}}


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
    warnings: List[str] = []
    extracted = os.path.join(run_dir, "extracted.wav")
    facts, step = extract_stream(skill, src, extracted, audio_stream)
    steps = [step]
    a = facts["audio"]
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


def validate_output(skill: FfmpegSkill, path: str, rate: int, expected_duration: float, engine_id: str,
                    channels: int = 1) -> Dict[str, Any]:
    """The engine's stem through ffmpeg-skill/probe: readable, `rate` Hz, `channels` channels (ISOLATE: mono), as long
    as the prepared input."""
    try:
        facts = skill.probe(path)
    except VoiceEnhanceError as e:
        raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine_id!r} wrote a file ffmpeg-skill cannot read: {e.message}",
                                {"engine": engine_id})
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    problems = []
    if int(a.get("sample_rate") or 0) != rate:
        problems.append(f"sample rate {a.get('sample_rate')}, expected {rate}")
    if int(a.get("channels") or 0) != channels:
        problems.append(f"{a.get('channels')} channels, expected {channels}")
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


def _check_common(req: Any) -> None:
    if not isinstance(req.timeout, (int, float)) or req.timeout <= 0:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--timeout must be > 0", {"timeout": req.timeout})
    if not isinstance(req.audio_stream, int) or req.audio_stream < 0:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--audio-stream must be an integer >= 0", {"audio_stream": req.audio_stream})


def _check_numbers(req: IsolateRequest) -> None:
    _check_common(req)
    if req.atten_lim_db is not None and not 0 < float(req.atten_lim_db) <= 100:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--atten-lim-db must be > 0 and <= 100 (100 = no limit)",
                                {"atten_lim_db": req.atten_lim_db})


def _usable_skill(skill: Optional[FfmpegSkill], directory: Optional[str], timeout: float) -> Tuple[FfmpegSkill, Any]:
    if skill is None:
        skill = FfmpegSkill.locate(directory, timeout=timeout)
    info = skill.info()
    if not info.supported:
        raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", "ffmpeg-skill is not usable: " + "; ".join(info.problems),
                                {"directory": str(info.directory), "problems": info.problems})
    return skill, info


def _run_dir(directory: str, output_path: str) -> str:
    """A private run directory in `directory` (the output's own directory), so the result is placed with one rename."""
    try:
        return tempfile.mkdtemp(prefix=".voice-enhance-", dir=directory)
    except OSError as e:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"cannot write to the output directory: {e}", {"output": output_path})


def _input_doc(src: str, facts: Dict[str, Any], audio_stream: int, stream_count: int, stream: Dict[str, Any]) -> Dict[str, Any]:
    return {"path": src, "duration": facts.get("duration"), "has_video": isinstance(facts.get("video"), dict),
            "audio_stream": audio_stream, "audio_stream_count": stream_count,
            "audio": {k: stream.get(k) for k in ("codec", "sample_rate", "channels", "channel_layout", "language")}}


def _output_doc(dst: str, out_facts: Dict[str, Any]) -> Dict[str, Any]:
    out_audio = out_facts.get("audio") or {}
    return {"path": dst, "format": "wav", "codec": out_audio.get("codec"), "sample_rate": out_audio.get("sample_rate"),
            "channels": out_audio.get("channels"), "duration": out_facts.get("duration")}


def isolate(req: IsolateRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """ISOLATE: the voice stem of one audio stream of `req.input_path`, as a 48 kHz mono WAV at `req.output_path`.
    `skill` is the ffmpeg-skill handle (located from `req.ffmpeg_skill_dir` when None)."""
    _check_numbers(req)
    src, dst = check_paths(req.input_path, req.output_path, req.overwrite)
    engine, report = select_engine("isolate", req.engine, req.offline)
    skill, info = _usable_skill(skill, req.ffmpeg_skill_dir, float(req.timeout))
    facts = skill.probe(src)
    stream, stream_count = select_stream(facts, req.audio_stream, os.path.basename(src))
    run_dir = _run_dir(os.path.dirname(dst), req.output_path)
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
    return {
        "schema": ISOLATE_SCHEMA,
        "ok": True,
        "op": "isolate",
        "capability": "voice.isolate",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": _output_doc(dst, out_facts),
        "input": _input_doc(src, facts, req.audio_stream, stream_count, stream),
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model", "model_sha256")},
        "parameters": parameters,
        "steps": steps,
        "provenance": provenance,
        "warnings": warnings,
    }


def _check_dereverb(req: DereverbRequest) -> None:
    _check_common(req)
    for key in ("taps", "delay", "iterations"):
        value = getattr(req, key)
        lo, hi = WPE_LIMITS[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi):
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"--{key} must be an integer from {lo} to {hi}", {key: value})


def dereverb(req: DereverbRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """DEREVERB: one audio stream of `req.input_path` with late reverberation removed by WPE, as a WAV at
    `req.output_path` with the stream's own sample rate and channels (32-bit float, sample-aligned with the input).
    `skill` is the ffmpeg-skill handle (located from `req.ffmpeg_skill_dir` when None)."""
    _check_dereverb(req)
    src, dst = check_paths(req.input_path, req.output_path, req.overwrite)
    engine, report = select_engine("dereverb", req.engine, req.offline)
    skill, info = _usable_skill(skill, req.ffmpeg_skill_dir, float(req.timeout))
    facts = skill.probe(src)
    stream, stream_count = select_stream(facts, req.audio_stream, os.path.basename(src))
    run_dir = _run_dir(os.path.dirname(dst), req.output_path)
    try:
        prepared = os.path.join(run_dir, PREPARED_NAME)
        prepared_facts, step = extract_stream(skill, src, prepared, req.audio_stream)
        pa = prepared_facts["audio"]
        rate, channels = int(pa.get("sample_rate") or 0), int(pa.get("channels") or 0)
        params: Dict[str, Any] = {"timeout": float(req.timeout)}
        params.update({k: getattr(req, k) for k in ("taps", "delay", "iterations") if getattr(req, k) is not None})
        stem = os.path.join(run_dir, "stem.wav")
        result = run_engine(engine.id, EngineRequest(op="dereverb", input_path=prepared, output_path=stem, sample_rate=rate,
                                                     params=params), float(req.timeout), run_dir)
        out_facts = validate_output(skill, result.output_path, rate, float(prepared_facts.get("duration") or 0.0), engine.id,
                                    channels=channels)
        engine_doc = dict(report, version=result.engine_version, model=result.model, model_sha256=None)
        x = result.extra
        parameters = {"audio_stream": req.audio_stream, "taps": x.get("taps"), "delay": x.get("delay"),
                      "iterations": x.get("iterations"), "method": x.get("method"), "statistics_mode": x.get("statistics_mode"),
                      "stft": x.get("stft"), "block_seconds": x.get("block_seconds"), "sample_rate": rate, "channels": channels,
                      "output_format": x.get("output_format")}
        provenance = build_provenance(op="dereverb", engine=engine_doc, input_path=src, output_path=result.output_path,
                                      parameters=parameters, ffmpeg_skill=info.to_dict())
        _place(result.output_path, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return {
        "schema": DEREVERB_SCHEMA,
        "ok": True,
        "op": "dereverb",
        "capability": "voice.dereverb",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": _output_doc(dst, out_facts),
        "input": _input_doc(src, facts, req.audio_stream, stream_count, stream),
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model")},
        "parameters": parameters,
        "steps": [step],
        "provenance": provenance,
        "warnings": [],
    }


def _check_bleed(req: BleedReduceRequest) -> None:
    if not isinstance(req.timeout, (int, float)) or req.timeout <= 0:
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--timeout must be > 0", {"timeout": req.timeout})
    if not req.input_paths or len(req.input_paths) > MAX_TRACKS:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"give 1 to {MAX_TRACKS} inputs", {"inputs": len(req.input_paths or [])})
    streams = req.audio_streams
    if streams is not None and (not streams or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in streams)):
        raise VoiceEnhanceError("INVALID_ARGUMENT", "--audio-streams must be integers >= 0", {"audio_streams": streams})
    if streams is not None and len(req.input_paths) > 1 and len(streams) != len(req.input_paths):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"with {len(req.input_paths)} inputs, --audio-streams needs one stream per input "
                                f"(got {len(streams)})", {"audio_streams": streams, "inputs": len(req.input_paths)})
    lo, hi = CROSSTALK_LIMITS["dominance_db"]
    d = req.dominance_db
    if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float)) or not lo <= float(d) <= hi):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"--dominance-db must be a number from {lo:g} to {hi:g}", {"dominance_db": d})


def _check_bleed_paths(req: BleedReduceRequest) -> Tuple[List[str], str]:
    """(absolute inputs, absolute output directory). The directory either exists or is created on success inside an
    existing parent; nothing is written here."""
    srcs = []
    for p in req.input_paths:
        if not os.path.isfile(p):
            raise VoiceEnhanceError("INPUT_NOT_FOUND", f"{p} is not a file", {"input": p})
        srcs.append(os.path.abspath(p))
    out_dir = os.path.abspath(req.output_dir)
    if os.path.lexists(out_dir) and not os.path.isdir(out_dir):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"{req.output_dir} exists and is not a directory (-o names a directory: one WAV "
                                "per track is written into it)", {"output_dir": req.output_dir})
    if not os.path.isdir(os.path.dirname(out_dir)):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"the parent directory of {req.output_dir} does not exist", {"output_dir": req.output_dir})
    return srcs, out_dir


def track_name(src: str, audio_stream: int) -> str:
    """The output file name of one track: <input name without extension>_a<stream>.wav."""
    return f"{os.path.splitext(os.path.basename(src))[0]}_a{audio_stream}.wav"


def _plan_tracks(skill: FfmpegSkill, srcs: List[str], wanted: Optional[List[int]]) -> List[Dict[str, Any]]:
    """One entry per track: the input, its probe, the chosen stream and its facts. One input without --audio-streams
    means all of its audio streams."""
    facts = {src: skill.probe(src) for src in dict.fromkeys(srcs)}
    if len(srcs) == 1:
        streams = wanted if wanted is not None else list(range(len(audio_streams(facts[srcs[0]]))))
        pairs = [(srcs[0], s) for s in streams]
    else:
        pairs = list(zip(srcs, wanted if wanted is not None else [0] * len(srcs)))
    if not MIN_TRACKS <= len(pairs) <= MAX_TRACKS:
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"bleed reduction needs {MIN_TRACKS} to {MAX_TRACKS} tracks (one per microphone); "
                                f"got {len(pairs)}. Give several files, or one file with several audio streams",
                                {"tracks": len(pairs)})
    if len(set(pairs)) != len(pairs):
        raise VoiceEnhanceError("INVALID_ARGUMENT", "the same audio stream of the same file is given twice",
                                {"tracks": [[os.path.basename(a), s] for a, s in pairs]})
    tracks = []
    for k, (src, s) in enumerate(pairs):
        stream, count = select_stream(facts[src], s, os.path.basename(src))
        tracks.append({"track": k, "src": src, "audio_stream": s, "facts": facts[src], "stream": stream, "stream_count": count,
                       "name": track_name(src, s)})
    names = [t["name"] for t in tracks]
    if len(set(names)) != len(names):
        raise VoiceEnhanceError("INVALID_ARGUMENT", f"two tracks would be written to the same file name: {names}; rename an input",
                                {"names": names})
    return tracks


def _check_outputs(tracks: List[Dict[str, Any]], srcs: List[str], out_dir: str, overwrite: bool) -> List[str]:
    dsts = [os.path.join(out_dir, t["name"]) for t in tracks]
    for dst in dsts:
        for src in srcs:
            if os.path.realpath(dst) == os.path.realpath(src) or (os.path.exists(dst) and os.path.samefile(dst, src)):
                raise VoiceEnhanceError("OUTPUT_IS_INPUT", f"{dst} is an input; the input is never modified", {"output": dst})
        if os.path.isdir(dst):
            raise VoiceEnhanceError("INVALID_ARGUMENT", f"{dst} is a directory", {"output": dst})
        if os.path.lexists(dst) and not overwrite:
            raise VoiceEnhanceError("OUTPUT_EXISTS", f"{dst} exists; pass --overwrite to replace it", {"output": dst})
    return dsts


def _check_rates(skill: FfmpegSkill, tracks: List[Dict[str, Any]]) -> int:
    """The common sample rate (the first track's). Another rate is converted only through audio --sample-rate."""
    rate = int(tracks[0]["stream"].get("sample_rate") or 0)
    other = [t for t in tracks if int(t["stream"].get("sample_rate") or 0) != rate]
    if other and not skill.info().takes_flag("audio", "sample_rate"):
        raise VoiceEnhanceError("FFMPEG_SKILL_UNAVAILABLE", f"the tracks have different sample rates ({rate} Hz for track 0, "
                                + ", ".join(f"{t['stream'].get('sample_rate')} Hz for track {t['track']}" for t in other)
                                + "); bleed reduction needs one rate and this ffmpeg-skill checkout has no audio --sample-rate to "
                                f"convert them. Convert every track to {rate} Hz first, or use an ffmpeg-skill release that declares "
                                "audio --sample-rate", {"tool": "ffmpeg-skill/audio", "needs_flag": "sample_rate",
                                                        "sample_rates": [t["stream"].get("sample_rate") for t in tracks],
                                                        "required_sample_rate": rate})
    return rate


def _place_all(stems: List[str], dsts: List[str], stem_dir: str, out_dir: str, overwrite: bool) -> None:
    """Every stem is already validated. A new output directory is the stem directory renamed whole; into an existing
    one each stem is renamed in turn, after a last check that none of the names appeared meanwhile."""
    if not os.path.lexists(out_dir):
        os.rename(stem_dir, out_dir)
        return
    for dst in dsts:
        if os.path.lexists(dst) and not overwrite:
            raise VoiceEnhanceError("OUTPUT_EXISTS", f"{dst} appeared while the engine ran; pass --overwrite to replace it", {"output": dst})
    for stem, dst in zip(stems, dsts):
        os.replace(stem, dst)


def bleed_reduce(req: BleedReduceRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """BLEED_REDUCE: two or more time-aligned microphone tracks (audio streams of one file, or one stream per file)
    with the other microphones' cross-talk reduced, one mono 32-bit float WAV per track in `req.output_dir`, each with
    its own length and sample-aligned with its input. `skill` is the ffmpeg-skill handle (located when None)."""
    _check_bleed(req)
    srcs, out_dir = _check_bleed_paths(req)
    engine, report = select_engine("bleed_reduce", req.engine, req.offline)
    skill, info = _usable_skill(skill, req.ffmpeg_skill_dir, float(req.timeout))
    tracks = _plan_tracks(skill, srcs, req.audio_streams)
    dsts = _check_outputs(tracks, srcs, out_dir, req.overwrite)
    rate = _check_rates(skill, tracks)
    # In the output directory, or next to it when it does not exist yet, so every stem is placed by a rename.
    run_dir = _run_dir(out_dir if os.path.isdir(out_dir) else os.path.dirname(out_dir), req.output_dir)
    try:
        steps: List[Dict[str, Any]] = []
        warnings: List[str] = []
        prepared, durations = [], []
        for t in tracks:
            track_dir = os.path.join(run_dir, f"track{t['track']}")
            os.mkdir(track_dir)
            path, pfacts, tsteps, twarn = prepare_wav(skill, t["src"], track_dir, t["audio_stream"], rate, t["stream"])
            prepared.append(path)
            durations.append(float(pfacts.get("duration") or 0.0))
            steps += [dict(st, track=t["track"]) for st in tsteps]
            warnings += [f"track {t['track']}: {w}" for w in twarn]
        if max(durations) - min(durations) > max(DURATION_TOLERANCE, 0.01 * max(durations)):
            warnings.append(f"the tracks differ in length ({min(durations):.3f}s to {max(durations):.3f}s); bleed reduction assumes "
                            "time-aligned tracks (one clock, one start). Align separately recorded tracks first (ffmpeg-skill sync)")
        stem_dir = os.path.join(run_dir, "out")
        os.mkdir(stem_dir)
        stems = [os.path.join(stem_dir, t["name"]) for t in tracks]
        params: Dict[str, Any] = {"timeout": float(req.timeout)}
        if req.dominance_db is not None:
            params["dominance_db"] = float(req.dominance_db)
        result = run_engine(engine.id, EngineRequest(op="bleed_reduce", input_path=prepared[0], output_path=stems[0], sample_rate=rate,
                                                     params=params, input_paths=prepared, output_paths=stems),
                            float(req.timeout), run_dir)
        if result.output_paths != stems:
            raise VoiceEnhanceError("INVALID_RESULT", f"engine {engine.id!r} returned {len(result.output_paths)} output paths for "
                                    f"{len(stems)} tracks", {"engine": engine.id})
        out_facts = [validate_output(skill, stem, rate, d, engine.id) for stem, d in zip(stems, durations)]
        x = result.extra
        coupling = x.get("coupling_db") or []
        for i, row in enumerate(coupling):
            for j, value in enumerate(row):
                if i != j and value is None:
                    warnings.append(f"track {j} was never louder than track {i} by {x.get('dominance_db')} dB, so no bleed of "
                                    f"track {j} could be measured in track {i}; nothing of it was removed")
        engine_doc = dict(report, version=result.engine_version, model=result.model, model_sha256=None)
        parameters = {"tracks": len(tracks), "audio_streams": [t["audio_stream"] for t in tracks],
                      "dominance_db": x.get("dominance_db"), "method": x.get("method"), "stft": x.get("stft"),
                      "block_seconds": x.get("block_seconds"), "coupling_db": coupling, "first_order_bins": x.get("first_order_bins"),
                      "sample_rate": rate, "channels": 1, "output_format": x.get("output_format")}
        provenance = build_provenance(op="bleed_reduce", engine=engine_doc, input_path=list(dict.fromkeys(srcs)), output_path=stems,
                                      parameters=parameters, ffmpeg_skill=info.to_dict())
        _place_all(stems, dsts, stem_dir, out_dir, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return {
        "schema": BLEED_REDUCE_SCHEMA,
        "ok": True,
        "op": "bleed_reduce",
        "capability": "voice.bleed_reduce",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output_dir": out_dir,
        "outputs": [dict(_output_doc(dst, f), track=t["track"]) for t, dst, f in zip(tracks, dsts, out_facts)],
        "inputs": [dict(_input_doc(t["src"], t["facts"], t["audio_stream"], t["stream_count"], t["stream"]), track=t["track"])
                   for t in tracks],
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model")},
        "parameters": parameters,
        "steps": steps,
        "provenance": provenance,
        "warnings": warnings,
    }
