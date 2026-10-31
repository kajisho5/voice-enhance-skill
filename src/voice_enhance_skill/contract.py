"""Machine-readable contract (`voice-enhance contract --json`), in the shape the AI Video Production OS registry reads
(skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict

from . import PACKAGE_NAME, SKILL_ID, __version__
from .doctor import DOCTOR_SCHEMA, IMPLEMENTED_OPS, OP_STEP
from .engines import DEFAULT_ENGINE, engine_ids, get_engine
from .errors import ERROR_TABLE
from .ffmpeg_skill import FLAGS_USED, OPTIONAL_FLAGS, SUPPORTED_CONTRACT_VERSION, TOOLS_USED
from .engines.crosstalk import DEFAULTS as CROSSTALK_DEFAULTS, MAX_TRACKS, MIN_TRACKS
from .ops import BLEED_REDUCE_SCHEMA, DEREVERB_SCHEMA, ISOLATE_RATE, ISOLATE_SCHEMA

CONTRACT_VERSION = "1.0"
CAPABILITY_IDS = {"isolate": "voice.isolate", "dereverb": "voice.dereverb", "bleed_reduce": "voice.bleed_reduce"}
_OP_TOOLS = {
    "isolate": {"description": "Voice stem of a recording with noise and background removed (AI, DeepFilterNet3)",
                "argv": ["isolate", "<media>", "-o <out.wav>", "[--audio-stream 0]", "[--engine deepfilternet]",
                         "[--atten-lim-db <0-100>]", "[--timeout 3600]", "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--offline]",
                         "--json"],
                "schema": ISOLATE_SCHEMA,
                "output": {"format": "wav", "sample_rate": ISOLATE_RATE, "channels": 1,
                           "alignment": "sample-aligned with the input (deep-filter -D compensates its delay)"},
                "result_keys": ["schema", "ok", "op", "capability", "skill", "output", "input", "engine", "parameters", "steps",
                                "provenance", "warnings"]},
    "dereverb": {"description": "Speech with room reverberation reduced (WPE, nara_wpe)",
                 "argv": ["dereverb", "<media>", "-o <out.wav>", "[--audio-stream 0]", "[--engine nara_wpe]", "[--taps 10]",
                          "[--delay 3]", "[--iterations 3]", "[--timeout 3600]", "[--overwrite]", "[--ffmpeg-skill <dir>]",
                          "[--offline]", "--json"],
                 "schema": DEREVERB_SCHEMA,
                 "output": {"format": "wav", "codec": "pcm_f32le", "sample_rate": "same as the input stream",
                            "channels": "same as the input stream (multi-channel WPE)",
                            "alignment": "sample-aligned with the input (WPE subtracts a prediction; no delay is added)"},
                 "result_keys": ["schema", "ok", "op", "capability", "skill", "output", "input", "engine", "parameters", "steps",
                                 "provenance", "warnings"]},
    "bleed_reduce": {"description": "Per-microphone tracks with cross-talk from the other microphones reduced (signal processing, "
                                    "no model weights)",
                     "argv": ["bleed-reduce", "<media>...", "-o <out_dir>", "[--audio-streams 0,1]", "[--engine crosstalk]",
                              f"[--dominance-db {CROSSTALK_DEFAULTS['dominance_db']:g}]", "[--timeout 3600]", "[--overwrite]",
                              "[--ffmpeg-skill <dir>]", "[--offline]", "--json"],
                     "schema": BLEED_REDUCE_SCHEMA,
                     "input": {"tracks": f"{MIN_TRACKS}-{MAX_TRACKS} time-aligned microphone tracks: the audio streams of one file "
                                         "(all, or --audio-streams), or one stream per file (stream 0, or one --audio-streams "
                                         "entry per file)", "sample_rate": "one rate for all tracks (the first track's)"},
                     "output": {"format": "wav", "codec": "pcm_f32le", "files": "one per track: <out_dir>/<input name>_a<stream>.wav",
                                "sample_rate": "the first track's", "channels": 1,
                                "alignment": "sample-aligned with each input (linear per-bin demixing; no delay is added)"},
                     "result_keys": ["schema", "ok", "op", "capability", "skill", "output_dir", "outputs", "inputs", "engine",
                                     "parameters", "steps", "provenance", "warnings"]},
}


def tools() -> list:
    out = []
    for op, spec in _OP_TOOLS.items():
        out.append({"tool_id": f"{SKILL_ID}/{op.replace('_', '-')}", "role": "transformation", "writes_media": True,
                    "mutates_input": False, "deterministic": True, "implemented": IMPLEMENTED_OPS[op], "roadmap_step": OP_STEP[op],
                    "engine": DEFAULT_ENGINE[op], **spec})
    out += [
        {"tool_id": f"{SKILL_ID}/doctor", "role": "diagnostic", "writes_media": False, "mutates_input": False, "deterministic": False,
         "implemented": True, "description": "What is installed and missing (engines, ffmpeg-skill) and how to get it; never downloads",
         "argv": ["doctor", "[--offline]", "[--ffmpeg-skill <dir>]", "--json"], "schema": DOCTOR_SCHEMA},
        {"tool_id": f"{SKILL_ID}/engines", "role": "diagnostic", "writes_media": False, "mutates_input": False, "deterministic": False,
         "implemented": True, "description": "Registered engines with status, version, licences and attribution",
         "argv": ["engines", "[--offline]", "--json"]},
        {"tool_id": f"{SKILL_ID}/contract", "role": "metadata", "writes_media": False, "mutates_input": False, "deterministic": True,
         "implemented": True, "description": "This contract", "argv": ["contract", "--json"]},
    ]
    return out


def contract() -> Dict[str, Any]:
    deps = [
        {"id": "ffmpeg-skill", "kind": "skill", "required": True, "for": [CAPABILITY_IDS[o] for o in CAPABILITY_IDS],
         "version_range": "any release whose contract declares the tools and flags below (capability detection)",
         "tools": list(TOOLS_USED), "flags": {t: list(f) for t, f in FLAGS_USED.items()},
         "optional_flags": {t: list(f) for t, f in OPTIONAL_FLAGS.items()},
         "contract_version": SUPPORTED_CONTRACT_VERSION,
         "located_by": ["--ffmpeg-skill", "$VOICE_ENHANCE_FFMPEG_SKILL_DIR", "$VIDEO_AGENT_FFMPEG_SKILL_DIR",
                        "~/.claude/skills/ffmpeg-skill", "./vendor/ffmpeg-skill", "../ffmpeg-skill"]},
    ]
    for eid in engine_ids():
        e = get_engine(eid)
        deps.append({"id": eid, "kind": "engine-" + e.kind, "required": False, "for": [CAPABILITY_IDS[o] for o in e.ops],
                     "version_range": e.version_range, "license": e.code_license, "weights_license": e.weights_license,
                     "install": e.install, "located_by": list(e.located_by), "downloaded_at_run_time": False})
    return {
        "skill_id": SKILL_ID,
        "id": SKILL_ID,
        "skill": PACKAGE_NAME,
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "provides": [{"id": CAPABILITY_IDS[op], "lifecycle": "EXPERIMENTAL" if IMPLEMENTED_OPS[op] else "PROPOSED",
                      "tool_id": f"{SKILL_ID}/{op.replace('_', '-')}"} for op in CAPABILITY_IDS],
        "dependencies": deps,
        "not_provided": ["traditional (non-AI) noise reduction, gain, loudness, mixing, silence and filler removal (audio-production-skill)",
                         "transcription (transcription-skill)", "speaker diarization (diarization-skill)",
                         "voice replacement, TTS, dubbing (dubbing-skill)", "music / BGM separation or generation (sound-design-skill)",
                         "generative re-synthesis of the voice (ADR-001: faithful, not generative)",
                         "downloading engines or model weights (one-time manual step; doctor says how)",
                         "network access of any kind at run time", "direct ffmpeg calls (every media step is an ffmpeg-skill tool)"],
        "tools": tools(),
        "engines": {"registered": {eid: get_engine(eid).spec() for eid in engine_ids()},
                    "default": dict(DEFAULT_ENGINE),
                    "selection_adr": "docs/decisions.md ADR-001",
                    "optional": True,
                    "statuses": ["OK", "MISSING", "INCOMPATIBLE"]},
        "provenance": {"fields": ["skill", "op", "engine.id", "engine.version", "engine.model", "engine.model_sha256", "engine.code_license",
                                  "engine.weights_license", "engine.attribution", "input.sha256", "output.sha256",
                                  "inputs[].sha256 (bleed_reduce)", "outputs[].sha256 (bleed_reduce)", "parameters",
                                  "ffmpeg_skill.version", "network"],
                       "status": "OBSERVED"},
        "execution": {"canonical_invocation": ["voice-enhance", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess in its own process group, killed on --timeout; Hugging Face libraries forced offline",
                      "media": "through ffmpeg-skill tools only", "input_is_read_only": True,
                      "overwrites": "never unless --overwrite; an output equal to an input is refused",
                      "offline_flag": "accepted everywhere; every registered engine already works with no network"},
        "errors": {code: {"exit_code": ex, "retryable": rt} for code, (ex, rt) in ERROR_TABLE.items()},
        "error_shape": {"ok": False, "error": {"code": "<ERRORS key>", "message": "str", "retryable": "bool", "details": "object"}},
    }
