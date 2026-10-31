"""voice-enhance CLI.

stdout: with --json exactly one JSON document, on success and failure alike. stderr: diagnostics only.
Exit code: 0 on success; doctor exits 1 when its status is `fail`; an error exits errors.EXIT_CODES[code].

  voice-enhance contract --json     the contract (alias: skill --json)
  voice-enhance doctor --json       engines, ffmpeg-skill, ops readiness [--offline] [--ffmpeg-skill DIR]
  voice-enhance engines --json      registered engines with status and licences [--offline]
  voice-enhance isolate <media> -o <out.wav> --json
                                    voice stem of one audio stream, 48 kHz mono WAV (ADR-003) [--audio-stream N]
                                    [--engine deepfilternet] [--atten-lim-db N] [--timeout S] [--overwrite]
                                    [--ffmpeg-skill DIR] [--offline]
  voice-enhance dereverb <media> -o <out.wav> --json
                                    one audio stream with late reverberation removed (WPE), same rate and channels,
                                    32-bit float WAV (ADR-004) [--audio-stream N] [--engine nara_wpe] [--taps N]
                                    [--delay N] [--iterations N] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline]
  voice-enhance bleed-reduce <media>... -o <out_dir> --json
                                    microphone tracks with the other mics' cross-talk reduced, one mono 32-bit float WAV
                                    per track (ADR-005) [--audio-streams 0,1] [--engine crosstalk] [--dominance-db N]
                                    [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline]
  voice-enhance remux <video> <stem> -o <out> --json
                                    the stem back into the video through ffmpeg-skill/audio --replace: video stream
                                    copied, the stem as the only audio stream, same container (ADR-006)
                                    [--drop-other-audio] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline]"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import PACKAGE_NAME, SKILL_ID, __version__
from .contract import contract
from .doctor import doctor, engines_report
from .errors import VoiceEnhanceError
from .engines.crosstalk import DEFAULTS as CROSSTALK_DEFAULTS, LIMITS as CROSSTALK_LIMITS, MAX_TRACKS
from .engines.nara_wpe_engine import DEFAULTS as WPE_DEFAULTS, LIMITS as WPE_LIMITS
from .ops import BleedReduceRequest, DereverbRequest, IsolateRequest, RemuxRequest, bleed_reduce, dereverb, isolate, remux
from .service import DEFAULT_TIMEOUT


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voice-enhance", description=f"{PACKAGE_NAME} {__version__}: AI voice isolation, "
                                 "dereverberation and mic-bleed reduction (local, offline)")
    ap.add_argument("--version", action="version", version=f"{PACKAGE_NAME} {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("contract", "skill"):
        c = sub.add_parser(name, help="print the machine-readable contract")
        c.add_argument("--json", action="store_true", help="JSON on stdout (exactly one document)")
    d = sub.add_parser("doctor", help="what is installed, what is missing, how to get it (never downloads)")
    d.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory (default: env / ~/.claude/skills/ffmpeg-skill / ./vendor / ..)")
    d.add_argument("--offline", action="store_true", help="report readiness with no network at all")
    d.add_argument("--json", action="store_true")
    e = sub.add_parser("engines", help="registered engines: status, version, licences, attribution")
    e.add_argument("--offline", action="store_true")
    e.add_argument("--json", action="store_true")
    i = sub.add_parser("isolate", help="voice stem of a recording, noise and background removed (DeepFilterNet3); 48 kHz mono WAV")
    i.add_argument("input", help="media file (video or audio); never modified")
    i.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    i.add_argument("--audio-stream", type=int, default=0, help="which audio stream, 0-based in file order (default 0)")
    i.add_argument("--engine", help="engine id (default: deepfilternet)")
    i.add_argument("--atten-lim-db", type=float, help="limit the noise attenuation to N dB, 0 < N <= 100 (default: no limit)")
    i.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"engine time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    i.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    i.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    i.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    i.add_argument("--json", action="store_true")
    r = sub.add_parser("dereverb", help="one audio stream with room reverberation reduced (WPE, nara_wpe); same rate and channels, "
                       "32-bit float WAV")
    r.add_argument("input", help="media file (video or audio); never modified")
    r.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    r.add_argument("--audio-stream", type=int, default=0, help="which audio stream, 0-based in file order (default 0)")
    r.add_argument("--engine", help="engine id (default: nara_wpe)")
    for key, what in (("taps", "prediction filter length in STFT frames"), ("delay", "frames skipped before the prediction "
                      "starts (direct sound and early reflections are kept)"), ("iterations", "WPE iterations")):
        lo, hi = WPE_LIMITS[key]
        r.add_argument(f"--{key}", type=int, help=f"{what}, {lo}-{hi} (default {WPE_DEFAULTS[key]})")
    r.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"engine time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    r.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    r.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    r.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    r.add_argument("--json", action="store_true")
    b = sub.add_parser("bleed-reduce", help="microphone tracks with the other mics' cross-talk reduced (signal processing, no weights); "
                       "one mono 32-bit float WAV per track")
    b.add_argument("inputs", nargs="+", help=f"media files (never modified): one file with 2-{MAX_TRACKS} audio streams, or one file "
                   "per microphone; the tracks must be time-aligned")
    b.add_argument("-o", "--output", required=True, help="output directory (created if missing); one <input name>_a<stream>.wav "
                   "per track, none replaced unless --overwrite")
    b.add_argument("--audio-streams", type=_int_list, help="which audio streams, 0-based, comma-separated: with one input, the "
                   "tracks (default: all of its streams); with several inputs, one per input (default: stream 0 of each)")
    b.add_argument("--engine", help="engine id (default: crosstalk)")
    lo, hi = CROSSTALK_LIMITS["dominance_db"]
    b.add_argument("--dominance-db", type=float, help=f"how much louder a track must be than another for a bin to count as its "
                   f"talker when the bleed is measured, {lo:g}-{hi:g} dB (default {CROSSTALK_DEFAULTS['dominance_db']:g})")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"engine time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    b.add_argument("--overwrite", action="store_true", help="allow replacing existing outputs")
    b.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    b.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    b.add_argument("--json", action="store_true")
    m = sub.add_parser("remux", help="put a stem back into its video through ffmpeg-skill/audio --replace: the video stream copied, "
                       "the stem as the only audio stream")
    m.add_argument("input", help="the video (never modified); one video stream")
    m.add_argument("stem", help="the audio to put under the picture (a stem from isolate, dereverb or bleed-reduce)")
    m.add_argument("-o", "--output", required=True, help="output file with the input's extension (refused if it exists, unless "
                   "--overwrite)")
    m.add_argument("--drop-other-audio", action="store_true", help="accept an input with several audio streams: the output keeps "
                   "only the stem (ffmpeg-skill cannot replace one stream and keep the others)")
    m.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"ffmpeg-skill time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    m.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    m.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    m.add_argument("--offline", action="store_true", help="accepted for symmetry; remux never uses the network")
    m.add_argument("--json", action="store_true")
    return ap


def _int_list(text: str) -> List[int]:
    try:
        return [int(v) for v in text.split(",") if v.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")


def _emit(doc: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        _human(doc)
    sys.stdout.flush()


def _human(doc: Dict[str, Any]) -> None:
    if doc.get("error"):
        err = doc["error"]
        sys.stderr.write(f"error [{err['code']}]: {err['message']}\n")
    elif doc.get("op") == "remux":
        o = doc["output"]
        print(f"wrote {o['path']} ({o['duration']:.2f}s, video {o['video']['codec']} copied, audio {o['audio']['codec']} "
              f"{o['audio']['channels']} ch)")
        for w in doc.get("warnings", []):
            print(f"  warning: {w}")
    elif "op" in doc and "outputs" in doc:
        e = doc["engine"]
        for o in doc["outputs"]:
            print(f"wrote {o['path']} ({o['duration']:.2f}s, {o['sample_rate']} Hz, {o['channels']} ch) with {e['id']} {e['version']}")
        for w in doc.get("warnings", []):
            print(f"  warning: {w}")
    elif "op" in doc and "output" in doc:
        o, e = doc["output"], doc["engine"]
        print(f"wrote {o['path']} ({o['duration']:.2f}s, {o['sample_rate']} Hz, {o['channels']} ch) with {e['id']} {e['version']}")
        for w in doc.get("warnings", []):
            print(f"  warning: {w}")
    elif "checks" in doc:
        print(f"{PACKAGE_NAME} {__version__}: {doc['status']}")
        fs = doc["checks"]["ffmpeg_skill"]
        print(f"  ffmpeg-skill  {fs['status']:12} {fs.get('version') or fs.get('detail') or ''}")
        for eid, e in doc["checks"]["engines"].items():
            print(f"  engine {eid:14} {e['status']:12} {e.get('detail') or ''}")
            if e.get("install"):
                print(f"    install: {e['install']}")
        for op, o in doc["checks"]["operations"].items():
            print(f"  op {op:14} implemented={o['implemented']} engine={o['engine']} ready={o['ready']}")
        for name, h in doc["checks"].get("helpers", {}).items():
            print(f"  helper {name:10} implemented={h['implemented']} ffmpeg-skill={h['ffmpeg_skill_tool']} ready={h['ready']}")
        for p in doc["problems"]:
            print(f"  problem: {p}")
    elif "engines" in doc and "skill_id" not in doc:
        for eid, e in doc["engines"].items():
            print(f"{eid:14} {e['status']:12} {e.get('version') or '-':10} {e['code_license']}")
    else:
        print(f"{doc['skill_id']} {doc['version']}: " + ", ".join(f"{p['id']} ({p['lifecycle']})" for p in doc["provides"]))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        if args.cmd in ("contract", "skill"):
            _emit(contract(), as_json)
            return 0
        if args.cmd == "doctor":
            doc = doctor(args.ffmpeg_skill, offline=args.offline)
            _emit(doc, as_json)
            return 1 if doc["status"] == "fail" else 0
        if args.cmd == "engines":
            _emit({"ok": True, "skill": {"id": SKILL_ID, "version": __version__}, "offline": bool(args.offline),
                   "engines": engines_report(offline=args.offline)}, as_json)
            return 0
        if args.cmd == "isolate":
            _emit(isolate(IsolateRequest(input_path=args.input, output_path=args.output, audio_stream=args.audio_stream,
                                         engine=args.engine, timeout=args.timeout, overwrite=args.overwrite,
                                         atten_lim_db=args.atten_lim_db, ffmpeg_skill_dir=args.ffmpeg_skill,
                                         offline=args.offline)), as_json)
            return 0
        if args.cmd == "dereverb":
            _emit(dereverb(DereverbRequest(input_path=args.input, output_path=args.output, audio_stream=args.audio_stream,
                                           engine=args.engine, timeout=args.timeout, overwrite=args.overwrite, taps=args.taps,
                                           delay=args.delay, iterations=args.iterations, ffmpeg_skill_dir=args.ffmpeg_skill,
                                           offline=args.offline)), as_json)
            return 0
        if args.cmd == "remux":
            _emit(remux(RemuxRequest(input_path=args.input, stem_path=args.stem, output_path=args.output,
                                     drop_other_audio=args.drop_other_audio, timeout=args.timeout, overwrite=args.overwrite,
                                     ffmpeg_skill_dir=args.ffmpeg_skill, offline=args.offline)), as_json)
            return 0
        if args.cmd == "bleed-reduce":
            _emit(bleed_reduce(BleedReduceRequest(input_paths=args.inputs, output_dir=args.output, audio_streams=args.audio_streams,
                                                  engine=args.engine, timeout=args.timeout, overwrite=args.overwrite,
                                                  dominance_db=args.dominance_db, ffmpeg_skill_dir=args.ffmpeg_skill,
                                                  offline=args.offline)), as_json)
            return 0
    except VoiceEnhanceError as e:
        _emit({"ok": False, "skill": {"id": SKILL_ID, "version": __version__}, "error": e.to_dict()}, as_json)
        return e.exit_code
    except Exception as e:  # pragma: no cover - a bug; still one JSON document
        err = VoiceEnhanceError("INTERNAL_ERROR", f"{type(e).__name__}: {e}")
        _emit({"ok": False, "skill": {"id": SKILL_ID, "version": __version__}, "error": err.to_dict()}, as_json)
        return err.exit_code
    return 2  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
