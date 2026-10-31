<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/app-icon-dark.png">
    <img src="brand/app-icon.png" width="128" alt="voice-enhance-skill app icon">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/voice-enhance-skill-wordmark-white.svg">
    <img src="brand/logo/voice-enhance-skill-wordmark-black.svg" height="48" alt="voice-enhance-skill">
  </picture>
</p>

<p align="center"><b>Speech enhancement for recordings</b></p>


AI voice isolation, dereverberation and mic-bleed reduction for recorded speech, for the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS) (department 音 / Audio). Local and offline: engines are optional, never downloaded by the skill, and every media step (probe, format conversion, putting the cleaned audio back into the video) goes through [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill).

**Status: skeleton (0.1.0).** `contract`, `doctor` and `engines` work; the ops are declared (`PROPOSED`) and arrive in [roadmap](ECOSYSTEM_ROADMAP.md) steps 3-5.

| Op | Capability | Engine ([ADR-001](docs/decisions.md)) | Licence (code / weights) | Step |
|---|---|---|---|---|
| ISOLATE | `voice.isolate` | DeepFilterNet3 via the standalone `deep-filter` binary (no PyTorch) | MIT OR Apache-2.0 / same | 3 |
| DEREVERB | `voice.dereverb` | nara_wpe (WPE, signal processing) | MIT / no weights | 4 |
| BLEED_REDUCE | `voice.bleed_reduce` | none: a weight-free method, decided in step 5 | - | 5 |

Traditional noise reduction, loudness and mixing belong to [audio-production-skill](https://github.com/kajisho5/audio-production-skill), not here.

```bash
pip install -e .                     # stdlib only, Python >= 3.9
voice-enhance doctor                 # what is installed, what is MISSING, and how to get it
voice-enhance engines --json         # engines with status, version, licences, attribution
voice-enhance contract --json        # machine-readable contract (alias: skill --json)
```

## Engines (optional)

- **DeepFilterNet3** (ISOLATE): download `deep-filter` for your platform and `DeepFilterNet3_onnx.tar.gz` from the [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases) (v0.5.x). Put `deep-filter` on `PATH` (or set `VOICE_ENHANCE_DEEP_FILTER`) and set `VOICE_ENHANCE_DFN_MODEL` to the model tarball.
- **nara_wpe** (DEREVERB): `pip install -e '.[wpe]'`.

`doctor` reports each engine `OK`, `MISSING` or `INCOMPATIBLE` (version outside the supported range). Engines run in a worker subprocess in their own process group, killed on `--timeout`.

## ffmpeg-skill

Located from `--ffmpeg-skill`, `VOICE_ENHANCE_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill` or `../ffmpeg-skill`. Compatibility is checked tool by tool and flag by flag against its contract (`probe`, `join --sample-rate --channels`, `audio --replace --audio-stream --overwrite`), not by version number.

## JSON and errors

With `--json`, stdout holds exactly one JSON document. Failures are `{"ok": false, "error": {"code", "message", "retryable", "details"}}`; the code fixes the exit code (`contract --json` lists them under `errors`). `TOOL_ERROR`, `ENGINE_TIMEOUT` and `CANCELLED` are retryable.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests need no engine, no weights and no FFmpeg (fake `deep-filter`, mocked package lookups). The integration tests run against a real ffmpeg-skill checkout (`VOICE_ENHANCE_FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg; the OS registry conformance checks run when `AI_VIDEO_PRODUCTION_OS_DIR` (or `../AI-video-production-OS`) is present.

## Licence

MIT. Engine attribution is listed by `voice-enhance engines --json` and goes into the provenance of every op result.
