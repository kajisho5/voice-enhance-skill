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

**Status: 0.1.0, unreleased.** `isolate` ([ADR-003](docs/decisions.md)) and `dereverb` ([ADR-004](docs/decisions.md)) work (`EXPERIMENTAL`). `contract`, `doctor` and `engines` work. `bleed-reduce` is declared (`PROPOSED`) and arrives in [roadmap](ECOSYSTEM_ROADMAP.md) step 5.

| Op | Capability | Engine ([ADR-001](docs/decisions.md)) | Licence (code / weights) | Step |
|---|---|---|---|---|
| ISOLATE | `voice.isolate` | DeepFilterNet3 via the standalone `deep-filter` binary (no PyTorch) | MIT OR Apache-2.0 / same | 3 (done) |
| DEREVERB | `voice.dereverb` | nara_wpe (WPE, signal processing) | MIT / no weights | 4 (done) |
| BLEED_REDUCE | `voice.bleed_reduce` | none: a weight-free method, decided in step 5 | - | 5 |

Traditional noise reduction, loudness and mixing belong to [audio-production-skill](https://github.com/kajisho5/audio-production-skill), not here.

```bash
pip install -e .                     # stdlib only, Python >= 3.9
voice-enhance doctor                 # what is installed, what is MISSING, and how to get it
voice-enhance engines --json         # engines with status, version, licences, attribution
voice-enhance contract --json        # machine-readable contract (alias: skill --json)
voice-enhance isolate talk.mp4 -o talk_voice.wav --json            # voice stem of audio stream 0, 48 kHz mono WAV
voice-enhance isolate talk.mp4 -o voice.wav --audio-stream 1 --atten-lim-db 20 --json
voice-enhance dereverb talk.mp4 -o talk_dry.wav --json          # reverb removed (WPE), same rate and channels, float WAV
voice-enhance dereverb hall.wav -o hall_dry.wav --taps 15 --iterations 5 --json
```

## isolate

`isolate` picks one audio stream (`--audio-stream`, 0-based, default 0) and has ffmpeg-skill extract it as PCM WAV. A stream with more than one channel is folded down to mono in a second ffmpeg-skill pass. The engine runs on that WAV in the worker, and the stem is checked through ffmpeg-skill/probe (48 kHz, mono, same duration) before it is renamed into place. The output is always a 48 kHz mono `.wav`, sample-aligned with the input (`deep-filter -D`). The input is never modified, and an existing output is never replaced without `--overwrite`. `--atten-lim-db N` limits how far the background is pushed down (default: no limit). `--timeout` (default 3600 s) kills the engine and the worker's whole process group.

The engine needs 48 kHz. A stream at another rate is converted only when the ffmpeg-skill checkout declares `audio --sample-rate`. Without that flag, `isolate` stops with `FFMPEG_SKILL_UNAVAILABLE` before doing any work, and the input has to be converted to 48 kHz first.

## dereverb

`dereverb` removes late reverberation from one audio stream (`--audio-stream`, default 0) with WPE (nara_wpe). WPE works at any sample rate and on any number of channels, so ffmpeg-skill extracts the stream as PCM WAV in a single `audio` pass. There is no fold-down and no resampling, so a 44.1 kHz stereo stream comes back as 44.1 kHz stereo and every channel helps predict the others. The engine runs a 32 ms / 8 ms STFT at the file's own rate and WPE on 30 s blocks, so memory does not grow with the length of the file. The result is a 32-bit float WAV with the input's rate, channels and length, sample-aligned with it, so nothing is clipped. It is checked through ffmpeg-skill/probe before it is renamed into place. `--taps` (default 10), `--delay` (3) and `--iterations` (3) are nara_wpe's parameters. `--delay` frames of direct sound and early reflections are always kept. The input is never modified, and `--overwrite` and `--timeout` work as for `isolate`.

## Engines (optional)

- **DeepFilterNet3** (ISOLATE): download `deep-filter` for your platform and `DeepFilterNet3_onnx.tar.gz` from the [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases) (v0.5.x). Put `deep-filter` on `PATH` (or set `VOICE_ENHANCE_DEEP_FILTER`) and set `VOICE_ENHANCE_DFN_MODEL` to the model tarball.
- **nara_wpe** (DEREVERB): `pip install -e '.[wpe]'`.

`doctor` reports each engine `OK`, `MISSING` or `INCOMPATIBLE` (version outside the supported range). Engines run in a worker subprocess in their own process group, killed on `--timeout`.

## ffmpeg-skill

Located from `--ffmpeg-skill`, `VOICE_ENHANCE_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill` or `../ffmpeg-skill`. Compatibility is checked tool by tool and flag by flag against its contract (`probe`, `audio --audio-stream --mono --replace --overwrite`, plus the optional `audio --sample-rate`), not by version number.

## JSON and errors

With `--json`, stdout holds exactly one JSON document. Failures are `{"ok": false, "error": {"code", "message", "retryable", "details"}}`; the code fixes the exit code (`contract --json` lists them under `errors`). `TOOL_ERROR`, `ENGINE_TIMEOUT` and `CANCELLED` are retryable.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests need no engine, no weights and no FFmpeg (fake `deep-filter`, a fake `nara_wpe` package, mocked package lookups, a stdlib stub of ffmpeg-skill for both ops). The ISOLATE SNR check runs on a synthetic voice+noise fixture. The DEREVERB check (signal-to-reverberation ratio) runs on a synthetic reverberant fixture and needs NumPy, because the fake nara_wpe computes on it; it is skipped without NumPy. The real-engine versions run only where the engine is installed. The integration tests run against a real ffmpeg-skill checkout (`VOICE_ENHANCE_FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg; the OS registry conformance checks run when `AI_VIDEO_PRODUCTION_OS_DIR` (or `../AI-video-production-OS`) is present.

## Licence

MIT. Engine attribution is listed by `voice-enhance engines --json` and goes into the provenance of every op result.
