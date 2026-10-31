# Changelog

## 0.1.0 - unreleased

ISOLATE op (roadmap step 3).

- ADR-003: `voice-enhance isolate <media> -o <out.wav>` gives the voice stem of one audio stream as a 48 kHz mono WAV, using DeepFilterNet3 (`deep-filter -D -m <model>`) in the worker. Optional `--audio-stream`, `--atten-lim-db`, `--timeout`, `--overwrite`, `--ffmpeg-skill`, `--offline`. Result schema `voice-enhance/isolate@1` with steps, warnings and provenance (now including the model's SHA-256). `voice.isolate` is `EXPERIMENTAL`.
- The input is prepared with ffmpeg-skill `audio` (`--audio-stream`, then `--mono` in a second pass) instead of `join`, which refuses a single clip. A sample rate other than 48 kHz needs `audio --sample-rate` from ffmpeg-skill (capability-detected); without it the op is refused before any work with `FFMPEG_SKILL_UNAVAILABLE`.
- The engine binary runs in the worker's process group, so `--timeout` kills it too. The stem is validated through ffmpeg-skill/probe (rate, channels, duration) and renamed into place from a private run directory.

Package skeleton (roadmap steps 1-2).

- ADR-001: engine selection. DeepFilterNet3 (`deep-filter` binary, MIT OR Apache-2.0 code and weights) for ISOLATE, nara_wpe (MIT, no weights) for DEREVERB, no AI engine for BLEED_REDUCE. Demucs, Resemble-Enhance and VoiceFixer were checked and not chosen (licence of code and weights, CPU, generative re-synthesis).
- ADR-002: skeleton. `voice-enhance contract` / `skill`, `doctor` (`--offline`, `--ffmpeg-skill`), `engines`; engine registry with `OK` / `MISSING` / `INCOMPATIBLE` availability checks; worker subprocess in its own process group with a timeout; typed errors with exit codes and `retryable`; provenance block; ffmpeg-skill adapter by capability detection.
- The three ops are declared with lifecycle `PROPOSED` and are not implemented yet.
