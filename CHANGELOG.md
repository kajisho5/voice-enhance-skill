# Changelog

## 0.1.0 - unreleased

Package skeleton (roadmap steps 1-2).

- ADR-001: engine selection. DeepFilterNet3 (`deep-filter` binary, MIT OR Apache-2.0 code and weights) for ISOLATE, nara_wpe (MIT, no weights) for DEREVERB, no AI engine for BLEED_REDUCE. Demucs, Resemble-Enhance and VoiceFixer were checked and not chosen (licence of code and weights, CPU, generative re-synthesis).
- ADR-002: skeleton. `voice-enhance contract` / `skill`, `doctor` (`--offline`, `--ffmpeg-skill`), `engines`; engine registry with `OK` / `MISSING` / `INCOMPATIBLE` availability checks; worker subprocess in its own process group with a timeout; typed errors with exit codes and `retryable`; provenance block; ffmpeg-skill adapter by capability detection.
- The three ops are declared with lifecycle `PROPOSED` and are not implemented yet.
