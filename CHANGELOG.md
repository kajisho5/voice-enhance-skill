# Changelog

## 0.1.0 - unreleased

DEREVERB op (roadmap step 4).

- ADR-004: `voice-enhance dereverb <media> -o <out.wav>` removes late reverberation from one audio stream with WPE (`nara_wpe.wpe.wpe_v8`, statistics mode `full`) in the worker. The result keeps the stream's own sample rate and channels (multi-channel WPE), is written as 32-bit float WAV and is sample-aligned with the input. Optional `--audio-stream`, `--taps`, `--delay`, `--iterations` (nara_wpe's defaults 10 / 3 / 3), `--timeout`, `--overwrite`, `--ffmpeg-skill`, `--offline`. Result schema `voice-enhance/dereverb@1`. `voice.dereverb` is `EXPERIMENTAL`.
- The input is prepared with a single ffmpeg-skill `audio --audio-stream N` pass. WPE needs no particular rate or layout, so there is no fold-down and no `audio --sample-rate`. The stem is validated through ffmpeg-skill/probe against the prepared input's rate, channels and duration.
- The engine does its own STFT (Hann, 32 ms / 8 ms at the file's rate) instead of `nara_wpe.utils`, which needs SciPy. It streams the file in 30 s blocks, so memory does not grow with the input's length.
- Tests: a fake `nara_wpe` package (a minimal WPE on NumPy) raises the signal-to-reverberation ratio of a synthetic reverberant fixture by more than 6 dB, through the stub and through the real ffmpeg-skill. The CI integration job installs NumPy for it.

ISOLATE op (roadmap step 3).

- ADR-003: `voice-enhance isolate <media> -o <out.wav>` gives the voice stem of one audio stream as a 48 kHz mono WAV, using DeepFilterNet3 (`deep-filter -D -m <model>`) in the worker. Optional `--audio-stream`, `--atten-lim-db`, `--timeout`, `--overwrite`, `--ffmpeg-skill`, `--offline`. Result schema `voice-enhance/isolate@1` with steps, warnings and provenance (now including the model's SHA-256). `voice.isolate` is `EXPERIMENTAL`.
- The input is prepared with ffmpeg-skill `audio` (`--audio-stream`, then `--mono` in a second pass) instead of `join`, which refuses a single clip. A sample rate other than 48 kHz needs `audio --sample-rate` from ffmpeg-skill (capability-detected); without it the op is refused before any work with `FFMPEG_SKILL_UNAVAILABLE`.
- The engine binary runs in the worker's process group, so `--timeout` kills it too. The stem is validated through ffmpeg-skill/probe (rate, channels, duration) and renamed into place from a private run directory.

Package skeleton (roadmap steps 1-2).

- ADR-001: engine selection. DeepFilterNet3 (`deep-filter` binary, MIT OR Apache-2.0 code and weights) for ISOLATE, nara_wpe (MIT, no weights) for DEREVERB, no AI engine for BLEED_REDUCE. Demucs, Resemble-Enhance and VoiceFixer were checked and not chosen (licence of code and weights, CPU, generative re-synthesis).
- ADR-002: skeleton. `voice-enhance contract` / `skill`, `doctor` (`--offline`, `--ffmpeg-skill`), `engines`; engine registry with `OK` / `MISSING` / `INCOMPATIBLE` availability checks; worker subprocess in its own process group with a timeout; typed errors with exit codes and `retryable`; provenance block; ffmpeg-skill adapter by capability detection.
- The three ops are declared with lifecycle `PROPOSED` and are not implemented yet.
