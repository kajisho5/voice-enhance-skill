---
name: voice-enhance-skill
description: Clean up recorded speech with local, offline engines - isolate the voice from noise and background (DeepFilterNet3), remove room reverberation (WPE, nara_wpe), reduce cross-talk between time-aligned microphones of a panel or interview (signal processing, no weights), and put the cleaned stem back into its video with the picture copied untouched. Use when asked to make a voice clearer, remove background noise or hum from speech with AI, make a recording in a hall or echoey room sound drier, stop each lavalier from picking up the other speakers, or replace a video's audio with the cleaned voice. All three ops are EXPERIMENTAL. Runs on ffmpeg-skill; engines are optional installs, never downloaded.
---

# voice-enhance-skill

Run `voice-enhance <command> ... --json` (installed by `pip install -e .`, Python >= 3.9, stdlib only). Every command prints exactly one JSON document. Media work (probe, extraction, fold-down, putting audio back) goes through ffmpeg-skill, found through `--ffmpeg-skill DIR`, `$VOICE_ENHANCE_FFMPEG_SKILL_DIR`, `$VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill` or `../ffmpeg-skill`. `voice-enhance doctor --json` says which engines are `OK`, `MISSING` or `INCOMPATIBLE` and how to install each; it never downloads anything.

| Task | Command |
|---|---|
| Voice stem, noise and background removed | `voice-enhance isolate talk.mp4 -o talk_voice.wav --json` (48 kHz mono WAV) |
| Same, another audio stream, keep some background | add `--audio-stream 1 --atten-lim-db 20` (0-based stream; 0 < dB <= 100, default no limit) |
| Reverb removed | `voice-enhance dereverb talk.mp4 -o talk_dry.wav --json` (same rate and channels, 32-bit float WAV) |
| Same, stronger | add `--taps 15 --iterations 5` (nara_wpe's `--taps` 10, `--delay` 3, `--iterations` 3 by default) |
| Cross-talk between the mics of one file | `voice-enhance bleed-reduce panel.mkv -o clean/ --json` (each audio stream is one mic; one mono float WAV per mic) |
| Same, one mic per file | `voice-enhance bleed-reduce lav_a.wav lav_b.wav -o clean/ --json` (add `--audio-streams 0,2` to pick streams, `--dominance-db 6` to be stricter) |
| The stem back under the picture | `voice-enhance remux talk.mp4 talk_voice.wav -o talk_clean.mp4 --json` (video stream copied, the stem the only audio) |
| Same, the input has several audio streams | add `--drop-other-audio` |
| What is installed | `voice-enhance doctor --json`, `voice-enhance engines --json` |
| What this skill declares | `voice-enhance contract --json` (alias `skill --json`) |

Every op takes `--overwrite` (an existing output is refused without it), `--timeout S` (default 3600; the engine's whole process group is killed), `--ffmpeg-skill DIR` and `--offline`. The input is never modified, and an output appears whole or not at all.

## When to use it

- **Noise, hum, air conditioning, a busy room behind the voice:** `isolate`. It removes everything that is not speech, including music under the voice.
- **A hall, a classroom, a bare room (the voice sounds far away, with a tail):** `dereverb`. It keeps the direct sound and early reflections and removes the late tail.
- **Several microphones recorded together (a panel, a two-person interview with lavaliers):** `bleed-reduce`, on tracks that are already time-aligned. Separately recorded tracks are lined up first with ffmpeg-skill `sync`; this skill does not align them.
- **Several of these:** `bleed-reduce` first (it needs every track), then `dereverb` or `isolate` per track.
- **Back into the video:** `remux` with the stem. Run qc-skill on the result if it is a delivery.
- Plain noise reduction at fixed settings, loudness, EQ and mixing are audio-production-skill's job, not this one: it is faster, needs no engine, and is enough for mild noise.

## Limits to tell the user

- **EXPERIMENTAL.** `voice.isolate`, `voice.dereverb` and `voice.bleed_reduce` have lifecycle `EXPERIMENTAL`. CI runs the real DeepFilterNet3 and nara_wpe on synthetic fixtures; none has been measured on real Japanese speech yet. Say the result is automatic and should be listened to.
- **48 kHz for `isolate`.** DeepFilterNet3 works at 48 kHz, and the stem is always 48 kHz mono. A stream at another rate (44.1 kHz, 16 kHz) is converted only when the ffmpeg-skill checkout declares `audio --sample-rate` (`doctor` shows it under `optional_flags`). Today's ffmpeg-skill does not, so `isolate` stops with `FFMPEG_SKILL_UNAVAILABLE` before doing any work. Tell the user the input has to be 48 kHz; do not resample it with a private ffmpeg call. `dereverb` works at any rate. `bleed-reduce` needs every track at the first track's rate.
- **Fold-down.** `isolate` and `bleed-reduce` fold a multi-channel stream to mono (a warning says so). Two microphones on the two channels of one stereo stream cannot be separated, and become one track.
- **`remux` re-encodes the audio** (AAC for `.mp4`, `.mov`, `.mkv`), keeps the input's container, and refuses a file with two video streams.
- Read and pass on every entry of `warnings` (fold-down, a length difference, an unmeasured microphone pair in `parameters.coupling_db`, an A/V offset). A `null` coupling means nothing was removed for that pair.

## Engines (optional)

| Op | Engine | Install |
|---|---|---|
| `isolate` | DeepFilterNet3 through the standalone `deep-filter` binary (v0.5.x, no PyTorch) | Download `deep-filter` for your platform from https://github.com/Rikorose/DeepFilterNet/releases (for example `deep-filter-0.5.6-x86_64-unknown-linux-musl`), and `DeepFilterNet3_onnx.tar.gz` from https://github.com/Rikorose/DeepFilterNet/raw/v0.5.6/models/DeepFilterNet3_onnx.tar.gz (it is not a release asset). Put `deep-filter` on `PATH` (or set `VOICE_ENHANCE_DEEP_FILTER`) and set `VOICE_ENHANCE_DFN_MODEL` to the tarball. |
| `dereverb` | nara_wpe 0.0.x (WPE, signal processing, no weights) | `pip install -e '.[wpe]'` |
| `bleed-reduce` | `crosstalk` (this package, NumPy, no weights) | `pip install -e '.[bleed]'` |

When an op fails with `ENGINE_UNAVAILABLE` or `MODEL_NOT_FOUND`, `error.details.install` holds the line to pass on. Do not download or install an engine without the user's go-ahead. Engine attribution is in every result's `provenance`; keep it with anything published.

## Errors

With `--json`, a failure is `{"ok": false, "error": {"code", "message", "retryable", "details"}}`, and the code fixes the exit code.

| Code | Exit | Retry? | Meaning, and what to do |
|---|---|---|---|
| `INVALID_ARGUMENT` | 2 | no | A flag value is malformed, `--audio-stream` names a stream that does not exist, or `remux` got a file with no video; fix the call. |
| `INPUT_NOT_FOUND` | 2 | no | An input does not exist or is not a file. |
| `UNSUPPORTED_MEDIA` | 2 | no | ffmpeg-skill cannot read the input as media, or it has no audio stream. |
| `OUTPUT_EXISTS` | 2 | no | The output exists; pick another path or pass `--overwrite` deliberately. |
| `OUTPUT_IS_INPUT` | 2 | no | The output would replace an input; never allowed. |
| `FFMPEG_SKILL_UNAVAILABLE` | 3 | no | No usable ffmpeg-skill, or it lacks a tool or flag this op needs (a non-48 kHz `isolate`, mixed rates in `bleed-reduce`, a second video stream or other audio streams in `remux`). The message names the gap and any workaround. |
| `ENGINE_UNAVAILABLE` | 3 | no | Unknown engine, or not installed; pass on `details.install`. |
| `MODEL_NOT_FOUND` | 3 | no | `deep-filter` is installed but `VOICE_ENHANCE_DFN_MODEL` is not set or not a file. |
| `TOOL_ERROR` | 4 | yes | An ffmpeg-skill tool failed or timed out. |
| `ENGINE_FAILED` | 4 | no | The engine ran and failed; `details.stderr_tail` says why. |
| `ENGINE_TIMEOUT` | 4 | yes | The engine exceeded `--timeout` and was killed; retry with a longer one. |
| `INVALID_RESULT` | 4 | no | The engine's output failed validation (rate, channels, length); nothing was written. |
| `NOT_IMPLEMENTED` | 5 | no | Declared in the contract but not built yet. |
| `CANCELLED` | 6 | yes | Interrupted by a signal. |
| `INTERNAL_ERROR` | 7 | no | A bug in this skill; report it. |

## Not this skill's job

Measuring audio (media-analysis-skill, qc-skill), loudness, EQ, fixed-setting noise reduction and mixing (audio-production-skill), aligning separately recorded tracks (ffmpeg-skill `sync`), changing the container or audio codec (ffmpeg-skill `export`), transcription (transcription-skill), who spoke when (diarization-skill), re-synthesising or cloning a voice (never: the engines remove, they do not generate).
