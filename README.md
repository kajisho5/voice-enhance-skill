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

**Status: 0.1.0, unreleased.** `isolate` ([ADR-003](docs/decisions.md)), `dereverb` ([ADR-004](docs/decisions.md)) and `bleed-reduce` ([ADR-005](docs/decisions.md)) work (`EXPERIMENTAL`). `remux` puts a cleaned stem back into its video ([ADR-006](docs/decisions.md)). `contract`, `doctor` and `engines` work. Next on the [roadmap](ECOSYSTEM_ROADMAP.md): step 7, SKILL.md and the CI matrix.

| Op | Capability | Engine ([ADR-001](docs/decisions.md)) | Licence (code / weights) | Step |
|---|---|---|---|---|
| ISOLATE | `voice.isolate` | DeepFilterNet3 via the standalone `deep-filter` binary (no PyTorch) | MIT OR Apache-2.0 / same | 3 (done) |
| DEREVERB | `voice.dereverb` | nara_wpe (WPE, signal processing) | MIT / no weights | 4 (done) |
| BLEED_REDUCE | `voice.bleed_reduce` | `crosstalk`: this repository's linear cross-talk demixing on NumPy, no AI model ([ADR-005](docs/decisions.md)) | MIT / no weights | 5 (done) |

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
voice-enhance bleed-reduce panel.mkv -o clean/ --json          # every audio stream is one mic; one float WAV per mic
voice-enhance bleed-reduce lav_a.wav lav_b.wav -o clean/ --dominance-db 6 --json
voice-enhance remux talk.mp4 talk_voice.wav -o talk_clean.mp4 --json   # the stem back under the picture, video stream copied
voice-enhance remux panel.mkv clean/panel_a0.wav -o panel_a0.mkv --drop-other-audio --json
```

## isolate

`isolate` picks one audio stream (`--audio-stream`, 0-based, default 0) and has ffmpeg-skill extract it as PCM WAV. A stream with more than one channel is folded down to mono in a second ffmpeg-skill pass. The engine runs on that WAV in the worker, and the stem is checked through ffmpeg-skill/probe (48 kHz, mono, same duration) before it is renamed into place. The output is always a 48 kHz mono `.wav`, sample-aligned with the input (`deep-filter -D`). The input is never modified, and an existing output is never replaced without `--overwrite`. `--atten-lim-db N` limits how far the background is pushed down (default: no limit). `--timeout` (default 3600 s) kills the engine and the worker's whole process group.

The engine needs 48 kHz. A stream at another rate is converted only when the ffmpeg-skill checkout declares `audio --sample-rate`. Without that flag, `isolate` stops with `FFMPEG_SKILL_UNAVAILABLE` before doing any work, and the input has to be converted to 48 kHz first.

## dereverb

`dereverb` removes late reverberation from one audio stream (`--audio-stream`, default 0) with WPE (nara_wpe). WPE works at any sample rate and on any number of channels, so ffmpeg-skill extracts the stream as PCM WAV in a single `audio` pass. There is no fold-down and no resampling, so a 44.1 kHz stereo stream comes back as 44.1 kHz stereo and every channel helps predict the others. The engine runs a 32 ms / 8 ms STFT at the file's own rate and WPE on 30 s blocks, so memory does not grow with the length of the file. The result is a 32-bit float WAV with the input's rate, channels and length, sample-aligned with it, so nothing is clipped. It is checked through ffmpeg-skill/probe before it is renamed into place. `--taps` (default 10), `--delay` (3) and `--iterations` (3) are nara_wpe's parameters. `--delay` frames of direct sound and early reflections are always kept. The input is never modified, and `--overwrite` and `--timeout` work as for `isolate`.

## bleed-reduce

`bleed-reduce` takes 2 to 8 time-aligned microphone tracks and removes from each one what it picked up of the other talkers. A track is one audio stream. With one input, its audio streams are the tracks (all of them, or `--audio-streams 0,2`). With several inputs, each input is one track (stream 0, or one `--audio-streams` entry per input). Each track is prepared through ffmpeg-skill like `isolate`'s input: one `audio --audio-stream` pass, and a second `--mono` pass if it has more than one channel. All tracks must share the first track's sample rate. A different rate needs ffmpeg-skill `audio --sample-rate`, and without it the op stops with `FFMPEG_SKILL_UNAVAILABLE` before doing any work.

The `crosstalk` engine is signal processing on NumPy, with no model and no weights. It measures, per frequency, how strongly each microphone hears each other talker. It measures this only where that talker is louder than the microphone's own by `--dominance-db` (default 3). It then undoes that mixing with one linear demix per frequency. Nothing is gated or generated, and a track's own voice is left as it was. The measured couplings are in `parameters.coupling_db`. A pair that could not be measured (a talker who never speaks alone) is `null` and gets a warning; nothing of it is removed.

Each track is written to `<out_dir>/<input name>_a<stream>.wav` as mono 32-bit float, with its own length and sample-aligned with its input. Every stem is checked through ffmpeg-skill/probe before any is placed. A new output directory appears whole or not at all. In an existing one no file is replaced without `--overwrite`. The op does not align tracks: separately recorded tracks have to be lined up first (ffmpeg-skill `sync`), and a length difference gives a warning. Two microphones on the two channels of one stream cannot be separated, because ffmpeg-skill has no channel-select flag yet, so such a stream is folded down to one track (with a warning). `--timeout` works as for `isolate`.

## remux

`remux` puts a stem (from `isolate`, `dereverb` or `bleed-reduce`, or any audio file) back into its video with one ffmpeg-skill `audio --replace` call. The video stream is copied packet for packet, not re-encoded. The stem becomes the output's only audio stream, encoded by the container (AAC for `.mp4`, `.mov` and `.mkv`), and is padded with silence or trimmed to the video's length. The output keeps the input's container (same extension). A container change is ffmpeg-skill `export`'s job.

`--replace` keeps only the first video stream and none of the input's audio streams, so `remux` refuses, before anything is written, a file with more than one video stream (`FFMPEG_SKILL_UNAVAILABLE`). A file with more than one audio stream is refused too, unless `--drop-other-audio` accepts an output whose only audio is the stem. The output is checked through ffmpeg-skill/probe: one video and one audio stream, and the video's codec, profile, size, pixel format, frame rate, frame count and length unchanged. Only then is it renamed into place. A stem whose length differs from the video's, or an input whose audio starts away from its video, gives a warning. The inputs are never modified, and `--overwrite` and `--timeout` (here the ffmpeg-skill time limit) work as for `isolate`. No engine is involved.

## Engines (optional)

- **DeepFilterNet3** (ISOLATE): download `deep-filter` for your platform and `DeepFilterNet3_onnx.tar.gz` from the [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases) (v0.5.x). Put `deep-filter` on `PATH` (or set `VOICE_ENHANCE_DEEP_FILTER`) and set `VOICE_ENHANCE_DFN_MODEL` to the model tarball.
- **nara_wpe** (DEREVERB): `pip install -e '.[wpe]'`.
- **crosstalk** (BLEED_REDUCE): part of this package; it needs NumPy, `pip install -e '.[bleed]'`.

`doctor` reports each engine `OK`, `MISSING` or `INCOMPATIBLE` (version outside the supported range). Engines run in a worker subprocess in their own process group, killed on `--timeout`.

## ffmpeg-skill

Located from `--ffmpeg-skill`, `VOICE_ENHANCE_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill` or `../ffmpeg-skill`. Compatibility is checked tool by tool and flag by flag against its contract (`probe`, `audio --audio-stream --mono --replace`, plus the optional `audio --sample-rate`), not by version number.

## JSON and errors

With `--json`, stdout holds exactly one JSON document. Failures are `{"ok": false, "error": {"code", "message", "retryable", "details"}}`; the code fixes the exit code (`contract --json` lists them under `errors`). `TOOL_ERROR`, `ENGINE_TIMEOUT` and `CANCELLED` are retryable.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests need no engine, no weights and no FFmpeg (fake `deep-filter`, a fake `nara_wpe` package, mocked package lookups, a stdlib stub of ffmpeg-skill for every op). The ISOLATE SNR check runs on a synthetic voice+noise fixture. The DEREVERB check (signal-to-reverberation ratio) runs on a synthetic reverberant fixture and needs NumPy, because the fake nara_wpe computes on it. The BLEED_REDUCE check (cross-talk energy) runs on a synthetic two-microphone fixture and needs NumPy, because the crosstalk engine computes on it. Both are skipped without NumPy. The REMUX checks use a stub whose videos carry a picture payload, and the integration tests check that the video packets are byte-identical through the real ffmpeg-skill. The real-engine versions run only where the engine is installed. The integration tests run against a real ffmpeg-skill checkout (`VOICE_ENHANCE_FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg; the OS registry conformance checks run when `AI_VIDEO_PRODUCTION_OS_DIR` (or `../AI-video-production-OS`) is present.

## Licence

MIT. Engine attribution is listed by `voice-enhance engines --json` and goes into the provenance of every op result.
