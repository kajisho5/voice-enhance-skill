# Architecture Decision Records

## ADR-001 Engines: DeepFilterNet3 for ISOLATE, nara_wpe for DEREVERB, no AI engine for BLEED_REDUCE

Checked 2026-10-31 against the repositories, model hosts and package indexes listed under Sources. "Code" and "weights" are judged separately: a permissive code licence says nothing about the checkpoint a tool downloads.

### Requirements

The skill cleans speech in field recordings (talks, interviews, panels, hybrid sessions; much of it Japanese), so the engine has to meet these:

- **Licence.** Code *and* weights allow commercial use of the output. Attribution is acceptable. Non-commercial (NC) licences, and weights with no stated licence or with research-only training data, are rejected.
- **Local and offline.** It runs with no network at run time. Weights are a file the user installs once, never fetched by the skill.
- **CPU.** It runs on a laptop CPU at or near real time. A GPU is optional, never required.
- **Faithful, not generative.** It removes noise / reverb / bleed from the speech that is there. It does not re-synthesise the voice, so it cannot change timbre or invent phonemes. This matters most for Japanese, because none of the candidates documents an evaluation on Japanese speech (see below).
- **Light dependencies.** No PyTorch in the default path. The skill itself stays stdlib-only, like its siblings; engines are optional extras behind availability checks.

### Candidates

| Candidate | Code licence | Weights licence | Offline / CPU | What it actually does | Verdict |
|---|---|---|---|---|---|
| **DeepFilterNet3** (Rikorose/DeepFilterNet, release v0.5.6) | MIT OR Apache-2.0 | The checkpoints (`models/DeepFilterNet3_onnx.tar.gz`, ...) are files in the same repository, whose README says "All code in this repository is dual-licensed" MIT / Apache-2.0. No separate weights licence exists; downstream packagers (Arch Linux `deepfilternet`) ship them as `Apache-2.0 OR MIT` | Yes. Pre-compiled `deep-filter` binary "with no python dependencies" (Rust + tract, no PyTorch); real-time design (the DeepFilterNet2 paper is "Real-Time Speech Enhancement on Embedded Devices") | Mask / deep-filtering noise suppression, full band 48 kHz. The binary accepts 48 kHz WAV only. Not a dereverberation tool | **Chosen for ISOLATE** |
| **nara_wpe** (fgnt/nara_wpe, PyPI 0.0.11, 2024-09-10) | MIT | None needed: signal processing, no weights | Yes. NumPy (+ soundfile, bottleneck, tqdm, click); offline iterative, block-online and frame-online variants | Weighted Prediction Error (WPE) dereverberation, single- and multi-channel | **Chosen for DEREVERB** |
| Demucs / HTDemucs (facebookresearch/demucs) | MIT | No separate statement. `htdemucs` was trained on MUSDB-HQ "plus an extra dataset of 800 songs" that is not described; `mdx_extra` also used the MUSDB test set | CPU ~1.5x track duration; PyTorch | **Music** source separation (vocals vs drums / bass / other). Not trained to separate speech from noise. Repository archived, "not maintained anymore" | Rejected: wrong task, undocumented training data behind the weights, unmaintained |
| Resemble-Enhance (resemble-ai/resemble-enhance) | MIT | MIT (Hugging Face `ResembleAI/resemble-enhance`); training data described only as "high-quality 44.1kHz speech data" | PyTorch; no CPU guidance | Denoiser + **generative** enhancer ("restoring distortions and extending audio bandwidth") | Not now. Licence is fine, but the enhancer re-generates content, and the default path would need PyTorch. Its denoiser-only mode is a possible second ISOLATE engine later, once measured on Japanese speech |
| VoiceFixer (haoheliu/voicefixer, PyPI 0.1.3, 2023-11-12) | MIT | CC-BY-4.0 (Zenodo record 5600188, "VoiceFixer Model CheckPoint", 1.6 GB) | CPU possible; PyTorch, librosa, streamlit; checkpoints downloaded automatically on first run | **Generative** restoration (noise, reverb, low resolution, clipping) re-synthesised through a 44.1 kHz neural vocoder | Rejected: re-synthesis can change the speaker's voice; auto-download on first run conflicts with "never fetched by the skill"; heavy; unmaintained since 2023 |

**Japanese speech.** None of the five projects publishes an evaluation on Japanese. This is why "faithful, not generative" is a requirement: WPE is language-independent by construction (linear prediction of the late reverberation), and DeepFilterNet only attenuates time-frequency bins of the existing signal, so neither can replace a phoneme with another. Steps 3 and 4 measure the result on a Japanese fixture (SNR / reverberation measures plus a transcription-skill round trip) before either op leaves `EXPERIMENTAL`.

### Decision

- **ISOLATE → `deepfilternet`.** The engine is the standalone `deep-filter` binary from the DeepFilterNet release, run with the DeepFilterNet3 ONNX model tarball (`DeepFilterNet3_onnx.tar.gz`) passed explicitly with `-m`, so the binary's built-in default (DeepFilterNet2) is never used silently. Located by `VOICE_ENHANCE_DEEP_FILTER` (else `deep-filter` on `PATH`) and `VOICE_ENHANCE_DFN_MODEL`. No PyTorch.
- **DEREVERB → `nara_wpe`.** The engine is the `nara_wpe` package (`pip install 'voice-enhance-skill[wpe]'`). Offline (iterative) WPE on the whole file by default; it has no weights.
- **BLEED_REDUCE → no AI engine.** None of the five candidates addresses cross-talk between microphones (Demucs separates instruments, the others enhance one channel). Step 5 decides the method in its own ADR; the preferred direction is a deterministic, weight-free reference-based suppression on time-aligned tracks (NumPy), with multi-channel WPE as a second option. Until then the op is declared but has no engine.
- **Engines are optional.** Neither is a hard dependency of the package. `doctor` reports each engine `OK`, `MISSING` or `INCOMPATIBLE`, with how to install it, and never downloads anything. The engine runs in a worker subprocess in its own process group with a timeout (transcription-skill ADR-003, diarization-skill ADR-001), so a hung engine is killed for real.
- **Media work goes through ffmpeg-skill.** Probing before and after, conversion to the engine's input format (48 kHz mono PCM for DeepFilterNet) and putting the cleaned stem back (`audio --replace`, step 6) are ffmpeg-skill tool calls. If ffmpeg-skill lacks a flag the skill needs, that is a gap reported to ffmpeg-skill, never a private ffmpeg call (audio-production-skill ADR-1).
- **Attribution and provenance.** Every result names the engine id, engine version, model file name and SHA-256, and the code and weights licences.
- **Tests.** CI downloads no weights and installs no engine. Unit tests use fake engines and a fake `deep-filter` executable. Real-engine tests are skipped unless the engine is installed.

### ecosystem.json

`ecosystem.json` in AI-video-production-OS lists `"engines": [{"name": "未選定", ...}]` for this repository. The value this ADR selects (to be applied in AI-video-production-OS, which this repository does not edit):

```json
"engines": [
  {"name": "DeepFilterNet3 (deep-filter)", "note": "声の分離 / MIT・Apache-2.0(コード・重み) / CPU実時間 / PyTorch不要"},
  {"name": "nara_wpe", "note": "残響除去 / MIT / 重みなし・信号処理 / CPU"}
]
```

### Follow-ups

- **Second ISOLATE engine.** Resemble-Enhance's denoiser-only mode, if it measures better than DeepFilterNet3 on the Japanese fixture without the enhancer.
- **BLEED_REDUCE method.** Its own ADR in step 5.

### Sources

- [Rikorose/DeepFilterNet (README, licence, `deep-filter` usage)](https://github.com/Rikorose/DeepFilterNet), [models/](https://github.com/Rikorose/DeepFilterNet/tree/main/models), [releases (v0.5.6)](https://github.com/Rikorose/DeepFilterNet/releases), [Arch Linux deepfilternet package (licence `Apache-2.0 OR MIT`)](https://archlinux.org/packages/extra/x86_64/deepfilternet/)
- [fgnt/nara_wpe](https://github.com/fgnt/nara_wpe), [PyPI nara-wpe](https://pypi.org/project/nara-wpe/)
- [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance), [ResembleAI/resemble-enhance (model card, licence MIT)](https://huggingface.co/ResembleAI/resemble-enhance)
- [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer), [PyPI voicefixer](https://pypi.org/project/voicefixer/), [Zenodo 5600188: VoiceFixer Model CheckPoint (CC-BY-4.0)](https://zenodo.org/record/5600188)

## ADR-002 Package skeleton: contract, doctor, engine registry and worker before any op

### Context

Roadmap step 2 asks for the package with `contract` and `doctor`, modelled on transcription-skill (engine registry, worker subprocess, `--offline`), so that the OS static checks pass and a missing engine is reported as `MISSING`, before any op exists.

### Decision

- **Stdlib-only core, Python >= 3.9.** `pip install voice-enhance-skill` installs no third-party package; engines are extras (`[wpe]`) or external binaries (`deep-filter`).
- **CLI.** `voice-enhance contract --json` (alias `skill --json`), `voice-enhance doctor --json [--offline] [--ffmpeg-skill DIR]`, `voice-enhance engines --json [--offline]`. With `--json`, stdout holds exactly one JSON document, on success and failure alike.
- **Contract in the OS registry shape** (`skill_id`, `version`, `contract_version`, `provides`, `dependencies` with `version_range`, `not_provided`), plus `tools`, `engines`, `ffmpeg_skill` and `errors`. The three ops are published as `provides` with lifecycle `PROPOSED` and their tools as `implemented: false`, because nothing runs them yet. Each op moves to `EXPERIMENTAL` in the PR that implements it.
- **Typed errors.** One `VoiceEnhanceError` with a code from a fixed table that decides the exit code and the `retryable` flag (audio-production-skill's error model). Retryable: `TOOL_ERROR`, `ENGINE_TIMEOUT`, `CANCELLED`.
- **Engine registry and worker.** Engines are classes registered in `engines/__init__.py`; only registered ids can run. `python -m voice_enhance_skill.engines.worker <request.json> <result.json>` runs one engine in a child process in its own process group, with a fixed argv, a minimal environment and the Hugging Face libraries forced offline; errors come back as structured JSON in the result file. `service.run_process_group` kills the whole group on timeout.
- **`--offline`.** Both selected engines need no network at any step, so `--offline` changes nothing today; it is accepted and reported so the agent can pass it unconditionally, and an engine that ever needs the network will be reported `MISSING` under it.
- **ffmpeg-skill by capability detection.** The adapter locates an ffmpeg-skill checkout (`--ffmpeg-skill`, `VOICE_ENHANCE_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill`, `../ffmpeg-skill`), reads its contract (`scripts/_contract.py --json --static`) and checks every tool and flag this skill will emit (`probe`, `audio --replace --audio-stream --overwrite`, `join --sample-rate --channels`), never the version number (audio-production-skill ADR-15). `doctor` reports what is missing.
- **Doctor status.** `ok` when ffmpeg-skill is usable and every engine is `OK`; `degraded` when an engine is `MISSING` / `INCOMPATIBLE` (exit 0: the skill itself works); `fail` when Python or ffmpeg-skill is not usable (exit 1).

## ADR-003 ISOLATE op: input prepared with ffmpeg-skill `audio`, deep-filter inside the worker's process group, stem validated before it is placed

### Context

Roadmap step 3: the voice stem through the engine worker, done when SNR rises on a synthetic voice+noise fixture. ADR-001 fixed the engine (`deep-filter` with the DeepFilterNet3 ONNX tarball, 48 kHz WAV only) and the rule that every media step is an ffmpeg-skill tool. ADR-002 planned the 48 kHz mono conversion with ffmpeg-skill `join --sample-rate --channels`. Checked against ffmpeg-skill 2.22.4 (main, 2026-10-31), that plan does not work: `join` refuses a single clip ("give at least two clips").

### Decision

- **Command.** `voice-enhance isolate <media> -o <out.wav> [--audio-stream N] [--engine deepfilternet] [--atten-lim-db N] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`. Result schema `voice-enhance/isolate@1`: `output` (path, 48 kHz, mono, duration), `input` (duration, has_video, the chosen stream), `engine` (id, version, model, model SHA-256), `parameters`, `steps` (the ffmpeg-skill calls made), `provenance`, `warnings`. Lifecycle `voice.isolate` → `EXPERIMENTAL`; it stays there until it is measured with the real engine on the Japanese fixture (ADR-001).
- **Order of checks, cheapest first.** Paths (input is a file; output is a `.wav`, not the input, not existing unless `--overwrite`, in an existing directory), then the engine (`OK`, else `ENGINE_UNAVAILABLE` / `MODEL_NOT_FOUND` with the install line), then ffmpeg-skill (usable), then `probe` (the stream exists), then the sample-rate route (below). Nothing is extracted and no engine starts until all of them pass.
- **Preparation through `ffmpeg-skill/audio`, not `join`.** `audio <media> -o extracted.wav --audio-stream N` writes the chosen stream as PCM WAV with the picture dropped. `join` leaves `TOOLS_USED`; `audio` gains `mono` in `FLAGS_USED`.
- **Mono in a second pass.** ffmpeg-skill `audio --mono` reads the channel count of the file's *first* audio stream, so `--audio-stream 1 --mono` on a file whose stream 0 is mono and stream 1 stereo writes a stereo WAV (observed on 2.22.4). The fold-down is therefore a second `audio --mono` call on the extracted one-stream WAV, where first stream and chosen stream are the same. It runs only when the stream has more than one channel, and the result says so in `warnings`. To be reported to ffmpeg-skill as a bug.
- **Sample rate: capability-detected, never improvised.** No ffmpeg-skill tool converts the sample rate of a single file: `join` needs two clips and `loudness` also changes the level. A stream already at 48 kHz (the norm for video) needs no conversion. For any other rate, `audio --sample-rate` is used when the located checkout declares it (`OPTIONAL_FLAGS`, reported by `doctor` under `optional_flags`). Otherwise the op stops with `FFMPEG_SKILL_UNAVAILABLE` before any work, and the message says to convert the input to 48 kHz first. A private ffmpeg call or a Python resampler would break ADR-001's rule, so neither is used. `audio --sample-rate` is a gap to report to ffmpeg-skill.
- **Engine argv.** `deep-filter -D -m <DeepFilterNet3_onnx.tar.gz> -o <out_dir> [--atten-lim-db N] <input.wav>`, a fixed list. `-D` compensates the STFT and lookahead delay, so the stem stays sample-aligned with the source (step 6 puts it back under the picture). `--atten-lim-db` (0 < N ≤ 100, default: no limit) keeps some of the background when a fully dry voice sounds wrong. The flags follow the v0.5.x `deep-filter` CLI. CI cannot check them because it installs no engine. The real-engine test checks them wherever the binary is installed.
- **The binary dies with the worker.** `procs.run_process_group(new_group=False)` keeps the binary in the worker's process group. The service's `--timeout` kills that whole group, binary included. Before this change every child got a session of its own, so a binary started by the worker would have outlived it. A test checks that the binary is gone after an `ENGINE_TIMEOUT`.
- **Validation and placement.** The stem is probed through ffmpeg-skill. It must be 48 kHz, mono, and as long as the prepared input (± max(0.1 s, 1 %)). Anything else is `INVALID_RESULT`, and nothing is written. Work happens in a private `.voice-enhance-*` directory inside the output's own directory, so the stem is moved into place with one rename: the output appears whole or not at all. The run directory is always removed. The output is the engine's WAV as written, never re-encoded. The input is never modified.
- **Provenance** adds `engine.model_sha256`, as ADR-001 requires ("model file name and SHA-256"). The engine's argv is recorded with placeholders instead of local paths.
- **Tests: "SNR rises on a synthetic voice+noise fixture".** The fixture (`tests/helpers.py`) is a voiced signal (140 Hz fundamental, harmonics up to 840 Hz, 4 Hz syllable envelope) with white noise at 0 dB SNR, written as WAV; the clean voice is kept as the reference. CI has no engine, so the fake `deep-filter` applies a zero-phase 2 kHz low-pass. That shows the op carries an engine's improvement end to end (more than +6 dB). The unit tests run it through a stub ffmpeg-skill. The integration tests run it through the real ffmpeg-skill on stream 1 of a two-stream video, where stream 0 is a decoy tone. The same SNR assertion runs against the real `deep-filter` when it is installed. CI skips it (ADR-001).

### Follow-ups

- ffmpeg-skill: `audio --sample-rate` (then non-48 kHz inputs work with no change here), and `--mono` reading the chosen stream's channel count.
- Real-engine measurement on the Japanese fixture (SNR and a transcription-skill round trip) before `voice.isolate` leaves `EXPERIMENTAL`.
