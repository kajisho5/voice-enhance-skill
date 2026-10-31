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
- **Tests.** Unit tests use fake engines and a fake `deep-filter` executable. Real-engine tests are skipped unless the engine is installed. (Superseded in part by ADR-007: one Linux CI job installs the real engines, pinned by SHA-256, and runs those tests.)

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
- **Engine argv.** `deep-filter -D -m <DeepFilterNet3_onnx.tar.gz> -o <out_dir> [--atten-lim-db N] <input.wav>`, a fixed list. `-D` compensates the STFT and lookahead delay, so the stem stays sample-aligned with the source (step 6 puts it back under the picture). `--atten-lim-db` (0 < N ≤ 100, default: no limit) keeps some of the background when a fully dry voice sounds wrong. The flags follow the v0.5.x `deep-filter` CLI. Since ADR-007, CI's engines job runs the real 0.5.6 binary and confirms them (`-m/--model`, `-D/--compensate-delay`, `-a/--atten-lim-db`, `-o/--output-dir`).
- **The binary dies with the worker.** `procs.run_process_group(new_group=False)` keeps the binary in the worker's process group. The service's `--timeout` kills that whole group, binary included. Before this change every child got a session of its own, so a binary started by the worker would have outlived it. A test checks that the binary is gone after an `ENGINE_TIMEOUT`.
- **Validation and placement.** The stem is probed through ffmpeg-skill. It must be 48 kHz, mono, and as long as the prepared input (± max(0.1 s, 1 %)). Anything else is `INVALID_RESULT`, and nothing is written. Work happens in a private `.voice-enhance-*` directory inside the output's own directory, so the stem is moved into place with one rename: the output appears whole or not at all. The run directory is always removed. The output is the engine's WAV as written, never re-encoded. The input is never modified.
- **Provenance** adds `engine.model_sha256`, as ADR-001 requires ("model file name and SHA-256"). The engine's argv is recorded with placeholders instead of local paths.
- **Tests: "SNR rises on a synthetic voice+noise fixture".** The fixture (`tests/helpers.py`) is a voiced signal (140 Hz fundamental, harmonics up to 840 Hz, 4 Hz syllable envelope) with white noise at 0 dB SNR, written as WAV; the clean voice is kept as the reference. CI has no engine, so the fake `deep-filter` applies a zero-phase 2 kHz low-pass. That shows the op carries an engine's improvement end to end (more than +6 dB). The unit tests run it through a stub ffmpeg-skill. The integration tests run it through the real ffmpeg-skill on stream 1 of a two-stream video, where stream 0 is a decoy tone. The same SNR assertion runs against the real `deep-filter` when it is installed: in CI's engines job since ADR-007.

### Follow-ups

- ffmpeg-skill: `audio --sample-rate` (then non-48 kHz inputs work with no change here), and `--mono` reading the chosen stream's channel count.
- Real-engine measurement on the Japanese fixture (SNR and a transcription-skill round trip) before `voice.isolate` leaves `EXPERIMENTAL`.

## ADR-004 DEREVERB op: WPE at the stream's own rate and layout, STFT and blocks owned by the engine, one ffmpeg-skill pass

### Context

Roadmap step 4: DEREVERB, "done when: same pattern" as step 3, so a reverberation measure has to rise on a synthetic reverberant fixture, through the engine worker and ffmpeg-skill. ADR-001 fixed the engine: nara_wpe, offline iterative WPE on the whole file by default, no weights. Checked against nara_wpe 0.0.11 (source on GitHub, 2026-10-31) and ffmpeg-skill 2.22.4 (main, 2026-10-31).

### Decision

- **Command.** `voice-enhance dereverb <media> -o <out.wav> [--audio-stream N] [--engine nara_wpe] [--taps N] [--delay N] [--iterations N] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`. Result schema `voice-enhance/dereverb@1`, with the same keys as `isolate@1`. `engine.model` is `null`, because WPE has no weights. Lifecycle `voice.dereverb` → `EXPERIMENTAL` until it is measured with the real nara_wpe on the Japanese fixture (ADR-001).
- **The stream's own rate and channels, so one ffmpeg-skill pass.** WPE is linear prediction per STFT bin. It has no rate of its own, and with more than one channel it uses all of them to predict each other (multi-channel WPE). The input is therefore prepared with one `ffmpeg-skill/audio <media> -o input.wav --audio-stream N` call and nothing else: no fold-down, so the `--mono` bug in ADR-003 does not apply, and no resampling, so `audio --sample-rate` is not needed. A 44.1 kHz stereo stream gives a 44.1 kHz stereo result. Nothing needed is missing from ffmpeg-skill. Folding down first would throw away the spatial information multi-channel WPE uses. (ADR-007: channels that are linear copies of each other are first reduced to their independent components.)
- **The engine owns the STFT, not `nara_wpe.utils`.** `nara_wpe.utils.stft` imports SciPy, which nara_wpe does not declare as a dependency. The engine does its own STFT on NumPy: a periodic Hann window of 32 ms and a hop of 8 ms at the file's own rate (512 / 128 at 16 kHz, as in nara_wpe's examples; 1536 / 384 at 48 kHz). The signal is padded so that every sample lies under all four overlapping frames. The inverse is a weighted overlap-add. With no filtering the input comes back sample for sample (a test checks this to 1e-6 across block borders and per channel). WPE subtracts a prediction from the current frame and adds no delay, so the output is sample-aligned with the source for step 6.
- **WPE call.** `nara_wpe.wpe.wpe_v8(Y, taps, delay, iterations, statistics_mode="full")` on `Y` of shape (frequency, channels, frames). `wpe_v8` loops over frequencies, so memory follows one bin, not the whole tensor. Defaults are nara_wpe's own (taps 10, delay 3, iterations 3). They are flags, checked by the op and again by the engine: taps 1-50, delay 1-20, iterations 1-20. `delay` frames of direct sound and early reflections are never predicted, so they are kept. Taps 10 and delay 3 at an 8 ms hop predict the reverberation from 24 ms to about 130 ms back.
- **Blocks of 30 s.** "The whole file" (ADR-001) would hold a one-hour talk's STFT in memory at once: about 5.5 GB for mono 48 kHz in complex128. The engine instead runs WPE on blocks of 30 s of frames. Each block gets the `taps + delay - 1` frames before it as context. The file is read from disk block by block, and the result is written as it goes. 30 s is far more data than the per-bin filter needs, so the statistics stay offline in character. Memory no longer grows with the file's length. The filter can change slightly at a block border, and the overlap-add smooths the change. This is not a flag.
- **Output: 32-bit float WAV, same rate, channels and length.** WPE output can exceed the input's peak. Writing it as float means nothing is clipped or requantised. Step 6 encodes it when it goes back under the picture. The engine reads PCM 16/24/32-bit and float WAV, plain or `WAVE_FORMAT_EXTENSIBLE`, with its own RIFF reader, because the stdlib `wave` module reads neither float nor 24-bit extensible on Python 3.9.
- **Same frame as ISOLATE.** Checks run cheapest first: numbers, paths, the engine (`ENGINE_UNAVAILABLE` with the `[wpe]` install line), ffmpeg-skill, `probe` (the stream exists). Then the work happens in a private run directory. The result is validated through `ffmpeg-skill/probe` against the prepared input's own rate, channel count and duration (`INVALID_RESULT` otherwise). It is placed with one rename. The input is never modified. `ops.validate_output` takes the expected channel count instead of assuming mono. `ops.extract_stream` is the shared first pass of both ops. nara_wpe runs inside the worker process, so the service's `--timeout` kills it with the worker's process group. No binary is involved.
- **Tests: "a reverberation measure rises on a synthetic reverberant fixture".** The fixture (`tests/helpers.py`) is a speech-like dry signal: syllables of 80-250 ms, each with its own gliding fundamental, harmonics, some breath noise, and gaps between them. It goes through two feedback comb filters in series (37 ms / 0.6 and 59 ms / 0.5), and the dry signal is kept as the reference. The measure is the signal-to-reverberation ratio (SRR) against the dry signal. The dry signal has to be non-stationary: WPE treats a steady tone as predictable, that is as reverberation, so ADR-003's steady 140 Hz voice would be removed along with the room. CI installs no engine (ADR-001), so the tests put a fake `nara_wpe` package on `sys.path` and on the worker's `PYTHONPATH`. It has a dist-info for the version check and a `wpe_v8` that is a minimal WPE on NumPy (delayed-frame stack, power-weighted least squares, subtract the prediction). It records every call (shape, taps, delay, iterations, statistics mode). It has `identity`, `fail` and `badshape` modes for the round-trip and error tests. Through the stub ffmpeg-skill (unit) and through the real ffmpeg-skill on the stereo stream 1 of a two-stream video (integration), SRR rises by more than 6 dB. A 44.1 kHz input works through the real ffmpeg-skill with no `audio --sample-rate`. The same SRR check runs against the real nara_wpe where it is installed: in CI's engines job since ADR-007, which also changed how channels that copy each other are handled.
- **NumPy in the integration job.** The fake computes on NumPy. The integration job, where nothing may skip, installs `numpy` (not nara_wpe). The unit matrix stays stdlib-only and skips the NumPy tests with "needs numpy". That proves the package still imports and `doctor` still runs with no NumPy at all.

### Follow-ups

- ffmpeg-skill: `audio` writes every `.wav` as `pcm_s16le`, so a 24-bit source reaches WPE as 16-bit. This is not blocking: WPE does not need more, and the output is float. Report to ffmpeg-skill as a wish for a bit-depth or codec choice on `.wav` output.
- Real-engine measurement on the Japanese fixture (SRR and a transcription-skill round trip) before `voice.dereverb` leaves `EXPERIMENTAL`. That includes whether 30 s blocks and the 32 ms / 8 ms STFT suit real rooms, and whether the defaults should differ at 48 kHz.

## ADR-005 BLEED_REDUCE op: linear cross-talk demixing of time-aligned microphone tracks, this repository's own signal processing

### Context

Roadmap step 5: BLEED_REDUCE on multi-mic input, done when the cross-talk energy falls on a 2-track fixture. ADR-001 found no AI engine for the job (none of the five candidates addresses cross-talk between microphones). It named the preferred direction: a deterministic, weight-free, reference-based suppression on time-aligned tracks (NumPy), with multi-channel WPE as a second option. Checked against ffmpeg-skill 2.22.5 (main, 2026-10-31).

### Decision

- **Command.** `voice-enhance bleed-reduce <media>... -o <out_dir> [--audio-streams 0,1] [--engine crosstalk] [--dominance-db N] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`. Result schema `voice-enhance/bleed-reduce@1`: `output_dir`, `outputs` and `inputs` (one entry per track, with `track`), `engine`, `parameters` (including the measured `coupling_db` matrix), `steps` (each with its `track`), `provenance`, `warnings`. Lifecycle `voice.bleed_reduce` → `EXPERIMENTAL` until it is measured on a real multi-mic recording (a Japanese panel or interview, ADR-001).
- **Tracks.** A track is one audio stream, one microphone. There are 2 to 8 tracks, and they must be time-aligned: one clock, one start. With one input, its audio streams are the tracks (all of them, or the ones `--audio-streams` lists, as a recorder or a camera with several lavaliers writes them). With several inputs, each input is one track: stream 0, or one `--audio-streams` entry per input. The same stream twice, or two tracks that would get the same output name, is `INVALID_ARGUMENT`. Each track is written to `<out_dir>/<input name without extension>_a<stream>.wav`.
- **Engine: `crosstalk`, written here, not an AI model.** It is registered like the other engines, runs in the worker and reports `OK` / `MISSING` / `INCOMPATIBLE`. It is `MISSING` without NumPy (`pip install 'voice-enhance-skill[bleed]'`). Its code is MIT (this repository), it has no weights, and it downloads nothing. Multi-channel WPE was not chosen: WPE predicts each frame from frames at least `delay` frames back, so it removes late reverberation. Bleed arrives in the same frame as the direct sound (a few milliseconds later), so WPE never predicts it. Spectral gating or masks were not chosen either: they are non-linear, they add musical noise, and during overlapping speech they cut the quieter talker's own voice. That conflicts with "faithful, not generative" (ADR-001).
- **Method: dominance-gated least squares, then a per-bin linear demix.** The model is that each microphone hears its own talker plus every other talker through one complex coupling per STFT bin: `X_i = S_i + sum_j H_ij S_j`. Pass 1 estimates `H_ij(f) = sum X_i conj(X_j) / sum |X_j|^2`, summed only over the bins where track j is louder than track i by more than `--dominance-db` (default 3 dB, range 0-30). Those bins hold talker j and its bleed into i, not talker i, so the estimate is not pulled towards i's own voice. Pass 2 applies `S = (I + H)^-1 X` per bin. Unlike the first-order `X_i - sum H_ij X_j`, this does not colour a track's own talker by the round trip `H_ij H_ji`. Where `I + H` is badly conditioned (condition number above 10), that bin falls back to first order, and the result reports how many bins did (`first_order_bins`). The process is linear and time-invariant, so nothing is gated and nothing is generated.
- **STFT of 64 ms / 16 ms.** Hann, window = 4 hops, at the tracks' own rate. One complex coefficient per bin models the delay between microphones well only while that delay is short against the window. 64 ms leaves room for about 1-3 m of extra path (3-9 ms). It is twice the window of DEREVERB's STFT, which wants time resolution instead. Both passes stream in blocks of 30 s of frames, so memory does not grow with the file's length. The statistics are summed over the whole file.
- **What it does not do, and says so.**
  - It does not align tracks. A length difference beyond the duration tolerance gives a warning to align separately recorded tracks first (ffmpeg-skill `sync`).
  - The late, reverberant part of the bleed stays: one coefficient per bin captures the direct path and early reflections only. DEREVERB can run on each track afterwards.
  - A coupling louder than `--dominance-db` cannot be measured, for example when a microphone is as close to the other talker as to its own. The same goes for a talker who never speaks alone. That pair's coupling is reported as `null`, nothing of it is removed, and a warning names the pair.
  - A talker who moves changes the coupling, and one estimate per file averages over the movement.
- **Preparation through ffmpeg-skill, as for ISOLATE.** Each track is prepared with `audio --audio-stream N`. A track with more than one channel gets a second `audio --mono` pass on the extracted one-stream WAV (ADR-003's `--mono` bug), with a warning. Every track has to be at the first track's sample rate. A different rate is converted only through `audio --sample-rate` when the checkout declares it. ffmpeg-skill 2.22.5 does not, so mixed rates are refused with `FFMPEG_SKILL_UNAVAILABLE` (details `needs_flag: sample_rate` and `sample_rates`) before anything is extracted. No private ffmpeg call and no Python resampler is used (ADR-001).
- **Two microphones on the channels of one stream** (dual-mono lavalier receivers, a stereo recorder with one mic per side) cannot be separated here: ffmpeg-skill `audio` has no flag to pick one channel of a stream. Such a stream is one track and is folded down with a warning, so its two microphones are mixed. Splitting it is left to ffmpeg-skill (gap to report: `audio --channel N`, or a split into one file per channel), never improvised here.
- **Validation and placement.** Every stem is probed through ffmpeg-skill and must have the common rate, one channel and its own prepared track's duration (± max(0.1 s, 1 %)). Otherwise the op fails with `INVALID_RESULT` and no output is written, not even the stems that passed. The run directory is private: `.voice-enhance-*` inside the output directory, or next to it when it does not exist yet. If the output directory does not exist, the finished stem directory is renamed into place whole, so it appears complete or not at all. If it exists, the names are checked again and then each stem is renamed in turn. Other files in it are left alone. No output name is replaced without `--overwrite`, an output that resolves to an input is refused (`OUTPUT_IS_INPUT`), and the inputs are never modified.
- **Output.** One mono 32-bit float WAV per track at the common rate. It keeps that track's own length and is sample-aligned with its input. The demix can exceed the input's peak, and float means nothing is clipped. Step 6 encodes it when it goes back under the picture.
- **Shared code.** The WAV reader and writer, the STFT and the overlap-add move from the nara_wpe engine to `engines/wavio.py` (`stft_process`, plus `TrackReader` / `TrackWriter` for several mono files read and written as the channels of one signal). DEREVERB's `dereverb_file` now calls `stft_process`. Its sample-exact round-trip test still passes. `EngineRequest` gains `input_paths` / `output_paths` and `EngineResult` gains `output_paths`: one call, all tracks. `input_path` / `output_path` name the first pair, so the single-file engines are unchanged. Provenance for several files has `inputs` / `outputs` (one SHA-256 each) in place of `input` / `output`.
- **Tests: "lower cross-talk energy on a 2-track fixture".** The fixture (`tests/helpers.py`, `crosstalk_fixture`) uses two speech-like talkers made by `syllables()`. Talker 0 speaks in the first 60 % of the time and talker 1 in the last 60 %, with an overlap in the middle fifth. Each microphone hears its own talker plus the other at -8 dB, 4 ms later, and a -60 dB independent noise floor. Cross-talk energy is a track's energy where only the other talker speaks. A track's own speech is its energy where only its own talker speaks (50 ms kept clear of each turn change).
  - Measured: cross-talk falls by 17.4-17.6 dB at 16 kHz and by 19.1-20.4 dB at 48 kHz. Own speech changes by less than 0.1 dB. The measured couplings are -8.3 to -8.5 dB (the fixture's -8 dB). No bin falls back to first order.
  - The tests assert more than 10 dB less cross-talk, own speech within 1 dB, and the coupling within 1.5 dB of -8 dB. They run through the worker, through the stub ffmpeg-skill (two files, one file whose channels the stub serves as streams, a stereo track folded down) and through the real ffmpeg-skill (the two audio streams of a video, and the CLI on two files with a stereo track).
  - Planning tests check every refusal before media work: the stub records no call. Tests also cover mixed rates (`FFMPEG_SKILL_UNAVAILABLE`), a missing NumPy (`ENGINE_UNAVAILABLE`), a stem failing validation (nothing placed, no directory created), a new versus an existing output directory, and `--overwrite`.
  - The unit matrix installs no NumPy and skips the engine tests with "needs numpy". The integration job installs NumPy, and nothing skips there.

### Follow-ups

- ffmpeg-skill: `audio --channel N` (or one file per channel) to split microphones that share a stream, and `audio --sample-rate` (ADR-003) for tracks at different rates. Both would work here with no change to the method.
- Alignment of separately recorded tracks as an optional preparation step through ffmpeg-skill `sync`. For two sources, `sync --trim-second` already writes the second file shifted to line up with the reference. With more sources, `sync` only reports the offsets and refuses `--trim-second`, so tracks 3 and up would need that from ffmpeg-skill first. Today the op only warns, and the caller aligns.
- Per-block coupling estimates for talkers who move, and couplings over several frames for strongly reverberant bleed, if the real-recording measurement shows the need.
- Real-recording measurement (a Japanese panel or interview, cross-talk energy and a transcription-skill round trip) before `voice.bleed_reduce` leaves `EXPERIMENTAL`.
- `ecosystem.json` in AI-video-production-OS: add `{"name": "crosstalk (signal processing)", "note": "マイク被り低減 / MIT / 重みなし・NumPy / CPU"}` to the engines that ADR-001 lists (pending there, as ADR-001's value is).

## ADR-006 REMUX helper: the stem back under the picture through ffmpeg-skill `audio --replace`, refused where that tool would lose a stream

### Context

Roadmap step 6: a remux helper that puts a cleaned stem back into the video, "via ffmpeg-skill audio --replace", done when the video stream is byte-identical. ADR-001 makes every media step an ffmpeg-skill tool call and forbids a private ffmpeg call. ISOLATE, DEREVERB and BLEED_REDUCE all write stems that are sample-aligned with their input stream (ADR-003, ADR-004, ADR-005), so that they can go back under the picture. Checked against ffmpeg-skill 2.22.5 (main, 2026-10-31), `scripts/audio.py`.

What `audio <video> --replace <file> -o <out>` does in 2.22.5:

- It maps `0:v:0` with `-c:v copy`. Only the **first** video stream is kept, and its packets are copied.
- The replacement file's first audio stream becomes the output's **only** audio stream. Every audio stream of the video is dropped.
- That audio is padded with silence or trimmed to the video's container duration (`apad,atrim=0:<duration>`), so the picture is never shortened.
- The audio codec comes from the output extension (AAC 192 kb/s for `.mp4`, `.mov`, `.mkv` and anything else that is not an audio extension). `-movflags +faststart` is added for `.mp4` / `.mov` / `.m4v`.
- Subtitle and data streams are copied when the container takes them. Otherwise they are dropped, and the result says `dropped_non_av_streams: true`.
- A silent replacement file (peak at or below -50 dBFS) is mixed anyway, with a note.
- The result document carries `encoder: "copy"` and a probe of the output.

### Decision

- **Command.** `voice-enhance remux <video> <stem> -o <out> [--drop-other-audio] [--timeout S] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`. Result schema `voice-enhance/remux@1`: `output` (path, format, duration, `video` with the checked properties and `stream_copied`, `audio`), `input`, `stem`, `parameters`, `steps`, `provenance`, `warnings`. The stem can be any audio file ffmpeg-skill reads. Its first audio stream is used.
- **A helper, not a capability.** Muxing is ffmpeg-skill's job (`ecosystem.json` gives this repository the three voice jobs only). `remux` is a tool in the contract (`voice-enhance/remux`, role `assembly`, `roadmap_step` 6, no engine), not a `provides` entry. `doctor` reports it under `checks.helpers.remux`, ready when ffmpeg-skill and FFmpeg are. It runs no engine and no worker, and its provenance has `engine: null`, with `inputs` (video, stem) and `outputs`.
- **One ffmpeg-skill call, nothing private.** The op runs `audio <video> --replace <stem> -o <run dir>/out<ext> --json` through the adapter (`FfmpegSkill.replace_audio`), with absolute paths and a fixed argv. The work happens in a private `.voice-enhance-*` directory in the output's directory, as for the ops. The validated file is renamed into place, so the output appears whole or not at all. `--overwrite` is never passed to ffmpeg-skill: its output is always a fresh file. `overwrite` is therefore dropped from the `audio` flags this skill requires (ADR-002 listed it for this step).
- **Same container.** The output must have the input's extension. The video stream is copied, not re-encoded, and a container change can fail for a codec the new container cannot hold. That is ffmpeg-skill `export`'s job. An input with an audio extension is refused (`INVALID_ARGUMENT`), because ffmpeg-skill drops the picture for those.
- **Refused before any media is written, where `--replace` would lose a stream.** The checks run cheapest first: numbers and paths, then ffmpeg-skill usable, then `probe` of the video and the stem.
  - **No video stream:** `INVALID_ARGUMENT`.
  - **More than one video stream:** `FFMPEG_SKILL_UNAVAILABLE` (`needs: audio --replace keeping every video stream`). `--replace` copies `0:v:0` only, and there is no flag to keep the others. A cover picture is not counted (ffmpeg-skill's `stream_counts`).
  - **More than one audio stream:** `FFMPEG_SKILL_UNAVAILABLE` (`needs: audio --replace of one audio stream, keeping the others`, `workaround: --drop-other-audio`). With `--drop-other-audio` the output has the stem as its only audio stream, and a warning says how many streams were left out. This is the BLEED_REDUCE or multi-lavalier case. Silently losing the other microphones, or a second language, would be a data loss the caller never asked for.
  - **A stem with no audio stream:** `UNSUPPORTED_MEDIA`.
- **Warnings, not refusals.** These cases still give a usable output, so the op warns:
  - The stem's length differs from the video's beyond max(0.1 s, 1 %): ffmpeg-skill pads or trims it.
  - The stem has several audio streams: only the first is used.
  - The input's first audio stream starts more than half a frame (or 20 ms) away from its video. A stem extracted from it starts at its first sample and is placed at the video's start, so it is shifted by that much.
  - Subtitle or data streams were dropped.
  - ffmpeg-skill's own notes (a silent stem).
- **Validation through `probe`, then byte identity in the tests.** ffmpeg-skill has no tool that hashes a stream's packets. So the op checks what `probe` can see. Anything else is `INVALID_RESULT`, and nothing is placed.
  - Exactly one video and one audio stream.
  - The video stream's codec, profile, width, height, pixel format and `r_frame_rate` are the input's.
  - Its frame count is the input's when both are known (MP4 / MOV report it; Matroska does not).
  - Its length is the input's (stream duration or Matroska's DURATION tag, ± max(0.1 s, 1 %)), and the file's duration is the input's.
  
  "Video stream byte-identical" is asserted by the integration tests, which hash every video packet (pts and MD5 of the data) of input and output with ffprobe. They do this on a Matroska / FFV1 file with two audio streams (ISOLATE's stem of stream 1 put back, with `--drop-other-audio`) and on an MP4 / MPEG-4 Part 2 file with AAC audio through the CLI. ffprobe is used there only to measure, like the ffmpeg calls that build the fixtures. The skill never calls either.
- **Audio encoding is ffmpeg-skill's choice.** The stem is float WAV (DEREVERB, BLEED_REDUCE) or 16-bit (ISOLATE). In an `.mp4`, `.mov` or `.mkv` it becomes AAC at 192 kb/s, at the stem's rate and channels. That is the first lossy step in the chain, and it happens once, at the end. `--bitrate` exists in ffmpeg-skill and is not exposed here yet. A caller who wants a different codec runs ffmpeg-skill `export` on the result.
- **Tests.** The unit tests use a stub whose "videos" are JSON files carrying a probe document and a picture payload. Its `replace_audio` copies the payload and puts the stem in as the only audio stream. They check:
  - Every refusal happens before any `--replace` call or file: paths, container, stream counts, a stem without audio, and an existing output.
  - The picture payload arrives unchanged and the result has the expected shape.
  - The warnings appear, including `--drop-other-audio`, a short stem, an A/V offset, dropped subtitles and a silent stem.
  - Each broken property gives `INVALID_RESULT` with nothing placed: frame count, codec, a second audio stream, duration.
  - `--overwrite` works, the CLI works, and an ISOLATE stem goes through `remux` end to end.
  
  The integration tests run the real ffmpeg-skill as described above. Nothing in them needs an engine or NumPy, so they run in the integration job, where nothing may skip.

### Follow-ups

- Gaps to report to ffmpeg-skill. Each would let the op accept more inputs with no change to its method:
  - Replacing one audio stream while copying the others (`audio --replace --audio-stream N`, or a `--keep-other-audio`). Then `--drop-other-audio` becomes optional.
  - Keeping every video stream under `--replace`.
  - A packet-hash comparison tool (for example `probe --packet-hashes v:0`). Then the op itself can assert byte identity instead of the properties `probe` reports.
  - The audio start offset reported for every stream in `audio_streams`, not only for the first. Then the A/V offset warning can name the stream the stem came from.
- Exposing ffmpeg-skill `audio --bitrate` (and, if ffmpeg-skill gains it, a codec choice such as PCM in `.mov` / `.mkv`), if callers ask for less lossy delivery than AAC 192 kb/s.
- A `--remux` convenience on `isolate` and `dereverb` that chains this step, once the agent adapter (step 8) shows whether callers want one command or two.

## ADR-007 SKILL.md and CI: an engines job runs the real deep-filter and nara_wpe on Linux; DEREVERB runs WPE on independent channel components

### Context

Roadmap step 7: SKILL.md, and CI with a unit job on three OSes and a Linux engine job, done when CI is green. The unit matrix (Linux, macOS, Windows × Python 3.9 / 3.12) and the integration job (real FFmpeg and ffmpeg-skill, fake engines) already existed. ADR-001 kept every engine out of CI, so the real-engine tests of ADR-003 and ADR-004 had never run anywhere, and the `deep-filter` flags ADR-003 relies on had been read from the documentation, not checked. Checked on 2026-10-31 against DeepFilterNet v0.5.6 (`libDF/src/bin/enhance_wav.rs` at the tag, and the release binary run in CI), nara_wpe 0.0.11 (PyPI wheel) and ffmpeg-skill main.

### Decision

- **SKILL.md** follows slide-sync-skill and diarization-skill: frontmatter `name` (the package name, `voice-enhance-skill`) and a `description` that says what it does and when to use it, then one command line per task, when to use which op, the limits to tell the user (`EXPERIMENTAL`, the 48 kHz rule of `isolate`, fold-down, remux re-encoding the audio), the engine install lines, every error code with its exit code and whether a retry can help, and what is not this skill's job. A unit test (`SkillMdTests`) holds it to the code: the frontmatter, a 30 kB budget, an example for every tool, the error table equal to `ERROR_TABLE`, the lifecycle of every capability, the 48 kHz figure and the install lines (`RELEASES`, `MODEL_URL`, `[wpe]`, `[bleed]`, the two environment variables).
- **The engines job is feasible on GitHub's Linux runners, and needs no PyTorch.** `deep-filter` 0.5.6 is a static musl binary of the release (36 MB). `DeepFilterNet3_onnx.tar.gz` (8 MB) is not a release asset, contrary to what the install hint said: it is a plain Git file of `models/` at the tag, fetched from `raw.githubusercontent.com`. Both are checked against a pinned SHA-256 in the workflow. nara_wpe comes from PyPI through `.[wpe]` (with NumPy, tqdm, soundfile, bottleneck, click; about 3 s to install). The job runs the whole suite. It fails if any test is skipped other than the one unit test that needs nara_wpe absent, and it names the six real-engine tests that must run and pass. `doctor` must report every engine `OK`. The job then measures the engines on the synthetic fixtures (`tests/measure_engines.py`) and writes the table to the job summary. macOS and Windows keep the unit job only: the engines are the same code there, and one Linux job is enough to keep the integration honest.
- **The `deep-filter` flags are as ADR-003 assumed.** `deep-filter --help` on 0.5.6 declares `-m, --model`, `-D, --compensate-delay`, `-a, --atten-lim-db` and `-o, --output-dir`, and a test checks every flag of `build_argv` against the installed binary's help. `--version` prints `deep_filter 0.5.6` (underscore, the Cargo package name); `parse_version` reads it. No code change was needed for the argv.
- **What the real engines measured** (CI, ubuntu-24.04, synthetic fixtures of ADR-003 and ADR-004):

  | Op | Fixture | Result |
  |---|---|---|
  | isolate | voice + white noise at 0 dB, 3 s, 48 kHz | SNR 0.0 → 18.2 dB; with `--atten-lim-db 12`, 11.2 dB |
  | isolate | the same at 10 dB | SNR 10.0 → 13.8 dB |
  | dereverb | speech-like + two combs, 2 s, mono, 16 / 48 kHz | SRR -1.1 → 11.9 dB / -0.8 → 19.0 dB |
  | dereverb | the same, stereo with a different room per channel | SRR -0.8 → 17.6 and 1.5 → 14.9 dB |
  | dereverb | the same, dual mono (both channels identical) | SRR -0.8 → **-48.4 dB** before the fix below, -0.8 → 19.0 dB after |

  Each run took about 1 s. The real-engine tests now require more than +6 dB, as the fake-engine tests do.
- **`-D` keeps the stem sample-aligned, and it ends 30 ms early.** The cross-correlation of the stem with the clean voice peaks at lag 0. With `-D` the binary drops the first `fft_size - hop + lookahead × hop` = 1440 samples of its output, and it does not process a last partial hop, so the stem is 1440 samples (30 ms) shorter than its input, at the end. The probe check (± max(0.1 s, 1 %)) accepts that, and `remux` pads the stem with silence to the video's length. A test asserts lag 0 and a shortfall of at most 1920 samples.
- **The 48 kHz rule is the skill's, not the binary's.** The 0.5.6 binary resamples any other rate to 48 kHz and back by itself (`enhance_wav.rs`; a 16 kHz file came back at 16 kHz in CI). `isolate` still converts through ffmpeg-skill or refuses, as ADR-003 decided: the output contract stays one format (48 kHz mono), and resampling stays a media step that ffmpeg-skill owns. Whether to let the binary resample instead is a follow-up, not part of this step.
- **DEREVERB: WPE on the independent channel components.** `nara_wpe.wpe.wpe_v8` solves its filter without regularisation. On channels that are copies of each other the correlation matrix is singular and the filter diverges: called directly on two identical channels of white noise, its output peak was 8.9 times the input's, and on the dual-mono fixture the result was 48 dB louder than the dry signal. Dual mono (one microphone recorded on both channels) is common in camera audio, so this matters. The engine now computes, per block, the broadband channel covariance of the STFT and its eigenvectors. Directions more than 60 dB (`RANK_TOLERANCE` 1e-6) below the strongest are treated as absent. When some are absent, WPE runs on the r independent components `c = Uᴴy`, and the result is `y + U(wpe(c) - c)`: linear, with whatever lies outside the components passed through unchanged. Dual mono therefore gets exactly the mono result on both channels. A channel pair 40 dB apart stays two components and measured fine (18.6 / 18.5 dB). A silent multi-channel block is passed through. The result reports `parameters.independent_channels` (the smallest count over the blocks) and a warning when it is below the channel count. The unit stub now keeps each channel as it is (it used to fold every stream to dual mono), and the multi-channel tests use a fixture with a different room per channel.

### Follow-ups

- Real-engine measurement on the Japanese fixture (SNR, SRR and a transcription-skill round trip) before any op leaves `EXPERIMENTAL` (ADR-001, ADR-003, ADR-004). CI now has the engines to run it once a fixture with a licence that allows it is chosen.
- Letting `deep-filter` resample non-48 kHz input itself, instead of waiting for ffmpeg-skill `audio --sample-rate`.
- A per-frequency rank check, if real recordings show channels that are dependent in one band only (mono bass under a stereo pair).
- Engines on macOS and Windows in CI, if a platform-specific failure ever shows up; the release has binaries for both.
