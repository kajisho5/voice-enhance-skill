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
