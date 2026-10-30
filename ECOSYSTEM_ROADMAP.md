# Roadmap — voice-enhance-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **音 (Audio)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold: one-line README, no package.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 声の分離(AI) | missing |  |
| 残響除去 | missing |  |
| マイク被り低減 | missing |  |

## Wiring

None. Expected: ffmpeg-skill probe before/after, audio --replace to put the cleaned stem back.

## Steps

- [ ] **1. Engine selection ADR** — candidates to verify: DeepFilterNet, Demucs/HTDemucs, Resemble-Enhance, VoiceFixer, nara_wpe -- licence of code AND weights, CPU feasibility, Japanese speech
  - Done when: ADR merged; ecosystem.json engines filled
- [ ] **2. Package skeleton + contract/doctor** — modelled on transcription-skill (engine registry, worker subprocess, --offline)
  - Done when: OS static checks PASS; doctor reports a missing engine as MISSING
- [ ] **3. ISOLATE op** — voice stem via the engine worker
  - Done when: SNR rises on a synthetic voice+noise fixture
- [ ] **4. DEREVERB op**
  - Done when: same pattern
- [ ] **5. BLEED_REDUCE op** — multi-mic input
  - Done when: lower cross-talk energy on a 2-track fixture
- [ ] **6. Remux helper** — via ffmpeg-skill audio --replace
  - Done when: video stream byte-identical
- [ ] **7. SKILL.md, CI (3 OSes unit, Linux engine job)**
  - Done when: CI green
- [ ] **8. (agent side) adapter + SkillSpec**
  - Done when: tracked in video-production-agent

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
