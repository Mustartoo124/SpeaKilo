# TECHNICAL PROPOSAL
**Phase 2 — Technical Submission**

| Field | Submission |
|---|---|
| Team / Project | SoftAI — SpeaKilo |
| Submission Date | 21 / 08 / 2026 |
| Version | v2.0 — Refactored Technical Proposal |
| Confidentiality | Restricted — Challenge Review Only |

---

## 1. Executive Summary

SpeaKilo is an offline, low-latency Edge-AI speech translation system designed for frontline industrial communication. The proposed product combines a Raspberry Pi 5 (8 GB) pocket compute unit, wireless earbud capture/playback, and a companion HUD. At runtime, audio remains on the device and flows through streaming VAD, noise conditioning, language-specific ASR, a stable-prefix commit buffer, machine translation, and queued TTS. The first deployment focus is multilingual manufacturing in Vietnam, especially Vietnamese–Korean operations, while the model stack also covers Vietnamese, English, Mandarin, and Korean.

The technical differentiator is system-level rather than a claim of a new foundation model: SpeaKilo combines specialist models per language pair, INT8 ONNX Runtime inference on the Raspberry Pi 5 CPU, industrial terminology biasing, confidence gating, and a stabilization policy that prevents uncommitted ASR text from propagating into translation or speech output. Phase 2 success is defined by measurable engineering gates: <2 s text turnaround, robust ASR under industrial noise, stable streaming with quantified revision/commit latency, full-shift battery operation, and zero runtime cloud dependency.

This proposal deliberately distinguishes published baselines, engineering estimates, acceptance targets, and team-measured results. Where Raspberry Pi 5 measurements are not yet available, they are presented as validation gates rather than completed results. That separation is central to the execution plan: the next milestones convert the current architecture into reproducible on-device evidence.

| Status Label | Meaning | How it is used in this proposal |
|---|---|---|
| PUBLISHED | Reported by model/dataset authors or public sources | Reference point only; not claimed as a SpeaKilo measurement |
| ESTIMATE | Engineering calculation before target-hardware profiling | Used for memory, power, and runtime planning |
| TARGET | Phase 2 / Phase 3 acceptance criterion | A requirement to be validated |
| MEASURED | Team result on a stated test setup | Only used after the test is completed and reproducible |

---

### 1.1 Problem Overview

Frontline manufacturing communication is time-sensitive, hands-busy, noisy, and often constrained by site connectivity or data-handling rules. Misunderstood instructions can affect safety, quality, and shift productivity. SpeaKilo therefore treats translation as an industrial edge-system problem rather than a generic mobile-app feature.

| Evidence | What it indicates | Interpretation for SpeaKilo |
|---|---|---|
| CBS–TNO multilingual-workplace survey: 46% reported that workers do not always understand each other; 10.4% reported mistakes/incomplete work; 1.2% reported dangerous situations. | Language barriers can have measurable operational consequences. | Directional analogue for industrial risk; not presented as Vietnam-specific incidence. |
| OECD / Vietnam MOIT references in §References document the scale and importance of FDI manufacturing and Korean industrial investment in Vietnam. | A multilingual manufacturing beachhead is commercially plausible. | Used to justify target segment selection, not to claim customer adoption. |

---

### 1.2 Proposed Solution

> **Figure 1.** SpeaKilo target real-time translation architecture. The diagram is a design view; performance claims are validated separately.

**Real-Time Translation Architecture — Private, low-latency AI processing on device**

The Pocket AI Unit (Raspberry Pi 5, 8 GB) processes audio through the following pipeline:

1. **Capture** — 16 kHz mono audio capture
2. **Streaming VAD** — Silero-VAD speech endpoints
3. **Audio Conditioning** — Suppress noise, reduce reverb
4. **Streaming ASR** — Zipformer-30M · per-language
5. **Stable Buffer** — Commit text at N=3 · L=2
6. **Incremental NMT** — envit5 & NLLB · by language

Output delivered via:
- **TTS Queue** — Supertonic 3 (EN/KO/VI) · Kokoro or MeloTTS (ZH)
- **Live UI** — EN → VI text
- **Companion HUD** — Wrist display

Audio input via **Earbud + Mic** (Bluetooth HFP).

---

### 1.3 Key Value Proposition

- **Fully offline runtime:** the production path is designed to complete ASR, translation, and TTS without a cloud API.
- **Predictable interaction:** the <2 s text-turnaround requirement is treated as an acceptance target and profiled per language pair.
- **Stable streaming:** only committed prefixes are translated; unstable partial text stays in the UI and never reaches MT/TTS.
- **Industrial robustness:** evaluation explicitly includes machine noise, babble, reverberation, code-switching, and safety terminology.
- **Edge deployment discipline:** Raspberry Pi 5 (CPU only), ONNX Runtime, quantization, memory budgeting, and power/thermal tests are part of the same system plan.

---

## 2. Problem Definition & Target Users

### 2.1 Problem Statement

SpeaKilo targets short, operational conversations where a delayed or unstable translation is more harmful than a polished translation that arrives too late. The primary beachhead is Vietnamese–Korean manufacturing in Vietnam, with Vietnamese–English and Vietnamese–Mandarin as adjacent deployment paths. The product is not intended to replace professional interpreters in negotiations or high-stakes legal/medical interactions; it is intended to reduce routine translation friction for production, training, QA, and safety communication.

---

### 2.2 Impact Analysis

| Impact Area | Current Pain Point | Operational Consequence | SpeaKilo Design Response |
|---|---|---|---|
| Safety / Operations | Misunderstood verbal instructions | Incorrect action, rework, compliance risk | Offline low-latency translation + terminology bias + confidence gating |
| Productivity | Translation pauses during shift handover / machine operations | Downtime and repeated clarification | Streaming partials + stable commits + hands-free workflow |
| Connectivity / Privacy | Internet may be unreliable or operational speech may be restricted from cloud processing | Cloud-dependent tools may be unavailable or disallowed | Zero runtime cloud dependency; egress validation in §6.5 |
| Usability | Phones require screen interaction in hands-busy environments | Workers interrupt tasks to operate the translator | Earbud push-to-talk/session mode + companion HUD |

---

### 2.3 Target Users & Use Cases

| User Segment | Language Need | Primary Context | Priority |
|---|---|---|---|
| Local operators | VI <-> EN / KO / CN | Assembly line, maintenance, machine operations | Critical |
| Foreign supervisors | KO / CN / EN <-> VI | Production management and shift handover | Critical |
| Technical trainers / QA | EN / KO <-> VI | Training, QA, troubleshooting | High |

---

### 2.4 Key Design Constraints

| Constraint | Acceptance Target | Evidence Status |
|---|---|---|
| Internet dependency | Zero runtime cloud dependency | TARGET — validate with network-egress test |
| Text translation turnaround | < 2 s end-to-end critical text path | TARGET — Raspberry Pi 5 measurement pending |
| Battery | ≥ 8 h at representative translation duty cycle | TARGET — current 7–9 h estimate makes this an open risk |
| Languages | VI, EN, KO, CN with fixed-direction routing in MVP | DESIGN — VI<->KO remains an interim pivot path |
| Noise robustness | Evaluate clean and 5–10 dB SNR machine/babble conditions | TARGET — regression suite in §4.6 |
| Industrial usability | Hands-busy interaction; single-button / session operation | DESIGN — field usability validation in Phase 3 |

> **Judge-facing acceptance principle:** SpeaKilo is considered technically successful only when latency, accuracy, stability, power, and offline behavior are measured together on the target device. A fast model that fails battery or noise requirements does not pass the system gate.

---

## 3. Business Solution & Innovation

### 3.1 Industry Problem & Solution Fit

Existing options solve different parts of the problem. Cloud translation apps offer strong general-purpose language coverage but depend on a phone/network workflow. Consumer translator devices optimize for travel and conversation. Human interpreters provide high semantic quality but do not scale as a per-worker operational layer. SpeaKilo is positioned around the intersection that matters on an industrial shift: offline operation, hands-busy interaction, site terminology, stable streaming, and an edge form factor.

---

### 3.2 Competitive Positioning

| Dimension | Cloud Translation App | Consumer Translator Device | Human Interpreter | SpeaKilo Design Target |
|---|---|---|---|---|
| Runtime internet | Often required for full capability | Varies by device/mode | No | No |
| Industrial terminology package | Generic by default | Limited / varies | Human knowledge | Site-specific lexicon and biasing |
| Hands-busy workflow | Phone interaction common | Device-dependent | Yes | Earbud + pocket unit + HUD |
| Streaming stability policy | Product-dependent | Product-dependent | Natural conversation | Explicit stable-prefix commit policy |
| Noise/safety evaluation | Not factory-specific by default | Not factory-specific by default | Human adaptation | Factory-noise + safety test suite |
| Enterprise data egress | Cloud processing may be incompatible with some site policies | Varies | No external service | Zero runtime cloud dependency by design |
| Scale across shifts/languages | Software scales; network required | Device fleet | Interpreter capacity constrained | Device fleet + offline terminology packs |

---

### 3.3 Commercial Beachhead

Go-to-market is B2B enterprise deployment, starting with manufacturing sites that have recurring Vietnamese–Korean communication needs. The current commercial structure uses three deployment tiers — Pilot (5–20 devices), Professional (20–100), and Enterprise (100+) — with revenue from hardware, software/service subscriptions, and enterprise AI customization.

---

### 3.4 ROI Validation Method

The proposal avoids presenting an unvalidated payback period as a fact. During pilot discovery, the team will measure translation-assisted minutes saved per shift, frequency of repeat clarifications, and interpreter/supervisor time displaced. Monthly benefit is then calculated from observed time saved × loaded labor cost × active users × working days. The business case is accepted only when the pilot-derived benefit exceeds device amortization plus service cost.

> **Commercial evidence gate:** replace internal labor-time assumptions with pilot-measured values before the final enterprise pricing claim.

---

## 4. AI Approach & Technical Design

### 4.1 System Pipeline Overview

SpeaKilo uses a streaming pipeline with a critical text path and an off-critical TTS path. The system renders ASR partials immediately, but only stable committed text is translated and synthesized. This decouples perceived responsiveness from downstream correctness and prevents repeated re-translation of unstable tokens.

| Stage | Role | Critical-Path Behavior |
|---|---|---|
| 1. Capture | 16 kHz mono audio from earbud microphone | Continuous / push-to-talk session input |
| 2. VAD | Detect speech start/end with pre- and post-roll | Avoid clipping first/last words |
| 3. Audio conditioning | Reduce stationary/non-stationary noise and reverberation | Improves ASR input quality |
| 4. Streaming ASR | Emit partial recognition hypotheses | Visible immediately; not yet trusted downstream |
| 5. Stable buffer | Commit prefixes only after repeated agreement | Only committed text can reach MT/TTS |
| 6. Translation | Translate committed chunks | No re-translation of unstable suffixes |
| 7. Live bilingual UI | Show source partials + committed translation | Independent updates |
| 8. TTS queue | Synthesize committed translations in order | Queued off critical text path |

---

### 4.2 Module-by-Module Design

| Module | Selected Model / Framework | Footprint (est.) | Coverage | Phase Status |
|---|---|---|---|---|
| VAD | Silero-VAD | ~1 MB | All languages | Selected |
| ASR — Vietnamese | Zipformer-30M RNNT streaming | ~30 MB INT8 | VI -> text | Selected / export path exists |
| ASR — EN/KO/CN | Compact streaming ASR, same architecture family | ~30–65 MB each INT8 | EN/KO/CN -> text | Phase 2 build candidate |
| MT — VI <-> EN | envit5-translation | ~275 MB INT8 | VI <-> EN | Selected |
| MT — VI <-> CN / KO | NLLB-200-distilled-600M | ~300 MB 4-bit | VI <-> CN; VI <-> KO | CN fine-tune planned; KO interim pivot |
| TTS — EN/KO/VI | Supertonic 3 | ~99M params, INT8 ONNX | EN/KO/VI output | Selected candidate; Pi 5 benchmark pending |
| TTS — CN | Kokoro-82M or MeloTTS | 82M params / ~163 MB | CN output | Second-priority candidates; benchmark pending |

**Estimated resident model set:** approximately 1 GB against 8 GB LPDDR4X on the Raspberry Pi 5. This is an engineering estimate; peak working memory and allocator behavior must be profiled on-device.

#### 4.2.1 Why this combination, not a single multilingual model

**ASR.** Vietnamese is the source language for the most frequent operator-side use cases, so the design prioritizes a compact Vietnamese-specialized streaming recognizer. EN/KO/CN use a companion streaming architecture to keep the export/runtime path consistent.

**Translation.** envit5 is reserved for VI<->EN, while NLLB covers VI<->CN and the current VI<->KO fallback. This avoids forcing one checkpoint to optimize every pair equally.

**VI<->KO gap.** Dedicated Vietnamese–Korean parallel training data is not yet available in the current data inventory. The MVP therefore uses VI->EN via envit5 and EN->KO via NLLB (and the reverse path accordingly). This is explicitly an interim fallback with added latency and quality risk; direct VI<->KO data/model acquisition is a Phase 2 priority.

**TTS.** Supertonic 3 is the primary engine because one compact model covers EN, KO and VI. Kokoro-82M has no Korean or Vietnamese voices, so it is only a candidate for EN/CN output. CN is a second-priority language served by Kokoro or MeloTTS, chosen by benchmark. Piper is not used: its engine is GPL-3.0 and Supertonic already covers Vietnamese.

**Licensing.** Supertonic 3 weights are OpenRAIL-M and must be reviewed for use restrictions before commercialization. Licensing risk is treated as an engineering release gate, not deferred to post-launch.

---

### 4.3 Streaming Stabilization Policy

The stabilization layer is a first-class system component because streaming ASR partials can revise earlier words. The policy prevents those revisions from causing translated-text flicker and repeated TTS.

1. Collect the last N ASR partial hypotheses (initial calibration value N = 3).
2. Compute the longest common prefix across those hypotheses.
3. Commit a prefix only after it remains identical for L consecutive updates (initial L = 2) and ends on a word or punctuation boundary.
4. Lock committed text permanently; only the uncommitted suffix may change on screen.
5. Send only committed text to MT and TTS.
6. Tune N and L on the stability-vs-latency Pareto frontier during field testing.

| Stability Metric | Definition | Why the Judge Should Care |
|---|---|---|
| Revision rate | Rewritten visible tokens / visible partial tokens | Direct measure of flicker |
| Commit latency | Time from spoken token completion to immutable commit | Captures the latency cost of stability |
| Committed-token integrity | Committed tokens that never require correction | Should be 100% by design; any violation is a pipeline defect |
| N/L Pareto curve | Revision rate vs commit latency across stabilization settings | Shows the trade-off rather than hiding it in one parameter choice |

---

### 4.4 On-Device Optimisation & Memory Management

| Model / Component | Optimisation Path | Rationale / Validation |
|---|---|---|
| Zipformer-30M | INT8 ONNX / sherpa-onnx on CPU | Compact streaming ASR; per-stage latency must be profiled on the Pi 5 CPU |
| Companion EN/KO/CN ASR | INT8; same export/runtime family as VI ASR | Reduces duplicated deployment tooling |
| envit5 | INT8 dynamic quantization; decoder cache reuse | Reduces autoregressive decoding cost |
| NLLB-distilled-600M | 4-bit quantization; language/domain fine-tuning | Fits current memory budget; accuracy must be re-baselined after quantization |
| Supertonic 3 | INT8 ONNX via sherpa-onnx; low denoising step count; synthesise per committed clause | Lowers time-to-first-audio; step count swept against quality |
| Kokoro / MeloTTS (CN) | Prebuilt sherpa-onnx packages; keep resident in RAM | Avoid reload latency on the CN output path |

> **Runtime decision:** standardize on ONNX Runtime with the CPU execution provider and INT8 models. The Raspberry Pi 5 has no NPU, so the QNN Execution Provider applies only if the platform changes to a Qualcomm board.

---

### 4.5 Robustness & Edge-Case Handling

- **Acoustic:** augment/evaluate stationary machine hum, non-stationary impacts/alarms, babble, and reverberation before ASR.
- **Language routing:** MVP uses explicit fixed-direction language selection instead of automatic LID; this removes an avoidable failure mode in high noise.
- **Confidence gating:** low-confidence committed chunks are held or marked uncertain instead of being spoken as if certain.
- **Code-switching:** Vietnamese industrial speech often includes English machinery terms and part codes; evaluate these explicitly rather than treating them as generic WER.
- **Linguistic safety:** evaluate negation, units, numbers, equipment IDs, and site-specific terminology separately from aggregate BLEU/chrF.

---

### 4.6 Accuracy Targets & Evaluation Plan

| Module / System | Metric | Current Evidence Status | Phase Target / Acceptance Test |
|---|---|---|---|
| VI ASR | WER | PUBLISHED model capability; SpeaKilo target-hardware result pending | <=15% clean industrial-vocabulary held-out speech; <=20% at 5–10 dB SNR |
| EN/KO/CN ASR | WER | Model candidate to be trained / selected | <=20% on held-out FLEURS/Common Voice/Zeroth-Korean; refine after training |
| VI<->EN MT | BLEU / chrF | Published envit5 PhoMT reference; SpeaKilo re-test pending | Stay within ~2 BLEU of the published baseline on held-out evaluation slice |
| VI<->CN MT | chrF / BLEU | No result for the planned exact fine-tune | Phase baseline: chrF >=50 / BLEU >=30 on FLORES+ VI–CN slice |
| VI<->KO MT | chrF + latency | Interim pivot path; no direct-model result | Interim target >=45 chrF; replace pivot when dedicated data/model is ready |
| TTS EN/KO/CN | MOS / intelligibility | Published/reference quality only | Internal panel target >=4.0 plus intelligibility regression |
| TTS VI | Round-trip intelligibility | No formal MOS result yet | <=10% WER after synthesized audio is re-transcribed by VI ASR |
| Streaming stability | Revision rate + commit latency | Policy designed; measurements pending | Tune N/L to minimize revisions while preserving <2 s text turnaround |
| Safety terminology | Critical-term / number / negation preservation | New domain gate; baseline pending | No release if safety test regresses vs previous checkpoint |
| End-to-end | Text turnaround + TTFA | Target only | <2 s critical text path on target device; report p50/p95 by language pair and noise level |

**Evaluation methodology:**

- Keep all fine-tuning and calibration data separate from held-out evaluation splits.
- Run automated regression for WER, BLEU/chrF, stability metrics, latency, and safety terminology after every checkpoint.
- Report p50 and p95 latency, not only a single best-case value.
- Measure quiet and industrial-noise conditions on the target device.
- Promote a model to the on-device build only when it does not regress the previous checkpoint on the agreed gate metrics.
- Run field validation with real operators/supervisors in Phase 3, combining transcript review, latency logs, and usability feedback.

---

### 4.7 Data Readiness for AI Training

#### Speech Data

| Dataset | Languages | Data Statistics | Use |
|---|---|---|---|
| FLEURS | EN, VI, KO, CN | EN 3,640; VI 4,120; KO 2,920; CN 4,600 samples | Cross-language ASR evaluation / supplemental training |
| Mozilla Common Voice | EN, VI, KO, CN | EN 1,180,618; VI 4,949; KO 1,756; CN 51,029 clips | Diverse ASR speech |
| ViASR | VI | 4,276 utterances; ~32 h | Vietnamese ASR |
| Zeroth-Korean | KO | 22,720 utterances; 52.8 h | Korean ASR supplement |

#### Machine Translation Data

| Dataset | Languages | Statistics | Use |
|---|---|---|---|
| FLORES+ | EN, VI, KO, CN | 2,009 sentences per listed language | Held-out multilingual MT validation |
| VLSP Chinese–Vietnamese MT | VI <-> CN | 300,348 train + 1,000 dev + 1,000 test pairs | VI–CN fine-tuning and evaluation |

#### Noise Augmentation

| Noise Category | Manufacturing Examples | Evaluation Role |
|---|---|---|
| Stationary | Motors, fans, HVAC, continuous machine hum | SNR sweep |
| Non-stationary | Machine operations, alarms, impacts | Transient robustness |
| Babble | Nearby worker conversations | Competing-speech robustness |
| Reverberation | Large rooms / enclosed workspaces | Acoustic-domain shift |

---

### 4.8 Technical Risk Register

| Risk | Impact | Mitigation | Exit Gate |
|---|---|---|---|
| VI<->KO lacks dedicated parallel data | Quality + latency from pivoting | Acquire/train direct pair; benchmark direct vs pivot | Direct path beats or justifies pivot on quality/latency |
| CPU-only inference over budget on 4 shared cores | Latency and power miss | Profile per stage; INT8; thread pinning; smaller models; measure under concurrent load | p95 latency and sustained power pass target |
| Battery estimate overlaps 8 h requirement | Shift-runtime failure | Measure duty cycle; power tune; hot-swap pack | ≥8 h representative workload or operational hot-swap plan |
| Bluetooth mic/noise degradation | ASR WER increase | Mic selection + conditioning + noise augmentation | Noise WER gate met at 5–10 dB |
| Safety term mistranslation | Operational risk | Domain lexicon + confidence gate + curated safety test | No regression on safety suite |
| TTS licensing path | Commercial release risk | Legal review / replace voice or engine if needed | License cleared before product release |

---

## 5. Hardware & Device Concept

### 5.1 Platform Selection & Justification

| Criterion | Jetson Orin Nano Super 8GB | Raspberry Pi 5 (8 GB) | Raspberry Pi 5 + Hailo AI HAT+ 2 |
|---|---|---|---|
| AI performance | 67 sparse / 33 dense INT8 TOPS | CPU only (4x Cortex-A76), no NPU | 40 TOPS at INT4 |
| Power | 7–25 W | To be measured on target | ~17–20 W system estimate |
| Memory / storage | 8 GB LPDDR5 | 8 GB LPDDR4X + microSD or NVMe | 8/16 GB host + accelerator memory |
| AI toolchain | CUDA / TensorRT / ONNX Runtime | ONNX Runtime CPU EP / sherpa-onnx | HailoRT / Model Zoo |
| Transformer deployment | Excellent ecosystem | INT8 on CPU; latency must be profiled | Model conversion compatibility is a constraint |
| Portable form factor | Small compute box | Compact, battery-friendly | Higher system integration/power burden |

**Platform decision:** Raspberry Pi 5 (8 GB) is selected for its compact form factor, low cost and mature Linux and ONNX Runtime support, not for raw TOPS. The trade-off is that all inference shares four CPU cores, so per-stage latency and concurrent-load behaviour are first-class validation gates.

---

### 5.2 Key Hardware Components & Power Budget

| Component | Prototype Specification | Power Status | Validation Note |
|---|---|---|---|
| SoC module | Raspberry Pi 5, 8 GB LPDDR4X | To be measured | Profile representative average and sustained peak |
| Audio I/O | Bluetooth 5.2 HFP earbud/headset | Independent battery | Measure added Bluetooth latency and packet robustness |
| Wireless | Integrated Wi-Fi + Bluetooth (confirm version and HFP support) | Included in board budget | Runtime internet disabled for core translation |
| Cooling | Passive heatsink + small fan | ~1–2 W estimate | Thermal throttle test under sustained inference |
| Storage | microSD or NVMe SSD (capacity to be decided) | Measure if NVMe is used | OS, models, terminology packages, logs |
| Power conversion | 12 V regulator, fuse, switch, LEDs | ~1 W loss/overhead | Measure conversion efficiency |
| Main battery | ~99 Wh V-mount prototype battery | Separate energy store | Full-shift discharge test |
| Companion HUD | ESP32-S3 + ~1.9–2.4 in IPS, own Li-Po | ~1.5–2 W peak, separate battery | BLE/Wi-Fi link and display runtime |

> **Battery status:** the earlier 7–9 h estimate was derived for the RB3 Gen 2 dev board and must be re-derived for the Raspberry Pi 5. Because the product requirement is ≥8 h, autonomy is an **OPEN RISK**, not a completed claim. Phase 2 must run a representative-duty-cycle discharge test; hot-swap capability is the operational mitigation.

---

### 5.3 Form Factor & Industrialisation Path

| Aspect | Phase 2 Prototype | Production Design Target |
|---|---|---|
| Compute enclosure | Raspberry Pi 5 development hardware + active cooling | Compact belt/pocket enclosure with controlled airflow |
| Battery | ~99 Wh V-mount prototype supply | Hot-swappable pack sized for full-shift continuity |
| User input | Earbud/session mode | Single-button, glove-friendly interaction |
| Environmental protection | Not yet certified | IP54 target; validate sealing after enclosure freeze |
| Mechanical robustness | Not yet qualified | 1.2 m drop target; formalize test method before claim |
| Thermal | Fan-assisted prototype cooling | Surface temperature + throttle limits measured under sustained load |
| Display | Companion HUD concept | Readable status/translation without requiring a handheld phone |

> **Important wording discipline:** IP54, drop resistance, and battery duration are design/qualification targets until a test report exists. The proposal does not present them as completed certifications.

---

### 5.4 Functional BOM Feasibility

| BOM Block | Selected / Candidate Component | Readiness | Procurement / Cost Action |
|---|---|---|---|
| Compute | Raspberry Pi 5 (8 GB) | Selected | Supplier availability and volume path to be confirmed |
| Audio | Bluetooth earbud/headset with microphone | Candidate class selected | Finalize microphone/SNR and battery requirements |
| Display | ESP32-S3 + compact IPS TFT | Candidate architecture | Finalize panel size, brightness, enclosure integration |
| Power | 99 Wh prototype battery + 12 V regulation | Prototype path defined | Cost custom/hot-swap production pack after power profile |
| Thermal | Heatsink + fan | Prototype path defined | Select fan/heatsink after sustained thermal test |
| Enclosure | Prototype / custom mechanical | Open | CAD, weight, mounting, sealing, drop test |

A costed BOM is a required commercialization artifact. This Phase 2 document provides a functional BOM and explicitly keeps supplier quotation as an execution gate rather than inventing unit prices.

---

### 5.5 Power & Thermal Validation

- Profile idle, active-listening, active-ASR, active-MT, and TTS states separately.
- Report average and p95 system power over representative multi-minute conversations.
- Run sustained inference until thermal steady state; record temperature, clock throttling, and latency drift.
- Repeat the <2 s latency test after thermal steady state — not only on a cold boot.
- Run a full battery discharge at representative duty cycle and report usable hours, not theoretical Wh/peak-W division.

---

## 6. System Architecture & Integration

### 6.1 Software Stack

| Layer | Component / Framework | Role |
|---|---|---|
| OS | Raspberry Pi OS / Ubuntu minimal build | Stable device runtime |
| AI runtime | ONNX Runtime (CPU EP) + sherpa-onnx | Unified local INT8 inference on the Pi 5 CPU |
| Audio processing | PulseAudio + VAD/noise-conditioning modules | Capture, buffering, suppression, endpointing |
| Orchestration | C++ | Pipeline scheduling, concurrency, queues, state handling |
| Connectivity | Bluetooth 5.2 HFP / BLE; Wi-Fi for maintenance only | Earbud audio + companion HUD; core translation does not require internet |
| Observability | Local structured logs / benchmark harness | Latency, confidence, power/thermal and regression evidence |

---

### 6.2 Architecture Diagram

**System Integration & Execution Mapping**  
*Target architecture on Raspberry Pi 5 (CPU only) — per-stage latency is profiled, not assumed.*

```
[EARBUD + MIC]                    [POCKET AI UNIT — Raspberry Pi 5]              [COMPANION HUD]
16 kHz capture          →  Audio I/O + VAD + conditioning (CPU)      →   Source + translation
Push-to-talk / session     Streaming ASR (ONNX Runtime, CPU)              BLE / Wi-Fi
Playback from TTS          Stable-prefix buffer + routing (C++)
[Bluetooth HFP]            NMT (ONNX Runtime, CPU)                  →   [TTS OUTPUT]
                           ~1 GB model set / 8 GB LPDDR4X                  Supertonic 3 / Kokoro

Offline runtime boundary: no cloud API or internet dependency.
All inference runs on the CPU; per-stage latency is profiled, not assumed.
```

> **Figure 2.** System integration view. All stages share the four Raspberry Pi 5 CPU cores; concurrent-load latency is validated per stage.

---

### 6.3 Offline-First Design Principles

- **Localized inference:** 100% of the core speech-to-speech path is designed to run without a cloud API.
- **Single deployment runtime:** ONNX Runtime (CPU EP, INT8) is the reference path for Raspberry Pi 5; TFLite is not used.
- **Deterministic queues:** bounded buffers prevent runaway TTS/backlog behavior.
- **Privacy by architecture:** operational audio/text is not required to leave the device during runtime.
- **Fail visible, not silently:** low confidence, disconnected peripherals, and resource pressure surface explicit states to the user.

---

### 6.4 Failure Handling

| Failure Mode | System Behavior | Recovery |
|---|---|---|
| Low ASR/MT confidence | Hold downstream commit or mark uncertain | User repeats / confirms; log event for evaluation |
| Bluetooth audio drop | Stop committing new audio; preserve last stable text | Reconnect peripheral without restarting models |
| TTS queue backlog | Prioritize newest safe committed chunks; bound queue | Drop stale non-critical playback while keeping text UI |
| Runtime/model error | Isolate failing stage and capture diagnostics | Restart stage/process; preserve session where possible |
| Low battery | Warn before performance collapse | Hot swap / external power; keep UI state |
| Thermal throttling | Expose diagnostic and latency drift | Reduce load or improve cooling before production acceptance |

---

### 6.5 Privacy & Security Validation

- Run the demo with internet disconnected and verify the complete translation path remains functional.
- Capture outbound network traffic during operation; the acceptance result is zero required external inference calls.
- Load terminology/model packages through controlled offline or signed-update procedures; production signing is a release target.
- Keep debug logs local and configurable so pilot data retention can follow site policy.

---

## 7. Team Profile & Execution Plan

### 7.1 Team Members

| Name | Role | Background | Primary Ownership |
|---|---|---|---|
| Nguyễn Văn Tú | Lead / Research Engineer | AI Engineer — Prudential | AI architecture, technical direction |
| Dương Trung Nghĩa | Research Engineer | AI Engineer — MoMo | Lightweight machine translation |
| Phạm Nguyên Hải Long | Research Engineer | AI Engineer — Marvell | ASR + efficient inference |
| Phạm Gia Bảo | Research Engineer | AI Engineer — Bosch | TTS + efficient inference |
| Phạm Vinh | Embedded Engineer | Embedded Engineer — Bosch | Embedded programming + hardware selection |
| Trần Phan Bảo Ngọc | Product Analyst | Social Marketing — upGrad | Product requirement scoring + tracking |
| Đinh Nguyễn Lan Anh | System Analyst | System Analyst — VPBank | Business analysis + stakeholder management |
| Tạ Minh Thư | Data Scientist | Data Scientist — Bosch | Data curation + model evaluation |
| Lê Nguyễn Hữu Trường | Software Engineer | Software Engineer — FPT | Frontend / mobile |
| Lê Quốc Khôi | Software Engineer | Software Engineer — FPT | System integration |
| Huỳnh Thái Tuấn | Embedded Engineer | — | Embedded programming |

---

### 7.2 Milestones With Exit Criteria

| Window | Milestone | Exit Criteria |
|---|---|---|
| Aug 2026 | Baseline freeze + evaluation harness | Reproducible held-out WER/MT/stability tests; target/estimate/measured labels applied consistently |
| Sep 2026 | Raspberry Pi 5 deployment + prototype integration | VI<->EN full path runs offline on target board; per-stage CPU latency profiled; p50/p95 latency reported |
| Sep–Oct 2026 | Four-language integration | EN/KO/CN ASR path integrated; VI<->CN baseline; VI<->KO pivot measured and direct-path plan frozen |
| Oct 2026 | Noise / thermal / battery field readiness | 5–10 dB noise results; sustained thermal test; representative battery discharge; failure-state handling |
| Oct–Nov 2026 | Field test + tuning | Operator/supervisor feedback; N/L stability calibration; terminology package validation |
| Nov 2026 | Final evaluation + challenge demo | Offline demo, benchmark dashboard, BOM/form-factor evidence, final video and reproducibility pack |

---

### 7.3 Execution Discipline

- One owner per model/system gate; integration is not considered complete when a model only runs on a workstation.
- Every performance claim includes hardware, language pair, noise condition, model version, quantization, and percentile.
- Checkpoint promotion is regression-gated; a latency win that causes accuracy/safety regression is rejected.
- Hardware and AI teams share the same acceptance dashboard for latency, power, temperature, memory, and error rates.

---

## 8. Rubric Coverage & Submission Readiness

### 8.1 Coverage Map

| Challenge Criterion | Weight | Where this Proposal Provides Evidence | Highest-Value Remaining Evidence |
|---|---|---|---|
| AI Approach & Technical Design | 35% | §4.1–§4.8: pipeline, models, stabilization, optimization, evaluation, data, risk register | Raspberry Pi 5 p50/p95 latency + WER/MT/stability results |
| Hardware & Device Concept | 25% | §5.1–§5.5: platform decision, functional BOM, power, industrialization, thermal plan | Costed BOM, form-factor dimensions/weight, full-shift battery test |
| Business Solution | 15% | §3: industry fit, differentiation, B2B model, ROI validation method | Customer discovery + pilot-derived ROI |
| Problem Definition & Impact | 15% | §1–§2: evidence, beachhead, personas, constraints | Vietnam-site interviews / pilot observations |
| Team & Execution Plan | 10% | §7: ownership, milestone exit criteria | Early target-hardware demo and regression dashboard |

---

### 8.2 Submission Checklist

| Item | Status | Action Before Final Submission |
|---|---|---|
| Executive summary and problem framing | READY | Keep evidence-status wording |
| Business differentiation | READY / VALIDATION NEEDED | Add customer-discovery evidence when available |
| AI pipeline + model specs + optimization | READY | Attach measured Raspberry Pi 5 benchmark dashboard |
| Evaluation plan | READY | Populate measured columns without replacing targets |
| Hardware platform + functional BOM | READY / PARTIAL | Add costed BOM, weight, dimensions, and photos |
| Architecture diagram | READY | Keep Figure 2; ensure final device routing matches implementation |
| Team + timeline | READY | Track exit criteria rather than activity-only milestones |
| Demo artifact | PENDING IF AVAILABLE | Insert video/QR/link + test setup + measured latency table |

---

## 9. Demo & Validation Evidence

### 9.1 Recommended Challenge Demo Script

The demo should prove the product claims in the same order a technical judge is likely to challenge them.

1. Show the device in offline mode and verify that the translation pipeline remains functional.
2. Run a clean Vietnamese -> English/Korean sentence and show ASR partials, stable committed text, translation, and TTS.
3. Repeat with industrial noise and display WER/latency logs from the same run.
4. Use a safety-critical sentence containing a negation, unit/number, and equipment/part code; show that these tokens are preserved.
5. Change the stabilization setting N/L and show the revision-rate vs commit-latency effect.
6. Display p50/p95 end-to-end latency, memory, temperature, and power from the Raspberry Pi 5 run.
7. Show outbound network capture demonstrating that no external inference API is required.

---

### 9.2 Evidence Pack to Attach

| Artifact | Minimum Content | Why it Matters |
|---|---|---|
| Demo video / QR | Offline run, target hardware, visible source/translation/TTS | Early demo bonus + credibility |
| Benchmark dashboard | Language pair, noise condition, WER/chrF/BLEU, revision rate, p50/p95 latency | Converts targets into evidence |
| Hardware photo / CAD | Compute unit, battery, cooling, HUD, scale reference | Form-factor credibility |
| Power / thermal log | Representative workload over time | Battery + sustained performance proof |
| BOM sheet | Part, supplier, quantity, prototype cost, production assumption | BOM feasibility |
| Field-test note | Participants, task, observations, issues, next changes | Real-world applicability |

> **Before submission,** replace this line with the final demo URL/QR and measured benchmark snapshot. Do not leave a 'demo' section containing only a duplicated software-stack table.

---

## References

[1] OECD. (2026). FDI Qualities Review of Viet Nam: Powering the Next Growth Phase. OECD Publishing. DOI: 10.1787/a3c78dac-en.

[2] OECD. (2026). Mobilising FDI for job quality and skills development: FDI Qualities Review of Viet Nam.

[3] Statistics Netherlands (CBS). (2025). Misunderstandings due to multilingualism in the workplace. Netherlands Working Conditions Survey (NEA), conducted with TNO.

[4] TNO. (2025). Bij meertaligheid vaakst misverstanden in bouw, landbouw en industrie. Netherlands Working Conditions Survey.

[5] Ministry of Industry and Trade of Viet Nam. (2026). Unblocking localisation bottlenecks, strengthening enterprise linkages in the automotive sector.

[6] Ministry of Industry and Trade of Viet Nam. (2025). MOIT leaders meets with Korean Minister of Trade, Industry and Energy.
