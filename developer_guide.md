# Developer Guide

Where to put things and how work flows through the repository. Each top-level area is self-contained: do not import code across areas.

## Workflow

```
data/  ->  playgrounds/  ->  models/  ->  evaluation/  ->  pipeline/
(acquire)  (experiment)     (export)     (gate)          (integrate + deploy)
```

1. Get data with `data/`.
2. Experiment on a component in its `playgrounds/<name>/`.
3. Export the candidate with `models/export/`, then validate it with `models/validate/`.
4. Measure it with `evaluation/`. The regression gate must pass.
5. Only then integrate into `pipeline/` and deploy to the device.

## Where do I...

| I want to... | Go to |
|---|---|
| Download or prepare a dataset | `data/download/`, `data/prepare/` |
| Add noise to audio (SNR, reverb) | `playgrounds/noise_augmentation/` |
| Try or tune VAD | `playgrounds/vad/` |
| Try an ASR model, streaming, WER, RTF, code-switching | `playgrounds/asr/` |
| Try MT, quantisation, safety terms, pivot vs direct | `playgrounds/mt/` |
| Try TTS, TTFA, round-trip WER, MOS samples | `playgrounds/tts/` |
| Tune the stable-prefix buffer (N/L) | `playgrounds/stability/` |
| Export a model to ONNX / quantise it for the device | `models/export/` |
| Check ONNX shapes or QNN operator offload | `models/validate/` |
| Register a model (version, licence, status) | `models/registry.yaml`, `models/README.md` |
| Compute an official metric | `evaluation/<area>_eval/` |
| Change a pass/fail threshold | `evaluation/regression_gate/gate_config.yaml` |
| Implement a pipeline stage | `pipeline/include/speakilo/<stage>.h` + `pipeline/src/<stage>.cpp` |
| Add a shared struct | `pipeline/include/speakilo/pipeline_types.h` |
| Change model paths, N/L, language pairs | `pipeline/config/` |
| Add domain terminology | `pipeline/config/terminology/` (build with `data/prepare/prepare_terminology.py`) |
| Add a C++ test | `pipeline/tests/unit/` or `pipeline/tests/integration/` |
| Deploy to / run on the device | `pipeline/scripts/` |
| Set up a new device | `infra/device_setup/` |
| Capture logs, power, benchmarks | `infra/observability/` |
| Change CI or Docker images | `infra/ci/`, `infra/docker/` |
| Implement mobile app | `mobile_app` |

## Adding a feature

**New or changed AI component**
1. Prototype in the matching playground. Keep its own `requirements.txt`; do not add a shared one.
2. Add an export script in `models/export/` and a `registry.yaml` entry.
3. Add or extend evaluation scripts in `evaluation/` and thresholds in `gate_config.yaml`.
4. Run the gate. Do not integrate if it regresses.
5. Implement the stage in `pipeline/`, add unit and integration tests, and update `pipeline/docs/`.

**New pipeline stage**
1. Add a header in `pipeline/include/speakilo/` (with include guard) and source in `pipeline/src/`.
2. Register it in `pipeline/CMakeLists.txt`.
3. Wire it in `session.cpp` / `main.cpp`.
4. Add a unit test; update `pipeline/docs/architecture.md` and `latency_budget.md`.

**New language pair**
1. Add the direction in `pipeline/config/language_pairs.yaml`.
2. Add routing in `language_router`, plus tests.
3. Add MT/ASR/TTS models (steps above) and a terminology pack.
4. Add evaluation data under `evaluation/datasets/README.md` and gate thresholds.

## Rules

- No logic across areas: playgrounds, pipeline, evaluation, data, models, infra must not import from each other. Share via files (exported models, manifests, logs).
- Never commit model weights, datasets, manifests, or `results/` contents (only `.gitkeep`).
- No cloud calls in `pipeline/`; verify with `pipeline/scripts/verify_offline.sh`.
- Python: `snake_case`, module docstring, `TODO:` on the first line of unimplemented bodies. C++: `snake_case` filenames, include guards.
- Label every numeric constant with `# STATUS: PUBLISHED | ESTIMATE | TARGET | MEASURED` (see `docs/evidence_status.md`).
- No notebooks, no web UI, no cloud infrastructure files.
- Report latency as p50/p95 with hardware, language pair, noise condition, model version, and quantisation.

## Running things

Each area documents its own commands in its `README.md` (currently `TODO`). Fill in "How to run" and "Dependencies" as you implement.

- Playground: create a separate venv in `playgrounds/<name>/`, then `pip install -r requirements.txt`.
- Evaluation: run scripts in `evaluation/`, then `python evaluation/regression_gate/run_gate.py`.
- Pipeline: build with CMake in `pipeline/` (output in `build/`), then use `pipeline/scripts/deploy.sh` and `run_on_device.sh`.
- Lint: `ruff` and `black` for Python (configured in root `pyproject.toml`), `clang-format` for C++.
