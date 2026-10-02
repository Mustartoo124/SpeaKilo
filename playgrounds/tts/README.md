# tts

Compare offline TTS candidates for EN, KO, VI and ZH on CPU-only Raspberry Pi 5 hardware, all through sherpa-onnx.

## Candidates

| Candidate | Languages | Role | Licence (verify before shipping) |
|---|---|---|---|
| `supertonic3` | EN, KO, VI | Primary engine (EN and KO are top priority) | Weights OpenRAIL-M, code MIT |
| `kokoro_multi_v1_0` | EN, ZH | EN comparison; CN candidate | Apache-2.0 |
| `melo_zh_en` | ZH (EN words in its lexicon) | CN candidate; large model (163 MB, 44.1 kHz) | MIT |
| `aishell3_zh` | ZH | Fast, small CN baseline (low sample rate) | TODO |

Kokoro has no Korean or Vietnamese voices, and the sherpa-onnx Melo package is Chinese+English only. Piper is not included: its engine is GPL-3.0 and Supertonic already covers Vietnamese. Candidate URLs and file layouts are taken from the sherpa-onnx docs; confirm they still resolve. See [LICENCE_CHECKLIST.md](LICENCE_CHECKLIST.md).

## How to run

```
pip install -r requirements.txt
python download_models.py                       # all candidates, into ./models (not committed)
python synthesize.py --candidate supertonic3 --lang ko --text "라인을 멈추세요." --num-steps 2
python bench_ttfa.py --hardware rpi5-8gb --threads 2 --background-load 2
python round_trip_wer.py --candidate supertonic3 --asr-encoder ... --asr-decoder ... --asr-tokens ...
python mos_manifest.py                          # blinded WAVs + rating sheet in ./results/mos
pytest                                          # text normalisation and chunking tests
python ui_server.py                             # local UI on http://127.0.0.1:8765
```

## Device emulator and UI

`ui_server.py` serves a small local page (stdlib only, localhost) to try each model on an emulated Raspberry Pi 5 (8 GB). Each model runs in its own worker process that `device_emulator.py` constrains:

- **Core pinning (real):** the worker is limited to 1-4 CPU cores, one per physical core where the host has hyper-threading.
- **Shared-core load (real):** busy processes occupy N of those cores to mimic ASR and MT running alongside TTS.
- **CPU slowdown (estimate):** the worker is suspended and resumed so it runs only 1/N of the time. The default of 3 is an unvalidated guess (`STATUS: ESTIMATE`); calibrate it by comparing one model's RTF on this PC and on the real board.
- **Memory (reported, not enforced):** worker RSS is shown against an 8 GB budget minus an estimated 1 GB for the OS.

This shows relative behaviour and catches gross problems (memory, starvation, chunking). It cannot reproduce Cortex-A76 timing, so UI numbers are never `MEASURED` evidence. The page also plays each chunk as it would be queued and can download missing models.

- `bench_ttfa.py` reports time-to-first-audio (committed text chunk in, first chunk synthesised), RTF, cold-start time and RSS per candidate and language. `--background-load N` runs N busy processes to mimic ASR/MT contention.
- Only runs with `--hardware rpi5-8gb` on the real board count as target-hardware evidence; desktop numbers are for development only.
- `round_trip_wer.py` scores the spoken (normalised) text, using CER for KO and ZH. It needs a sherpa-onnx Whisper model you download separately.
- `text_normalize.py` expands units, spells equipment IDs character by character, and applies an optional `term<TAB>spoken` lexicon. Numbers are left as digits for the model front-end; check them in the round-trip test.

## Decisions pending

- Custom voices: Supertonic ships fixed preset voices, so none are planned for the prototype.
- Supertonic `--num-steps` (2 vs more) is a speed/quality trade-off to settle with the benchmark.

## Dependencies

`sherpa-onnx`, `numpy`, `soundfile`, `psutil`, `jiwer`, `pytest` (see `requirements.txt`). Python 3.9 or newer.
