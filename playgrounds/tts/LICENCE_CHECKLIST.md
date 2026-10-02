# Licence review checklist

Stub for engineering triage; not legal advice.

- [ ] Supertonic 3: read the OpenRAIL-M LICENSE (use restrictions) in the model repo before shipping.
- [ ] Supertonic 3: confirm whether custom voice styles need Supertone's Voice Builder terms.
- [ ] Kokoro-82M: Apache-2.0 weights; confirm the sherpa-onnx multi-lang build's own notices.
- [ ] MeloTTS: MIT code; confirm the converted sherpa-onnx package licence file.
- [ ] AISHELL-3 / icefall model: confirm dataset and model licence.
- [ ] sherpa-onnx runtime: Apache-2.0.
- [ ] Piper (if ever added): engine is GPL-3.0 and bundles espeak-ng (GPL); check each voice's licence.
- [ ] MMS-TTS: CC-BY-NC, not usable commercially.
- [ ] Training or fine-tuning data (viVoice, PhoAudiobook, VIVOS): research-only; do not ship models derived from them.
