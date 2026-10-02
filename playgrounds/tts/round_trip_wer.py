"""Round-trip intelligibility: synthesise, re-transcribe with a sherpa-onnx Whisper model, score WER/CER."""
import argparse
import json
import re
from pathlib import Path

import jiwer

from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, load_lexicon, normalize

SENTENCES = Path(__file__).parent / "test_sentences.json"
RESULTS = Path(__file__).parent / "results"
CHARACTER_LANGS = {"ko", "zh"}


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _transcribe(recognizer, samples, rate) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def main() -> None:
    import numpy as np
    import sherpa_onnx as so

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--asr-encoder", required=True, help="Whisper encoder .onnx")
    parser.add_argument("--asr-decoder", required=True, help="Whisper decoder .onnx")
    parser.add_argument("--asr-tokens", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--lexicon")
    args = parser.parse_args()

    cand = CANDIDATES[args.candidate]
    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    tts = load_tts(args.candidate, args.threads)
    rows = []
    for lang in (l for l in args.langs if l in cand.languages):
        recognizer = so.OfflineRecognizer.from_whisper(
            encoder=args.asr_encoder,
            decoder=args.asr_decoder,
            tokens=args.asr_tokens,
            language=lang,
            num_threads=args.threads,
        )
        metric = jiwer.cer if lang in CHARACTER_LANGS else jiwer.wer
        for text in sentences[lang]:
            spoken = normalize(text, lang, lexicon)
            audio = [synth(tts, cand, c, lang, num_steps=args.num_steps) for c in chunk_text(spoken)]
            rate = audio[0][1]
            heard = _transcribe(recognizer, np.concatenate([a[0] for a in audio]), rate)
            error = metric(_clean(spoken), _clean(heard))
            rows.append({"lang": lang, "spoken": spoken, "heard": heard, "error": round(error, 4),
                         "metric": "cer" if lang in CHARACTER_LANGS else "wer"})
            print(json.dumps(rows[-1], ensure_ascii=False))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"round_trip_{args.candidate}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
