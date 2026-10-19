"""Round-trip check: synthesise, re-transcribe with sherpa-onnx Whisper, score error and protected terms."""
import argparse
import re
from pathlib import Path

import jiwer
import numpy as np

from benchmark_schema import DEFAULT_TESTSET, AsrCheck, load_testset, term_heard, write_jsonl
from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, load_lexicon, normalize

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
    import sherpa_onnx as so

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--asr-encoder", required=True, help="Whisper encoder .onnx")
    parser.add_argument("--asr-decoder", required=True, help="Whisper decoder .onnx")
    parser.add_argument("--asr-tokens", required=True)
    parser.add_argument("--asr-name", default="whisper", help="label stored with each result")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--lexicon")
    args = parser.parse_args()

    cand = CANDIDATES[args.candidate]
    steps = args.num_steps if cand.uses_lang_extra else None
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    tts = load_tts(args.candidate, args.threads)
    rows = []
    for lang in (l for l in args.langs if l in cand.languages):
        recognizer = so.OfflineRecognizer.from_whisper(
            encoder=args.asr_encoder, decoder=args.asr_decoder, tokens=args.asr_tokens,
            language=lang, num_threads=args.threads,
        )
        metric_name = "cer" if lang in CHARACTER_LANGS else "wer"
        metric = jiwer.cer if metric_name == "cer" else jiwer.wer
        for utt in load_testset(args.testset, langs=[lang], split=args.split):
            spoken = normalize(utt.text, lang, lexicon)
            audio = [synth(tts, cand, c, lang, num_steps=steps) for c in chunk_text(spoken)]
            heard = _transcribe(recognizer, np.concatenate([a[0] for a in audio]), audio[0][1])
            hits = tuple(term_heard(spec, heard) for spec in utt.must_preserve)
            rows.append(
                AsrCheck(
                    candidate=args.candidate, utt_id=utt.id, lang=lang, asr_model=args.asr_name,
                    metric=metric_name, error=round(metric(_clean(spoken), _clean(heard)), 4),
                    spoken=spoken, heard=heard, must_preserve_hit=hits, all_preserved=all(hits),
                )
            )
            print(f"{utt.id} {metric_name}={rows[-1].error} preserved={rows[-1].all_preserved}")

    out = RESULTS / f"roundtrip_{args.candidate}_{args.asr_name}.jsonl"
    write_jsonl(out, rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
