"""Synthesise one sentence to a WAV file: normalise, chunk, synthesise chunk by chunk."""
import argparse

import numpy as np
import soundfile as sf

from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, load_lexicon, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--lang", required=True, choices=["en", "ko", "vi", "zh"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--sid", type=int)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, help="Supertonic denoising steps; fewer is faster")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lexicon", help="TSV file of 'term<TAB>spoken form'")
    args = parser.parse_args()

    cand = CANDIDATES[args.candidate]
    if args.lang not in cand.languages:
        parser.error(f"{args.candidate} does not support {args.lang}: {cand.languages}")
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    spoken = normalize(args.text, args.lang, lexicon)
    print(f"spoken form: {spoken}")

    tts = load_tts(args.candidate, args.threads)
    parts, rate = [], None
    for chunk in chunk_text(spoken):
        samples, rate = synth(tts, cand, chunk, args.lang, args.sid, args.speed, args.num_steps)
        parts.append(samples)
    sf.write(args.out, np.concatenate(parts), rate)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
