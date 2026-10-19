"""Write blinded WAV samples plus a rating sheet and answer key for a listening panel (MOS proxy)."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmark_schema import DEFAULT_TESTSET, ERROR_TYPES, load_testset
from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, normalize

OUT = Path(__file__).parent / "results" / "mos"
RATING_COLUMNS = ["sample_id", "utt_id", "lang", "text", "rater", "mos", "intelligibility",
                  "pronunciation_error", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    testset = load_testset(args.testset, langs=args.langs, split=args.split)
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for name in args.candidates:
        cand = CANDIDATES[name]
        steps = args.num_steps if cand.uses_lang_extra else None
        tts = load_tts(name, args.threads)
        for utt in (u for u in testset if u.lang in cand.languages):
            parts = [synth(tts, cand, c, utt.lang, num_steps=steps)
                     for c in chunk_text(normalize(utt.text, utt.lang))]
            items.append((name, utt, np.concatenate([p[0] for p in parts]), parts[0][1]))

    random.Random(args.seed).shuffle(items)
    with open(OUT / "rating_sheet.csv", "w", newline="", encoding="utf-8") as sheet, \
         open(OUT / "answer_key.csv", "w", newline="", encoding="utf-8") as key:
        rating, answers = csv.writer(sheet), csv.writer(key)
        rating.writerow(RATING_COLUMNS)
        answers.writerow(["sample_id", "candidate", "utt_id", "lang"])
        for number, (name, utt, samples, rate) in enumerate(items):
            sample_id = f"s{number:03d}"
            sf.write(OUT / f"{sample_id}.wav", samples, rate)
            rating.writerow([sample_id, utt.id, utt.lang, utt.text, "", "", "", "", ""])
            answers.writerow([sample_id, name, utt.id, utt.lang])
    print(f"wrote {len(items)} samples to {OUT}")
    print("mos and intelligibility: 1-5; pronunciation_error one of:", ", ".join(e or "(blank)" for e in ERROR_TYPES))


if __name__ == "__main__":
    main()
