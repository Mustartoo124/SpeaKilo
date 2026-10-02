"""Write blinded WAV samples and a rating sheet so a panel can score candidates (MOS proxy)."""
import argparse
import csv
import json
import random
from pathlib import Path

import soundfile as sf
import numpy as np

from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, normalize

SENTENCES = Path(__file__).parent / "test_sentences.json"
OUT = Path(__file__).parent / "results" / "mos"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for name in args.candidates:
        cand = CANDIDATES[name]
        tts = load_tts(name, args.threads)
        for lang in (l for l in args.langs if l in cand.languages):
            for idx, text in enumerate(sentences[lang]):
                parts = [synth(tts, cand, c, lang, num_steps=args.num_steps)
                         for c in chunk_text(normalize(text, lang))]
                items.append((name, lang, idx, text, np.concatenate([p[0] for p in parts]), parts[0][1]))

    random.Random(args.seed).shuffle(items)
    with open(OUT / "rating_sheet.csv", "w", newline="", encoding="utf-8") as sheet, \
         open(OUT / "answer_key.csv", "w", newline="", encoding="utf-8") as key:
        rating, answers = csv.writer(sheet), csv.writer(key)
        rating.writerow(["sample_id", "lang", "text", "mos_1_to_5", "notes"])
        answers.writerow(["sample_id", "candidate", "lang", "sentence_index"])
        for number, (name, lang, idx, text, samples, rate) in enumerate(items):
            sample_id = f"s{number:03d}"
            sf.write(OUT / f"{sample_id}.wav", samples, rate)
            rating.writerow([sample_id, lang, text, "", ""])
            answers.writerow([sample_id, name, lang, idx])
    print(f"wrote {len(items)} samples to {OUT}")


if __name__ == "__main__":
    main()
