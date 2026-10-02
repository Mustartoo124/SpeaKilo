"""Benchmark time-to-first-audio, RTF and memory per candidate and language, optionally under CPU load."""
import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
import psutil

from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, load_lexicon, normalize

SENTENCES = Path(__file__).parent / "test_sentences.json"
RESULTS = Path(__file__).parent / "results"


def _burn() -> None:
    while True:
        pass


def _pct(values: list, q: float) -> float:
    return round(float(np.percentile(values, q)), 3)


def bench_language(tts, cand, lang, sentences, args, lexicon, proc) -> dict:
    ttfa_ms, rtf, rss_max = [], [], 0
    cold_ms = None
    for text in sentences:
        for run in range(args.runs + 1):  # first run per sentence is a warm-up
            start = time.perf_counter()
            chunks = chunk_text(normalize(text, lang, lexicon), args.max_chars)
            first, audio_s = None, 0.0
            for chunk in chunks:
                samples, rate = synth(tts, cand, chunk, lang, num_steps=args.num_steps)
                if first is None:
                    first = time.perf_counter() - start
                audio_s += len(samples) / rate
            total = time.perf_counter() - start
            rss_max = max(rss_max, proc.memory_info().rss)
            if cold_ms is None:
                cold_ms = first * 1000
            if run:
                ttfa_ms.append(first * 1000)
                rtf.append(total / audio_s)
    return {
        "candidate": cand.name,
        "lang": lang,
        "n": len(ttfa_ms),
        "cold_first_ms": round(cold_ms, 1),
        "ttfa_ms_p50": _pct(ttfa_ms, 50),
        "ttfa_ms_p95": _pct(ttfa_ms, 95),
        "rtf_p50": _pct(rtf, 50),
        "rtf_p95": _pct(rtf, 95),
        "rss_max_mb": round(rss_max / 2**20, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2, help="Supertonic denoising steps")
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--background-load", type=int, default=0, help="busy processes to mimic ASR/MT")
    parser.add_argument("--lexicon")
    parser.add_argument("--hardware", default="unlabelled", help="e.g. rpi5-8gb; only target-board runs count")
    args = parser.parse_args()

    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    proc = psutil.Process()
    burners = [mp.Process(target=_burn, daemon=True) for _ in range(args.background_load)]
    rows = []
    try:
        for p in burners:
            p.start()
        for name in args.candidates:
            cand = CANDIDATES[name]
            start = time.perf_counter()
            tts = load_tts(name, args.threads)
            load_s = time.perf_counter() - start
            rss_load = proc.memory_info().rss / 2**20
            for lang in (l for l in args.langs if l in cand.languages):
                row = bench_language(tts, cand, lang, sentences[lang], args, lexicon, proc)
                row.update(load_s=round(load_s, 2), rss_after_load_mb=round(rss_load, 1))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
            del tts
    finally:
        for p in burners:
            p.terminate()

    RESULTS.mkdir(exist_ok=True)
    meta = {
        "hardware": args.hardware,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": args.threads,
        "background_load": args.background_load,
        "num_steps": args.num_steps,
        "runs": args.runs,
    }
    out = RESULTS / f"ttfa_{args.hardware}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
