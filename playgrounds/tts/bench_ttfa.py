"""Benchmark TTS candidates on the test set; writes a run record, per-utterance rows and a summary per candidate."""
import argparse
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from benchmark_schema import (
    DEFAULT_TESTSET,
    SCHEMA_VERSION,
    TARGET_HARDWARE,
    RunRecord,
    UtteranceResult,
    load_testset,
    write_jsonl,
)
from candidates import CANDIDATES, load_tts, synth
from text_normalize import chunk_text, load_lexicon, normalize

RESULTS = Path(__file__).parent / "results"


def _burn() -> None:
    while True:
        pass


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def _throttled():
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, check=True)
        return bool(int(out.stdout.split("=")[1], 16) & 0x4)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def _stats(rows: list) -> dict:
    ttfa = [r.ttfa_ms for r in rows]
    rtf = [r.rtf for r in rows]
    return {
        "n": len(rows),
        "ttfa_ms_p50": round(float(np.percentile(ttfa, 50)), 1),
        "ttfa_ms_p95": round(float(np.percentile(ttfa, 95)), 1),
        "rtf_p50": round(float(np.percentile(rtf, 50)), 3),
        "rtf_p95": round(float(np.percentile(rtf, 95)), 3),
        "rss_mb_max": max(r.rss_mb for r in rows),
    }


def summarise(rows: list, utterances: dict) -> dict:
    warm = [r for r in rows if r.warm]
    by_lang, by_tag = defaultdict(list), defaultdict(list)
    for row in warm:
        by_lang[row.lang].append(row)
        for tag in utterances[row.utt_id].tags:
            by_tag[f"{row.lang}:{tag}"].append(row)
    return {
        "cold_first_ttfa_ms": round(rows[0].ttfa_ms, 1),
        "overall": _stats(warm),
        "by_lang": {k: _stats(v) for k, v in sorted(by_lang.items())},
        "by_lang_tag": {k: _stats(v) for k, v in sorted(by_tag.items())},
    }


def run_candidate(cand, utterances, args, lexicon, run: RunRecord, out_dir: Path) -> list:
    tts = load_tts(cand.name, args.threads)
    proc = psutil.Process()
    steps = args.num_steps if cand.uses_lang_extra else None
    rows = []
    for utt in utterances:
        for rep in range(args.reps + 1):  # rep 0 is a warm-up
            start = time.perf_counter()
            spoken = normalize(utt.text, utt.lang, lexicon)
            chunks = chunk_text(spoken, args.max_chars)
            t_norm = (time.perf_counter() - start) * 1000
            first, parts, rate = None, [], None
            for chunk in chunks:
                samples, rate = synth(tts, cand, chunk, utt.lang, speed=args.speed, num_steps=steps)
                parts.append(samples)
                if first is None:
                    first = time.perf_counter() - start
            total = time.perf_counter() - start
            audio_s = sum(len(p) for p in parts) / rate
            wav_path = None
            if args.save_audio and rep == args.reps:
                wav_path = str(out_dir / "wav" / f"{utt.id}.wav")
                Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_path, np.concatenate(parts), rate)
            rows.append(
                UtteranceResult(
                    run_id=run.run_id, utt_id=utt.id, lang=utt.lang, rep=rep, warm=rep > 0,
                    t_norm_ms=round(t_norm, 2), ttfa_ms=round(first * 1000, 1),
                    total_ms=round(total * 1000, 1), audio_s=round(audio_s, 3),
                    rtf=round(total / audio_s, 3), n_chunks=len(chunks),
                    rss_mb=round(proc.memory_info().rss / 2**20, 1),
                    temp_c=_temp_c(), throttled=_throttled(), wav_path=wav_path,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--langs", nargs="+", default=["en", "ko", "vi", "zh"])
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2, help="Supertonic denoising steps")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--background-load", type=int, default=0, help="busy processes to mimic ASR/MT")
    parser.add_argument("--lexicon")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--hardware", default="unlabelled", help=f"{TARGET_HARDWARE} only on the real board")
    args = parser.parse_args()

    testset = load_testset(args.testset, langs=args.langs, split=args.split)
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    by_id = {u.id: u for u in testset}
    evidence = "MEASURED" if args.hardware == TARGET_HARDWARE else "ESTIMATE"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    burners = [mp.Process(target=_burn, daemon=True) for _ in range(args.background_load)]
    try:
        for burner in burners:
            burner.start()
        for name in args.candidates:
            cand = CANDIDATES[name]
            utterances = [u for u in testset if u.lang in cand.languages]
            if not utterances:
                continue
            run = RunRecord(
                run_id=f"{stamp}_{args.hardware}_{name}",
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                hardware=args.hardware, evidence=evidence, candidate=name,
                model_version=cand.archive, threads=args.threads,
                background_load=args.background_load,
                num_steps=args.num_steps if cand.uses_lang_extra else None,
                speed=args.speed, testset=Path(args.testset).name, split=args.split,
                reps=args.reps, git_commit=_git_commit(), platform=platform.platform(),
                machine=platform.machine(), lexicon=args.lexicon,
            )
            out_dir = RESULTS / run.run_id
            rows = run_candidate(cand, utterances, args, lexicon, run, out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run.json").write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
            write_jsonl(out_dir / "utterances.jsonl", rows)
            summary = summarise(rows, by_id)
            (out_dir / "summary.json").write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, **summary}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"{name}: {json.dumps(summary['overall'])} -> {out_dir}")
    finally:
        for burner in burners:
            burner.terminate()


if __name__ == "__main__":
    main()
