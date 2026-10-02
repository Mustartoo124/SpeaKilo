"""Worker process that owns one loaded TTS model, so the emulated device limits apply to it alone."""
from __future__ import annotations

import base64
import io
import time
import wave


def _wav_b64(samples, rate: int) -> str:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _synthesize(tts, cand, msg: dict) -> dict:
    import psutil

    from candidates import synth
    from text_normalize import chunk_text, normalize

    start = time.perf_counter()
    spoken = normalize(msg["text"], msg["lang"]) if msg["normalize"] else msg["text"]
    chunks = []
    for text in chunk_text(spoken, msg["max_chars"]):
        samples, rate = synth(
            tts, cand, text, msg["lang"], msg["sid"], msg["speed"], msg["num_steps"]
        )
        chunks.append(
            {
                "text": text,
                "ready_ms": round((time.perf_counter() - start) * 1000, 1),
                "audio_s": round(len(samples) / rate, 3),
                "wav_b64": _wav_b64(samples, rate),
            }
        )
    total_ms = (time.perf_counter() - start) * 1000
    audio_s = sum(c["audio_s"] for c in chunks)
    return {
        "ok": True,
        "spoken": spoken,
        "chunks": chunks,
        "ttfa_ms": chunks[0]["ready_ms"] if chunks else None,
        "total_ms": round(total_ms, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None,
        "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
    }


def worker_main(requests, responses) -> None:
    import psutil

    from candidates import CANDIDATES, load_tts

    tts, cand = None, None
    while True:
        msg = requests.get()
        if msg["cmd"] == "quit":
            return
        try:
            if msg["cmd"] == "load":
                start = time.perf_counter()
                cand = CANDIDATES[msg["candidate"]]
                tts = load_tts(msg["candidate"], msg["threads"])
                responses.put(
                    {
                        "ok": True,
                        "load_s": round(time.perf_counter() - start, 2),
                        "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
                    }
                )
            elif msg["cmd"] == "synth":
                responses.put(_synthesize(tts, cand, msg))
        except Exception as exc:  # report to the UI instead of killing the worker
            responses.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
