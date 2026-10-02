"""Local web UI for trying TTS candidates on an emulated Raspberry Pi 5 (stdlib server, localhost only)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from candidates import CANDIDATES, MODELS_ROOT, ensure_model
from device_emulator import DeviceProfile, EmulatedDevice
from tts_worker import worker_main

HERE = Path(__file__).parent
MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 1000
LOAD_TIMEOUT_S = 900
SYNTH_TIMEOUT_S = 300
DEFAULTS = DeviceProfile()


def _clamp(value, low, high, cast=float):
    return max(low, min(high, cast(value)))


class Controller:
    """Owns the worker process and restarts it when the candidate or device settings change."""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._worker = None
        self._device = None
        self._key = None
        self._load = {}
        self._requests = self._responses = None

    def _reply(self, timeout: float) -> dict:
        waited = 0.0
        while waited < timeout:
            try:
                return self._responses.get(timeout=1)
            except queue.Empty:
                waited += 1
                if not self._worker.is_alive():
                    raise RuntimeError("TTS worker exited unexpectedly")
        raise TimeoutError("TTS worker did not answer in time")

    def _stop(self) -> None:
        if self._device:
            self._device.detach()
        if self._worker and self._worker.is_alive():
            self._requests.put({"cmd": "quit"})
            self._worker.join(5)
            if self._worker.is_alive():
                self._worker.terminate()
        self._worker = self._device = self._key = None

    def _start(self, candidate: str, profile: DeviceProfile, threads: int) -> None:
        self._stop()
        self._requests, self._responses = self._ctx.Queue(), self._ctx.Queue()
        self._worker = self._ctx.Process(
            target=worker_main, args=(self._requests, self._responses), daemon=True
        )
        self._worker.start()
        self._device = EmulatedDevice(profile)
        self._device.attach(self._worker.pid)
        self._requests.put({"cmd": "load", "candidate": candidate, "threads": threads})
        reply = self._reply(LOAD_TIMEOUT_S)
        if not reply["ok"]:
            self._stop()
            raise RuntimeError(reply["error"])
        self._key = (candidate, profile, threads)
        self._load = reply

    def synthesize(self, params: dict) -> dict:
        key = (params["candidate"], params["profile"], params["threads"])
        with self._lock:
            if self._key != key:
                self._start(*key)
            self._requests.put({"cmd": "synth", **params["synth"]})
            reply = self._reply(SYNTH_TIMEOUT_S)
            if reply["ok"]:
                reply["load_s"] = self._load["load_s"]
                reply["ram_budget_mb"] = round(params["profile"].ram_budget_mb)
            return reply

    def close(self) -> None:
        with self._lock:
            self._stop()


CONTROLLER = Controller()
DOWNLOAD_LOCK = threading.Lock()


def parse_synthesize(body: dict) -> dict:
    name = body.get("candidate")
    if name not in CANDIDATES:
        raise ValueError("Unknown candidate")
    cand = CANDIDATES[name]
    lang = body.get("lang")
    if lang not in cand.languages:
        raise ValueError(f"{name} does not support {lang}")
    text = str(body.get("text", "")).strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text must be 1-{MAX_TEXT_CHARS} characters")
    device = body.get("device", {})
    cores = _clamp(device.get("cores", DEFAULTS.cores), 1, DEFAULTS.cores, int)
    profile = replace(
        DEFAULTS,
        cores=cores,
        background_busy_cores=_clamp(device.get("background", 0), 0, cores - 1, int),
        cpu_scale=_clamp(device.get("cpu_scale", 1.0), 1.0, 10.0),
    )
    sid = body.get("sid")
    return {
        "candidate": name,
        "profile": profile,
        "threads": _clamp(device.get("threads", 2), 1, cores, int),
        "synth": {
            "text": text,
            "lang": lang,
            "sid": None if sid in (None, "") else _clamp(sid, 0, 999, int),
            "speed": _clamp(body.get("speed", 1.0), 0.5, 2.0),
            "num_steps": _clamp(body.get("num_steps", 2), 1, 16, int) if cand.uses_lang_extra else None,
            "normalize": bool(body.get("normalize", True)),
            "max_chars": 80,
        },
    }


def state() -> dict:
    sentences = json.loads((HERE / "test_sentences.json").read_text(encoding="utf-8"))
    return {
        "candidates": [
            {
                "name": c.name,
                "languages": list(c.languages),
                "licence": c.licence,
                "downloaded": (MODELS_ROOT / c.dirname).is_dir(),
                "has_steps": c.uses_lang_extra,
            }
            for c in CANDIDATES.values()
        ],
        "sentences": sentences,
        "host": {
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
        },
        "defaults": {
            "cores": DEFAULTS.cores,
            "background": DEFAULTS.background_busy_cores,
            "cpu_scale": DEFAULTS.cpu_scale,
            "ram_gb": DEFAULTS.ram_gb,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload, content_type="application/json") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            raise ValueError("Request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path == "/":
            html = (HERE / "ui" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/api/synthesize":
                reply = CONTROLLER.synthesize(parse_synthesize(body))
                self._send(200 if reply["ok"] else 500, reply)
            elif self.path == "/api/download":
                name = body.get("candidate")
                if name not in CANDIDATES:
                    raise ValueError("Unknown candidate")
                with DOWNLOAD_LOCK:
                    ensure_model(CANDIDATES[name])
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # surface model or worker failures in the UI
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *_args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"SpeaKilo TTS lab on {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CONTROLLER.close()
        server.server_close()


if __name__ == "__main__":
    main()
