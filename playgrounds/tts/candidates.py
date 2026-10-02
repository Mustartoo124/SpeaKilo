"""Registry of TTS candidates (all run through sherpa-onnx) plus download and load helpers."""
from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
MODELS_ROOT = Path(__file__).parent / "models"


@dataclass(frozen=True)
class Candidate:
    name: str
    archive: str
    languages: tuple
    licence: str
    default_sid: dict = field(default_factory=dict)
    uses_lang_extra: bool = False

    @property
    def dirname(self) -> str:
        return self.archive.removesuffix(".tar.bz2")


CANDIDATES = {
    c.name: c
    for c in (
        Candidate(
            name="supertonic3",
            archive="sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
            languages=("en", "ko", "vi"),
            licence="Weights OpenRAIL-M, code MIT: read use restrictions before shipping",
            uses_lang_extra=True,
        ),
        Candidate(
            name="kokoro_multi_v1_0",
            archive="kokoro-multi-lang-v1_0.tar.bz2",
            languages=("en", "zh"),
            licence="Apache-2.0 (no Korean or Vietnamese voices)",
            default_sid={"zh": 18},  # from the sherpa-onnx zh+en example
        ),
        Candidate(
            name="melo_zh_en",
            archive="vits-melo-tts-zh_en.tar.bz2",
            languages=("zh",),
            licence="MIT (MeloTTS); one speaker, English only for words in lexicon.txt",
        ),
        Candidate(
            name="aishell3_zh",
            archive="vits-icefall-zh-aishell3.tar.bz2",
            languages=("zh",),
            licence="TODO: confirm AISHELL-3 / icefall licence",
        ),
    )
}


def _supertonic(d: Path, threads: int):
    import sherpa_onnx as so

    return so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            supertonic=so.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin"),
            ),
            num_threads=threads,
            provider="cpu",
        )
    )


def _kokoro(d: Path, threads: int):
    import sherpa_onnx as so

    kw = dict(
        model=str(d / "model.onnx"),
        voices=str(d / "voices.bin"),
        tokens=str(d / "tokens.txt"),
        data_dir=str(d / "espeak-ng-data"),
        lexicon=",".join(str(d / f) for f in ("lexicon-us-en.txt", "lexicon-zh.txt")),
    )
    if (d / "dict").is_dir():
        kw["dict_dir"] = str(d / "dict")
    try:
        kokoro = so.OfflineTtsKokoroModelConfig(**kw)
    except TypeError:  # older sherpa-onnx without dict_dir
        kw.pop("dict_dir", None)
        kokoro = so.OfflineTtsKokoroModelConfig(**kw)
    return so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(kokoro=kokoro, num_threads=threads, provider="cpu")
    )


def _vits_zh(rule_fsts: tuple):
    def build(d: Path, threads: int):
        import sherpa_onnx as so

        return so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(
                    model=str(d / "model.onnx"),
                    lexicon=str(d / "lexicon.txt"),
                    tokens=str(d / "tokens.txt"),
                ),
                num_threads=threads,
                provider="cpu",
            ),
            rule_fsts=",".join(str(d / f) for f in rule_fsts),
        )

    return build


_BUILDERS = {
    "supertonic3": _supertonic,
    "kokoro_multi_v1_0": _kokoro,
    "melo_zh_en": _vits_zh(("date.fst", "number.fst")),
    "aishell3_zh": _vits_zh(("phone.fst", "date.fst", "number.fst")),
}


def ensure_model(cand: Candidate, root: Path = MODELS_ROOT) -> Path:
    """Download and extract the candidate archive if it is not already present."""
    target = root / cand.dirname
    if target.is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / cand.archive
    urllib.request.urlretrieve(f"{RELEASE_URL}/{cand.archive}", archive)
    with tarfile.open(archive, "r:bz2") as tar:
        base = root.resolve()
        for member in tar.getmembers():
            if not (root / member.name).resolve().is_relative_to(base):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(root)
    archive.unlink()
    return target


def load_tts(name: str, threads: int = 2, root: Path = MODELS_ROOT):
    """Build a sherpa-onnx OfflineTts; the model must already be downloaded."""
    import sherpa_onnx as so

    cand = CANDIDATES[name]
    config = _BUILDERS[name](root / cand.dirname, threads)
    if not config.validate():
        raise ValueError(f"Invalid config for {name}; run download_models.py first")
    return so.OfflineTts(config)


def synth(tts, cand: Candidate, text: str, lang: str, sid=None, speed=1.0, num_steps=None):
    """Return (float32 samples, sample_rate) for one chunk of text."""
    import numpy as np
    import sherpa_onnx as so

    gen = so.GenerationConfig()
    gen.sid = cand.default_sid.get(lang, 0) if sid is None else sid
    gen.speed = speed
    if num_steps is not None:
        gen.num_steps = num_steps
    if cand.uses_lang_extra:
        gen.extra["lang"] = lang
    audio = tts.generate(text, gen)
    return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate
