"""Data schema for TTS benchmarking: test set, run records, per-utterance results, ASR checks, ratings."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "1"
TARGET_HARDWARE = "rpi5-8gb"
DEFAULT_TESTSET = Path(__file__).parent / "data" / "testset_v1.jsonl"

LANGS = ("en", "ko", "vi", "zh")
CATEGORIES = ("warning", "instruction", "status")
TAGS = ("negation", "unit", "equipment_id", "number", "code_switch", "short", "long")
SPLITS = ("dev", "heldout")
EVIDENCE = ("PUBLISHED", "ESTIMATE", "TARGET", "MEASURED")
METRICS = ("wer", "cer")
ERROR_TYPES = (
    "",
    "unit_misread",
    "id_misread",
    "number_misread",
    "negation_lost",
    "word_skipped",
    "word_repeated",
    "wrong_language",
    "artifact",
    "other",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _alnum(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


@dataclass(frozen=True)
class Preserve:
    """A token that must still be audible after synthesis and re-transcription."""

    term: str
    accept: tuple = ()  # other spoken or written forms an ASR may return

    def __post_init__(self):
        object.__setattr__(self, "accept", tuple(self.accept))
        _require(
            self.term.isdigit() or len(_alnum(self.term)) >= 2,
            f"term too short to check reliably: {self.term!r}",
        )


@dataclass(frozen=True)
class BenchUtterance:
    id: str
    lang: str
    text: str
    category: str
    tags: tuple
    must_preserve: tuple
    negation: bool
    source: str
    licence: str
    split: str
    version: str

    def __post_init__(self):
        _require(self.lang in LANGS, f"{self.id}: unknown lang {self.lang!r}")
        _require(re.fullmatch(rf"{self.lang}-\d{{4}}", self.id) is not None, f"bad id {self.id!r}")
        _require(bool(self.text.strip()), f"{self.id}: empty text")
        _require(self.category in CATEGORIES, f"{self.id}: unknown category {self.category!r}")
        _require(set(self.tags) <= set(TAGS), f"{self.id}: unknown tag in {self.tags}")
        _require(self.negation == ("negation" in self.tags), f"{self.id}: negation flag and tag disagree")
        _require(self.split in SPLITS, f"{self.id}: unknown split {self.split!r}")
        for spec in self.must_preserve:
            _require(spec.term in self.text, f"{self.id}: term {spec.term!r} is not in the text")

    @property
    def length_chars(self) -> int:
        return len(self.text)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchUtterance":
        data = dict(data)
        data["tags"] = tuple(data.get("tags", ()))
        data["must_preserve"] = tuple(Preserve(**p) for p in data.get("must_preserve", ()))
        return cls(**data)


@dataclass(frozen=True)
class RunRecord:
    """One benchmark sweep of one candidate on one device setup."""

    run_id: str
    created_utc: str
    hardware: str
    evidence: str
    candidate: str
    model_version: str
    threads: int
    background_load: int
    num_steps: int | None
    speed: float
    testset: str
    split: str
    reps: int
    git_commit: str | None
    platform: str
    machine: str
    lexicon: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _require(self.evidence in EVIDENCE, f"unknown evidence label {self.evidence!r}")
        _require(
            self.evidence != "MEASURED" or self.hardware == TARGET_HARDWARE,
            f"MEASURED requires hardware={TARGET_HARDWARE!r}, got {self.hardware!r}",
        )
        _require(self.threads >= 1 and self.reps >= 1, "threads and reps must be >= 1")
        _require(self.background_load >= 0, "background_load must be >= 0")

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        return cls(**data)


@dataclass(frozen=True)
class UtteranceResult:
    """One timed synthesis of one test utterance; rep 0 is a warm-up and has warm=False."""

    run_id: str
    utt_id: str
    lang: str
    rep: int
    warm: bool
    t_norm_ms: float
    ttfa_ms: float
    total_ms: float
    audio_s: float
    rtf: float
    n_chunks: int
    rss_mb: float
    temp_c: float | None = None
    throttled: bool | None = None
    wav_path: str | None = None

    def __post_init__(self):
        _require(self.lang in LANGS, f"unknown lang {self.lang!r}")
        _require(self.rep >= 0 and self.n_chunks >= 1, "rep must be >= 0 and n_chunks >= 1")
        _require(min(self.ttfa_ms, self.total_ms, self.audio_s, self.rtf) > 0, "timings must be positive")
        _require(self.ttfa_ms <= self.total_ms + 1e-6, "ttfa_ms cannot exceed total_ms")


@dataclass(frozen=True)
class AsrCheck:
    """Round-trip result: synthesised speech re-transcribed by ASR and scored."""

    candidate: str
    utt_id: str
    lang: str
    asr_model: str
    metric: str
    error: float
    spoken: str
    heard: str
    must_preserve_hit: tuple
    all_preserved: bool

    def __post_init__(self):
        _require(self.metric in METRICS, f"unknown metric {self.metric!r}")
        _require(self.error >= 0, "error must be >= 0")
        object.__setattr__(self, "must_preserve_hit", tuple(self.must_preserve_hit))

    @classmethod
    def from_dict(cls, data: dict) -> "AsrCheck":
        return cls(**data)


@dataclass(frozen=True)
class Rating:
    """One listener's score for one blinded sample; join to the answer key for the candidate."""

    sample_id: str
    utt_id: str
    rater: str
    mos: int | None = None
    intelligibility: int | None = None
    pronunciation_error: str = ""
    notes: str = ""

    def __post_init__(self):
        for name in ("mos", "intelligibility"):
            value = getattr(self, name)
            _require(value is None or 1 <= value <= 5, f"{name} must be 1-5 or empty")
        _require(self.pronunciation_error in ERROR_TYPES, f"unknown error type {self.pronunciation_error!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "Rating":
        return cls(**data)


def term_heard(spec: Preserve, heard: str) -> bool:
    """Weak check that a protected term appears in an ASR transcript (case, spacing, hyphens ignored)."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", heard.lower())
    for candidate in (spec.term, *spec.accept):
        if candidate.isdigit():
            if re.search(rf"(?<!\d){candidate}(?!\d)", text):
                return True
        elif _alnum(candidate) and _alnum(candidate) in _alnum(heard):
            return True
    return False


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, cls) -> list:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            rows.append(cls.from_dict(data) if hasattr(cls, "from_dict") else cls(**data))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def load_testset(path: str | Path = DEFAULT_TESTSET, langs=None, split: str | None = None) -> list:
    items = read_jsonl(path, BenchUtterance)
    ids = [item.id for item in items]
    _require(len(ids) == len(set(ids)), "duplicate utterance ids in test set")
    return [
        item
        for item in items
        if (langs is None or item.lang in langs) and (split is None or item.split == split)
    ]
