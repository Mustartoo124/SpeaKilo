"""Spoken-form normalisation (units, equipment IDs, domain lexicon) and clause chunking for TTS."""
from __future__ import annotations

import re
from pathlib import Path

# unit -> spoken form; "{n}" templates are used where the number sits inside the phrase.
_PLAIN = {
    "en": {
        "mm": "millimeters", "cm": "centimeters", "m": "meters", "kg": "kilograms",
        "g": "grams", "V": "volts", "kV": "kilovolts", "mA": "milliamps", "W": "watts",
        "kW": "kilowatts", "Hz": "hertz", "kHz": "kilohertz", "bar": "bar",
        "kPa": "kilopascals", "MPa": "megapascals", "psi": "psi",
        "rpm": "revolutions per minute",
    },
    "ko": {
        "mm": "밀리미터", "cm": "센티미터", "m": "미터", "kg": "킬로그램", "g": "그램",
        "V": "볼트", "kV": "킬로볼트", "mA": "밀리암페어", "W": "와트", "kW": "킬로와트",
        "Hz": "헤르츠", "kHz": "킬로헤르츠", "bar": "바", "kPa": "킬로파스칼",
        "MPa": "메가파스칼", "psi": "피에스아이", "rpm": "알피엠",
    },
    "vi": {
        "mm": "milimét", "cm": "xăng-ti-mét", "m": "mét", "kg": "ki-lô-gam", "g": "gam",
        "V": "vôn", "kV": "ki-lô-vôn", "mA": "mi-li-am-pe", "W": "oát", "kW": "ki-lô-oát",
        "Hz": "héc", "kHz": "ki-lô-héc", "bar": "ba", "kPa": "ki-lô-pascal",
        "MPa": "mê-ga-pascal", "psi": "pi-ét-ai", "rpm": "vòng trên phút",
    },
    "zh": {
        "mm": "毫米", "cm": "厘米", "m": "米", "kg": "千克", "g": "克", "V": "伏特",
        "kV": "千伏", "mA": "毫安", "W": "瓦", "kW": "千瓦", "Hz": "赫兹", "kHz": "千赫兹",
        "bar": "巴", "kPa": "千帕", "MPa": "兆帕", "psi": "磅力每平方英寸", "rpm": "转每分钟",
    },
}
_TEMPLATES = {
    "en": {"°C": "{n} degrees Celsius", "°F": "{n} degrees Fahrenheit", "%": "{n} percent"},
    "ko": {"°C": "섭씨 {n}도", "°F": "화씨 {n}도", "%": "{n} 퍼센트"},
    "vi": {"°C": "{n} độ C", "°F": "{n} độ F", "%": "{n} phần trăm"},
    "zh": {"°C": "{n}摄氏度", "°F": "{n}华氏度", "%": "百分之{n}"},
}

_UNIT_KEYS = sorted({*_TEMPLATES["en"], *_PLAIN["en"]}, key=len, reverse=True)
_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?<![A-Za-z0-9]-)(-?\d+(?:[.,]\d+)?)\s*("
    + "|".join(re.escape(u) for u in _UNIT_KEYS)
    + r")(?![A-Za-z0-9])"
)
# Tokens mixing letters and digits (e.g. M3-205B); ASCII-only lookarounds so Hangul/Han particles may follow.
_ID_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9-])"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；\n])\s*")
_CLAUSE_SPLIT = re.compile(r"(?<=[,:])\s+|(?<=[，、：])\s*")


def load_lexicon(path: str | Path) -> dict:
    """Read 'term<TAB>spoken form' lines; '#' starts a comment."""
    lexicon = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "\t" in line:
            term, spoken = line.split("\t", 1)
            lexicon[term.strip()] = spoken.strip()
    return lexicon


def _apply_lexicon(text: str, lexicon: dict) -> str:
    for term in sorted(lexicon, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        text = re.sub(pattern, lexicon[term], text)
    return text


def _expand_units(text: str, lang: str) -> str:
    templates, plain = _TEMPLATES[lang], _PLAIN[lang]

    def repl(match: re.Match) -> str:
        number, unit = match.groups()
        if unit in templates:
            return templates[unit].format(n=number)
        return f"{number} {plain[unit]}"

    return _UNIT_RE.sub(repl, text)


def _spell_ids(text: str) -> str:
    return _ID_RE.sub(lambda m: " ".join(re.findall(r"[A-Za-z]|\d", m.group())), text)


def normalize(text: str, lang: str, lexicon: dict | None = None) -> str:
    """Rewrite units and equipment IDs into forms a TTS front-end reads unambiguously."""
    if lang not in _PLAIN:
        raise ValueError(f"Unsupported language: {lang}")
    if lexicon:
        text = _apply_lexicon(text, lexicon)
    return _spell_ids(_expand_units(text, lang))


def _pack(sentence: str, max_chars: int) -> list:
    out, current = [], ""
    for part in filter(None, _CLAUSE_SPLIT.split(sentence)):
        if current and len(current) + 1 + len(part) > max_chars:
            out.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        out.append(current)
    return out


def chunk_text(text: str, max_chars: int = 80, min_chars: int = 12) -> list:
    """Split into sentence/clause chunks; the first chunk stays short to cut time-to-first-audio."""
    pieces = []
    for sentence in filter(None, (p.strip() for p in _SENTENCE_SPLIT.split(text))):
        pieces.extend(_pack(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    chunks, carry = [], ""
    for piece in pieces:
        piece = f"{carry} {piece}".strip()
        if len(piece) < min_chars:
            carry = piece
        else:
            chunks.append(piece)
            carry = ""
    if carry:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {carry}"
        else:
            chunks.append(carry)
    return chunks
