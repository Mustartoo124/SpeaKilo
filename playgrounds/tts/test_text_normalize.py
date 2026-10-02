"""Safety-term tests: normalisation must keep negations, numbers, units and equipment IDs audible."""
import re

import pytest

from text_normalize import chunk_text, load_lexicon, normalize

NEGATIONS = {
    "en": "Do not open the guard.",
    "ko": "보호 덮개를 열지 마세요.",
    "vi": "Không được mở nắp bảo vệ.",
    "zh": "不要打开防护罩。",
}


@pytest.mark.parametrize("lang", NEGATIONS)
def test_negation_text_is_untouched(lang):
    assert normalize(NEGATIONS[lang], lang) == NEGATIONS[lang]


def test_units_keep_their_numbers():
    assert normalize("Set 5 bar", "en") == "Set 5 bar"
    assert normalize("Limit 70 °C", "en") == "Limit 70 degrees Celsius"
    assert normalize("24V supply", "en") == "24 volts supply"
    assert normalize("温度50%", "zh") == "温度百分之50"
    assert normalize("온도 85 °C이며", "ko") == "온도 섭씨 85도이며"
    assert normalize("Đặt 5 bar", "vi") == "Đặt 5 ba"


def test_decimal_numbers_survive():
    assert normalize("pressure 3.5 bar", "en") == "pressure 3.5 bar"


def test_equipment_id_is_spelled_character_by_character():
    assert normalize("check valve M3-205B", "en") == "check valve M 3 2 0 5 B"
    assert normalize("밸브 M3-205B를 확인", "ko") == "밸브 M 3 2 0 5 B를 확인"


def test_id_with_embedded_unit_is_not_split_as_a_unit():
    assert normalize("M3-2m", "en") == "M 3 2 m"


def test_lexicon_replaces_whole_terms_only(tmp_path):
    path = tmp_path / "lex.txt"
    path.write_text("# comment\nPLC\tprogrammable logic controller\n", encoding="utf-8")
    lexicon = load_lexicon(path)
    assert normalize("Reset the PLC now", "en", lexicon) == "Reset the programmable logic controller now"
    assert normalize("PLCX stays", "en", lexicon) == "PLCX stays"


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        normalize("hello", "fr")


@pytest.mark.parametrize(
    "text",
    [
        "Stop the line. Do not open the guard.",
        "Pressure is 3.5 bar, which is within limits. Continue.",
        "停止生产线。不要打开防护罩。",
    ],
)
def test_chunking_loses_no_content(text):
    chunks = chunk_text(text, max_chars=40)
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", text)


def test_decimal_point_is_not_a_sentence_boundary():
    assert chunk_text("Set pressure to 3.5 bar now.") == ["Set pressure to 3.5 bar now."]


def test_first_chunk_is_short_for_low_ttfa():
    chunks = chunk_text("Stop the line. Then check every valve on the manifold before restart.")
    assert chunks[0] == "Stop the line."
