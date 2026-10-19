"""Checks that the benchmark data schema and the shipped test set stay consistent."""
import pytest

from benchmark_schema import (
    LANGS,
    TARGET_HARDWARE,
    AsrCheck,
    BenchUtterance,
    Preserve,
    Rating,
    RunRecord,
    UtteranceResult,
    load_testset,
    read_jsonl,
    term_heard,
    write_jsonl,
)


def _run(**overrides):
    base = dict(
        run_id="r1", created_utc="2026-10-19T00:00:00+00:00", hardware=TARGET_HARDWARE,
        evidence="MEASURED", candidate="supertonic3", model_version="m", threads=2,
        background_load=0, num_steps=2, speed=1.0, testset="testset_v1.jsonl", split="heldout",
        reps=5, git_commit=None, platform="p", machine="aarch64",
    )
    return RunRecord(**{**base, **overrides})


def test_shipped_testset_loads_and_covers_every_language():
    items = load_testset()
    for lang in LANGS:
        assert sum(1 for i in items if i.lang == lang) >= 6


def test_every_language_has_negation_unit_and_id_cases():
    items = load_testset()
    for lang in LANGS:
        tags = {t for i in items if i.lang == lang for t in i.tags}
        assert {"negation", "unit", "equipment_id", "number"} <= tags


def test_term_not_in_text_is_rejected():
    with pytest.raises(ValueError):
        BenchUtterance.from_dict({
            "id": "en-0099", "lang": "en", "text": "Stop.", "category": "warning",
            "tags": [], "must_preserve": [{"term": "valve"}], "negation": False,
            "source": "authored", "licence": "own", "split": "dev", "version": "v1",
        })


def test_negation_flag_must_match_tag():
    with pytest.raises(ValueError):
        BenchUtterance.from_dict({
            "id": "en-0099", "lang": "en", "text": "Do not stop.", "category": "warning",
            "tags": [], "must_preserve": [], "negation": True,
            "source": "authored", "licence": "own", "split": "dev", "version": "v1",
        })


def test_measured_evidence_requires_target_hardware():
    assert _run().evidence == "MEASURED"
    with pytest.raises(ValueError):
        _run(hardware="dev-pc")


def test_result_rejects_ttfa_above_total():
    with pytest.raises(ValueError):
        UtteranceResult(run_id="r", utt_id="en-0001", lang="en", rep=1, warm=True, t_norm_ms=1,
                        ttfa_ms=900, total_ms=500, audio_s=2, rtf=0.25, n_chunks=1, rss_mb=100)


def test_rating_validates_scale_and_error_type():
    Rating(sample_id="s000", utt_id="en-0001", rater="r1", mos=4, pronunciation_error="unit_misread")
    with pytest.raises(ValueError):
        Rating(sample_id="s000", utt_id="en-0001", rater="r1", mos=6)
    with pytest.raises(ValueError):
        Rating(sample_id="s000", utt_id="en-0001", rater="r1", pronunciation_error="bad")


def test_short_non_numeric_terms_are_rejected():
    with pytest.raises(ValueError):
        Preserve(term="°C")


def test_term_heard_handles_spacing_case_and_digits():
    assert term_heard(Preserve("M3-205B"), "check valve M 3 2 0 5 B now")
    assert term_heard(Preserve("1200"), "run at 1,200 rpm")
    assert not term_heard(Preserve("5"), "set it to 85")
    assert term_heard(Preserve("5", ("five",)), "set it to five bar")
    assert not term_heard(Preserve("not"), "open the guard")


def test_jsonl_round_trip(tmp_path):
    check = AsrCheck(candidate="c", utt_id="en-0001", lang="en", asr_model="w", metric="wer",
                     error=0.1, spoken="a", heard="a", must_preserve_hit=(True,), all_preserved=True)
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [check])
    assert read_jsonl(path, AsrCheck) == [check]
