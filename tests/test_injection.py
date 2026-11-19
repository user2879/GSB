import json

from depedit.config import AppConfig, Injection
from depedit.injection import InjectionEngine
from depedit.pipeline import Pipeline
from depedit.relations import load_mapping


def _pipeline(sample_dir, injections):
    cfg = AppConfig()
    mapping = load_mapping(sample_dir / "relation_map.json")
    return Pipeline(cfg, mapping, InjectionEngine(injections)), cfg, mapping


def test_all_six_faults_isolated(sample_dir):
    injections = Injection.load_list(
        Path := __import__("pathlib").Path("config/injections.json")
    )
    pipeline, _, _ = _pipeline(sample_dir, injections)
    _, failures = pipeline.parse_v1(sample_dir / "parse_v1.txt")
    codes = {f["business_code"] for f in failures}
    # v1 解析路径覆盖定宽/越界/得分/关系四类，转换覆盖非有限数。
    assert {
        "E_FIXED_WIDTH_FIELD",
        "E_HEAD_OUT_OF_RANGE",
        "E_SCORE_RANGE",
        "E_RELATION_UNKNOWN",
    }.issubset(codes)
    assert all(f["inject_id"] for f in failures)


def test_non_finite_v2(sample_dir):
    injections = [
        Injection("INJ-X", "S001", "parse", "v2_non_finite_score", target_word=1)
    ]
    pipeline, _, _ = _pipeline(sample_dir, injections)
    _, failures = pipeline.parse_v2(sample_dir / "parse_v2.json")
    assert failures and failures[0]["business_code"] == "E_SCORE_NON_FINITE"
    assert failures[0]["inject_id"] == "INJ-X"


def test_cycle_in_edit_stage(sample_dir):
    injections = [Injection("INJ-C", "S001", "edit", "cycle_dependency")]
    pipeline, _, _ = _pipeline(sample_dir, injections)
    corpus, failures = pipeline.parse_v1(sample_dir / "parse_v1.txt")
    assert not failures
    inj = pipeline.injections.inject_edit_cycle(corpus.sentences["S001"])
    assert inj is not None
    from depedit.errors import ECycleDetected
    from depedit.validation import validate_sentence

    try:
        validate_sentence(corpus.sentences["S001"])
        assert False, "应当检测到环"
    except ECycleDetected:
        pass


def test_other_sentences_continue_after_failure(sample_dir):
    injections = [Injection("INJ-1", "S001", "parse", "score_over_1000")]
    pipeline, _, _ = _pipeline(sample_dir, injections)
    corpus, failures = pipeline.parse_v1(sample_dir / "parse_v1.txt")
    assert len(failures) == 1
    assert "S002" in corpus.sentences


def test_v1_faults_ignored_on_v2_channel(sample_dir):
    # v1 专用注入在 v2 解析路径不应抛出。
    v1_only = [
        Injection("I1", "S001", "parse", "fixed_width_missing"),
        Injection("I2", "S002", "parse", "score_over_1000"),
        Injection("I3", "S003", "parse", "relation_not_in_map"),
    ]
    pipeline, _, _ = _pipeline(sample_dir, v1_only)
    corpus, failures = pipeline.parse_v2(sample_dir / "parse_v2.json")
    assert failures == []
    assert len(corpus.sentences) == 8


def test_convert_guard_version_scoped():
    from depedit.errors import DepEditError

    engine = InjectionEngine(
        [Injection("I5", "S004", "convert", "v2_non_finite_score")]
    )
    # v1 通道放行，v2 通道命中。
    engine.convert_stage_guard("S004", "v1")
    try:
        engine.convert_stage_guard("S004", "v2")
        assert False, "v2 通道应命中非有限数注入"
    except DepEditError as exc:
        assert exc.code == "E_SCORE_NON_FINITE"
        assert exc.inject_id == "I5"
