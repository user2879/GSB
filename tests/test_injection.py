import os

from workbench import codes, injection
from workbench.cache import StageCache
from workbench.pipeline import Pipeline

INJECTION_PATH = os.path.join(os.path.dirname(__file__), "..", "config",
                              "injection.yaml")


def run_injected(dataset):
    injections = injection.load_injections(INJECTION_PATH)
    turns, acoustics, tools, config, hits, config_failures = \
        injection.apply_injections(dataset["turns"], dataset["acoustics"],
                                   dataset["tools"], dataset["config"], injections)
    report = Pipeline(turns, acoustics, tools, dataset["lexicon"], config,
                      cache=StageCache()).run()
    return hits, config_failures, report


def test_four_injection_ids_recorded(dataset):
    hits, _, _ = run_injected(dataset)
    ids = [h["id"] for h in hits]
    assert ids == ["INJ-ACOUSTIC-NONFINITE-001", "INJ-TOOL-DIMENSION-002",
                   "INJ-PRECISION-RANGE-003", "INJ-WEIGHTS-ZERO-004"]


def test_each_injection_produces_business_code(dataset):
    _, config_failures, report = run_injected(dataset)
    failure_codes = {f["code"] for f in config_failures + report["failures"]}
    turn_codes = {c for t in report["turns"] for c in t["codes"]}
    assert codes.E_LOAD_ACOUSTIC_NONFINITE in failure_codes
    assert codes.E_FUSE_DIM_INCOMPAT in failure_codes | turn_codes
    assert codes.E_LOAD_PRECISION in failure_codes
    assert codes.E_FUSE_WEIGHT_SUM_ZERO in turn_codes


def test_injection_ids_attached_to_failures(dataset):
    _, config_failures, report = run_injected(dataset)
    all_failures = config_failures + report["failures"]
    injected = [f for f in all_failures if "injection_id" in f]
    assert any(f["injection_id"] == "INJ-ACOUSTIC-NONFINITE-001" for f in injected)
    assert any(f["injection_id"] == "INJ-PRECISION-RANGE-003" for f in injected)


def test_single_turn_fault_isolated(dataset):
    # only the nonfinite-acoustic injection; other turns must still complete
    injections = [{"id": "INJ-X", "type": "nonfinite_acoustic",
                   "target_turn": "T007", "field": "pitch_mean"}]
    turns, acoustics, tools, config, hits, config_failures = \
        injection.apply_injections(dataset["turns"], dataset["acoustics"],
                                   dataset["tools"], dataset["config"], injections)
    report = Pipeline(turns, acoustics, tools, dataset["lexicon"], config,
                      cache=StageCache()).run()
    t007 = next(t for t in report["turns"] if t["turn_id"] == "T007")
    assert codes.E_LOAD_ACOUSTIC_NONFINITE in t007["codes"]
    others = [t for t in report["turns"] if t["turn_id"] != "T007"]
    assert all(t["confidence_int"] is not None for t in others)
    failure = next(f for f in report["failures"] if f["turn_id"] == "T007")
    assert failure["stage"] == "score_acoustic"
    assert failure["injection_id"] == "INJ-X"
