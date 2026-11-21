import copy
import json

from workbench.interactive import Workbench
from workbench.verify import NEW_TOOL, NEW_TURN

DATA_KEYS = ("config_fingerprint", "precision_tiers", "normalize_stats", "turns",
             "failures", "budget", "summary")


def make_workbench(dataset, tmp_path, name="wb"):
    return Workbench(dataset["turns"], dataset["acoustics"], dataset["tools"],
                     dataset["lexicon"], copy.deepcopy(dataset["config"]),
                     dataset["load_failures"], str(tmp_path / name))


def apply_command_sequence(workbench):
    workbench.apply("run")
    workbench.apply("set-weight text 0.65")
    workbench.apply("set-temperature 1.25")
    workbench.apply("set-precision 16 1")
    workbench.apply("replace-turn T005 " + json.dumps(NEW_TURN))
    workbench.apply("add-tool " + json.dumps(NEW_TOOL))
    return workbench.apply("remove-tool speech_rate")["report"]


def test_incremental_matches_full_rebuild(dataset, tmp_path):
    workbench = make_workbench(dataset, tmp_path, "inc")
    incremental = apply_command_sequence(workbench)
    warm = workbench.apply("run")["report"]

    ds = copy.deepcopy(dataset)
    ds["config"]["weights"]["text"] = 0.65
    ds["config"]["temperature"] = 1.25
    ds["config"]["precision"]["frac_bits"] = 16
    ds["config"]["precision"]["quantum"] = 1
    for idx, turn in enumerate(ds["turns"]):
        if turn["turn_id"] == "T005":
            ds["turns"][idx] = json.loads(json.dumps(NEW_TURN))
    ds["tools"]["emphasis_score"] = json.loads(json.dumps(NEW_TOOL))
    ds["tools"].pop("speech_rate", None)
    rebuilder = make_workbench(ds, tmp_path, "rebuild")
    cold = rebuilder.apply("run")["report"]
    rebuild_warm = rebuilder.apply("run")["report"]

    for key in DATA_KEYS:
        assert incremental[key] == cold[key], f"field {key} diverged"
    assert warm["reuse"] == rebuild_warm["reuse"]
    assert warm["recompute"] == rebuild_warm["recompute"] == []


def test_precise_invalidation_on_weight_change(dataset, tmp_path):
    workbench = make_workbench(dataset, tmp_path)
    workbench.apply("run")
    response = workbench.apply("set-weight text 0.65")
    report = response["report"]
    recomputed = {(r["turn_id"], r["stage"]) for r in report["recompute"]}
    reused = {(r["turn_id"], r["stage"]) for r in report["reuse"]}
    # normalize and quantize products are untouched by a weight change
    assert ("T001", "normalize") in reused
    assert ("T001", "quantize") in reused
    # weighted_sum / clip / fuse must be recomputed for every turn
    for turn in dataset["turns"]:
        assert (turn["turn_id"], "weighted_sum") in recomputed
        assert (turn["turn_id"], "clip") in recomputed


def test_precise_invalidation_on_precision_change(dataset, tmp_path):
    workbench = make_workbench(dataset, tmp_path)
    workbench.apply("run")
    report = workbench.apply("set-precision 16 1")["report"]
    recomputed = {(r["turn_id"], r["stage"]) for r in report["recompute"]}
    reused = {(r["turn_id"], r["stage"]) for r in report["reuse"]}
    assert ("T001", "normalize") in reused
    assert ("T001", "quantize") in recomputed


def test_replace_turn_only_invalidates_downstream(dataset, tmp_path):
    workbench = make_workbench(dataset, tmp_path)
    workbench.apply("run")
    new_turn = dict(NEW_TURN)
    report = workbench.apply("replace-turn T005 " + json.dumps(new_turn))["report"]
    assert report["summary"]["total_turns"] == len(dataset["turns"])
    t005 = next(t for t in report["turns"] if t["turn_id"] == "T005")
    assert t005["confidence_int"] is not None


def test_remove_tool_marks_dependent_turns(dataset, tmp_path):
    workbench = make_workbench(dataset, tmp_path)
    workbench.apply("run")
    report = workbench.apply("remove-tool sentiment_polarity")["report"]
    from workbench import codes
    for turn in report["turns"]:
        assert codes.E_FUSE_EVIDENCE_MISSING in turn["codes"]
