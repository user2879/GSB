"""Acceptance tests for the RoPE extrapolation diagnostic component."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rope_diag import scheduler as sched
from rope_diag.error_trace import trace_sample
from rope_diag.pipeline import Pipeline
from rope_diag.pruning import select_policy
from rope_diag.report import COMPARE_SECTIONS, canonical
from rope_diag.samples import (
    apply_edits,
    build_config,
    build_injections,
    build_policy_table,
    build_samples,
    build_edits,
)
from rope_diag.schemes import (
    SCHEMES,
    SchemeFault,
    SchemeParams,
    apply_scheme,
    boundary_scan,
    position_angles,
)
from rope_diag.scoring import compute_scores, stratified_stats
from rope_diag.sensitivity import diagnose


@pytest.fixture(scope="module")
def inputs():
    config = build_config()
    samples = build_samples(config)
    return {
        "config": config,
        "samples": samples,
        "edits": build_edits(samples),
        "injections": build_injections(samples),
        "policy_table": build_policy_table(config),
    }


def _params(config, scheme):
    return SchemeParams(
        scheme=scheme,
        base=config["base"],
        scale=config["scale"],
        dim=config["dim"],
        train_len=config["train_len"],
        target_len=config["target_len"],
    )


# ---------------------------------------------------------------- requirement 1
def test_three_schemes_unified_interface_and_reproducibility(inputs):
    config, sample = inputs["config"], inputs["samples"][0]
    for scheme in SCHEMES:
        params = _params(config, scheme)
        out1 = apply_scheme(params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"])
        out2 = apply_scheme(params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"])
        assert out1.scheme == scheme
        assert out1.param_fingerprint == params.fingerprint
        for a, b in ((out1.rotated_query, out2.rotated_query), (out1.rotated_keys, out2.rotated_keys)):
            np.testing.assert_array_equal(a, b)  # element-wise reproducible


def test_scheme_angle_rules(inputs):
    config = inputs["config"]
    pos = np.array([100.0, 2000.0])
    standard = position_angles(pos, _params(config, "standard"))
    linear = position_angles(pos, _params(config, "linear"))
    ntk = position_angles(pos, _params(config, "ntk"))
    np.testing.assert_allclose(linear, standard / config["scale"])
    assert not np.allclose(ntk, standard)  # NTK alters angle growth via base


# ---------------------------------------------------------------- requirement 2
def test_stratified_scoring(inputs):
    config, sample = inputs["config"], inputs["samples"][0]
    params = _params(config, "standard")
    out = apply_scheme(params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"])
    scores = compute_scores(out.rotated_query, out.rotated_keys)
    stats = stratified_stats(scores, sample["query_pos"], sample["key_positions"], config["train_len"])
    assert set(stats) == {"in_train", "extrapolation"}
    assert stats["extrapolation"], "expected extrapolation distances"
    for stratum, bound in (("in_train", config["train_len"]),):
        assert all(int(d) <= bound for d in stats[stratum])
    assert all(int(d) > config["train_len"] for d in stats["extrapolation"])
    for entry in stats["extrapolation"].values():
        assert {"mean", "min", "max", "count"} <= set(entry)


# ---------------------------------------------------------------- requirement 3
def test_error_trace_deterministic_and_staged(inputs):
    config, sample = inputs["config"], inputs["samples"][0]
    params = _params(config, "linear")
    out = apply_scheme(params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"])
    scores = compute_scores(out.rotated_query, out.rotated_keys)
    args = (params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"],
            out.rotated_query, out.rotated_keys, scores, config["error_tolerance"])
    t1 = trace_sample(*args)
    t2 = trace_sample(*args)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    assert t1["n_extrapolation_keys"] > 0
    for rec in t1["records"]:
        assert set(rec["deviations"]) == {"position_vector", "score", "weight"}
        assert rec["first_exceeded_stage"] in (None, "position_vector", "score", "weight")


# ---------------------------------------------------------------- requirement 4
def test_boundary_sensitivity_marks_unusable(inputs):
    config, samples = inputs["config"], inputs["samples"]
    diag = diagnose(config, samples[:3], (0, config["dim"]), 16)
    for scheme in SCHEMES:
        tiers = diag[scheme]["unusable_tiers"]
        assert any("E_DIM_TRUNCATION" in t["reasons"] for t in tiers), scheme
        ranking = diag[scheme]["sensitivity_ranking"]
        assert ranking == sorted(ranking, key=lambda r: -r["sensitivity"])
        # boundary-sensitive samples recorded with trigger conditions
        for entry in diag[scheme]["entries"]:
            if entry["unusable"]:
                assert entry["boundary_samples"]
                assert all("trigger" in b for b in entry["boundary_samples"])


def test_scheme_fault_codes():
    base = dict(scheme="standard", base=10000.0, scale=4.0, dim=64, train_len=512, target_len=2048)
    flags = boundary_scan(SchemeParams(**{**base, "base": 0.0}))
    assert flags["unusable"] and "E_BASE_ZERO" in flags["reasons"]
    flags = boundary_scan(SchemeParams(**{**base, "scale": -1.0}))
    assert flags["unusable"] and "E_SCALE_NONPOSITIVE" in flags["reasons"]
    with pytest.raises(SchemeFault) as exc:
        apply_scheme(
            SchemeParams(**{**base, "base": 0.0}),
            np.zeros(64), np.zeros((2, 64)), 10, np.array([1, 2]),
        )
    assert exc.value.code == "E_BASE_ZERO"


# ---------------------------------------------------------------- requirement 5
def test_pruning_policy_changes_compute_path(inputs):
    config, sample = inputs["config"], inputs["samples"][0]
    table = inputs["policy_table"]
    params = _params(config, "standard")
    out = apply_scheme(params, sample["query"], sample["keys"], sample["query_pos"], sample["key_positions"])
    results = {}
    for policy in table["policies"]:
        scores = compute_scores(out.rotated_query, out.rotated_keys,
                                tuple(policy["dim_range"]), int(policy["score_chunk"]))
        results[policy["policy_id"]] = scores
    assert not np.allclose(results["P_FULL_16"], results["P_PRUNE_32"])
    # chunked vs unchunked over the same range must agree
    whole = compute_scores(out.rotated_query, out.rotated_keys, (0, 64), None)
    chunked = compute_scores(out.rotated_query, out.rotated_keys, (0, 64), 16)
    np.testing.assert_allclose(whole, chunked)
    chosen, record = select_policy(table, config, len(inputs["samples"]))
    assert record["selected_policy"] == chosen["policy_id"]
    assert record["n_samples_referenced"] == len(inputs["samples"])
    assert record["extrapolation_ratio"] == config["target_len"] / config["train_len"]
    assert record["candidate_scores"] and record["rationale"]


# ---------------------------------------------------------------- requirement 7
def test_budget_scheduler_pending_and_order():
    units = [sched.ScheduledUnit("ntk", "S1", 30), sched.ScheduledUnit("linear", "S1", 30),
             sched.ScheduledUnit("linear", "S0", 25), sched.ScheduledUnit("ntk", "S0", 25)]
    plan = sched.schedule(units, point_budget=60)
    order = [(u.scheme, u.sample_id) for u in plan["admitted"]]
    assert order == sorted(order)  # stable (scheme, sample) consumption
    assert plan["spent_points"] <= 60
    assert plan["pending"] and all(p["code"] == "E_POINT_BUDGET_PENDING" for p in plan["pending"])


# ---------------------------------------------------------- requirements 6,7,8
def _run(pipe, config, samples, injections, policy, decision, mode):
    return pipe.run(config, samples, injections, policy, decision, mode=mode)


def test_pipeline_end_to_end(tmp_path, inputs):
    config, samples = inputs["config"], inputs["samples"]
    injections, table = inputs["injections"], inputs["policy_table"]
    policy, decision = select_policy(table, config, len(samples))

    pipe = Pipeline(tmp_path / "main")
    run1 = _run(pipe, config, samples, injections, policy, decision, "full")

    # serial vs concurrent identical
    serial_cfg = dict(config, concurrency=1)
    run_serial = _run(Pipeline(tmp_path / "serial"), serial_cfg, samples, injections, policy, decision, "full")
    assert canonical(run1, COMPARE_SECTIONS) == canonical(run_serial, COMPARE_SECTIONS)

    # repeat run reproducible
    run_rep = _run(Pipeline(tmp_path / "rep"), config, samples, injections, policy, decision, "full")
    assert canonical(run1, COMPARE_SECTIONS) == canonical(run_rep, COMPARE_SECTIONS)

    # edits -> incremental == fresh full
    config2, samples2 = apply_edits(config, samples, inputs["edits"])
    policy2, decision2 = select_policy(table, config2, len(samples2))
    run2 = _run(pipe, config2, samples2, injections, policy2, decision2, "incremental")
    run3 = _run(Pipeline(tmp_path / "full2"), config2, samples2, injections, policy2, decision2, "full")
    assert canonical(run2, COMPARE_SECTIONS) == canonical(run3, COMPARE_SECTIONS)
    assert len(run2["reused"]) > 0  # unaffected stages reused from snapshots

    # injection business codes, one per unique injection id
    codes = {(f["injection_id"], f["code"]) for f in run1["failures"]}
    assert ("INJ-001", "E_NONFINITE_FEATURE") in codes
    assert ("INJ-002", "E_KEY_POS_OOB") in codes
    assert ("INJ-003", "E_BASE_ZERO") in codes
    assert ("INJ-004", "E_SCALE_NONPOSITIVE") in codes

    # budget pending marked
    assert run1["budget"]["pending"]
    assert all(p["code"] == "E_POINT_BUDGET_PENDING" for p in run1["budget"]["pending"])

    # manifest audit fields
    manifest = json.loads((tmp_path / "main" / "snapshots" / "manifest.json").read_text())
    assert manifest["global_version"] == 2
    assert manifest["config_hash"] and manifest["injection_hash"]
    assert manifest["completed"]["schemes"] and manifest["completed"]["samples"]

    # single-sample replacement invalidates only that sample's downstream stages
    config3, samples3 = apply_edits(config, samples, {"commands": [
        {"op": "replace_sample", "sample_id": samples[0]["sample_id"],
         "sample": {**samples[0], "query": list(np.zeros(config["dim"]))}}]})
    run4 = _run(pipe, config3, samples3, injections, policy, decision, "incremental")
    recomputed_downstream = {
        (e["scheme"], e["sample_id"]) for e in run4["recomputed"] if e["stage"] != "load_validate"
    }
    assert recomputed_downstream and all(sid == samples[0]["sample_id"] for _, sid in recomputed_downstream)
