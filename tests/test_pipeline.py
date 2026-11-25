"""Requirements 4/6/8 integration: cross-parallelism determinism, invalidation
propagation with byte-identical unaffected windows, end-to-end run."""

import copy

from obs_pipeline import gen
from obs_pipeline.ledger import Ledger
from obs_pipeline.pipeline import (
    commit_report,
    compute_invalidation,
    content_digest,
    run_pipeline,
)
from obs_pipeline.spec import validate_spec


def _records(spec):
    return {s.id: gen.generate_source_records(s) for s in spec.sources}


def test_parallelism_and_batch_order_bit_identical(tmp_path, sample_spec):
    records = _records(sample_spec)
    ledger_a = Ledger(str(tmp_path / "a.sqlite"))
    ledger_b = Ledger(str(tmp_path / "b.sqlite"))
    run_a = run_pipeline(sample_spec, ledger_a, records,
                         global_budget=10 ** 6, workers=1)
    run_b = run_pipeline(sample_spec, ledger_b, records,
                         global_budget=60, stage_quotas={s.id: 30 for s in sample_spec.stages},
                         workers=8)
    for key in run_a.stage_results:
        assert run_a.stage_results[key].summary == run_b.stage_results[key].summary
        assert run_a.stage_results[key].outputs == run_b.stage_results[key].outputs
    for key in run_a.metric_stats:
        assert run_a.metric_stats[key].canonical() == run_b.metric_stats[key].canonical()
    ledger_a.close()
    ledger_b.close()


def test_invalidation_recomputes_only_affected(tmp_path, sample_spec_dict):
    spec = validate_spec(sample_spec_dict)
    records = _records(spec)
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    base = run_pipeline(spec, ledger, records)
    commit_report(ledger, base)

    evolved_dict = copy.deepcopy(sample_spec_dict)
    evolved_dict["spec_version"] = 2
    evolved_dict["stages"][1]["params"]["window"] = 6  # s_avg param edit
    evolved = validate_spec(evolved_dict)
    invalidation = compute_invalidation(spec, evolved)
    assert "s_avg" in invalidation.affected_stages
    assert "s_bucket" in invalidation.affected_stages  # downstream closure
    assert "s_norm" in invalidation.affected_stages
    assert "s_diff" not in invalidation.affected_stages  # upstream untouched
    assert invalidation.basis == tuple(sorted(invalidation.basis))

    rerun = run_pipeline(evolved, ledger, records,
                         affected=invalidation.as_sets(),
                         prior_results=base.stage_results,
                         prior_stats=base.metric_stats)
    # Unaffected stage summaries are byte-identical.
    for window in evolved.windows:
        key = ("s_diff", window.id)
        assert rerun.stage_results[key].summary == base.stage_results[key].summary
    # Affected stage summaries actually changed.
    changed = [
        key for key in rerun.stage_results
        if key[0] != "s_diff"
        and rerun.stage_results[key].summary != base.stage_results[key].summary
    ]
    assert changed, "param edit must change downstream summaries"
    ledger.close()


def test_window_split_keeps_unaffected_windows(tmp_path, sample_spec_dict):
    spec = validate_spec(sample_spec_dict)
    records = _records(spec)
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    base = run_pipeline(spec, ledger, records)

    evolved_dict = copy.deepcopy(sample_spec_dict)
    evolved_dict["spec_version"] = 2
    last = evolved_dict["windows"][-1]
    mid = (last["start"] + last["end"]) // 2
    evolved_dict["windows"][-1] = {"id": "w4a", "start": last["start"], "end": mid}
    evolved_dict["windows"].append(
        {"id": "w4b", "start": mid, "end": last["end"]})
    evolved = validate_spec(evolved_dict)
    invalidation = compute_invalidation(spec, evolved)
    assert set(invalidation.affected_windows) == {"w4a", "w4b"}

    rerun = run_pipeline(evolved, ledger, records,
                         affected=invalidation.as_sets(),
                         prior_results=base.stage_results,
                         prior_stats=base.metric_stats)
    for stage in evolved.stages:
        for window in evolved.windows[:4]:
            key = (stage.id, window.id)
            assert rerun.stage_results[key].summary == base.stage_results[key].summary
    ledger.close()


def test_drift_verdicts_have_basis_and_cause(tmp_path, sample_spec):
    records = _records(sample_spec)
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    report = run_pipeline(sample_spec, ledger, records)
    assert report.drift_verdicts
    for verdict in report.drift_verdicts:
        assert verdict.cause in (
            "certain-exceeded", "certain-not-exceeded", "error-indeterminate")
        assert "order=obs_seq-ascending" in verdict.basis
        if verdict.cause == "error-indeterminate":
            assert verdict.error_affected
    ledger.close()


def test_null_counts_reported_per_stage(tmp_path, sample_spec):
    records = _records(sample_spec)
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    report = run_pipeline(sample_spec, ledger, records)
    total_nulls = sum(
        1 for recs in records.values() for r in recs if r.value_fixed is None)
    assert total_nulls > 0, "sample must contain nulls"
    diff_nulls = sum(
        report.stage_results[("s_diff", w.id)].null_count for w in sample_spec.windows)
    assert diff_nulls > 0
    for key, result in report.stage_results.items():
        assert result.null_rule
    ledger.close()
