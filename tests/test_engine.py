"""端到端业务码、策略冲突、真值对照与临界标记测试。"""

from __future__ import annotations

from audit_engine.budget import BudgetConfig
from audit_engine.errors import (
    E_BUDGET,
    E_DENY_SHADOW,
    E_INHERIT,
    E_LINK_ESCAPE,
    E_PARSE,
    E_POLICY_CONFLICT,
    E_TIMEOUT,
    E_TRAVERSAL,
    E_TRUTH_MISMATCH,
)
from audit_engine.parser import load_manifest_file
from audit_engine.engine import audit_file
from audit_engine.runner import (
    EXIT_BUDGET_OR_TIMEOUT,
    EXIT_OK,
    EXIT_VIOLATIONS,
    run_batch,
)
from audit_engine.truth import verify_record
from audit_engine.engine import audit_record


def _run_manifest(path, policies, budget=None):
    records, rejections, metadata = load_manifest_file(path)
    return audit_file(
        file_id=metadata["file_id"],
        path=path,
        root=metadata["root"],
        records=records,
        rejections=rejections,
        policies=policies,
        budget=budget or BudgetConfig(),
    )


def test_traversal_manifest(manifests, policies):
    result = _run_manifest(manifests["manifest_traversal"], policies)
    assert E_TRAVERSAL in result.codes
    traversal_verdicts = [v for v in result.verdicts if E_TRAVERSAL in v.codes]
    assert {v.traversal_depth for v in traversal_verdicts} == {2}


def test_link_escape_manifest(manifests, policies):
    result = _run_manifest(manifests["manifest_link_escape"], policies)
    assert E_LINK_ESCAPE in result.codes
    assert result.policy_conflicts


def test_representation_mix_critical_flag(manifests, policies):
    result = _run_manifest(manifests["manifest_representation_mix"], policies)
    critical = [v for v in result.verdicts if v.critical]
    assert critical
    peggy = [v for v in result.verdicts if v.record_id == "MIX-003"]
    assert all(v.critical for v in peggy)
    # 临界记录只标记不确定，不直接产出确定性越权码
    for verdict in peggy:
        assert E_TRAVERSAL not in verdict.codes
        assert verdict.perm_epsilon == 189 / 64


def test_deny_shadow_manifest(manifests, policies):
    result = _run_manifest(manifests["manifest_deny_shadow"], policies)
    assert E_DENY_SHADOW in result.codes


def test_inherit_manifest(manifests, policies):
    result = _run_manifest(manifests["manifest_inherit"], policies)
    assert E_INHERIT in result.codes


def test_policy_conflict_lists_policy_pairs(manifests, policies):
    result = _run_manifest(manifests["manifest_policy_conflict"], policies)
    assert E_POLICY_CONFLICT in result.codes
    involved = set()
    for conflict in result.policy_conflicts:
        involved.update(conflict["policies"])
    assert "strict" in involved and len(involved) >= 2


def test_parse_errors_manifest(manifests, policies):
    result = _run_manifest(manifests["manifest_parse_errors"], policies)
    assert E_PARSE in result.codes
    assert len(result.rejections) == 3
    assert result.processed_records == 2


def test_truth_consistency_counts(manifests, policies):
    result = _run_manifest(manifests["manifest_traversal"], policies)
    assert result.truth["consistent_count"] == 3
    assert result.truth["inconsistent_count"] == 0


def test_truth_mismatch_detected(manifests, policies):
    records, rejections, metadata = load_manifest_file(manifests["manifest_traversal"])
    record = records[0]
    verdicts = audit_record(record, metadata["root"], policies)
    tampered = type(record)(
        **{**record.__dict__, "perm_value": record.perm_value ^ 1}
    )
    verification = verify_record(tampered, verdicts[0])
    assert verification["code"] == E_TRUTH_MISMATCH
    assert any(item["field"] == "raw_perm_value" for item in verification["mismatches"])


def test_budget_enforced_with_code(manifests, policies):
    result = _run_manifest(
        manifests["manifest_budget"],
        policies,
        BudgetConfig(max_records_per_file=5, per_file_timeout_seconds=30),
    )
    assert result.status == "budget"
    assert E_BUDGET in result.codes
    assert result.processed_records == 5
    assert result.dropped_records == 7


def test_timeout_enforced_with_real_wall_clock(manifests, policies):
    def slow_probe(_):
        import time
        end = time.monotonic() + 0.03
        while time.monotonic() < end:
            pass

    records, rejections, metadata = load_manifest_file(manifests["manifest_timeout"])
    result = audit_file(
        metadata["file_id"], manifests["manifest_timeout"], metadata["root"],
        records, rejections, policies,
        BudgetConfig(per_file_timeout_seconds=0.05, max_records_per_file=1000),
        work_probe=slow_probe,
    )
    assert result.status == "timeout"
    assert E_TIMEOUT in result.codes
    assert result.elapsed_seconds >= 0.05
    assert result.dropped_records > 0


def test_batch_exit_codes(manifests, policies, tmp_path):
    all_paths = sorted(manifests.values())
    report, code = run_batch(
        all_paths, policies, BudgetConfig(concurrency=1),
        str(tmp_path / "ck"), str(tmp_path / "report.json"), backend="serial",
    )
    assert code == EXIT_VIOLATIONS

    clean = [manifests["manifest_budget"]]
    _, clean_code = run_batch(
        clean, policies, BudgetConfig(concurrency=1),
        str(tmp_path / "ck2"), str(tmp_path / "clean.json"), backend="serial",
    )
    assert clean_code == EXIT_OK

    _, budget_code = run_batch(
        clean, policies, BudgetConfig(concurrency=1, max_records_per_file=2),
        str(tmp_path / "ck3"), str(tmp_path / "budget.json"), backend="serial",
    )
    assert budget_code == EXIT_BUDGET_OR_TIMEOUT
