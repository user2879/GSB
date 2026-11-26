from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from .edits import validate_command_stream
from .errors import SchemaViolation, SnapshotUnavailable
from .incremental import IncrementalEngine
from .pipeline import (
    assignment_fingerprint,
    compare_quantization_gears,
    resume_from_snapshot_state,
    run_corpus,
)
from .reference import attribution_for_risk_flip
from .samplegen import generate_sample_dir
from .scanner import scan_ticket
from .schema import load_jsonl, load_sample, validate_bundle
from .settlement import settle_ticket
from .composition import compose_ticket
from .snapshot import SnapshotStore, canonical_result_for_compare


def json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def _scope_brief(scan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scope_id": scope["scope_id"],
            "start": scope["start"],
            "end": scope["end"],
            "length": scope["length"],
            "trigger_order": scope["trigger_order"],
            "attributed_hit_ids": scope["attributed_hit_ids"],
        }
        for scope in scan["scopes"]
    ]


def run_acceptance(sample_dir: str | Path = "samples", artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    sample_path = generate_sample_dir(sample_dir)
    bundle = load_sample(sample_path)
    commands = load_jsonl(Path(sample_path) / "edits.jsonl")
    validate_command_stream(commands)

    report: dict[str, Any] = {"sample_dir": str(sample_path), "checks": {}}

    # 1. Schema validation failure must be aggregated and carry E_SCHEMA_VIOLATION.
    bad_bundle = copy.deepcopy(bundle)
    bad_bundle["tickets"][1]["ticket_id"] = bad_bundle["tickets"][0]["ticket_id"]
    bad_bundle["config"]["fixed_point"]["decimal_width"] = 3
    schema_code = None
    schema_issue_fields: list[str] = []
    try:
        validate_bundle(bad_bundle)
        raise AssertionError("invalid bundle unexpectedly passed validation")
    except SchemaViolation as exc:
        schema_code = exc.code
        schema_issue_fields = sorted({issue["field"] for issue in exc.issues})
    assert schema_code == "E_SCHEMA_VIOLATION"
    assert "ticket_id" in schema_issue_fields
    assert "fixed_point.decimal_width" in schema_issue_fields
    report["checks"]["schema_violation"] = {
        "code": schema_code,
        "issue_fields": schema_issue_fields,
        "command_count": len(commands),
    }

    # 2-4. Canonical serial result and concrete scope/settlement fixtures.
    serial = run_corpus(bundle, use_scheduler=False)
    concurrent = run_corpus(bundle, budget_points=24, max_workers=4)
    assert concurrent["complete"]
    assert json_equal(concurrent["conclusions"], serial["conclusions"])
    assert json_equal(concurrent["scope_ledger"], serial["scope_ledger"])

    scans = serial["scans"]
    tk1 = scans["TK-0001"]
    tk15 = scans["TK-0015"]
    assert len(tk1["scopes"]) == 2
    assert all(hit["effective_scope_id"] == "TK-0001:S001" for hit in tk1["hits"])
    assert len(tk1["suppressions"]) == 2
    assert len(tk15["scopes"]) == 2
    assert any(item["suppressed_scope_id"] == "TK-0015:S003" for item in tk15["suppressions"])
    assert [hit["scan_order"] for hit in tk1["hits"]] == [0, 1]

    tk5_settlement = serial["settlements"]["TK-0005"]
    assert tk5_settlement["saturation_event_count"] >= 2
    assert any(hit["canceled_to_zero"] and hit["cancellation_steps"] for hit in tk5_settlement["hit_results"])

    isolation_ticket = {
        "ticket_id": "TK-ISO",
        "tokens": ["延迟", "。", "不", "满意", "但是", "。"],
        "channel_id": "app",
        "annotator_version": 1,
        "completion_status": "closed",
    }
    iso_scan = scan_ticket(isolation_ticket, bundle)
    iso_settlement = settle_ticket(iso_scan, isolation_ticket, bundle)
    iso_conclusion = compose_ticket(isolation_ticket, iso_scan, iso_settlement, bundle)
    outside = next(hit for hit in iso_settlement["hit_results"] if hit["outside_scope"])
    inside = next(hit for hit in iso_settlement["hit_results"] if not hit["outside_scope"])
    assert outside["term"] == "延迟"
    assert outside["cancellation_steps"] == []
    assert outside["final_cancellation_q"] == outside["initial_evidence_q"]
    assert inside["term"] == "满意"
    assert iso_conclusion["referenced_scope_ids"] == ["TK-ISO:S002"]

    report["checks"]["negation_scope"] = {
        "nested_ticket": "TK-0001",
        "nested_scopes": _scope_brief(tk1),
        "nested_suppression_count": len(tk1["suppressions"]),
        "overlap_ticket": "TK-0015",
        "overlap_scopes": _scope_brief(tk15),
        "overlap_suppression_count": len(tk15["suppressions"]),
        "scan_order_example": tk1["scan_order_hit_ids"],
    }
    report["checks"]["cancellation_and_isolation"] = {
        "saturation_ticket": "TK-0005",
        "saturation_event_count": tk5_settlement["saturation_event_count"],
        "retained_zeroed_hits": [
            hit["hit_id"] for hit in tk5_settlement["hit_results"] if hit["canceled_to_zero"]
        ],
        "isolation_ticket": "TK-ISO",
        "outside_hit_id": outside["hit_id"],
        "outside_score_unchanged": outside["final_cancellation_q"] == outside["initial_evidence_q"],
        "inside_hit_id": inside["hit_id"],
    }

    # 5. Float64 reference, stage error propagation, and gear flip attribution.
    coarse = run_corpus(bundle, decimal_width=2, use_scheduler=False)
    gear_report = compare_quantization_gears(bundle)
    assert coarse["error_summary"]["first_exceeded_stage_counts"]["cancellation"] > 0
    assert gear_report["source_counts"]["threshold_comparison"] >= 1

    scope_variant_bundle = copy.deepcopy(bundle)
    scope_variant_bundle["config"]["scope_max_tokens"] = 4
    scope_variant = run_corpus(scope_variant_bundle, use_scheduler=False)
    scope_flip_id = next(
        tid for tid in serial["conclusions_by_id"]
        if serial["conclusions_by_id"][tid]["risk_level"] != scope_variant["conclusions_by_id"][tid]["risk_level"]
        and assignment_fingerprint(serial["scans"][tid]) != assignment_fingerprint(scope_variant["scans"][tid])
    )
    scope_source = attribution_for_risk_flip(
        serial["conclusions_by_id"][scope_flip_id],
        scope_variant["conclusions_by_id"][scope_flip_id],
        left_assignment_fingerprint=assignment_fingerprint(serial["scans"][scope_flip_id]),
        right_assignment_fingerprint=assignment_fingerprint(scope_variant["scans"][scope_flip_id]),
    )
    assert scope_source == "scope_attribution"

    saturation_bundle = copy.deepcopy(bundle)
    saturation_bundle["config"]["cancellation_strength_points"] = 150
    saturation_variant = run_corpus(saturation_bundle, use_scheduler=False)
    saturation_flip_id = next(
        tid for tid in serial["conclusions_by_id"]
        if serial["conclusions_by_id"][tid]["risk_level"] != saturation_variant["conclusions_by_id"][tid]["risk_level"]
        and serial["settlements"][tid]["saturation_event_count"]
        != saturation_variant["settlements"][tid]["saturation_event_count"]
    )
    saturation_source = attribution_for_risk_flip(
        serial["conclusions_by_id"][saturation_flip_id],
        saturation_variant["conclusions_by_id"][saturation_flip_id],
        left_assignment_fingerprint=assignment_fingerprint(serial["scans"][saturation_flip_id]),
        right_assignment_fingerprint=assignment_fingerprint(saturation_variant["scans"][saturation_flip_id]),
    )
    assert saturation_source == "cancellation_saturation"

    controlled_sources = {
        "scope_attribution": [scope_flip_id],
        "cancellation_saturation": [saturation_flip_id],
        "threshold_comparison": [flip["ticket_id"] for flip in gear_report["risk_flips"]],
    }
    demonstrated_source_counts = {
        "scope_attribution": len(controlled_sources["scope_attribution"]),
        "cancellation_saturation": len(controlled_sources["cancellation_saturation"]),
        "threshold_comparison": len(controlled_sources["threshold_comparison"]),
    }
    demonstrated_source_counts["total"] = sum(
        value for key, value in demonstrated_source_counts.items() if key != "total"
    )
    report["checks"]["error_propagation"] = {
        "fine_error_summary": serial["error_summary"],
        "coarse_error_summary": coarse["error_summary"],
        "gear_report": gear_report,
        "controlled_source_examples": controlled_sources,
        "demonstrated_source_counts": demonstrated_source_counts,
    }

    # 6. Content fingerprints and incremental reuse.
    engine = IncrementalEngine(bundle)
    incremental_steps: list[dict[str, Any]] = []
    for command in commands:
        step = engine.apply(command)
        fresh = run_corpus(engine.bundle, use_scheduler=False)
        assert json_equal(engine.result["conclusions"], fresh["conclusions"])
        assert json_equal(engine.result["scans"], fresh["scans"])
        assert json_equal(engine.result["settlements"], fresh["settlements"])
        incremental_steps.append({
            "seq": command["seq"],
            "op": command["op"],
            "affected_ticket_ids": step["affected_ticket_ids"],
            "reused_ticket_count": step["reused_ticket_count"],
            "ledger_counts": step["ledger_counts"],
        })

    replace_engine = IncrementalEngine(bundle)
    replacement = copy.deepcopy(next(t for t in bundle["tickets"] if t["ticket_id"] == "TK-0004"))
    replacement["tokens"] = ["客服", "耐心", "处理", "延迟", "。"]
    replace_report = replace_engine.apply({"seq": "acc-replace", "op": "replace_ticket", "ticket": replacement})
    assert replace_report["affected_ticket_ids"] == ["TK-0004"]

    trigger_engine = IncrementalEngine(bundle)
    trigger_report = trigger_engine.apply({
        "seq": "acc-trigger",
        "op": "upsert_neg_trigger",
        "trigger": {"trigger_id": "NEG_BIE", "term": "别", "order": 8},
    })
    assert trigger_report["affected_ticket_ids"] == ["TK-0003"]
    report["checks"]["incremental_reuse"] = {
        "edit_stream_steps": incremental_steps,
        "single_ticket_replace": {
            "affected": replace_report["affected_ticket_ids"],
            "reused": replace_report["reused_ticket_count"],
            "ledger_counts": replace_report["ledger_counts"],
        },
        "single_trigger_edit": {
            "affected": trigger_report["affected_ticket_ids"],
            "reused": trigger_report["reused_ticket_count"],
            "ledger_counts": trigger_report["ledger_counts"],
        },
    }

    # 7. Concurrency/budget matrix and canonical ledger equality.
    budget_matrix: list[dict[str, Any]] = []
    for budget in (24, 1000):
        for workers in (1, 2, 4, 8):
            candidate = run_corpus(bundle, budget_points=budget, max_workers=workers)
            assert candidate["complete"]
            assert json_equal(candidate["conclusions"], serial["conclusions"])
            assert json_equal(candidate["scope_ledger"], serial["scope_ledger"])
            peaks = [batch.get("peak_in_flight", 0) for batch in candidate["scheduler_report"]["batches"]]
            assert max(peaks, default=0) <= workers
            budget_matrix.append({
                "budget": budget,
                "workers": workers,
                "batches": len(candidate["scheduler_report"]["batches"]),
                "peak_in_flight": max(peaks, default=0),
                "matches_serial": True,
            })
    pending = run_corpus(bundle, budget_points=1, max_workers=2, drain_budget=False)
    assert pending["pending_tickets"]
    assert all(row["status"] == "E_SCOPE_BUDGET_PENDING" for row in pending["scope_ledger"] if row["status"] != "settled")
    report["checks"]["budget_concurrency"] = {
        "matrix": budget_matrix,
        "budget_one_drain_budget_false": {
            "pending_ticket_count": len(pending["pending_tickets"]),
            "pending_scope_count": len(pending["pending_items"]),
            "marker": "E_SCOPE_BUDGET_PENDING",
        },
    }

    # 8. Atomic snapshot and recovery.
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    snapshot_root = artifact_path / "snapshot-demo"
    store = SnapshotStore(snapshot_root)
    state: dict[str, Any] = {
        "bundle": bundle,
        "global_version": serial["global_version"],
        "config_hash": serial["config_hash"],
    }
    store.commit_state(state, "validation")
    state["scans"] = serial["scans"]
    hit_manifest = store.commit_state(state, "hit_scan")
    loaded_hit = store.load_current(expected_config_hash=serial["config_hash"])
    resumed_hit = resume_from_snapshot_state(loaded_hit)
    assert json_equal(
        canonical_result_for_compare(resumed_hit), canonical_result_for_compare(serial)
    )

    state.update({
        "settlements": serial["settlements"],
        "conclusions": serial["conclusions"],
        "ordered_conclusions": serial["ordered_conclusions"],
        "errors": serial["errors"],
        "error_summary": serial["error_summary"],
        "pending_items": serial["pending_items"],
        "pending_tickets": serial["pending_tickets"],
    })
    complete_manifest = store.commit_state(state, "composition")
    loaded_complete = store.load_current(expected_global_version=1, expected_config_hash=serial["config_hash"])
    resumed_complete = resume_from_snapshot_state(loaded_complete)
    assert json_equal(
        canonical_result_for_compare(resumed_complete), canonical_result_for_compare(serial)
    )

    with tempfile.TemporaryDirectory() as tmp:
        missing_store = SnapshotStore(Path(tmp) / "missing")
        unavailable_code = None
        try:
            missing_store.load_current()
        except SnapshotUnavailable as exc:
            unavailable_code = exc.code
        assert unavailable_code == "E_SNAPSHOT_UNAVAILABLE"

    report["checks"]["snapshot_recovery"] = {
        "partial_manifest_id": hit_manifest["snapshot_id"],
        "complete_manifest_id": complete_manifest["snapshot_id"],
        "partial_recovery_matches_rebuild": True,
        "complete_recovery_matches_rebuild": True,
        "unavailable_code": unavailable_code,
    }
    report["ok"] = True

    report_path = artifact_path / "acceptance-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
