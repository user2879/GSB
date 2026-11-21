"""JSON diagnostic report construction and atomic emission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import STAGE_ORDER
from .numeric import write_json


def empty_stage_stats() -> dict[str, dict[str, int]]:
    return {
        stage: {"reused": 0, "computed": 0, "invalidated": 0, "cache_miss": 0}
        for stage in STAGE_ORDER
    }


def merge_stage_stats(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    for stage in STAGE_ORDER:
        for key, value in source.get(stage, {}).items():
            target[stage][key] = target[stage].get(key, 0) + int(value)


def build_diagnostic_report(
    *,
    mode: str,
    config_hash: str,
    rules_fingerprint: str,
    tickets: list[dict],
    failures: list[dict],
    stage_stats: dict[str, dict[str, int]],
    concurrency: dict,
    snapshot_version: int,
    recovery: dict,
    degradation_details: dict,
    conversion_roundtrip: dict,
    extra: dict | None = None,
) -> dict:
    upgraded = [ticket for ticket in tickets if int(ticket["upgrade_code"]) == 1]
    boundary = [ticket["ticket_id"] for ticket in tickets if ticket.get("boundary")]
    stage_reused = sum(stage_stats.get(stage, {}).get("reused", 0) for stage in STAGE_ORDER)
    stage_computed = sum(stage_stats.get(stage, {}).get("computed", 0) for stage in STAGE_ORDER)
    tickets_reused = sum(
        1
        for ticket in tickets
        if all(stage_stats.get(stage, {}).get("reused", 0) > 0 for stage in STAGE_ORDER)
    )
    report = {
        "schema": "local-upgrade-arbitration-diagnostic/v1",
        "mode": mode,
        "config_hash": config_hash,
        "rules_fingerprint": rules_fingerprint,
        "counts": {
            "ticket_count": len(tickets) + len({failure["ticket_id"] for failure in failures}),
            "arbitrated_ticket_count": len(tickets),
            "upgraded_ticket_count": len(upgraded),
            "not_upgraded_ticket_count": len(tickets) - len(upgraded),
            "failure_count": len(failures),
        },
        "tickets": [
            {
                "ticket_id": ticket["ticket_id"],
                "score": ticket["score"],
                "score_fixed": ticket["score_fixed"],
                "upgrade_code": ticket["upgrade_code"],
                "priority_level": ticket["priority_level"],
                "priority_name": ticket["priority_name"],
                "boundary": bool(ticket.get("boundary", False)),
                "reason_codes": list(ticket["reason_codes"]),
                "missing": int(ticket.get("missing", 0)),
                "precision": ticket["precision"],
                "threshold": ticket.get("threshold"),
                "fusion": ticket.get("fusion"),
                "raw_total_fixed": ticket.get("raw_total_fixed"),
                "relation_delta_fixed": ticket.get("relation_delta_fixed"),
                "relation_contributions": list(ticket["relation_contributions"]),
                "evidence_scores": list(ticket["evidence_scores"]),
                "adjusted_evidence_scores": list(ticket.get("adjusted_evidence_scores", [])),
            }
            for ticket in tickets
        ],
        "boundary_tickets": boundary,
        "degradation_details": degradation_details,
        "failures": failures,
        "invalidation_and_reuse": {
            "stages": stage_stats,
            "total_stage_reuse": stage_reused,
            "total_stage_compute": stage_computed,
            "tickets_with_all_stages_reused": tickets_reused,
        },
        "concurrency": concurrency,
        "snapshot": {
            "version": snapshot_version,
            "recovery": recovery,
        },
        "version_conversion": {
            "roundtrip": conversion_roundtrip,
            "degradations": degradation_details,
        },
    }
    if extra:
        report.update(extra)
    return report


def write_diagnostic_report(path: str | Path, report: dict, *, temp_dir: str | Path | None = None) -> None:
    write_json(path, report, temp_dir=temp_dir, pretty=True)
