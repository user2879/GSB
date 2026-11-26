from __future__ import annotations

from typing import Any

from .composition import compose_ticket, order_conclusions
from .hashing import fingerprint
from .reference import (
    STAGE_ORDER,
    attribution_for_risk_flip,
    compare_ticket_error,
    reference_ticket,
    summarize_flip_sources,
)
from .scanner import scan_corpus
from .scheduler import drain_all
from .settlement import assemble_settlement, settle_scope, settle_ticket


def config_hash(bundle: dict[str, Any]) -> str:
    return fingerprint(bundle["config"])


def build_work_items(scans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ticket_id in sorted(scans):
        scan = scans[ticket_id]
        for scope in sorted(scan["scopes"], key=lambda item: (item["start"], item["scope_id"])):
            items.append({
                "ticket_id": ticket_id,
                "scope_id": scope["scope_id"],
                "start": scope["start"],
                "point_cost": scope["length"] * len(scope["attributed_hit_ids"]),
            })
    return items


def build_scope_ledger(
    scans: dict[str, dict[str, Any]],
    processed_scope_ids: set[str],
    pending_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pending_by_id = {item["scope_id"]: item for item in pending_items}
    ledger: list[dict[str, Any]] = []
    for seq, item in enumerate(build_work_items(scans), 1):
        scope_id = item["scope_id"]
        pending = pending_by_id.get(scope_id)
        ledger.append({
            "ledger_seq": seq,
            "ticket_id": item["ticket_id"],
            "scope_id": scope_id,
            "start": item["start"],
            "point_cost": item["point_cost"],
            "status": "E_SCOPE_BUDGET_PENDING" if pending else "settled",
            "error_code": "E_SCOPE_BUDGET_PENDING" if pending else None,
        })
    return ledger


def run_corpus(
    bundle: dict[str, Any],
    *,
    decimal_width: int | None = None,
    use_scheduler: bool = True,
    budget_points: int | None = None,
    max_workers: int | None = None,
    drain_budget: bool = True,
) -> dict[str, Any]:
    tickets_by_id = {ticket["ticket_id"]: ticket for ticket in bundle["tickets"]}
    scans = scan_corpus(bundle)
    width = int(bundle["config"]["fixed_point"]["decimal_width"] if decimal_width is None else decimal_width)
    work_items = build_work_items(scans)
    scheduler_report: dict[str, Any] | None = None
    scope_results_by_id: dict[str, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []

    if use_scheduler and work_items:
        budget = int(bundle["config"]["batch_budget_points"] if budget_points is None else budget_points)
        workers = int(bundle["config"]["max_workers"] if max_workers is None else max_workers)

        def worker(item: dict[str, Any]) -> dict[str, Any]:
            ticket_id = item["ticket_id"]
            scope_id = item["scope_id"]
            scope = next(scope for scope in scans[ticket_id]["scopes"] if scope["scope_id"] == scope_id)
            return settle_scope(scans[ticket_id], scope, tickets_by_id[ticket_id], bundle, width)

        if drain_budget:
            drained = drain_all(
                work_items,
                worker,
                budget_points=budget,
                max_workers=workers,
                larger_budget_on_retry=None,
            )
            scope_results_by_id = {item["scope_id"]: item for item in drained["results"]}
            pending_items = drained["pending"]
            scheduler_report = {
                "complete": drained["complete"],
                "consumed_budget_points": drained["consumed_budget_points"],
                "batches": drained["batches"],
                "pending": drained["pending"],
            }
        else:
            from .scheduler import BudgetScheduler

            scheduler = BudgetScheduler(budget, workers)
            batch = scheduler.submit_batch(work_items, worker)
            scope_results_by_id = {item["scope_id"]: item for item in batch["results"]}
            pending_items = batch["pending"]
            scheduler_report = {key: value for key, value in batch.items() if key not in {"results"}}
    else:
        for ticket in bundle["tickets"]:
            settlement = settle_ticket(scans[ticket["ticket_id"]], ticket, bundle, width)
            for scope_result in settlement["scope_results"]:
                scope_results_by_id[scope_result["scope_id"]] = scope_result

    processed_scope_ids = set(scope_results_by_id)
    conclusions: list[dict[str, Any]] = []
    settlements: dict[str, dict[str, Any]] = {}
    pending_tickets: list[dict[str, Any]] = []

    for ticket in sorted(bundle["tickets"], key=lambda item: item["ticket_id"]):
        ticket_id = ticket["ticket_id"]
        scan = scans[ticket_id]
        missing = [scope["scope_id"] for scope in scan["scopes"] if scope["scope_id"] not in processed_scope_ids]
        if missing:
            pending_tickets.append({
                "ticket_id": ticket_id,
                "status": "E_SCOPE_BUDGET_PENDING",
                "pending_scope_ids": missing,
            })
            continue
        scope_results = [scope_results_by_id[scope["scope_id"]] for scope in scan["scopes"]]
        settlement = assemble_settlement(scan, scope_results, bundle, width)
        settlements[ticket_id] = settlement
        conclusions.append(compose_ticket(ticket, scan, settlement, bundle, width))

    ordered = order_conclusions(conclusions)
    scope_ledger = build_scope_ledger(scans, processed_scope_ids, pending_items)
    references = {
        ticket_id: reference_ticket(tickets_by_id[ticket_id], scans[ticket_id], bundle)
        for ticket_id in settlements
    }
    tolerances = bundle["config"]["error_tolerances_points"]
    errors = [
        compare_ticket_error(ticket_id, references[ticket_id], settlements[ticket_id], next(
            item for item in conclusions if item["ticket_id"] == ticket_id
        ), tolerances)
        for ticket_id in sorted(settlements)
    ]

    return {
        "global_version": bundle["config"]["global_version"],
        "config_hash": config_hash(bundle),
        "decimal_width": width,
        "gear": f"Q{width}",
        "scans": scans,
        "settlements": settlements,
        "conclusions": conclusions,
        "conclusions_by_id": {item["ticket_id"]: item for item in conclusions},
        "ordered_conclusions": ordered,
        "scope_ledger": scope_ledger,
        "references": references,
        "errors": errors,
        "error_summary": summarize_errors(errors),
        "scheduler_report": scheduler_report,
        "pending_items": pending_items,
        "pending_tickets": pending_tickets,
        "complete": not pending_tickets,
    }


def summarize_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    first_stage_counts = {stage: 0 for stage in STAGE_ORDER}
    max_diffs = {stage: 0.0 for stage in STAGE_ORDER}
    exceeded_tickets = 0
    risk_mismatches = 0
    for error in errors:
        if error["first_exceeded_stage"]:
            first_stage_counts[error["first_exceeded_stage"]] += 1
            exceeded_tickets += 1
        if not error["risk_matches_float64"]:
            risk_mismatches += 1
        for stage in STAGE_ORDER:
            max_diffs[stage] = max(max_diffs[stage], error["stage_diffs"][stage])
    return {
        "ticket_count": len(errors),
        "first_exceeded_stage_counts": first_stage_counts,
        "exceeded_ticket_count": exceeded_tickets,
        "risk_mismatch_float64_count": risk_mismatches,
        "max_stage_diffs": max_diffs,
    }


def assignment_fingerprint(scan: dict[str, Any]) -> str:
    return fingerprint([
        (hit["hit_id"], hit["effective_scope_id"], hit["candidate_scope_ids"])
        for hit in scan["hits"]
    ])


def saturation_fingerprint(settlement: dict[str, Any]) -> str:
    return fingerprint([
        (hit["hit_id"], hit["final_cancellation_q"], hit["saturation_positions"])
        for hit in settlement["hit_results"]
    ])


def compare_quantization_gears(bundle: dict[str, Any]) -> dict[str, Any]:
    fine = run_corpus(bundle, decimal_width=6, use_scheduler=False)
    coarse = run_corpus(bundle, decimal_width=2, use_scheduler=False)
    flips: list[dict[str, Any]] = []
    attributions: list[str] = []
    for ticket in sorted(bundle["tickets"], key=lambda item: item["ticket_id"]):
        ticket_id = ticket["ticket_id"]
        fine_conclusion = fine["conclusions_by_id"][ticket_id]
        coarse_conclusion = coarse["conclusions_by_id"][ticket_id]
        if fine_conclusion["risk_level"] == coarse_conclusion["risk_level"]:
            continue
        source = attribution_for_risk_flip(
            fine_conclusion,
            coarse_conclusion,
            left_assignment_fingerprint=assignment_fingerprint(fine["scans"][ticket_id]),
            right_assignment_fingerprint=assignment_fingerprint(coarse["scans"][ticket_id]),
        )
        flips.append({
            "ticket_id": ticket_id,
            "fine_risk_level": fine_conclusion["risk_level"],
            "coarse_risk_level": coarse_conclusion["risk_level"],
            "fine_score": fine_conclusion["weighted_sentiment_points"],
            "coarse_score": coarse_conclusion["weighted_sentiment_q"] / coarse_conclusion["scale"],
            "source": source,
            "boundary_fine": fine_conclusion["boundary"],
            "boundary_coarse": coarse_conclusion["boundary"],
        })
        attributions.append(source)
    return {
        "fine_error_summary": fine["error_summary"],
        "coarse_error_summary": coarse["error_summary"],
        "risk_flips": flips,
        "source_counts": summarize_flip_sources(attributions),
    }


def resume_from_snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    """Continue from committed stage products without reading source sample files."""
    bundle = state["bundle"]
    width = int(bundle["config"]["fixed_point"]["decimal_width"])
    scans = state.get("scans") or scan_corpus(bundle)

    settlements = state.get("settlements")
    if settlements is None:
        settlements = {}
        for ticket in bundle["tickets"]:
            ticket_id = ticket["ticket_id"]
            settlements[ticket_id] = settle_ticket(scans[ticket_id], ticket, bundle, width)

    conclusions = state.get("conclusions")
    if conclusions is None:
        conclusions = []
        for ticket in sorted(bundle["tickets"], key=lambda item: item["ticket_id"]):
            ticket_id = ticket["ticket_id"]
            conclusions.append(compose_ticket(
                ticket, scans[ticket_id], settlements[ticket_id], bundle, width
            ))

    conclusions_by_id = {item["ticket_id"]: item for item in conclusions}
    ordered = order_conclusions(conclusions)
    references = {
        ticket["ticket_id"]: reference_ticket(ticket, scans[ticket["ticket_id"]], bundle)
        for ticket in bundle["tickets"]
    }
    tolerances = bundle["config"]["error_tolerances_points"]
    errors = [
        compare_ticket_error(
            ticket_id, references[ticket_id], settlements[ticket_id], conclusions_by_id[ticket_id], tolerances
        )
        for ticket_id in sorted(conclusions_by_id)
    ]
    return {
        "global_version": state.get("global_version", bundle["config"]["global_version"]),
        "config_hash": state.get("config_hash", config_hash(bundle)),
        "decimal_width": width,
        "gear": f"Q{width}",
        "scans": scans,
        "settlements": settlements,
        "conclusions": conclusions,
        "conclusions_by_id": conclusions_by_id,
        "ordered_conclusions": ordered,
        "references": references,
        "errors": errors,
        "error_summary": summarize_errors(errors),
        "scope_ledger": build_scope_ledger(scans, set(settlements), []),
        "scheduler_report": None,
        "pending_items": state.get("pending_items", []),
        "pending_tickets": state.get("pending_tickets", []),
        "complete": True,
        "recovered_from_snapshot": True,
        "manifest": state.get("manifest"),
    }
