from __future__ import annotations

from collections import defaultdict
from typing import Any

from .canonical import canonical_bytes, sha256_hex
from .engine import run_event_path, run_window_path
from .errors import QosDiagError
from .fixedpoint import FixedPoint, classify_difference
from .model import STAGES


def select_path(bundle: dict[str, Any]) -> dict[str, Any]:
    config = bundle["config"]
    pipeline_count = len(bundle["pipelines"])
    event_count = len(bundle["events"])
    events_by_pipeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in bundle["events"]:
        events_by_pipeline[event["pipeline_id"]].append(event)

    total_span_ms = 0
    gap_count = 0
    active_windows: set[tuple[str, int]] = set()
    for pipeline in bundle["pipelines"]:
        pid = pipeline["pipeline_id"]
        items = events_by_pipeline[pid]
        if len(items) >= 2:
            total_span_ms += int(items[-1]["arrival_ms"]) - int(items[0]["arrival_ms"])
            gap_count += len(items) - 1
        for event in items:
            start = int(event["arrival_ms"]) - (int(event["arrival_ms"]) % int(config["slide_ms"]))
            active_windows.add((pid, start))

    fp = FixedPoint(int(config["scale"]))
    avg_gap_exact = {
        "total_span_ms": total_span_ms,
        "gap_count": gap_count,
        "numerator_ms": total_span_ms,
        "denominator": max(1, gap_count),
    }
    avg_gap_q = (fp.q * total_span_ms) // max(1, gap_count) if gap_count else 0
    window_count = len(active_windows)
    event_work = 3 * event_count + pipeline_count
    window_work = event_count + 3 * window_count + pipeline_count
    advantage = event_work - window_work
    deadband = max(int(config["selection_deadband_work"]), pipeline_count, event_count // 20)
    selected = "window" if advantage > deadband else "event"
    density_q = (event_count * fp.q) // max(1, window_count)
    return {
        "selected_path": selected,
        "rule": "select window only when event_work - window_work > deadband_work; otherwise stable-select event",
        "tie_break": "event path is the stable canonical tie winner",
        "parameters": {
            "pipeline_count": pipeline_count,
            "event_count": event_count,
            "gap_count": gap_count,
            "average_arrival_interval_ms_q": avg_gap_q,
            "average_arrival_interval_exact": avg_gap_exact,
            "window_ms": int(config["window_ms"]),
            "slide_ms": int(config["slide_ms"]),
            "active_window_count": window_count,
            "events_per_window_q": density_q,
        },
        "estimate": {
            "event_path_work_3E_plus_P": event_work,
            "window_path_work_E_plus_3W_plus_P": window_work,
            "advantage_window_minus_event_saving": advantage,
            "deadband_work": deadband,
            "switch_boundary_advantage_gt": deadband,
            "switched": advantage > deadband,
            "boundary_value_if_not_switched": {
                "required_advantage_at_least": deadband + 1,
                "current_advantage": advantage,
                "margin_work": max(0, deadband + 1 - advantage),
            },
        },
    }


def build_window_aggregates(
    results: list[dict[str, Any]], fp: FixedPoint, window_ms: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["pipeline_id"], int(result["window_start_ms"]))].append(result)

    windows: list[dict[str, Any]] = []
    for pipeline_id, window_start in sorted(grouped, key=lambda key: (key[0], key[1])):
        items = sorted(grouped[(pipeline_id, window_start)], key=lambda item: item["event_seq"])
        stage_aggs: list[dict[str, Any]] = []
        chain_bound = 0
        for stage in STAGES:
            records = [next(record for record in item["stages"] if record["stage"] == stage) for item in items]
            evaluated = [record for record in records if record["evaluated"]]
            allowed = sum(record["verdict"] == "allow" for record in evaluated)
            shaped = sum(record["decision"] == "shaped" for record in evaluated)
            skipped = sum(not record["evaluated"] for record in records)
            required_q = sum(int(record["required_q"]) for record in evaluated)
            consumed_q = sum(
                int(record["balance_before_decision_q"]) - int(record["balance_after_decision_q"])
                for record in evaluated
                if record["verdict"] == "allow"
            )
            bound_q = len(evaluated) + 1
            chain_bound += bound_q
            stage_aggs.append(
                {
                    "stage": stage,
                    "event_count": len(records),
                    "evaluated_count": len(evaluated),
                    "allowed": allowed,
                    "shaped": shaped,
                    "skipped": skipped,
                    "required_q_sum": required_q,
                    "consumed_q_sum": consumed_q,
                    "start_balance_q": records[0]["balance_before_refill_q"],
                    "end_balance_q": records[-1]["balance_after_decision_q"],
                    "aggregation_error_bound_q": bound_q,
                }
            )
        windows.append(
            {
                "pipeline_id": pipeline_id,
                "window_start_ms": window_start,
                "window_end_ms": window_start + window_ms,
                "event_count": len(items),
                "event_seqs": [item["event_seq"] for item in items],
                "stages": stage_aggs,
                "three_stage_bound_q": chain_bound,
                "window_aggregation_error_bound_q": len(items) * 3 + 3,
            }
        )
    return windows


def build_stage_stats(results: list[dict[str, Any]], bundle: dict[str, Any], fp: FixedPoint) -> list[dict[str, Any]]:
    shapers = {(item["pipeline_id"], item["stage"]): item for item in bundle["shapers"]}
    stats: list[dict[str, Any]] = []
    for pipeline in bundle["pipelines"]:
        pid = pipeline["pipeline_id"]
        pipeline_results = [item for item in results if item["pipeline_id"] == pid]
        for stage in STAGES:
            records = [
                next(record for record in item["stages"] if record["stage"] == stage)
                for item in pipeline_results
            ]
            evaluated = [record for record in records if record["evaluated"]]
            allowed = sum(record["verdict"] == "allow" for record in evaluated)
            shaped = sum(record["decision"] == "shaped" for record in evaluated)
            skipped = len(records) - len(evaluated)
            balances = [int(record["balance_after_decision_q"]) for record in evaluated]
            sum_balance = sum(balances)
            mean_q = sum_balance // len(balances) if balances else 0
            mean_remainder = sum_balance % len(balances) if balances else 0
            ratio_q, ratio_remainder = fp.floor_ratio(shaped, len(evaluated)) if evaluated else (0, 0)
            shaper = shapers[(pid, stage)]
            stats.append(
                {
                    "pipeline_id": pid,
                    "stage": stage,
                    "shaper_id": shaper["shaper_id"],
                    "observations": len(evaluated),
                    "allowed": allowed,
                    "shaped": shaped,
                    "skipped": skipped,
                    "shape_ratio_defined": bool(evaluated),
                    "shape_ratio_q": ratio_q,
                    "shape_ratio_numerator": shaped,
                    "shape_ratio_denominator": len(evaluated),
                    "shape_ratio_truncation_remainder": ratio_remainder,
                    "min_balance_after_q": min(balances) if balances else None,
                    "max_balance_after_q": max(balances) if balances else None,
                    "mean_balance_after_q": mean_q,
                    "mean_balance_remainder_q": mean_remainder,
                    "final_balance_q": records[-1]["balance_after_decision_q"] if records else None,
                    "final_state_clock_ms": records[-1]["state_clock_ms"] if records else None,
                    "mean_error_bound_q_exclusive": 1 if balances else 0,
                    "ratio_error_bound_q_exclusive": 1 if evaluated else 0,
                }
            )
    return stats


def build_error_bounds(
    results: list[dict[str, Any]], windows: list[dict[str, Any]], fp: FixedPoint
) -> dict[str, Any]:
    evaluated_count = sum(
        1 for result in results for record in result["stages"] if record["evaluated"]
    )
    max_window_bound = max((window["window_aggregation_error_bound_q"] for window in windows), default=0)
    max_chain_bound = max((window["three_stage_bound_q"] for window in windows), default=0)
    tolerance_q = fp.q
    uncertainty_q = max(1, max_chain_bound)
    examples = [
        {
            "cause": "difference_definitely_exceeds_tolerance",
            "observed_abs_q": 10 * fp.q,
            "tolerance_q": tolerance_q,
            "uncertainty_q": 1,
            "classification": classify_difference(10 * fp.q, tolerance_q, 1),
        },
        {
            "cause": "difference_definitely_within_tolerance",
            "observed_abs_q": 0,
            "tolerance_q": tolerance_q,
            "uncertainty_q": 1,
            "classification": classify_difference(0, tolerance_q, 1),
        },
        {
            "cause": "quantization_error_makes_cause_indistinguishable",
            "observed_abs_q": tolerance_q,
            "tolerance_q": tolerance_q,
            "uncertainty_q": max(1, fp.q),
            "classification": classify_difference(tolerance_q, tolerance_q, max(1, fp.q)),
        },
    ]
    return {
        **fp.error_bounds(),
        "evaluated_stage_records": evaluated_count,
        "single_event_upper_bound_q": 1,
        "max_window_aggregation_bound_q": max_window_bound,
        "max_three_stage_chain_bound_q": max_chain_bound,
        "conservative_total_bound_q": evaluated_count + len(windows) * 3,
        "observed_integer_quantization_error_q": 0,
        "default_tolerance_q": tolerance_q,
        "classification_rule": "lower=abs(delta)-u; upper=abs(delta)+u; exceeds if lower>tol, within if upper<=tol, otherwise indistinguishable",
        "classification_examples": examples,
    }


def diagnose(bundle: dict[str, Any], *, forced_path: str = "auto") -> dict[str, Any]:
    fp = FixedPoint(int(bundle["config"]["scale"]))
    selection = select_path(bundle)
    event_results = run_event_path(bundle, fp)
    window_results = run_window_path(bundle, fp)
    event_bytes = canonical_bytes(event_results)
    window_bytes = canonical_bytes(window_results)
    parity = event_bytes == window_bytes
    if not parity:
        raise QosDiagError("event and window paths produced non-identical canonical records")

    selected_path = selection["selected_path"] if forced_path == "auto" else forced_path
    if selected_path not in ("event", "window"):
        raise QosDiagError(f"unknown path {selected_path}")
    results = window_results if selected_path == "window" else event_results
    windows = build_window_aggregates(results, fp, int(bundle["config"]["window_ms"]))
    stage_stats = build_stage_stats(results, bundle, fp)
    error_bounds = build_error_bounds(results, windows, fp)
    return {
        "schema": "qos-diag/diagnosis@1",
        "spec_digest": bundle.get("spec_digest"),
        "path_selection": {**selection, "requested_path": forced_path, "active_path": selected_path},
        "path_parity": {
            "event_and_window_canonical_bytes_identical": parity,
            "event_sha256": sha256_hex(event_results),
            "window_sha256": sha256_hex(window_results),
            "event_byte_count": len(event_bytes),
            "window_byte_count": len(window_bytes),
        },
        "events": results,
        "windows": windows,
        "stage_stats": stage_stats,
        "error_bounds": error_bounds,
    }
