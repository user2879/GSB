from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from typing import Any

from .codes import E_SESSION_QUOTA_EXCEEDED, E_WINDOW_BUDGET_PENDING
from .ledger import LedgerBook, LedgerEntry
from .model import QUANT_STEP_UNITS, VERDICT_NAMES, Config, PreferenceRecord
from .util import (
    bucket_vector,
    fingerprint,
    fraction_decimal_text,
    ordered_dot,
    ordered_sum,
    project_feature_vector,
    round_half_up,
    session_order_key,
    stable_task_hash,
)


def record_id(record: PreferenceRecord | dict[str, Any]) -> str:
    if isinstance(record, PreferenceRecord):
        return f"{record.session_id}#{record.turn:06d}"
    return f"{record['session_id']}#{int(record['turn']):06d}"


def weights_fingerprint(weights: dict[str, int]) -> str:
    return fingerprint({key: weights[key] for key in sorted(weights)})


def annotator_weight_fingerprint(annotator_id: str, weight_bp: int) -> str:
    return fingerprint({"annotator_id": annotator_id, "weight_bp": int(weight_bp)})


def feature_cache_key(record: PreferenceRecord, config: Config) -> str:
    return fingerprint(
        {
            "stage": "bucket_vector",
            "record": record.to_dict(),
            "rule": config.subword_rule_version,
            "normalize": config.normalize,
            "H": config.hash_bucket_count,
            "D": config.feature_dim,
            "exchange_version": config.exchange_version,
        }
    )


def score_cache_key(record: PreferenceRecord, config: Config, feature_key: str, all_weights: dict[str, int]) -> str:
    weight_bp = int(all_weights[record.annotator_id])
    step_unit = QUANT_STEP_UNITS[config.quant_scheme]
    weight_fp = annotator_weight_fingerprint(record.annotator_id, weight_bp)
    return fingerprint(
        {
            "stage": "weighted_score",
            "feature_key": feature_key,
            "annotator_weight_fingerprint": weight_fp,
            "annotator": record.annotator_id,
            "weight_bp": weight_bp,
            "places": config.fixed_decimal_places,
            "quant_scheme": config.quant_scheme,
            "step_unit": step_unit,
            "exchange_version": config.exchange_version,
            "bucket_usage": "modulo_project:prompt_gate:ascending_index",
        }
    )


def feature_stage(record: PreferenceRecord, config: Config) -> dict[str, Any]:
    common = {
        "rule": config.subword_rule_version,
        "normalize": config.normalize,
        "H": config.hash_bucket_count,
        "D": config.feature_dim,
    }
    prompt_buckets = bucket_vector(record.prompt, config.hash_bucket_count, config.subword_rule_version, config.normalize)
    chosen_buckets = bucket_vector(record.chosen, config.hash_bucket_count, config.subword_rule_version, config.normalize)
    rejected_buckets = bucket_vector(record.rejected, config.hash_bucket_count, config.subword_rule_version, config.normalize)
    vectors = {
        "prompt": prompt_buckets,
        "chosen": chosen_buckets,
        "rejected": rejected_buckets,
    }
    bucket_digest = fingerprint(vectors)
    key = feature_cache_key(record, config)
    return {
        "record_id": record_id(record),
        "session_id": record.session_id,
        "turn": record.turn,
        "vectors": vectors,
        "bucket_digest": bucket_digest,
        "nonzero": {name: sum(1 for value in vec if value) for name, vec in vectors.items()},
        "feature_key": key,
        "cache_key": key,
    }


def score_stage(
    record: PreferenceRecord,
    config: Config,
    features: dict[str, Any],
    all_weights: dict[str, int],
) -> dict[str, Any]:
    weight_bp = int(all_weights[record.annotator_id])
    p_features = project_feature_vector(features["vectors"]["prompt"], config.feature_dim)
    c_features = project_feature_vector(features["vectors"]["chosen"], config.feature_dim)
    r_features = project_feature_vector(features["vectors"]["rejected"], config.feature_dim)

    chosen_score = 0
    rejected_score = 0
    operation_order: list[int] = []
    for index in range(config.feature_dim):
        prompt_factor = 1 + p_features[index]
        chosen_term = config.score_weights[index] * c_features[index] * prompt_factor
        rejected_term = config.score_weights[index] * r_features[index] * prompt_factor
        chosen_score += chosen_term
        rejected_score += rejected_term
        operation_order.append(index)
    raw_diff = chosen_score - rejected_score

    places = config.fixed_decimal_places
    scale = 10**places
    step_unit = QUANT_STEP_UNITS[config.quant_scheme]
    exact_scaled = Fraction(raw_diff * weight_bp * scale, 10_000)
    grid_index = round_half_up(exact_scaled.numerator, exact_scaled.denominator * step_unit)
    quantized_scaled = grid_index * step_unit
    residual = exact_scaled - quantized_scaled
    weight_fp = annotator_weight_fingerprint(record.annotator_id, weight_bp)
    key = score_cache_key(record, config, features["feature_key"], all_weights)
    half_step = Fraction(step_unit, 2)
    critical = abs(exact_scaled) <= half_step
    category = "conflict" if raw_diff < 0 else ("critical" if critical else "stable")
    return {
        "record_id": record_id(record),
        "session_id": record.session_id,
        "turn": record.turn,
        "annotator_id": record.annotator_id,
        "event_ms": record.event_ms,
        "weight_bp": weight_bp,
        "weight_fingerprint": weight_fp,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "raw_score_diff": raw_diff,
        "operation_order": operation_order,
        "scale": scale,
        "decimal_places": places,
        "step_unit_scaled": step_unit,
        "step_decimal": fraction_decimal_text(Fraction(step_unit, scale), places + 2),
        "exact_scaled_num": exact_scaled.numerator,
        "exact_scaled_den": exact_scaled.denominator,
        "grid_index": grid_index,
        "quantized_scaled": quantized_scaled,
        "residual_num": residual.numerator,
        "residual_den": residual.denominator,
        "residual_decimal": fraction_decimal_text(residual / scale, places + 4),
        "category": category,
        "feature_key": features["feature_key"],
        "cache_key": key,
    }


def boundary_decision(
    sum_scaled: int,
    count: int,
    config: Config,
    step_unit: int | None = None,
) -> dict[str, Any]:
    scale = 10**config.fixed_decimal_places
    unit = QUANT_STEP_UNITS[config.quant_scheme] if step_unit is None else step_unit
    mean_fixed = round_half_up(sum_scaled, count)
    exact_permille = Fraction(sum_scaled * 1000, count * scale)
    mean_permille = round_half_up(exact_permille.numerator, exact_permille.denominator)
    delta_permille = exact_permille - config.decision_threshold_permille
    half_step_permille = Fraction(unit * 1000, 2 * scale)
    exact_match = delta_permille == 0
    boundary = exact_match or abs(delta_permille) <= half_step_permille
    if boundary:
        verdict = 0
    elif delta_permille > 0:
        verdict = 1
    else:
        verdict = -1
    return {
        "mean_fixed_scaled": mean_fixed,
        "mean_permille": mean_permille,
        "mean_permille_exact_num": exact_permille.numerator,
        "mean_permille_exact_den": exact_permille.denominator,
        "mean_decimal": fraction_decimal_text(exact_permille / 1000, config.fixed_decimal_places + 3),
        "delta_permille_num": delta_permille.numerator,
        "delta_permille_den": delta_permille.denominator,
        "half_step_permille_num": half_step_permille.numerator,
        "half_step_permille_den": half_step_permille.denominator,
        "exact_threshold_match": exact_match,
        "boundary": boundary,
        "verdict_code": verdict,
        "verdict_name": VERDICT_NAMES[verdict],
    }


def high_precision_reference(score_rows: list[dict[str, Any]], length: int, config: Config) -> dict[str, Any]:
    ref_places = config.fixed_decimal_places + 1
    ref_scale = 10**ref_places
    total = 0
    for row in score_rows:
        exact = Fraction(row["raw_score_diff"] * row["weight_bp"] * ref_scale, 10_000)
        total += round_half_up(exact.numerator, exact.denominator)
    ref_permille = Fraction(total * 1000, length * ref_scale)
    current = Fraction(sum(row["quantized_scaled"] for row in score_rows) * 1000, length * (10**config.fixed_decimal_places))
    difference = ref_permille - current
    return {
        "reference_decimal_places": ref_places,
        "reference_mean_permille_num": ref_permille.numerator,
        "reference_mean_permille_den": ref_permille.denominator,
        "reference_mean_decimal": fraction_decimal_text(ref_permille / 1000, ref_places + 3),
        "difference_permille_num": difference.numerator,
        "difference_permille_den": difference.denominator,
        "difference_decimal": fraction_decimal_text(difference / 1000, ref_places + 4),
    }


def window_task_id(session_id: str, length: int, end_turn: int) -> str:
    return f"{session_id}:L{length}:E{end_turn:04d}"


def build_window_tasks(score_rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        by_session.setdefault(row["session_id"], []).append(row)
    tasks: list[dict[str, Any]] = []
    for session_id in sorted(by_session):
        rows = sorted(by_session[session_id], key=lambda row: row["turn"])
        by_turn = {int(row["turn"]): row for row in rows}
        max_turn = max(by_turn)
        for length in config.window_lengths:
            window_fp = fingerprint({"length": length, "tie_rule": "extreme_choose_lowest_turn", "advance": "turn_ascending"})
            for end_turn in range(length, max_turn + 1):
                turns = list(range(end_turn - length + 1, end_turn + 1))
                if any(turn not in by_turn for turn in turns):
                    continue
                ordered_rows = [by_turn[turn] for turn in turns]
                score_keys = [row["cache_key"] for row in ordered_rows]
                scope_id = window_task_id(session_id, length, end_turn)
                key = fingerprint(
                    {
                        "stage": "sliding_window",
                        "scope_id": scope_id,
                        "score_keys": score_keys,
                        "window_param_fingerprint": window_fp,
                        "quant_scheme": config.quant_scheme,
                        "places": config.fixed_decimal_places,
                        "exchange_version": config.exchange_version,
                    }
                )
                tasks.append(
                    {
                        "scope_id": scope_id,
                        "session_id": session_id,
                        "length": length,
                        "span": length,
                        "start_turn": turns[0],
                        "end_turn": end_turn,
                        "turns": turns,
                        "rows": ordered_rows,
                        "score_keys": score_keys,
                        "point_cost": length * config.hash_bucket_count,
                        "order_key": session_order_key(session_id, end_turn, length),
                        "stable_hash": stable_task_hash(session_id, end_turn, length),
                        "window_param_fingerprint": window_fp,
                        "cache_key": key,
                    }
                )
    tasks.sort(key=lambda task: task["order_key"])
    return tasks


def compute_window(task: dict[str, Any], config: Config) -> dict[str, Any]:
    rows = task["rows"]
    values = [int(row["quantized_scaled"]) for row in rows]
    total = ordered_sum(values)
    decision = boundary_decision(total, len(values), config)
    max_value = max(values)
    min_value = min(values)
    max_candidates = [row["turn"] for row, value in zip(rows, values) if value == max_value]
    min_candidates = [row["turn"] for row, value in zip(rows, values) if value == min_value]
    max_turn = min(max_candidates)
    min_turn = min(min_candidates)
    tie_basis: list[str] = []
    if len(max_candidates) > 1:
        tie_basis.append(f"max_tie_turns={max_candidates}:choose_lowest={max_turn}")
    if len(min_candidates) > 1:
        tie_basis.append(f"min_tie_turns={min_candidates}:choose_lowest={min_turn}")
    reference = high_precision_reference(rows, task["length"], config)
    detail = {
        "turn_summaries": [
            {
                "turn": row["turn"],
                "annotator_id": row["annotator_id"],
                "quantized_scaled": row["quantized_scaled"],
                "residual_decimal": row["residual_decimal"],
                "category": row["category"],
            }
            for row in rows
        ]
    }
    return {
        "scope_id": task["scope_id"],
        "session_id": task["session_id"],
        "kind": "window",
        "length": task["length"],
        "span": task["span"],
        "start_turn": task["start_turn"],
        "end_turn": task["end_turn"],
        "turns": task["turns"],
        "window_refs": [],
        "sum_scaled": total,
        "values_scaled": values,
        "max_value_scaled": max_value,
        "max_turn": max_turn,
        "max_tie": len(max_candidates) > 1,
        "min_value_scaled": min_value,
        "min_turn": min_turn,
        "min_tie": len(min_candidates) > 1,
        "tie_rule": "ascending_turn_scan;extreme_ties_choose_lowest_turn",
        "tie_basis": tie_basis,
        "detail_summary": detail,
        "reference": reference,
        "cache_key": task["cache_key"],
        "score_keys": task["score_keys"],
        "window_param_fingerprint": task["window_param_fingerprint"],
        **decision,
    }


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope_id": task["scope_id"],
        "session_id": task["session_id"],
        "order_key": list(task["order_key"]),
        "stable_hash": task["stable_hash"],
        "point_cost": task["point_cost"],
        "cache_key": task["cache_key"],
        "turns": task["turns"],
        "length": task["length"],
    }


def schedule_windows(
    tasks: list[dict[str, Any]],
    config: Config,
    budget: int | None = None,
    session_quota: int | None = None,
    workers: int = 1,
    cached_windows: dict[str, dict[str, Any]] | None = None,
    pending_task_ids: list[str] | None = None,
    initial_session_points: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
    budget_total = int(config.window_budget if budget is None else budget)
    quota = int(config.session_quota if session_quota is None else session_quota)
    remaining = budget_total
    cached_windows = cached_windows or {}
    pending_id_set = set(pending_task_ids or [])
    session_points = dict(initial_session_points or {})
    issues: list[Any] = []
    reuse_hits: list[dict[str, str]] = []

    by_id = {task["scope_id"]: task for task in tasks}
    priority = [by_id[task_id] for task_id in pending_task_ids or [] if task_id in by_id]
    priority_ids = {task["scope_id"] for task in priority}
    ordinary = [task for task in tasks if task["scope_id"] not in priority_ids]
    ordered = sorted(priority, key=lambda task: task["order_key"]) + sorted(ordinary, key=lambda task: task["order_key"])

    computable: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    aborted_sessions: set[str] = set()
    budget_pending = False

    for task in ordered:
        scope_id = task["scope_id"]
        session_id = task["session_id"]
        if scope_id in cached_windows:
            reuse_hits.append({"scope_id": scope_id, "cache_key": task["cache_key"], "stage": "sliding_window"})
            continue
        if session_id in aborted_sessions:
            pending.append({**_public_task(task), "marker": E_SESSION_QUOTA_EXCEEDED})
            continue
        if budget_pending:
            pending.append({**_public_task(task), "marker": E_WINDOW_BUDGET_PENDING})
            issues.append(_scheduler_issue(E_WINDOW_BUDGET_PENDING, scope_id, "queued after budget exhaustion"))
            continue
        cost = int(task["point_cost"])
        used = int(session_points.get(session_id, 0))
        if used + cost > quota:
            aborted_sessions.add(session_id)
            issues.append(
                _scheduler_issue(
                    E_SESSION_QUOTA_EXCEEDED,
                    f"session:{session_id}",
                    f"session points {used}+{cost} exceed quota {quota} at {scope_id}",
                )
            )
            pending.append({**_public_task(task), "marker": E_SESSION_QUOTA_EXCEEDED})
            continue
        if remaining < cost:
            budget_pending = True
            pending.append({**_public_task(task), "marker": E_WINDOW_BUDGET_PENDING})
            issues.append(_scheduler_issue(E_WINDOW_BUDGET_PENDING, scope_id, f"remaining budget {remaining} < cost {cost}"))
            continue
        remaining -= cost
        session_points[session_id] = used + cost
        computable.append(task)

    if workers and workers > 1 and computable:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_window, task, config): task["scope_id"] for task in computable}
            computed = []
            for future in as_completed(futures):
                computed.append(future.result())
    else:
        computed = [compute_window(task, config) for task in computable]

    windows: list[dict[str, Any]] = list(cached_windows.values()) + computed
    windows.sort(key=lambda item: session_order_key(item["session_id"], item["end_turn"], item["length"]))
    pending.sort(key=lambda item: tuple(item["order_key"]))
    completed_sessions = sorted(
        {
            task["session_id"]
            for task in tasks
            if task["scope_id"] not in {item["scope_id"] for item in pending}
            and task["session_id"] not in aborted_sessions
        }
    )
    scheduler_state = {
        "budget_total": budget_total,
        "budget_consumed": budget_total - remaining,
        "budget_remaining": remaining,
        "session_points": dict(sorted(session_points.items())),
        "session_quota": quota,
        "pending": pending,
        "pending_task_ids": [item["scope_id"] for item in pending if item["marker"] == E_WINDOW_BUDGET_PENDING],
        "quota_pending_task_ids": [item["scope_id"] for item in pending if item["marker"] == E_SESSION_QUOTA_EXCEEDED],
        "aborted_sessions": sorted(aborted_sessions),
        "completed_session_cursor": completed_sessions[-1] if completed_sessions else "",
        "completed_sessions": completed_sessions,
        "reuse_hits": reuse_hits,
        "workers": workers,
    }
    return windows, scheduler_state, issues


def _scheduler_issue(code: str, target: str, message: str) -> Any:
    from .model import Issue

    return Issue(code, "window_scheduler", target, message)


def build_partitions(
    windows: list[dict[str, Any]],
    max_turns_by_session: dict[str, int],
    config: Config,
    cached_partitions: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cached_partitions = cached_partitions or {}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        by_session.setdefault(window["session_id"], []).append(window)
    partitions: list[dict[str, Any]] = []
    reuse: list[dict[str, str]] = []

    for session_id in sorted(max_turns_by_session):
        max_turn = max_turns_by_session[session_id]
        intervals: list[tuple[str, int, int]] = []
        start = 1
        for index, boundary in enumerate(config.partition_boundaries, start=1):
            if start > max_turn:
                break
            end = min(int(boundary), max_turn)
            if end >= start:
                intervals.append((f"{session_id}:P{index:02d}", start, end))
            start = int(boundary) + 1
        if start <= max_turn:
            intervals.append((f"{session_id}:P{len(intervals) + 1:02d}", start, max_turn))

        session_windows = sorted(by_session.get(session_id, []), key=lambda w: (w["end_turn"], w["length"]))
        for partition_index, (scope_id, start_turn, end_turn) in enumerate(intervals, start=1):
            members = [w for w in session_windows if start_turn <= int(w["end_turn"]) <= end_turn]
            if not members:
                continue
            acc = 0
            weight_sum = 0
            for window in members:
                span = int(window["span"])
                acc += int(window["mean_fixed_scaled"]) * span
                weight_sum += span
            decision = boundary_decision(acc, weight_sum, config)
            boundary_fp = fingerprint({"boundaries": list(config.partition_boundaries), "interval": [start_turn, end_turn]})
            window_keys = [w["cache_key"] for w in members]
            key = fingerprint(
                {
                    "stage": "partition",
                    "scope_id": scope_id,
                    "window_keys": window_keys,
                    "weighted_acc": acc,
                    "weight_sum": weight_sum,
                    "boundary_fingerprint": boundary_fp,
                    "quant_scheme": config.quant_scheme,
                    "places": config.fixed_decimal_places,
                    "exchange_version": config.exchange_version,
                }
            )
            if key in cached_partitions:
                partitions.append(cached_partitions[key])
                reuse.append({"scope_id": scope_id, "cache_key": key, "stage": "partition"})
                continue
            matching = sum(1 for w in members if w["verdict_code"] == decision["verdict_code"])
            partition = {
                "scope_id": scope_id,
                "session_id": session_id,
                "kind": "partition",
                "partition_index": partition_index,
                "start_turn": start_turn,
                "end_turn": end_turn,
                "span": end_turn - start_turn + 1,
                "turns": list(range(start_turn, end_turn + 1)),
                "window_refs": [w["scope_id"] for w in members],
                "window_keys": window_keys,
                "weighted_acc_scaled": acc,
                "weight_span_sum": weight_sum,
                "synthesis_residual_num": Fraction(acc, weight_sum).numerator - decision["mean_fixed_scaled"] * Fraction(1, 1).denominator,
                "synthesis_residual_den": weight_sum,
                "consistency_score": matching / len(members),
                "consistency_matching_windows": matching,
                "consistency_window_count": len(members),
                "boundary_fingerprint": boundary_fp,
                "detail_summary": {
                    "window_refs": [w["scope_id"] for w in members],
                    "weighted_by": "window_span",
                    "accumulation_order": "session_id:asc,end_turn:asc,length:asc",
                },
                "cache_key": key,
                **decision,
            }
            # Residual is acc/weight_sum minus rounded mean_fixed.
            residual = Fraction(acc, weight_sum) - decision["mean_fixed_scaled"]
            partition["synthesis_residual_num"] = residual.numerator
            partition["synthesis_residual_den"] = residual.denominator
            partitions.append(partition)
    partitions.sort(key=lambda item: session_order_key(item["session_id"], item["end_turn"], item["partition_index"]))
    return partitions, reuse


def summarize(
    records: list[PreferenceRecord],
    scores: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    partitions: list[dict[str, Any]],
    weights: dict[str, int],
) -> dict[str, Any]:
    scores_by_annotator: dict[str, list[dict[str, Any]]] = {annotator: [] for annotator in sorted(weights)}
    for score in scores:
        scores_by_annotator.setdefault(score["annotator_id"], []).append(score)

    annotator_summary: list[dict[str, Any]] = []
    for annotator_id in sorted(scores_by_annotator):
        rows = sorted(scores_by_annotator.get(annotator_id, []), key=lambda r: (r["session_id"], r["turn"]))
        count = len(rows)
        stable = sum(1 for row in rows if row["category"] == "stable")
        critical = sum(1 for row in rows if row["category"] == "critical")
        conflict = sum(1 for row in rows if row["category"] == "conflict")
        q_total = ordered_sum(int(row["quantized_scaled"]) for row in rows)
        mean_fixed = round_half_up(q_total, count) if count else 0
        entry = {
            "annotator_id": annotator_id,
            "weight_bp": weights.get(annotator_id),
            "participated": count,
            "stable_count": stable,
            "critical_count": critical,
            "conflict_count": conflict,
            "stable_ratio": stable / count if count else 0.0,
            "critical_ratio": critical / count if count else 0.0,
            "conflict_ratio": conflict / count if count else 0.0,
            "weighted_score_mean_fixed_scaled": mean_fixed,
            "weighted_score_mean_permille": round_half_up(q_total * 1000, count * (10 ** int(rows[0]["decimal_places"]))) if count else 0,
            "degenerate_all_conflict": count > 0 and conflict == count,
            "summary_order": len(annotator_summary),
        }
        annotator_summary.append(entry)

    windows_by_session: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        windows_by_session.setdefault(window["session_id"], []).append(window)
    partitions_by_session: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        partitions_by_session.setdefault(partition["session_id"], []).append(partition)

    session_summary: list[dict[str, Any]] = []
    for session_id in sorted(set(windows_by_session) | set(partitions_by_session)):
        session_windows = sorted(windows_by_session.get(session_id, []), key=lambda w: (w["end_turn"], w["length"]))
        session_partitions = sorted(partitions_by_session.get(session_id, []), key=lambda p: p["end_turn"])
        means = sorted(int(w["mean_permille"]) for w in session_windows)
        verdict_distribution = {str(code): 0 for code in (-1, 0, 1)}
        for window in session_windows:
            verdict_distribution[str(window["verdict_code"])] += 1
        partition_consistency = [
            {
                "scope_id": partition["scope_id"],
                "consistency_score": partition["consistency_score"],
                "verdict_code": partition["verdict_code"],
            }
            for partition in session_partitions
        ]
        overall_consistency = (
            sum(p["consistency_score"] for p in session_partitions) / len(session_partitions)
            if session_partitions
            else 0.0
        )
        middle = means[len(means) // 2] if means else 0
        session_summary.append(
            {
                "session_id": session_id,
                "window_count": len(session_windows),
                "partition_count": len(session_partitions),
                "mean_permille_values": means,
                "mean_distribution": {
                    "min": means[0] if means else 0,
                    "median": middle,
                    "max": means[-1] if means else 0,
                    "verdict_codes": verdict_distribution,
                },
                "partition_consistency": partition_consistency,
                "consistency_score": overall_consistency,
            }
        )

    batch_summary = {
        "record_count": len(records),
        "session_count": len({r.session_id for r in records}),
        "annotator_count": len(weights),
        "window_count": len(windows),
        "partition_count": len(partitions),
        "boundary_window_count": sum(1 for w in windows if w["boundary"]),
        "boundary_partition_count": sum(1 for p in partitions if p["boundary"]),
        "degenerate_annotators": [a["annotator_id"] for a in annotator_summary if a["degenerate_all_conflict"]],
        "verdict_code_counts": {
            "windows": {str(code): sum(1 for w in windows if w["verdict_code"] == code) for code in (-1, 0, 1)},
            "partitions": {str(code): sum(1 for p in partitions if p["verdict_code"] == code) for code in (-1, 0, 1)},
        },
        "mean_partition_consistency": (
            sum(p["consistency_score"] for p in partitions) / len(partitions) if partitions else 0.0
        ),
        "accumulation_order": "annotator_id:asc;session_id:asc;turn:asc",
    }
    return {
        "annotator_summary": annotator_summary,
        "session_summary": session_summary,
        "batch_summary": batch_summary,
    }


def make_ledger_book(
    windows: list[dict[str, Any]],
    partitions: list[dict[str, Any]],
    summary: dict[str, Any],
    config: Config,
) -> LedgerBook:
    entries: list[LedgerEntry] = []
    for item in [*windows, *partitions]:
        exact = Fraction(item["mean_permille_exact_num"], item["mean_permille_exact_den"])
        entries.append(
            LedgerEntry(
                scope_id=item["scope_id"],
                kind=item["kind"],
                session_id=item["session_id"],
                mean_permille=int(item["mean_permille"]),
                verdict_code=int(item["verdict_code"]),
                span=int(item["span"]),
                turn_refs=tuple(item["turns"]),
                start_turn=int(item["start_turn"]),
                end_turn=int(item["end_turn"]),
                window_refs=tuple(item.get("window_refs", [])),
                float_mean=float(exact / 1000),
                verdict_name=item["verdict_name"],
                quant_scheme=config.quant_scheme,
                detail_summary=item.get("detail_summary"),
                precision_label=f"{config.quant_scheme}:P{config.fixed_decimal_places}:half_up_permille",
                cache_key=item["cache_key"],
                source_exchange="engine",
                source_generation=config.ledger_generation,
                missing_fields=(),
            )
        )
    book = LedgerBook(
        entries=entries,
        generation=config.ledger_generation,
        exchange_version=config.exchange_version,
        metadata={"batch_summary": summary["batch_summary"], "config_fingerprint": fingerprint(config.to_dict())},
    )
    book.metadata["score_weights"] = list(config.score_weights)
    book.metadata["quant_scheme"] = config.quant_scheme
    book.metadata["fixed_decimal_places"] = config.fixed_decimal_places
    return book
