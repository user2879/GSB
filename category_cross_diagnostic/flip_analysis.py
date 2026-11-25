"""Adjacent width-gear recomputation and flip attribution."""

from __future__ import annotations

from typing import Any

from .budget import adjudicate_combo
from .encoding import EncodingResult, encode_combo
from .enumeration import Combo
from .sample_data import TableData
from .config import Strategy
from .widths import Gear, adjacent_pairs, encoding_ladder, quantization_ladder


def build_flip_report(
    combos: list[Combo], table: TableData, strategy: Strategy
) -> dict[str, Any]:
    encoding_gears = encoding_ladder(strategy)
    profile_gears = quantization_ladder(strategy)
    encoding_views = {
        gear.gear_id: _compute_view(combos, table, strategy, gear)
        for gear in encoding_gears
    }
    profile_views = {
        gear.gear_id: _compute_view(combos, table, strategy, gear)
        for gear in profile_gears
    }

    combo_records: dict[str, list[dict[str, Any]]] = {
        combo.combo_id: [] for combo in combos
    }
    axis_records: dict[str, list[dict[str, Any]]] = {
        "encoding_width": [],
        "fixed_point_quantization": [],
    }

    for axis, gears, views in (
        ("encoding_width", encoding_gears, encoding_views),
        ("fixed_point_quantization", profile_gears, profile_views),
    ):
        for left_gear, right_gear in adjacent_pairs(gears):
            for combo in combos:
                record = _attribute_pair(
                    axis=axis,
                    combo=combo,
                    left_gear=left_gear,
                    right_gear=right_gear,
                    left=views[left_gear.gear_id][combo.combo_id],
                    right=views[right_gear.gear_id][combo.combo_id],
                    strategy=strategy,
                )
                combo_records[combo.combo_id].append(record)
                axis_records[axis].append(record)

    summary = {
        "adjacent_pair_checks": sum(len(records) for records in combo_records.values()),
        "flip_count": 0,
        "unique_count_flips": 0,
        "collision_count_flips": 0,
        "decision_flips": 0,
        "combinations_with_flips": [],
    }
    for combo in combos:
        records = combo_records[combo.combo_id]
        combo_flip = any(record["flip"] for record in records)
        if combo_flip:
            summary["combinations_with_flips"].append(combo.combo_id)
        for record in records:
            summary["flip_count"] += int(record["flip"])
            summary["unique_count_flips"] += int(record["unique_count_changed"])
            summary["collision_count_flips"] += int(record["collision_count_changed"])
            summary["decision_flips"] += int(record["decision_changed"])

    return {
        "schema": "width-flip-attribution/v1",
        "encoding_gears": [_gear_dict(gear) for gear in encoding_gears],
        "quantization_gears": [_gear_dict(gear) for gear in profile_gears],
        "summary": summary,
        "axes": axis_records,
        "combo_flip_attribution": combo_records,
    }


def _compute_view(
    combos: list[Combo], table: TableData, strategy: Strategy, gear: Gear
) -> dict[str, tuple[EncodingResult, dict[str, Any]]]:
    view: dict[str, tuple[EncodingResult, dict[str, Any]]] = {}
    consumed = 0
    for combo in combos:
        encoding = encode_combo(
            combo,
            table,
            encoding_width=gear.encoding_width,
            fixed_point_width=gear.fixed_point_width,
            quantization_levels=gear.quantization_levels,
            preview_limit=0,
        )
        decision = adjudicate_combo(
            combo,
            table.column_fingerprints,
            encoding.cardinality_diagnosis(),
            encoding.entries,
            row_count=table.row_count,
            encoding_width=gear.encoding_width,
            consumed_points=consumed,
            strategy=strategy,
            width_gear_fp=gear.fingerprint,
        )
        # What-if ladder continues accounting so every combination has an
        # adjacent-gear diagnosis.  The production pending queue separately
        # stops at the first point-budget exhaustion.
        consumed = int(decision["point"]["consumed_after"])
        view[combo.combo_id] = (encoding, decision)
    return view


def _attribute_pair(
    *,
    axis: str,
    combo: Combo,
    left_gear: Gear,
    right_gear: Gear,
    left: tuple[EncodingResult, dict[str, Any]],
    right: tuple[EncodingResult, dict[str, Any]],
    strategy: Strategy,
) -> dict[str, Any]:
    encoding_left, decision_left = left
    encoding_right, decision_right = right
    card_left = encoding_left.cardinality_diagnosis()
    card_right = encoding_right.cardinality_diagnosis()

    unique_delta = encoding_right.unique_code_count - encoding_left.unique_code_count
    collision_delta = encoding_right.collision_count - encoding_left.collision_count
    pair_collision_delta = (
        encoding_right.colliding_pair_count - encoding_left.colliding_pair_count
    )
    unique_changed = unique_delta != 0
    collision_changed = collision_delta != 0

    decision_keys = (
        "cardinality_verdict",
        "disposition",
        "point_exhausted",
        "dominant_constraint",
    )
    decision_before = {key: decision_left[key] for key in decision_keys}
    decision_after = {key: decision_right[key] for key in decision_keys}
    decision_changed = decision_before != decision_after

    width_changed = left_gear.encoding_width != right_gear.encoding_width
    profile_changed = (
        left_gear.fixed_point_width != right_gear.fixed_point_width
        or left_gear.quantization_levels != right_gear.quantization_levels
    )

    card_margin_before = strategy.cardinality_budget - encoding_left.unique_code_count
    card_margin_after = strategy.cardinality_budget - encoding_right.unique_code_count
    cardinality_boundary_crossed = (card_margin_before >= 0) != (
        card_margin_after >= 0
    )
    point_margin_before = (
        strategy.point_budget - decision_left["point"]["consumed_after"]
    )
    point_margin_after = (
        strategy.point_budget - decision_right["point"]["consumed_after"]
    )
    point_boundary_crossed = (
        decision_left["point_exhausted"] != decision_right["point_exhausted"]
    )
    budget_boundary_crossed = (
        cardinality_boundary_crossed or point_boundary_crossed
    )

    space_pressure_before = max(
        0, encoding_left.true_unique_count - encoding_left.code_space_size
    )
    space_pressure_after = max(
        0, encoding_right.true_unique_count - encoding_right.code_space_size
    )

    source_contributions = {
        "encoding_space_compression": {
            "axis_applies": width_changed,
            "code_space_before": encoding_left.code_space_size,
            "code_space_after": encoding_right.code_space_size,
            "code_space_delta": encoding_right.code_space_size
            - encoding_left.code_space_size,
            "space_pressure_before": space_pressure_before,
            "space_pressure_after": space_pressure_after,
            "space_pressure_delta": space_pressure_after - space_pressure_before,
            "unique_count_effect": unique_delta if width_changed else 0,
            "contribution": space_pressure_after - space_pressure_before,
            "contribution_unit": "true_tuples_without_code_space",
        },
        "collision_count_change": {
            "collision_count_before": encoding_left.collision_count,
            "collision_count_after": encoding_right.collision_count,
            "collision_count_delta": collision_delta,
            "colliding_pair_count_delta": pair_collision_delta,
            "net_excess_before": encoding_left.true_unique_count
            - encoding_left.unique_code_count,
            "net_excess_after": encoding_right.true_unique_count
            - encoding_right.unique_code_count,
            "contribution": collision_delta,
            "contribution_unit": "collapsed_distinct_tuple_excess",
        },
        "budget_boundary": {
            "cardinality_budget": strategy.cardinality_budget,
            "cardinality_margin_before": card_margin_before,
            "cardinality_margin_after": card_margin_after,
            "cardinality_margin_delta": card_margin_after - card_margin_before,
            "cardinality_boundary_crossed": cardinality_boundary_crossed,
            "point_budget": strategy.point_budget,
            "point_margin_before": point_margin_before,
            "point_margin_after": point_margin_after,
            "point_boundary_crossed": point_boundary_crossed,
            "boundary_crossed": budget_boundary_crossed,
            "contribution": (
                card_margin_after - card_margin_before
                if cardinality_boundary_crossed
                else (point_margin_after - point_margin_before if point_boundary_crossed else 0)
            ),
            "contribution_unit": "budget_margin_change",
        },
        "fixed_point_quantization_profile": {
            "axis_applies": profile_changed,
            "salt_changed": profile_changed,
            "unique_count_effect": unique_delta if profile_changed else 0,
            "collision_count_effect": collision_delta if profile_changed else 0,
            "contribution": collision_delta if profile_changed else 0,
            "contribution_unit": "salt_redistribution_collision_delta",
        },
    }

    attribution_sources: list[str] = []
    if width_changed and (unique_changed or collision_changed):
        attribution_sources.append("encoding_space_compression")
    if collision_changed:
        attribution_sources.append("collision_count_change")
    if budget_boundary_crossed and decision_changed:
        attribution_sources.append("budget_boundary")
    if profile_changed and (unique_changed or collision_changed):
        attribution_sources.append("fixed_point_quantization_profile")

    if decision_changed and budget_boundary_crossed:
        decision_driver = "budget_boundary"
    elif width_changed and (unique_changed or collision_changed):
        decision_driver = "encoding_space_compression"
    elif profile_changed and (unique_changed or collision_changed):
        decision_driver = "fixed_point_quantization_profile"
    elif collision_changed:
        decision_driver = "collision_count_change"
    else:
        decision_driver = "none"

    return {
        "axis": axis,
        "gear_before": left_gear.gear_id,
        "gear_after": right_gear.gear_id,
        "gear_fingerprint_before": left_gear.fingerprint,
        "gear_fingerprint_after": right_gear.fingerprint,
        "combo_id": combo.combo_id,
        "combo_fingerprint_before": combo.width_fingerprint(left_gear.fingerprint),
        "combo_fingerprint_after": combo.width_fingerprint(right_gear.fingerprint),
        "fingerprint_changed": left_gear.fingerprint != right_gear.fingerprint,
        "code_sample_before": list(encoding_left.first_row_codes[:8]),
        "code_sample_after": list(encoding_right.first_row_codes[:8]),
        "code_sample_changed": list(encoding_left.first_row_codes[:8])
        != list(encoding_right.first_row_codes[:8]),
        "true_unique_count": encoding_left.true_unique_count,
        "unique_count_before": encoding_left.unique_code_count,
        "unique_count_after": encoding_right.unique_code_count,
        "unique_count_delta": unique_delta,
        "unique_count_changed": unique_changed,
        "collision_count_before": encoding_left.collision_count,
        "collision_count_after": encoding_right.collision_count,
        "collision_count_delta": collision_delta,
        "collision_count_changed": collision_changed,
        "decision_before": decision_before,
        "decision_after": decision_after,
        "decision_changed": decision_changed,
        "flip": unique_changed or collision_changed or decision_changed,
        "attribution_sources": attribution_sources,
        "primary_driver": decision_driver,
        "source_contributions": source_contributions,
    }


def _gear_dict(gear: Gear) -> dict[str, Any]:
    return {
        "gear_id": gear.gear_id,
        "encoding_width": gear.encoding_width,
        "fixed_point_width": gear.fixed_point_width,
        "quantization_levels": gear.quantization_levels,
        "fingerprint": gear.fingerprint,
    }
