"""Deterministic conflict arbitration and feature-set merging."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fingerprint import cell_cache_key, dependency_fingerprint, rule_fingerprint
from .models import Cell, MergeConfig


@dataclass
class MergeOutput:
    rows: list[dict]
    conflicts: list[dict]
    suppressed: list[dict]
    provenance: list[dict]
    dependency_keys: dict


def _source_rank(source: str, config: MergeConfig) -> int:
    return config.source_order.index(source)


def _precision_rank(label: str | None, config: MergeConfig) -> int:
    if label is None:
        return len(config.precision_rank)
    try:
        return config.precision_rank.index(label)
    except ValueError:
        return len(config.precision_rank)


def candidate_dicts(cells: list[Cell]) -> list[dict]:
    return [
        {
            "source_version": cell.source_version,
            "value": None if cell.missing else float(cell.value),
            "precision_label": cell.precision_label,
            "missing": cell.missing,
            "filled": cell.filled,
            "partition_id": cell.partition_id,
        }
        for cell in cells
    ]


def choose_winner(cells: list[Cell], config: MergeConfig) -> tuple[Cell, str]:
    present = [cell for cell in cells if not cell.missing]
    if not present:
        # All sources mark the slot missing: highest-priority source wins so
        # provenance stays deterministic.
        winner = sorted(
            cells, key=lambda c: _source_rank(c.source_version, config)
        )[0]
        return winner, "all_sources_missing"

    values = [cell.value for cell in present]
    spread = max(values) - min(values)
    if spread <= config.numeric_tolerance:
        winner = sorted(
            present, key=lambda c: _source_rank(c.source_version, config)
        )[0]
        return winner, (
            "within_tolerance"
            if len(present) > 1 and spread > 0.0
            else "single_present_value"
            if len(present) == 1
            else "exact_match"
        )

    if config.conflict_priority == "source_version":
        ordered = sorted(
            present, key=lambda c: _source_rank(c.source_version, config)
        )
        return ordered[0], "conflict_source_version_priority"

    if config.conflict_priority == "precision_label":
        ordered = sorted(
            present,
            key=lambda c: (
                _precision_rank(c.precision_label, config),
                _source_rank(c.source_version, config),
            ),
        )
        return ordered[0], "conflict_precision_label_priority"

    # abs_value: largest absolute value wins, source order breaks exact ties
    ordered = sorted(
        present,
        key=lambda c: (-abs(c.value), _source_rank(c.source_version, config)),
    )
    return ordered[0], "conflict_abs_value_priority"


def merge_tables(
    tables: list,
    config: MergeConfig,
    config_hash: str,
    entity_ids: list[str] | None = None,
) -> MergeOutput:
    """Merge one deterministic batch of entities.

    Output ordering matches a single-threaded run over entities sorted by id
    and, within an entity, the whitelist feature order.
    """
    grouped: dict[tuple[str, str], list[Cell]] = {}
    for table in tables:
        for cell in table.cells:
            if cell.feature not in config.feature_whitelist:
                continue
            grouped.setdefault((cell.entity_id, cell.feature), []).append(cell)

    if entity_ids is None:
        entity_ids = sorted({key[0] for key in grouped})

    rows: list[dict] = []
    conflicts: list[dict] = []
    suppressed: list[dict] = []
    provenance: list[dict] = []
    dependency_keys: dict[str, str] = {}

    for entity_id in entity_ids:
        for feature in config.feature_whitelist:
            cells = grouped.get((entity_id, feature))
            if not cells:
                continue
            ordered_cells = sorted(
                cells, key=lambda c: _source_rank(c.source_version, config)
            )
            winner, reason = choose_winner(ordered_cells, config)
            cache_keys = {
                cell.source_version: cell_cache_key(cell, config_hash)
                for cell in ordered_cells
            }
            dep_key = dependency_fingerprint(
                cache_keys.values(), rule_fingerprint(config)
            )
            dependency_keys[f"{entity_id}|{feature}"] = dep_key

            rows.append(
                {
                    "entity_id": entity_id,
                    "feature": feature,
                    "value": winner.value_json(),
                    "missing": winner.missing,
                    "winning_source": winner.source_version,
                    "precision_label": winner.precision_label,
                    "filled": winner.filled,
                    "dependency_key": dep_key,
                }
            )

            present = [cell for cell in ordered_cells if not cell.missing]
            if len(present) > 1:
                values = [cell.value for cell in present]
                spread = max(values) - min(values)
                if spread > config.numeric_tolerance:
                    conflicts.append(
                        {
                            "entity_id": entity_id,
                            "feature": feature,
                            "candidates": candidate_dicts(ordered_cells),
                            "winner_source": winner.source_version,
                            "winner_value": winner.value_json(),
                            "winning_reason": reason,
                            "numeric_tolerance": config.numeric_tolerance,
                            "observed_spread": spread,
                        }
                    )
            for cell in ordered_cells:
                if cell is not winner:
                    suppressed.append(
                        {
                            "entity_id": entity_id,
                            "feature": feature,
                            "suppressed_source": cell.source_version,
                            "suppressed_value": cell.value_json(),
                            "winning_source": winner.source_version,
                            "reason": reason,
                        }
                    )

            provenance.append(
                {
                    "entity_id": entity_id,
                    "feature": feature,
                    "winning_source": winner.source_version,
                    "winning_reason": reason,
                    "candidate_sources": [
                        cell.source_version for cell in ordered_cells
                    ],
                    "candidate_cache_keys": cache_keys,
                    "dependency_key": dep_key,
                }
            )

    sort_identity = lambda item: (item["entity_id"], item["feature"])
    rows.sort(key=sort_identity)
    conflicts.sort(key=sort_identity)
    suppressed.sort(
        key=lambda item: (
            item["entity_id"],
            item["feature"],
            item["suppressed_source"],
        )
    )
    provenance.sort(key=sort_identity)
    return MergeOutput(rows, conflicts, suppressed, provenance, dependency_keys)


def overall_dependency_fingerprint(merge_output: MergeOutput) -> str:
    return dependency_fingerprint(
        merge_output.dependency_keys.values(), "merge-result"
    )
