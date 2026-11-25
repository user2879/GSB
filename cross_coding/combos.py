from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from .hashing import fingerprint, sha256_hex


def pair_fingerprint(pair: tuple[str, str], table: dict[str, Any]) -> str:
    left, right = pair
    return fingerprint(
        "candidate-pair",
        {
            "pair": list(pair),
            "column_fingerprints": {
                left: table["columns"][left]["fingerprint"],
                right: table["columns"][right]["fingerprint"],
            },
        },
    )


def _is_connected(cols: tuple[str, ...], adjacency: dict[str, set[str]]) -> bool:
    if len(cols) <= 1:
        return True
    chosen = set(cols)
    seen = {cols[0]}
    stack = [cols[0]]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor in chosen and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == chosen


def _ordered_unique(values: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    unique = set(values)
    return sorted(unique, key=lambda item: (len(item), item))


def enumerate_combinations(
    policy: dict[str, Any],
    table: dict[str, Any],
    *,
    encoding_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw_pairs = [tuple(pair) for pair in policy.get("candidate_pairs", [])]
    pairs = sorted({tuple(sorted(pair)) for pair in raw_pairs})
    active_columns = sorted({name for pair in pairs for name in pair})
    adjacency: dict[str, set[str]] = {name: set() for name in active_columns}
    pair_fingerprints: dict[str, str] = {}
    for pair in pairs:
        left, right = pair
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        pair_fingerprints["\0".join(pair)] = pair_fingerprint(pair, table)

    max_order = int(policy["max_order"])
    mode = policy.get("combination_enumeration", "all_subsets")
    max_size = min(max_order, len(active_columns))
    enumerated: list[tuple[str, ...]] = []

    if mode == "all_subsets":
        universe = active_columns
        for size in range(1, max_size + 1):
            enumerated.extend(combinations(universe, size))
    elif mode == "pair_seeded":
        if max_order >= 1:
            enumerated.extend((name,) for name in active_columns)
        if max_order >= 2:
            enumerated.extend(pairs)
        for size in range(3, max_size + 1):
            for candidate in combinations(active_columns, size):
                candidate_set = set(candidate)
                if any(set(pair).issubset(candidate_set) for pair in pairs):
                    enumerated.append(candidate)
    else:
        if max_order >= 1:
            enumerated.extend((name,) for name in active_columns)
        if max_order >= 2:
            enumerated.extend(pairs)
        for size in range(3, max_size + 1):
            for candidate in combinations(active_columns, size):
                if _is_connected(candidate, adjacency):
                    enumerated.append(candidate)

    ordered = _ordered_unique(enumerated)
    profile = encoding_profile or {
        "encoding_width_bits": policy.get("encoding_width_bits"),
        "fixed_point_bits": policy.get("fixed_point_bits"),
        "quantization_levels": policy.get("quantization_levels"),
    }

    result: list[dict[str, Any]] = []
    for columns in ordered:
        reusable_pairs = [
            pair for pair in pairs if pair[0] in columns and pair[1] in columns
        ]
        reusable_pair_fps = {
            "\0".join(pair): pair_fingerprints["\0".join(pair)]
            for pair in reusable_pairs
        }
        content_fp = fingerprint(
            "combination-content",
            {
                "columns": list(columns),
                "order": len(columns),
                "column_fingerprints": {
                    name: table["columns"][name]["fingerprint"] for name in columns
                },
                "reusable_pairs": [list(pair) for pair in reusable_pairs],
                "pair_fingerprints": reusable_pair_fps,
        "enumeration_mode": mode,
            },
        )
        effective_fp = fingerprint(
            "combination",
            {"content_fingerprint": content_fp, "encoding_profile": profile},
        )
        result.append(
            {
                "columns": list(columns),
                "order": len(columns),
                "canonical_key": "|".join(columns),
                "content_fingerprint": content_fp,
                "combo_fingerprint": effective_fp,
                "fingerprint": effective_fp,
                "column_fingerprints": {
                    name: table["columns"][name]["fingerprint"] for name in columns
                },
                "reusable_pairs": [list(pair) for pair in reusable_pairs],
                "reusable_pair_fingerprints": reusable_pair_fps,
                "reusable_pair_salts": {
                    key: int(sha256_hex(value)[:16], 16)
                    for key, value in reusable_pair_fps.items()
                },
                "task_sort_key": task_sort_key(effective_fp, "combo-task"),
                "task_sort_keys": {
                    stage: task_sort_key(effective_fp, stage)
                    for stage in ("enumeration", "coding", "cardinality", "adjudication")
                },
            }
        )
    return result


def task_sort_key(combo_fingerprint: str, stage: str) -> str:
    return sha256_hex({"combo_fingerprint": combo_fingerprint, "stage": stage})


def find_combo(combinations_list: list[dict[str, Any]], columns: Iterable[str]) -> dict[str, Any] | None:
    target = tuple(sorted(columns))
    for combo in combinations_list:
        if tuple(combo["columns"]) == target:
            return combo
    return None
