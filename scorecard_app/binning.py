from __future__ import annotations

from typing import Any

from .config import EffectiveColumn
from .fingerprint import stable_float


def _is_bad(value: Any, positive_value: Any) -> bool:
    return value == positive_value


def _stats_for_rows(
    values: list[Any],
    target: list[Any],
    positive_value: Any,
) -> tuple[int, int, int, int]:
    total = len(values)
    bad = sum(1 for value in target if _is_bad(value, positive_value))
    good = total - bad
    return total, good, bad, sum(1 for value in values if value is None)


def _chi_square(good_left: int, bad_left: int, good_right: int, bad_right: int) -> float:
    total = good_left + bad_left + good_right + bad_right
    if total == 0:
        return 0.0
    good_total = good_left + good_right
    bad_total = bad_left + bad_right
    expected = (
        good_total * (good_left + bad_left) / total,
        good_total * (good_right + bad_right) / total,
        bad_total * (good_left + bad_left) / total,
        bad_total * (good_right + bad_right) / total,
    )
    observed = (good_left, good_right, bad_left, bad_right)
    value = 0.0
    for obs, exp in zip(observed, expected):
        if exp > 0.0:
            value += (obs - exp) ** 2 / exp
    return stable_float(value)


def _finalize_numeric_indices(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, bin_value in enumerate(bins):
        bin_value["index"] = index
    return bins


def _merge_numeric_pair(bins: list[dict[str, Any]], left_index: int, right_index: int) -> None:
    left = bins[left_index]
    right = bins[right_index]
    left["upper"] = right["upper"]
    left["count"] += right["count"]
    left["good"] += right["good"]
    left["bad"] += right["bad"]
    del bins[right_index]
    _finalize_numeric_indices(bins)


def _numeric_limit(max_bins: int, has_missing: bool) -> int:
    # For max_bins >= 2, reserve one total slot for the mandatory missing box.
    if has_missing and max_bins > 1:
        return max_bins - 1
    return max_bins


def _numeric_prebins(
    values: list[Any],
    target: list[Any],
    positive_value: Any,
    prebins: int,
) -> tuple[list[dict[str, Any]], int, int, int, bool]:
    total = len(values)
    total_bad = sum(1 for value in target if _is_bad(value, positive_value))
    total_good = total - total_bad

    by_value: dict[int, list[int]] = {}
    for value, target_value in zip(values, target):
        if value is None:
            continue
        number = int(value)
        stats = by_value.setdefault(number, [0, 0, 0])
        stats[0] += 1
        if _is_bad(target_value, positive_value):
            stats[1] += 1
        else:
            stats[2] += 1

    if not by_value:
        bins: list[dict[str, Any]] = []
    else:
        ordered_values = sorted(by_value)
        wanted = min(int(prebins), len(ordered_values))
        target_size = max(1.0, sum(by_value[value][0] for value in ordered_values) / wanted)
        chunks: list[list[int]] = []
        current: list[int] = []
        current_count = 0
        for position, value in enumerate(ordered_values):
            current.append(value)
            current_count += by_value[value][0]
            remaining = len(ordered_values) - position - 1
            if current and len(chunks) < wanted - 1 and current_count >= target_size and remaining:
                chunks.append(current)
                current = []
                current_count = 0
        if current:
            chunks.append(current)

        bins = []
        for chunk in chunks:
            count = sum(by_value[value][0] for value in chunk)
            bad = sum(by_value[value][1] for value in chunk)
            good = sum(by_value[value][2] for value in chunk)
            bins.append(
                {
                    "index": len(bins),
                    "lower": int(chunk[0]),
                    "upper": int(chunk[-1]),
                    "count": int(count),
                    "good": int(good),
                    "bad": int(bad),
                    "is_missing": False,
                }
            )

    missing_count = sum(1 for value in values if value is None)
    has_missing = missing_count > 0
    if has_missing:
        missing_bad = sum(
            1
            for value, target_value in zip(values, target)
            if value is None and _is_bad(target_value, positive_value)
        )
        bins.append(
            {
                "index": len(bins),
                "lower": None,
                "upper": None,
                "count": int(missing_count),
                "good": int(missing_count - missing_bad),
                "bad": int(missing_bad),
                "is_missing": True,
            }
        )
    return bins, total_bad, total_good, total, has_missing


def bin_numeric(
    column: str,
    values: list[Any],
    target: list[Any],
    positive_value: Any,
    params: EffectiveColumn,
    prebins: int,
) -> dict[str, Any]:
    bins, total_bad, total_good, total, has_missing = _numeric_prebins(
        values, target, positive_value, prebins
    )
    numeric_limit = _numeric_limit(params.max_bins, has_missing)
    total_rows = max(1, total)

    while True:
        numeric_bins = [bin_value for bin_value in bins if not bin_value["is_missing"]]
        too_many = len(numeric_bins) > numeric_limit
        too_small = any(bin_value["count"] / total_rows < params.min_bin_ratio for bin_value in numeric_bins)
        if (not too_many and not too_small) or len(numeric_bins) < 2:
            break

        candidates: list[tuple[float, int, int, int]] = []
        for index in range(len(numeric_bins) - 1):
            left = numeric_bins[index]
            right = numeric_bins[index + 1]
            statistic = _chi_square(left["good"], left["bad"], right["good"], right["bad"])
            candidates.append((statistic, int(left["lower"]), int(right["upper"]), index))
        # Ties are resolved by the left boundary ascending, then by the original
        # adjacent position as a final stable guard.
        _, _, _, selected = min(candidates, key=lambda item: (item[0], item[1], item[3]))
        _merge_numeric_pair(bins, selected, selected + 1)

    return {
        "column": column,
        "type": "numeric",
        "params": {
            "max_bins": params.max_bins,
            "min_bin_ratio": stable_float(params.min_bin_ratio),
            "prebins": prebins,
        },
        "bins": _finalize_numeric_indices(bins),
        "category_to_group": {},
        "missing_index": next((item["index"] for item in bins if item["is_missing"]), None),
        "total": total,
        "total_bad": total_bad,
        "total_good": total_good,
        "nonmissing_bin_count": sum(1 for item in bins if not item["is_missing"]),
    }


def _group_key(group: dict[str, Any]) -> str:
    return min(group["categories"])


def _group_rate(group: dict[str, Any]) -> float:
    return group["bad"] / group["count"] if group["count"] else 0.0


def _merge_category_groups(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "categories": sorted(left["categories"] + right["categories"]),
        "count": int(left["count"] + right["count"]),
        "good": int(left["good"] + right["good"]),
        "bad": int(left["bad"] + right["bad"]),
    }


def bin_categorical(
    column: str,
    values: list[Any],
    target: list[Any],
    positive_value: Any,
    params: EffectiveColumn,
    prebins: int,
) -> dict[str, Any]:
    total = len(values)
    total_bad = sum(1 for value in target if _is_bad(value, positive_value))
    total_good = total - total_bad

    category_stats: dict[str, list[int]] = {}
    missing_count = 0
    missing_bad = 0
    for value, target_value in zip(values, target):
        if value is None:
            missing_count += 1
            if _is_bad(target_value, positive_value):
                missing_bad += 1
            continue
        label = str(value)
        stats = category_stats.setdefault(label, [0, 0, 0])
        stats[0] += 1
        if _is_bad(target_value, positive_value):
            stats[1] += 1
        else:
            stats[2] += 1

    groups: list[dict[str, Any]] = [
        {
            "categories": [label],
            "count": int(stats[0]),
            "bad": int(stats[1]),
            "good": int(stats[2]),
        }
        for label, stats in sorted(category_stats.items())
    ]
    has_missing = missing_count > 0
    group_limit = _numeric_limit(params.max_bins, has_missing)
    total_rows = max(1, total)

    while len(groups) > 1:
        over_limit = len(groups) > group_limit
        under_ratio = any(group["count"] / total_rows < params.min_bin_ratio for group in groups)
        if not over_limit and not under_ratio:
            break

        ordered = sorted(groups, key=lambda group: (stable_float(_group_rate(group)), _group_key(group)))
        victim = min(ordered, key=lambda group: (group["count"], _group_key(group)))
        victim_index = ordered.index(victim)
        neighbor_indices = []
        if victim_index > 0:
            neighbor_indices.append(victim_index - 1)
        if victim_index + 1 < len(ordered):
            neighbor_indices.append(victim_index + 1)
        neighbor_index = min(
            neighbor_indices,
            key=lambda index: (
                stable_float(abs(_group_rate(victim) - _group_rate(ordered[index]))),
                ordered[index]["count"],
                _group_key(ordered[index]),
            ),
        )

        left_index = min(victim_index, neighbor_index)
        right_index = max(victim_index, neighbor_index)
        merged = _merge_category_groups(ordered[left_index], ordered[right_index])
        del ordered[right_index]
        del ordered[left_index]
        ordered.append(merged)
        groups = ordered

    groups = sorted(groups, key=_group_key)
    bins: list[dict[str, Any]] = []
    category_to_group: dict[str, int] = {}
    for index, group in enumerate(groups):
        categories = sorted(group["categories"])
        categories_count = len(categories)
        label = f"group_{index:03d}"
        if categories_count == 1:
            label = categories[0]
        bins.append(
            {
                "index": index,
                "label": label,
                "categories": categories,
                "count": int(group["count"]),
                "good": int(group["good"]),
                "bad": int(group["bad"]),
                "target_rate": stable_float(_group_rate(group)),
                "is_missing": False,
            }
        )
        for category in categories:
            category_to_group[category] = index

    missing_index = None
    if has_missing:
        missing_index = len(bins)
        bins.append(
            {
                "index": missing_index,
                "label": "__missing__",
                "categories": [],
                "count": int(missing_count),
                "good": int(missing_count - missing_bad),
                "bad": int(missing_bad),
                "target_rate": stable_float(missing_bad / missing_count) if missing_count else 0.0,
                "is_missing": True,
            }
        )

    return {
        "column": column,
        "type": "categorical",
        "params": {
            "max_bins": params.max_bins,
            "min_bin_ratio": stable_float(params.min_bin_ratio),
            "prebins": prebins,
        },
        "bins": bins,
        "category_to_group": category_to_group,
        "missing_index": missing_index,
        "total": total,
        "total_bad": total_bad,
        "total_good": total_good,
        "nonmissing_bin_count": sum(1 for item in bins if not item["is_missing"]),
    }


def bin_column(
    name: str,
    column_type: str,
    values: list[Any],
    target: list[Any],
    positive_value: Any,
    params: EffectiveColumn,
    prebins: int,
) -> dict[str, Any]:
    if column_type == "numeric":
        return bin_numeric(name, values, target, positive_value, params, prebins)
    if column_type == "categorical":
        return bin_categorical(name, values, target, positive_value, params, prebins)
    raise ValueError(f"unsupported column type for binning: {column_type!r}")
