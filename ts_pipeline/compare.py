"""Field-level report comparison for incremental/full parity evidence."""

from __future__ import annotations

from typing import Any

from .canonical import to_builtin


SEMANTIC_CONFIG_FIELDS = ("backtest", "models", "quantization", "scheduler", "snapshot")


def parity_view(report: dict[str, Any]) -> dict[str, Any]:
    view = {
        "schema_version": report.get("schema_version"),
        "report_version": report.get("report_version"),
        "samples": report.get("samples"),
        "windows": report.get("windows"),
        "models": report.get("models"),
        "metric_matrix": report.get("metric_matrix"),
        "over_tolerance_windows": report.get("over_tolerance_windows"),
        "cache_plan": report.get("cache_plan"),
        "scheduler": report.get("scheduler"),
        "failures": report.get("failures"),
        "injections": report.get("injections"),
        "summary": report.get("summary"),
    }
    config = report.get("config")
    if isinstance(config, dict):
        config_semantics = {
            key: config.get(key) for key in SEMANTIC_CONFIG_FIELDS
        }
        scheduler_config = config_semantics.get("scheduler")
        if isinstance(scheduler_config, dict):
            scheduler_config = dict(scheduler_config)
            scheduler_config.pop("concurrency", None)
            config_semantics["scheduler"] = scheduler_config
        view["config_semantics"] = config_semantics
    scheduler_view = view.get("scheduler")
    if isinstance(scheduler_view, dict):
        scheduler_view = dict(scheduler_view)
        scheduler_view.pop("concurrency", None)
        view["scheduler"] = scheduler_view
    return to_builtin(view)


def deep_diff(left: Any, right: Any, path: str = "$", limit: int = 80) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if len(differences) >= limit:
        return differences
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right, "kind": "type"}]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append({"path": f"{path}.{key}", "left": "<missing>", "right": right[key], "kind": "missing_left"})
            elif key not in right:
                differences.append({"path": f"{path}.{key}", "left": left[key], "right": "<missing>", "kind": "missing_right"})
            else:
                differences.extend(deep_diff(left[key], right[key], f"{path}.{key}", limit - len(differences)))
            if len(differences) >= limit:
                break
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            differences.append({"path": f"{path}.length", "left": len(left), "right": len(right), "kind": "length"})
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(deep_diff(left_item, right_item, f"{path}[{index}]", limit - len(differences)))
            if len(differences) >= limit:
                break
        return differences
    if left != right:
        differences.append({"path": path, "left": left, "right": right, "kind": "value"})
    return differences


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_view = parity_view(left)
    right_view = parity_view(right)
    differences = deep_diff(left_view, right_view)
    return {
        "equal": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "left_failures": len(left.get("failures", [])),
        "right_failures": len(right.get("failures", [])),
    }
