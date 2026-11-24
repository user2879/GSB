from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import CONFIG_INVALID, Failure
from .fixedpoint import GEAR_ORDER
from .hashing import fingerprint


STAGE_TYPES = {
    "standardize",
    "bucket",
    "log",
    "interaction",
    "clip",
}
REQUIRED_STAGE_ORDER = list(STAGE_TYPES)
MAX_THRESHOLD_BP = 10_000
MAX_DECIMAL_WIDTH = 18
MAX_WORKERS = 32


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @property
    def channels(self) -> list[str]:
        return list(self.raw["data_source"]["channels"])

    @property
    def stages(self) -> list[dict[str, Any]]:
        return list(self.raw["stages"])

    @property
    def stage_names(self) -> list[str]:
        return [stage["name"] for stage in self.stages]

    @property
    def max_workers(self) -> int:
        return int(self.raw.get("concurrency", {}).get("max_workers", 1))

    @property
    def point_budget(self) -> int:
        return int(self.raw.get("point_budget_per_batch", 10**12))

    @property
    def gear(self) -> str:
        return self.raw["fixed_point"]["quantization_gear"]

    @property
    def decimal_width(self) -> int:
        return int(self.raw["fixed_point"]["decimal_width"])

    @property
    def tolerance_q(self) -> int:
        fp = self.raw.get("fixed_point", {})
        return int(fp.get("attribution_tolerance_q", 0))

    def canonical(self) -> dict[str, Any]:
        return _canonical_config(self.raw)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def read_json_file(path: str | Path) -> tuple[Any | None, str | None]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    try:
        return (
            json.loads(
                text,
                parse_constant=_reject_constant,
                parse_float=__import__("decimal").Decimal,
            ),
            None,
        )
    except json.JSONDecodeError as exc:
        return None, f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except ValueError as exc:
        return None, str(exc)


def load_and_validate_config(
    path: str | Path,
    injection_id: str | None = None,
) -> tuple[LoadedConfig | None, list[Failure]]:
    path = Path(path)
    failures: list[Failure] = []
    raw, parse_error = read_json_file(path)
    if parse_error is not None:
        failures.append(
            Failure(
                CONFIG_INVALID,
                f"config JSON parse error: {parse_error}",
                path=str(path),
                injection_id=injection_id,
            )
        )
        return None, failures
    if not isinstance(raw, dict):
        failures.append(
            Failure(
                CONFIG_INVALID,
                "top-level config value must be a JSON object",
                path="$",
                injection_id=injection_id,
            )
        )
        return None, failures

    for path_expr, message in validate_config_object(raw):
        failures.append(
            Failure(
                CONFIG_INVALID,
                message,
                path=path_expr,
                injection_id=injection_id,
            )
        )
    if failures:
        return None, failures
    return LoadedConfig(path=path, raw=raw, fingerprint=fingerprint(_canonical_config(raw))), []


def validate_config_object(raw: dict[str, Any]) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []

    def require(condition: bool, path: str, message: str) -> None:
        if not condition:
            errors.append((path, message))

    require(isinstance(raw, dict), "$", "config must be an object")
    if not isinstance(raw, dict):
        return errors

    ds = raw.get("data_source")
    require(isinstance(ds, dict), "$.data_source", "data_source must be an object")
    if isinstance(ds, dict):
        require_int(errors, ds.get("seed"), "$.data_source.seed")
        channels = ds.get("channels")
        require(
            isinstance(channels, list)
            and len(channels) > 0
            and all(isinstance(x, str) and x for x in channels),
            "$.data_source.channels",
            "channels must be a non-empty list of strings",
        )
        if isinstance(channels, list):
            require(
                len(channels) == len(set(channels)),
                "$.data_source.channels",
                "channel names must be unique",
            )
        require_positive_int(
            errors, ds.get("samples_per_batch"), "$.data_source.samples_per_batch"
        )
        rate = ds.get("missing_rate")
        require(
            isinstance(rate, (int, float, Decimal))
            and not isinstance(rate, bool)
            and Decimal(0) <= rate <= Decimal(1),
            "$.data_source.missing_rate",
            "missing_rate must be between 0 and 1",
        )
        bitmap = ds.get("null_bitmap", [])
        require(
            isinstance(bitmap, list),
            "$.data_source.null_bitmap",
            "null_bitmap must be a list",
        )

    stages = raw.get("stages")
    require(isinstance(stages, list) and len(stages) == 5, "$.stages", "five stages are required")
    if isinstance(stages, list):
        names: list[str] = []
        for i, stage in enumerate(stages):
            p = f"$.stages[{i}]"
            if not isinstance(stage, dict):
                errors.append((p, "stage must be an object"))
                continue
            name = stage.get("name")
            if not isinstance(name, str) or not name:
                errors.append((f"{p}.name", "stage name must be a non-empty string"))
            else:
                names.append(name)
            stype = stage.get("type")
            if stype not in STAGE_TYPES:
                errors.append((f"{p}.type", f"stage type must be one of {sorted(STAGE_TYPES)}"))
            require_positive_int(errors, stage.get("scale"), f"{p}.scale")
            params = stage.get("params")
            require(isinstance(params, dict), f"{p}.params", "params must be an object")
            upstream = stage.get("upstream", stage.get("upstream_stage"))
            if upstream is not None:
                require(
                    isinstance(upstream, (str, list)),
                    f"{p}.upstream",
                    "upstream must be a stage name, list of names, or 'source'",
                )
        if len(names) != len(set(names)):
            errors.append(("$.stages", "stage names must be unique"))
        if not errors or all(isinstance(s, dict) for s in stages):
            errors.extend(validate_stage_graph(stages))

    windows = raw.get("windows")
    require(isinstance(windows, dict), "$.windows", "windows must be an object")
    if isinstance(windows, dict):
        for window_name in ("reference", "observation"):
            window = windows.get(window_name)
            p = f"$.windows.{window_name}"
            require(isinstance(window, dict), p, f"{window_name} window is required")
            if isinstance(window, dict):
                require_positive_int(errors, window.get("start_seq"), f"{p}.start_seq")
                require_positive_int(errors, window.get("end_seq"), f"{p}.end_seq")
                start = window.get("start_seq")
                end = window.get("end_seq")
                if isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool):
                    require(start <= end, p, "start_seq must not exceed end_seq")

    thresholds = raw.get("thresholds_bp")
    require(isinstance(thresholds, dict) and thresholds, "$.thresholds_bp", "thresholds_bp must be a non-empty object")
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            require_int(errors, value, f"$.thresholds_bp.{key}")
            if isinstance(value, int) and not isinstance(value, bool):
                require(
                    0 <= value <= MAX_THRESHOLD_BP,
                    f"$.thresholds_bp.{key}",
                    f"threshold must be an integer in [0, {MAX_THRESHOLD_BP}]",
                )

    fp = raw.get("fixed_point")
    require(isinstance(fp, dict), "$.fixed_point", "fixed_point must be an object")
    if isinstance(fp, dict):
        width = fp.get("decimal_width")
        require_positive_int(errors, width, "$.fixed_point.decimal_width")
        if isinstance(width, int) and not isinstance(width, bool):
            require(
                width <= MAX_DECIMAL_WIDTH,
                "$.fixed_point.decimal_width",
                f"decimal_width must not exceed {MAX_DECIMAL_WIDTH}",
            )
        gear = fp.get("quantization_gear")
        require(
            gear in GEAR_ORDER,
            "$.fixed_point.quantization_gear",
            f"quantization_gear must be one of {GEAR_ORDER}",
        )
        tol = fp.get("attribution_tolerance_q", 0)
        require_nonnegative_int(errors, tol, "$.fixed_point.attribution_tolerance_q")

    concurrency = raw.get("concurrency", {})
    require(isinstance(concurrency, dict), "$.concurrency", "concurrency must be an object")
    if isinstance(concurrency, dict):
        workers = concurrency.get("max_workers", 1)
        require_positive_int(errors, workers, "$.concurrency.max_workers")
        if isinstance(workers, int) and not isinstance(workers, bool):
            require(workers <= MAX_WORKERS, "$.concurrency.max_workers", f"max_workers must be <= {MAX_WORKERS}")

    require_positive_int(errors, raw.get("point_budget_per_batch"), "$.point_budget_per_batch")
    require_positive_int(errors, raw.get("snapshot_interval_batches"), "$.snapshot_interval_batches")
    return errors


def require_int(errors: list[tuple[str, str]], value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append((path, "value must be an integer"))


def require_positive_int(errors: list[tuple[str, str]], value: Any, path: str) -> None:
    require_int(errors, value, path)
    if isinstance(value, int) and not isinstance(value, bool) and value <= 0:
        errors.append((path, "value must be a positive integer"))


def require_nonnegative_int(errors: list[tuple[str, str]], value: Any, path: str) -> None:
    require_int(errors, value, path)
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        errors.append((path, "value must be a non-negative integer"))


def validate_stage_graph(stages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    names = {s.get("name") for s in stages if isinstance(s, dict)}
    graph: dict[str, list[str]] = {}
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            continue
        name = stage["name"]
        upstream = stage.get("upstream", stage.get("upstream_stage"))
        if upstream is None:
            refs: list[str] = []
        elif isinstance(upstream, str):
            refs = [] if upstream == "source" else [upstream]
        else:
            refs = [x for x in upstream if x != "source"]
        graph[name] = refs
        for ref in refs:
            if ref not in names:
                errors.append((f"$.stages.{name}.upstream", f"upstream stage does not exist: {ref}"))

    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        if node in state:
            if state[node] == 1:
                cycle = " -> ".join(stack + [node])
                errors.append(("$.stages", f"stage reference cycle detected: {cycle}"))
            return
        if node not in graph:
            return
        state[node] = 1
        for nxt in graph[node]:
            visit(nxt, stack + [node])
        state[node] = 2

    for name in graph:
        visit(name, [])
    return sorted(set(errors))


def topological_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {s["name"]: s for s in stages}
    visited: set[str] = set()
    result: list[dict[str, Any]] = []

    def visit(name: str) -> None:
        if name in visited or name not in by_name:
            return
        stage = by_name[name]
        upstream = stage.get("upstream", stage.get("upstream_stage"))
        refs: list[str] = []
        if isinstance(upstream, str) and upstream != "source":
            refs = [upstream]
        elif isinstance(upstream, list):
            refs = [x for x in upstream if x != "source"]
        for ref in refs:
            visit(ref)
        visited.add(name)
        result.append(stage)

    for stage in stages:
        visit(stage["name"])
    return result


def _canonical_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only injection annotations before hashing."""

    clone = json.loads(json.dumps(raw, default=str))
    clone.pop("injection_id", None)
    clone.pop("fault_injections", None)
    return clone
