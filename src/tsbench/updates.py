"""Loading and per-batch application of the JSON update plan."""

from __future__ import annotations

import json

from .errors import E_CONFIG_INVALID, ConfigError, SeriesFailure


def load_update_plan(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            plan = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"update plan not found: {path}", E_CONFIG_INVALID) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid update plan JSON: {exc}", E_CONFIG_INVALID) from exc

    if not isinstance(plan, dict) or not isinstance(plan.get("batches"), list):
        raise ConfigError("update plan must contain a 'batches' list", E_CONFIG_INVALID)
    batches = plan["batches"]
    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("batch_id"), int):
            raise ConfigError("each batch needs an integer batch_id", E_CONFIG_INVALID)
        operations = batch.get("operations", [])
        if not isinstance(operations, list):
            raise ConfigError(
                f"batch {batch['batch_id']} operations must be a list",
                E_CONFIG_INVALID,
            )
        for op in operations:
            if not isinstance(op, dict) or not isinstance(op.get("series_id"), str):
                raise ConfigError(
                    f"batch {batch['batch_id']} has an invalid operation",
                    E_CONFIG_INVALID,
                )
            if op.get("type") not in ("append", "tail_correction", "replace_range"):
                raise ConfigError(
                    f"batch {batch['batch_id']}: unknown operation type",
                    E_CONFIG_INVALID,
                )
    return batches


def apply_batch(
    series_map: dict[str, "SeriesData"],  # noqa: F821
    known_ids: set[str],
    batch: dict,
    block_size: int,
) -> dict[str, dict]:
    """Apply every operation of one batch; returns per-series change records."""
    changes: dict[str, dict] = {}
    for op in sorted(batch["operations"], key=lambda o: o["series_id"]):
        sid = op["series_id"]
        if sid not in known_ids:
            raise SeriesFailure(sid, "update targets an unknown series")
        series = series_map[sid]
        changed, fragments = apply_operation_guarded(series, op, block_size)
        record = changes.setdefault(
            sid,
            {"indices": set(), "fragments": set(), "operations": []},
        )
        record["indices"].update(changed)
        record["fragments"].update(fragments)
        record["operations"].append(op["type"])
    return changes


def apply_operation_guarded(series, op: dict, block_size: int):
    from .synthetic import apply_operation

    return apply_operation(series, op, block_size)
