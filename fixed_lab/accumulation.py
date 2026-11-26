"""Fixed-order integer accumulation for weighted normal equations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cache import ContentCache
from .dataset import Dataset
from .features import QuantizedDataset
from .fixedpoint import FixedParams, round_div
from .hashing import canonical_sha256, decimal_fingerprint_value, short_key
from .ledger import ReuseLedger
from .scheduler import ScheduledTask, plan_batches, run_scheduler


@dataclass
class Quadratics:
    g: list[list[int]]
    b: list[int]
    accum_scale: int
    operations: list[dict[str, Any]] = field(default_factory=list)
    schedule: dict[str, Any] | None = None


def _term_key(material: dict[str, Any]) -> str:
    return short_key("mac-term", material)


def build_quadratics(
    dataset: Dataset,
    qd: QuantizedDataset,
    *,
    cache: ContentCache,
    ledger: ReuseLedger,
    worker_count: int = 1,
    mac_budget: int = 10_000,
    column_quota: int = 10_000,
) -> Quadratics:
    params: FixedParams = qd.params
    sample_ids = qd.sample_ids
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    col_fps = dataset.column_fingerprints()
    p = len(qd.columns)

    def feature_cell_fp(sample_id: str, col: int) -> str:
        row = dataset.rows[sample_id]
        return canonical_sha256({
            "kind": "feature_cell",
            "sample_id": sample_id,
            "column": qd.columns[col],
            "value": decimal_fingerprint_value(row.features[col]),
        })

    def scalar_cell_fp(sample_id: str, field_name: str) -> str:
        value = dataset.rows[sample_id].target if field_name == "target" else dataset.rows[sample_id].weight
        return canonical_sha256({
            "kind": field_name,
            "sample_id": sample_id,
            "value": decimal_fingerprint_value(value),
        })

    requests: dict[str, dict[str, Any]] = {}
    term_values: dict[str, dict[str, int]] = {}
    tasks: list[ScheduledTask] = []
    plan_tasks: list[ScheduledTask] = []
    sample_material = [[sample_id, dataset.rows[sample_id].fingerprint] for sample_id in sample_ids]

    def make_term(kind: str, i: int, j: int, sample_id: str) -> None:
        idx = sample_index[sample_id]
        artifact = f"mac:{kind}[i={i:02d},j={j:02d}]:sample={sample_id}"
        key_material = {
            "kind": kind,
            "i_name": qd.columns[i],
            "j_name": qd.columns[j],
            "i_cell_fingerprint": feature_cell_fp(sample_id, i),
            "j_cell_fingerprint": feature_cell_fp(sample_id, j),
            "weight_cell_fingerprint": scalar_cell_fp(sample_id, "weight"),
            "fixed": params.fingerprint_material(),
            "accum_digits": params.accum_digits,
        }
        if kind == "rhs":
            key_material["target_cell_fingerprint"] = scalar_cell_fp(sample_id, "target")
        cache_key = _term_key(key_material)
        requests[artifact] = {
            "kind": kind,
            "i": i,
            "j": j,
            "sample_id": sample_id,
            "cache_key": cache_key,
            "material": key_material,
        }
        sort_key = f"{j:02d}:{kind}:{i:02d}:{sample_id}"
        plan_tasks.append(ScheduledTask(
            key=artifact,
            sort_key=sort_key,
            column=qd.columns[j],
            cost=1,
            fn=lambda: None,
        ))
        cached = cache.get(cache_key)
        if cached is not None:
            term_values[artifact] = cached
            ledger.record(
                artifact_id=artifact,
                stage="accumulation_mac",
                cache_key=cache_key,
                cache_hit=True,
                basis="sample_fingerprint+column_fingerprints+fixed_params_match",
                sample_ids=[sample_id],
                columns=[qd.columns[i], qd.columns[j]],
                material={
                    "fixed": params.fingerprint_material(),
                    "sample_fingerprints": [[sample_id, dataset.rows[sample_id].fingerprint]],
                    "column_fingerprints": [col_fps[i], col_fps[j]],
                },
            )
            return

        def compute() -> dict[str, int]:
            row = qd.x[idx]
            weight = qd.w[idx]
            if kind == "g":
                raw = weight * row[i] * row[j]
            else:
                raw = weight * row[i] * qd.y[idx]
            rounded, remainder, direction = round_div(raw, params.accum_divisor)
            return {
                "raw": raw,
                "rounded": rounded,
                "remainder": remainder,
                "direction_code": {"exact": 0, "down": 1, "toward_zero": 2, "up": 3, "half_up": 4}.get(direction, 99),
            }

        tasks.append(ScheduledTask(
            key=artifact,
            sort_key=sort_key,
            column=qd.columns[j],
            cost=1,
            fn=compute,
        ))

    # Stable double order: target column j ascending, source i ascending, sample id ascending.
    for j in range(p):
        for i in range(j + 1):
            for sample_id in sample_ids:
                make_term("g", i, j, sample_id)
        for sample_id in sample_ids:
            make_term("rhs", j, j, sample_id)

    scheduler_result = run_scheduler(
        tasks,
        worker_count=worker_count,
        mac_budget=mac_budget,
        column_quota=column_quota,
    )
    for artifact, value in scheduler_result.values.items():
        req = requests[artifact]
        direction = {0: "exact", 1: "down", 2: "toward_zero", 3: "up", 4: "half_up"}.get(value["direction_code"], "unknown")
        value = dict(value)
        value["direction"] = direction
        term_values[artifact] = value
        cache.put(req["cache_key"], {k: v for k, v in value.items() if k != "direction"})
        ledger.record(
            artifact_id=artifact,
            stage="accumulation_mac",
            cache_key=req["cache_key"],
            cache_hit=False,
            basis="content_key_miss",
            sample_ids=[req["sample_id"]],
            columns=[qd.columns[req["i"]], qd.columns[req["j"]]],
            material={
                "fixed": params.fingerprint_material(),
                "sample_fingerprints": [[req["sample_id"], dataset.rows[req["sample_id"]].fingerprint]],
                "column_fingerprints": [col_fps[req["i"]], col_fps[req["j"]]],
            },
        )

    g = [[0 for _ in range(p)] for _ in range(p)]
    b = [0 for _ in range(p)]
    operations: list[dict[str, Any]] = []

    for j in range(p):
        for i in range(j + 1):
            total = 0
            for sample_id in sample_ids:
                artifact = f"mac:g[i={i:02d},j={j:02d}]:sample={sample_id}"
                value = term_values[artifact]
                total += value["rounded"]
                operations.append({
                    "element": "G",
                    "i": i,
                    "j": j,
                    "sample_id": sample_id,
                    **value,
                })
            g[i][j] = total
            g[j][i] = total

        total_b = 0
        for sample_id in sample_ids:
            artifact = f"mac:rhs[i={j:02d},j={j:02d}]:sample={sample_id}"
            value = term_values[artifact]
            total_b += value["rounded"]
            operations.append({
                "element": "b",
                "i": j,
                "j": j,
                "sample_id": sample_id,
                **value,
            })
        b[j] = total_b

    schedule = {
        "task_order": [task.key for task in sorted(plan_tasks, key=lambda task: (task.sort_key, task.key))],
        "budget_plan": plan_batches(
            plan_tasks,
            mac_budget=mac_budget,
            column_quota=column_quota,
        ),
        "execution_batches": scheduler_result.batches,
        "miss_task_count": len(tasks),
        "budget": {"mac_budget_per_batch": mac_budget, "single_column_quota": column_quota},
    }
    return Quadratics(g=g, b=b, accum_scale=params.accum_scale, operations=operations, schedule=schedule)
