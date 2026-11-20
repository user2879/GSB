from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import load_json
from .errors import fail


PARTITION_MODES = ("modulo", "contiguous")
CONFIG_FIELDS = {
    "seed",
    "shard_count",
    "partition_mode",
    "steps",
    "sample_count",
    "input_dim",
    "hidden_dim",
    "classes",
}


@dataclass(frozen=True)
class InteropConfig:
    seed: int
    shard_count: int
    partition_mode: str
    steps: tuple[int, ...]
    sample_count: int
    input_dim: int
    hidden_dim: int
    classes: int
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shard_count": self.shard_count,
            "partition_mode": self.partition_mode,
            "steps": list(self.steps),
            "sample_count": self.sample_count,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "classes": self.classes,
        }


def _require_int(raw: dict[str, Any], name: str, *, minimum: int, maximum: int) -> int:
    if name not in raw:
        fail("E_CONFIG", f"missing config field {name}", path=name, details={"field": name})
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int):
        fail(
            "E_CONFIG",
            f"config field {name} must be an integer",
            path=name,
            details={"field": name, "value": value},
        )
    if value < minimum or value > maximum:
        fail(
            "E_CONFIG",
            f"config field {name} is outside [{minimum}, {maximum}]",
            path=name,
            details={"field": name, "value": value, "min": minimum, "max": maximum},
        )
    return value


def load_config(source: str | Path | dict[str, Any]) -> InteropConfig:
    source_path: Path | None = None
    if isinstance(source, dict):
        raw = dict(source)
    else:
        source_path = Path(source)
        loaded = load_json(source_path, code="E_CONFIG")
        if not isinstance(loaded, dict):
            fail("E_CONFIG", "config root must be a JSON object", path="$", details={"field": "$"})
        raw = loaded

    unknown = sorted(set(raw) - CONFIG_FIELDS)
    if unknown:
        fail(
            "E_CONFIG",
            f"unknown config field {unknown[0]}",
            path=unknown[0],
            details={"field": unknown[0]},
        )

    seed = _require_int(raw, "seed", minimum=0, maximum=2**63 - 1)
    shard_count = _require_int(raw, "shard_count", minimum=1, maximum=256)
    sample_count = _require_int(raw, "sample_count", minimum=shard_count, maximum=1_000_000)
    input_dim = _require_int(raw, "input_dim", minimum=1, maximum=4096)
    hidden_dim = _require_int(raw, "hidden_dim", minimum=1, maximum=4096)
    classes = _require_int(raw, "classes", minimum=2, maximum=1024)

    partition_mode = raw.get("partition_mode")
    if not isinstance(partition_mode, str):
        fail(
            "E_CONFIG",
            "config field partition_mode must be a string",
            path="partition_mode",
            details={"field": "partition_mode", "value": partition_mode},
        )
    if partition_mode not in PARTITION_MODES:
        fail(
            "E_CONFIG",
            f"partition_mode must be one of {list(PARTITION_MODES)}",
            path="partition_mode",
            details={"field": "partition_mode", "value": partition_mode},
        )

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        fail(
            "E_CONFIG",
            "config field steps must be a non-empty array",
            path="steps",
            details={"field": "steps", "value": raw_steps},
        )
    if len(raw_steps) > 10_000:
        fail(
            "E_CONFIG",
            "config field steps has too many entries",
            path="steps",
            details={"field": "steps", "max_entries": 10_000},
        )
    steps: list[int] = []
    for index, value in enumerate(raw_steps):
        path = f"steps[{index}]"
        if isinstance(value, bool) or not isinstance(value, int):
            fail(
                "E_CONFIG",
                "step must be a non-negative integer",
                path=path,
                details={"field": path, "value": value},
            )
        if value < 0 or value > 1_000_000_000:
            fail(
                "E_CONFIG",
                "step is outside [0, 1000000000]",
                path=path,
                details={"field": path, "value": value, "min": 0, "max": 1_000_000_000},
            )
        if value in steps:
            fail("E_CONFIG", "steps must be unique", path=path, details={"field": path, "value": value})
        steps.append(value)
    steps.sort()
    if steps[0] != 0:
        fail(
            "E_CONFIG",
            "steps must include the genesis step 0",
            path="steps",
            details={"field": "steps", "missing": 0},
        )

    return InteropConfig(
        seed=seed,
        shard_count=shard_count,
        partition_mode=partition_mode,
        steps=tuple(steps),
        sample_count=sample_count,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        classes=classes,
        source_path=source_path,
    )


def step_id(step: int) -> str:
    return f"step-{step:08d}"
