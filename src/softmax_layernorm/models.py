from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FORMAT_VERSION = 1


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, int]
    seed: int
    distribution: str
    distribution_params: dict[str, Any]
    special_values: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "seed": self.seed,
            "distribution": self.distribution,
            "distribution_params": dict(self.distribution_params),
            "special_values": [dict(item) for item in self.special_values],
        }


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    sequence: int
    rows: int
    cols: int
    softmax_axis: int
    layernorm_axis: int
    epsilon: float
    stabilize: bool
    layernorm_scale: float = 1.0
    reference_precision: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CallSpec":
        return cls(
            call_id=str(raw["call_id"]),
            sequence=int(raw["sequence"]),
            rows=int(raw["rows"]),
            cols=int(raw["cols"]),
            softmax_axis=int(raw["softmax_axis"]),
            layernorm_axis=int(raw["layernorm_axis"]),
            epsilon=float(raw["epsilon"]),
            stabilize=bool(raw["stabilize"]),
            layernorm_scale=float(raw.get("layernorm_scale", 1.0)),
            reference_precision=raw.get("reference_precision"),
        )


@dataclass(frozen=True)
class RunConfig:
    interaction_budget_ms: float = 50.0
    thread_pool_workers: int = 4
    snapshot_interval_commands: int = 3
    layernorm_scale: float = 1.0
    snapshot_root: str = "output"
    failure_dir: str = "output/failures"
    report_dir: str = "output/reports"
    resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedRun:
    config: RunConfig
    tensor_spec: TensorSpec
    tensor: Any
    calls: list[CallSpec]
    config_hash: str
    source_input_fingerprint: str
    current_input_fingerprint: str
    call_description_hash: str
    tensor_description_hash: str


@dataclass(frozen=True)
class CommandSpec:
    index: int
    command_id: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def command_type(self) -> str:
        return str(self.raw.get("type", self.raw.get("action", self.raw.get("op", "set_params"))))

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "command_id": self.command_id, **self.raw}
