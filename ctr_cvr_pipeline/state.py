from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .config import FeatureConfig, TrainingConfig
from .errors import Failure


@dataclass(frozen=True)
class Sample:
    request_id: str
    item_id: str
    dense: dict[str, float]
    sparse: dict[str, float]
    click: int
    conversion: int
    conversion_delay_ms: int | None
    event_time_ms: int
    batch_id: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.item_id)

    @property
    def sample_id(self) -> str:
        return f"{self.request_id}:{self.item_id}"

    def with_labels(
        self,
        *,
        click: int | None = None,
        conversion: int | None = None,
        conversion_delay_ms: int | None | object = ...,
    ) -> "Sample":
        delay = (
            self.conversion_delay_ms
            if conversion_delay_ms is ...
            else conversion_delay_ms
        )
        if delay is not None and not isinstance(delay, int):
            raise TypeError("conversion_delay_ms must be int or None")
        return replace(
            self,
            click=self.click if click is None else click,
            conversion=self.conversion if conversion is None else conversion,
            conversion_delay_ms=delay,
        )

    def feature_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "item_id": self.item_id,
            "dense": self.dense,
            "sparse": self.sparse,
            "event_time_ms": self.event_time_ms,
            "batch_id": self.batch_id,
        }


@dataclass
class PipelineState:
    samples: dict[tuple[str, str], Sample]
    explicit_edges: dict[str, tuple[float, ...]]
    quantization_tier: str
    fixed_fractional_bits: int
    conversion_delay_cutoff_ms: int
    current_event_time_ms: int
    failures: list[Failure] = field(default_factory=list)
    applied_commands: list[dict[str, Any]] = field(default_factory=list)
    batch_event_times: dict[str, int] = field(default_factory=dict)

    def sorted_samples(self) -> list[Sample]:
        return [self.samples[key] for key in sorted(self.samples)]

    def clone(self) -> "PipelineState":
        return PipelineState(
            samples=dict(self.samples),
            explicit_edges=dict(self.explicit_edges),
            quantization_tier=self.quantization_tier,
            fixed_fractional_bits=self.fixed_fractional_bits,
            conversion_delay_cutoff_ms=self.conversion_delay_cutoff_ms,
            current_event_time_ms=self.current_event_time_ms,
            failures=list(self.failures),
            applied_commands=[dict(item) for item in self.applied_commands],
            batch_event_times=dict(self.batch_event_times),
        )


def initial_state(
    samples: list[Sample],
    config: FeatureConfig,
    training: TrainingConfig,
    base_failures: list[Failure] | None = None,
    base_batch_id: str = "b0000",
) -> PipelineState:
    by_key: dict[tuple[str, str], Sample] = {}
    failures = list(base_failures or [])
    for sample in samples:
        if sample.key in by_key:
            failures.append(
                Failure(
                    code="E_SAMPLE_INVALID",
                    message="duplicate request/item pair",
                    request_id=sample.request_id,
                    item_id=sample.item_id,
                    sample_id=sample.sample_id,
                    field="item_id",
                )
            )
            continue
        by_key[sample.key] = sample
    return PipelineState(
        samples=by_key,
        explicit_edges={},
        quantization_tier=config.quantization_tier,
        fixed_fractional_bits=config.fixed_fractional_bits,
        conversion_delay_cutoff_ms=training.conversion_delay_cutoff_ms,
        current_event_time_ms=training.initial_event_time_ms,
        failures=failures,
        batch_event_times={base_batch_id: training.initial_event_time_ms},
    )
