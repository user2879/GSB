"""Parameterized haptic effect graph model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import cos, pi, sin
from typing import Any, Iterable, Mapping

from .canonical import stable_hash
from .constants import CHANNELS, CHANNEL_INDEX

PRIMITIVE_KINDS = ("sine", "pulse_train", "envelope")
BLEND_MODES = ("replace", "max", "sum")
PARAMETER_KEYS = (
    "amplitude",
    "frequency_hz",
    "duration_us",
    "start_us",
    "channels",
    "phase_deg",
    "duty_cycle",
    "attack_us",
    "release_us",
)


@dataclass(frozen=True)
class Primitive:
    id: str
    kind: str
    channels: tuple[str, ...]
    start_us: int
    duration_us: int
    amplitude: float
    frequency_hz: float
    phase_deg: float = 0.0
    duty_cycle: float = 0.5
    attack_us: int = 0
    release_us: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("primitive id must be non-empty")
        if self.kind not in PRIMITIVE_KINDS:
            raise ValueError(f"unsupported primitive kind: {self.kind!r}")
        if not self.channels:
            raise ValueError(f"primitive {self.id!r} must bind at least one channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"primitive {self.id!r} has duplicate channels")
        unknown = [channel for channel in self.channels if channel not in CHANNELS]
        if unknown:
            raise ValueError(f"primitive {self.id!r} has unknown channels: {unknown}")
        if self.start_us < 0 or self.duration_us <= 0:
            raise ValueError(f"primitive {self.id!r} has invalid time interval")
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError(f"primitive {self.id!r} amplitude must be in [0,1]")
        if not 0.0 <= self.frequency_hz <= 2000.0:
            raise ValueError(f"primitive {self.id!r} frequency must be in [0,2000]")
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError(f"primitive {self.id!r} duty_cycle must be in [0,1]")
        if self.attack_us < 0 or self.release_us < 0:
            raise ValueError(f"primitive {self.id!r} envelope stages must be non-negative")

    @property
    def end_us(self) -> int:
        return self.start_us + self.duration_us

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def active_at(self, timestamp_us: int) -> bool:
        return self.start_us <= timestamp_us < self.end_us

    def contribution(self, timestamp_us: int) -> float:
        """Return the non-negative amplitude contributed at a microsecond tick."""
        if not self.active_at(timestamp_us):
            return 0.0
        local_us = timestamp_us - self.start_us
        phase = (self.phase_deg % 360.0) * pi / 180.0
        if self.kind == "sine":
            return self.amplitude * (0.5 + 0.5 * sin(2.0 * pi * self.frequency_hz * local_us / 1_000_000.0 + phase))
        if self.kind == "pulse_train":
            if self.frequency_hz == 0.0:
                return 0.0
            period_us = 1_000_000.0 / self.frequency_hz
            fraction = (local_us % period_us) / period_us
            return self.amplitude if fraction < self.duty_cycle else 0.0
        # Envelope fragment: a shaped amplitude contour. Attack and release are
        # clipped so that malformed source values can never yield >1 amplitude.
        attack = min(self.attack_us, self.duration_us)
        release = min(self.release_us, self.duration_us - attack)
        if attack > 0 and local_us < attack:
            envelope = local_us / attack
        elif release > 0 and local_us >= self.duration_us - release:
            envelope = max(0.0, (self.duration_us - local_us) / release)
        else:
            envelope = 1.0
        # A cosine-shaped carrier makes the envelope fragment useful for
        # exercising per-frame amplitude quantization.
        carrier = 0.5 + 0.5 * sin(2.0 * pi * self.frequency_hz * local_us / 1_000_000.0 + phase)
        return self.amplitude * max(0.0, min(1.0, envelope * carrier))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Primitive":
        data = dict(value)
        data["channels"] = tuple(data["channels"])
        return cls(**data)


@dataclass(frozen=True)
class TimeSlice:
    id: str
    start_us: int
    end_us: int
    primitive_ids: tuple[str, ...]
    blend: str = "max"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("slice id must be non-empty")
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError(f"slice {self.id!r} has invalid interval")
        if not self.primitive_ids:
            raise ValueError(f"slice {self.id!r} must contain primitives")
        if len(set(self.primitive_ids)) != len(self.primitive_ids):
            raise ValueError(f"slice {self.id!r} has duplicate primitive ids")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"slice {self.id!r} has invalid blend mode")

    def overlaps_frame(self, timestamp_us: int, frame_period_us: int) -> bool:
        frame_end = timestamp_us + frame_period_us
        return self.start_us < frame_end and self.end_us > timestamp_us

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimeSlice":
        data = dict(value)
        data["primitive_ids"] = tuple(data["primitive_ids"])
        return cls(**data)


@dataclass(frozen=True)
class CompositionLayer:
    id: str
    slice_ids: tuple[str, ...]
    blend: str = "max"
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("layer id must be non-empty")
        if not self.slice_ids:
            raise ValueError(f"layer {self.id!r} must contain slices")
        if len(set(self.slice_ids)) != len(self.slice_ids):
            raise ValueError(f"layer {self.id!r} has duplicate slice ids")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"layer {self.id!r} has invalid blend mode")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompositionLayer":
        data = dict(value)
        data["slice_ids"] = tuple(data["slice_ids"])
        return cls(**data)


@dataclass(frozen=True)
class EffectGraph:
    id: str
    duration_us: int
    frame_period_us: int
    primitives: tuple[Primitive, ...]
    slices: tuple[TimeSlice, ...]
    layers: tuple[CompositionLayer, ...]
    batch_size: int = 16

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("graph id must be non-empty")
        if self.duration_us <= 0 or self.frame_period_us <= 0:
            raise ValueError("graph duration and frame period must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        primitive_ids = [primitive.id for primitive in self.primitives]
        if len(set(primitive_ids)) != len(primitive_ids):
            raise ValueError("duplicate primitive ids")
        slice_ids = [time_slice.id for time_slice in self.slices]
        if len(set(slice_ids)) != len(slice_ids):
            raise ValueError("duplicate slice ids")
        known_primitives = set(primitive_ids)
        known_slices = set(slice_ids)
        for time_slice in self.slices:
            missing = set(time_slice.primitive_ids) - known_primitives
            if missing:
                raise ValueError(f"slice {time_slice.id!r} references unknown primitives: {sorted(missing)}")
        used_slices: set[str] = set()
        for layer in self.layers:
            missing = set(layer.slice_ids) - known_slices
            if missing:
                raise ValueError(f"layer {layer.id!r} references unknown slices: {sorted(missing)}")
            used_slices.update(layer.slice_ids)
        unused = known_slices - used_slices
        if unused:
            raise ValueError(f"unreferenced slices: {sorted(unused)}")
        for primitive in self.primitives:
            if primitive.start_us >= self.duration_us:
                raise ValueError(f"primitive {primitive.id!r} starts after graph duration")

    @property
    def frame_count(self) -> int:
        return (self.duration_us + self.frame_period_us - 1) // self.frame_period_us

    @property
    def primitive_by_id(self) -> dict[str, Primitive]:
        return {primitive.id: primitive for primitive in self.primitives}

    @property
    def slice_by_id(self) -> dict[str, TimeSlice]:
        return {time_slice.id: time_slice for time_slice in self.slices}

    def validate_integrity(self) -> None:
        for primitive in self.primitives:
            if primitive.channel_count != len(primitive.channels):
                raise AssertionError("channel count mismatch")
            if set(primitive.channels) - set(CHANNELS):
                raise AssertionError("unknown channel")

    def with_primitive_changes(self, changes: Mapping[str, Mapping[str, Any]]) -> "EffectGraph":
        unknown_primitive_ids = set(changes) - set(self.primitive_by_id)
        if unknown_primitive_ids:
            raise ValueError(f"unknown primitive ids: {sorted(unknown_primitive_ids)}")
        updated: list[Primitive] = []
        for primitive in self.primitives:
            patch = changes.get(primitive.id)
            if not patch:
                updated.append(primitive)
                continue
            unknown_keys = set(patch) - set(PARAMETER_KEYS)
            if unknown_keys:
                raise ValueError(f"unsupported primitive parameters: {sorted(unknown_keys)}")
            values = dict(patch)
            if "channels" in values:
                values["channels"] = tuple(values["channels"])
            updated.append(replace(primitive, **values))
        return replace(self, primitives=tuple(updated))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "duration_us": self.duration_us,
            "frame_period_us": self.frame_period_us,
            "batch_size": self.batch_size,
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "slices": [time_slice.to_dict() for time_slice in self.slices],
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.canonical_dict())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectGraph":
        data = dict(value)
        return cls(
            id=data["id"],
            duration_us=int(data["duration_us"]),
            frame_period_us=int(data["frame_period_us"]),
            batch_size=int(data.get("batch_size", 16)),
            primitives=tuple(Primitive.from_dict(item) for item in data["primitives"]),
            slices=tuple(TimeSlice.from_dict(item) for item in data["slices"]),
            layers=tuple(CompositionLayer.from_dict(item) for item in data["layers"]),
        )


@dataclass(frozen=True)
class Contribution:
    primitive_id: str
    channel: str
    amplitude: float
    frequency_hz: float
    slice_id: str
    layer_id: str


def _combine_contributions(contributions: list[Contribution], blend: str) -> tuple[float, float] | None:
    usable = [item for item in contributions if item.amplitude > 0.0]
    if not usable:
        return None
    if blend == "replace":
        selected = usable[-1]
        return selected.amplitude, selected.frequency_hz
    if blend == "max":
        selected = max(usable, key=lambda item: (item.amplitude, CHANNEL_INDEX[item.channel], item.primitive_id))
        return selected.amplitude, selected.frequency_hz
    total = sum(item.amplitude for item in usable)
    if total <= 0.0:
        return None
    amplitude = min(1.0, total)
    frequency = sum(item.amplitude * item.frequency_hz for item in usable) / total
    return amplitude, frequency


def evaluate_channel(graph: EffectGraph, channel: str, timestamp_us: int) -> tuple[float, float] | None:
    """Evaluate a channel at one microsecond tick using layers and blend modes."""
    graph_primitives = graph.primitive_by_id
    graph_slices = graph.slice_by_id
    layer_results: list[tuple[int, float, float, int]] = []
    for layer_order, layer in enumerate(graph.layers):
        contributions: list[Contribution] = []
        for slice_order, slice_id in enumerate(layer.slice_ids):
            time_slice = graph_slices[slice_id]
            if not (time_slice.start_us <= timestamp_us < time_slice.end_us):
                continue
            for primitive_order, primitive_id in enumerate(time_slice.primitive_ids):
                primitive = graph_primitives[primitive_id]
                if channel not in primitive.channels:
                    continue
                amplitude = primitive.contribution(timestamp_us)
                if amplitude > 0.0:
                    contributions.append(
                        Contribution(
                            primitive_id=primitive_id,
                            channel=channel,
                            amplitude=amplitude,
                            frequency_hz=primitive.frequency_hz,
                            slice_id=slice_id,
                            layer_id=layer.id,
                        )
                    )
                # Preserve deterministic replace order even when all values are
                # currently zero by allowing a zero contribution marker.
                elif layer.blend == "replace":
                    contributions.append(
                        Contribution(
                            primitive_id=primitive_id,
                            channel=channel,
                            amplitude=0.0,
                            frequency_hz=primitive.frequency_hz,
                            slice_id=slice_id,
                            layer_id=layer.id,
                        )
                    )
        combined = _combine_contributions(contributions, layer.blend)
        if combined is not None:
            layer_results.append((layer.priority, combined[0], combined[1], layer_order))
    if not layer_results:
        return None
    # Higher-priority layer replaces lower-priority layers.  Equal priorities
    # resolve by declaration order, making the result fully deterministic.
    selected = max(layer_results, key=lambda item: (item[0], item[3]))
    return selected[1], selected[2]


def iter_active_primitives(graph: EffectGraph, timestamp_us: int) -> Iterable[Primitive]:
    for primitive in graph.primitives:
        if primitive.active_at(timestamp_us):
            yield primitive
