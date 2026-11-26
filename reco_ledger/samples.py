"""Synthetic exposure samples, batch envelopes and command-stream validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .codes import Codes
from .config import AppConfig, canonical_hash
from .diagnostics import Diagnostic, DiagnosticBag


SAMPLE_FIELDS = {
    "request_id",
    "candidate_id",
    "dense_features",
    "sparse_counts",
    "click",
    "conversion",
    "conversion_callback_ms",
    "sample_version",
    "event_time_ms",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def stable_int(seed: int, *parts: Any, modulo: int) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


@dataclass(frozen=True)
class Sample:
    request_id: str
    candidate_id: str
    dense_features: tuple[float, ...]
    sparse_counts: tuple[int, ...]
    click: int
    conversion: int
    conversion_callback_ms: int
    sample_version: int
    event_time_ms: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.candidate_id)

    @property
    def sample_ref(self) -> str:
        return f"{self.request_id}|{self.candidate_id}|v{self.sample_version}"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"d{index:02d}" for index in range(len(self.dense_features))) + tuple(
            f"s{index:02d}" for index in range(len(self.sparse_counts))
        )

    @property
    def content_fingerprint(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "dense_features": list(self.dense_features),
            "sparse_counts": list(self.sparse_counts),
            "click": self.click,
            "conversion": self.conversion,
            "conversion_callback_ms": self.conversion_callback_ms,
            "sample_version": self.sample_version,
            "event_time_ms": self.event_time_ms,
        }


@dataclass(frozen=True)
class Batch:
    batch_id: str
    training_event_time_ms: int
    samples: tuple[Sample, ...]
    processed_at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "training_event_time_ms": self.training_event_time_ms,
            "processed_at_ms": self.processed_at_ms,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class Command:
    operation: str
    request_id: str
    candidate_id: str
    sample: Sample | None = None
    command_id: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "op": self.operation,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
        }
        if self.sample is not None:
            value["sample"] = self.sample.to_dict()
        if self.command_id is not None:
            value["command_id"] = self.command_id
        return value


@dataclass
class SampleLoadResult:
    samples: list[Sample]
    diagnostics: DiagnosticBag
    accepted_events: list[Sample] = field(default_factory=list)


@dataclass
class CommandLoadResult:
    commands: list[Command]
    diagnostics: DiagnosticBag


def validate_sample_record(
    record: Mapping[str, Any],
    *,
    config: AppConfig,
    source: str,
    entry_index: int | None,
    bag: DiagnosticBag,
    injection_id: str | None = None,
) -> Sample | None:
    if not isinstance(record, Mapping):
        bag.add(
            Codes.TYPE_MISMATCH,
            "sample must be an object",
            source=source,
            entry_index=entry_index,
            injection_id=injection_id,
        )
        return None

    bad = False
    for key in record:
        if key not in SAMPLE_FIELDS:
            bag.add(
                Codes.FIELD_UNKNOWN,
                f"unknown sample field {key}",
                source=source,
                entry_index=entry_index,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True

    for key in ("request_id", "candidate_id", "dense_features", "sparse_counts", "click",
                "conversion", "conversion_callback_ms", "sample_version"):
        if key not in record:
            bag.add(
                Codes.FIELD_MISSING,
                f"missing sample field {key}",
                source=source,
                entry_index=entry_index,
                field_name=key,
                injection_id=injection_id,
            )
            bad = True
    if bad:
        return None

    request_id = record["request_id"]
    candidate_id = record["candidate_id"]
    if not isinstance(request_id, str) or not request_id:
        bag.add(
            Codes.TYPE_MISMATCH,
            "request_id must be a non-empty string",
            source=source,
            entry_index=entry_index,
            field_name="request_id",
            injection_id=injection_id,
        )
        bad = True
    if not isinstance(candidate_id, str) or not candidate_id:
        bag.add(
            Codes.TYPE_MISMATCH,
            "candidate_id must be a non-empty string",
            source=source,
            entry_index=entry_index,
            field_name="candidate_id",
            injection_id=injection_id,
        )
        bad = True

    dense_raw = record["dense_features"]
    if not isinstance(dense_raw, list):
        bag.add(
            Codes.TYPE_MISMATCH,
            "dense_features must be a list",
            source=source,
            entry_index=entry_index,
            field_name="dense_features",
            injection_id=injection_id,
        )
        dense: list[float] | None = None
        bad = True
    else:
        dense = []
        for feature_index, value in enumerate(dense_raw):
            if not _is_number(value):
                bag.add(
                    Codes.TYPE_MISMATCH,
                    f"dense_features[{feature_index}] must be numeric",
                    source=source,
                    entry_index=entry_index,
                    field_name=f"dense_features[{feature_index}]",
                    injection_id=injection_id,
                )
                bad = True
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                bag.add(
                    Codes.VALUE_RANGE,
                    f"dense_features[{feature_index}] must be finite",
                    source=source,
                    entry_index=entry_index,
                    field_name=f"dense_features[{feature_index}]",
                    injection_id=injection_id,
                )
                bad = True
                continue
            if numeric < 0:
                bag.add(
                    Codes.VALUE_RANGE,
                    f"dense_features[{feature_index}] must be non-negative",
                    source=source,
                    entry_index=entry_index,
                    field_name=f"dense_features[{feature_index}]",
                    injection_id=injection_id,
                )
                bad = True
                continue
            dense.append(numeric)
        if len(dense) != config.dense_dim:
            bag.add(
                Codes.VALUE_RANGE,
                f"dense_features length {len(dense)} != configured {config.dense_dim}",
                source=source,
                entry_index=entry_index,
                field_name="dense_features",
                injection_id=injection_id,
            )
            bad = True

    sparse_raw = record["sparse_counts"]
    if not isinstance(sparse_raw, list):
        bag.add(
            Codes.TYPE_MISMATCH,
            "sparse_counts must be a list",
            source=source,
            entry_index=entry_index,
            field_name="sparse_counts",
            injection_id=injection_id,
        )
        sparse: list[int] | None = None
        bad = True
    else:
        sparse = []
        for count_index, value in enumerate(sparse_raw):
            if not _is_int(value):
                bag.add(
                    Codes.TYPE_MISMATCH,
                    f"sparse_counts[{count_index}] must be an integer",
                    source=source,
                    entry_index=entry_index,
                    field_name=f"sparse_counts[{count_index}]",
                    injection_id=injection_id,
                )
                bad = True
                continue
            if value < 0:
                bag.add(
                    Codes.VALUE_RANGE,
                    f"sparse_counts[{count_index}] must be non-negative",
                    source=source,
                    entry_index=entry_index,
                    field_name=f"sparse_counts[{count_index}]",
                    injection_id=injection_id,
                )
                bad = True
                continue
            sparse.append(value)
        if len(sparse) != config.sparse_dim:
            bag.add(
                Codes.VALUE_RANGE,
                f"sparse_counts length {len(sparse)} != configured {config.sparse_dim}",
                source=source,
                entry_index=entry_index,
                field_name="sparse_counts",
                injection_id=injection_id,
            )
            bad = True

    labels: dict[str, int] = {}
    for field_name in ("click", "conversion"):
        value = record[field_name]
        if not _is_int(value):
            bag.add(
                Codes.TYPE_MISMATCH,
                f"{field_name} must be an integer label",
                source=source,
                entry_index=entry_index,
                field_name=field_name,
                injection_id=injection_id,
            )
            bad = True
        elif value not in (0, 1):
            bag.add(
                Codes.VALUE_RANGE,
                f"{field_name} must be 0 or 1",
                source=source,
                entry_index=entry_index,
                field_name=field_name,
                injection_id=injection_id,
            )
            bad = True
        else:
            labels[field_name] = value

    callback = record["conversion_callback_ms"]
    if not _is_int(callback):
        bag.add(
            Codes.TYPE_MISMATCH,
            "conversion_callback_ms must be an integer",
            source=source,
            entry_index=entry_index,
            field_name="conversion_callback_ms",
            injection_id=injection_id,
        )
        bad = True
    elif callback < 0:
        bag.add(
            Codes.VALUE_RANGE,
            "conversion_callback_ms must be non-negative",
            source=source,
            entry_index=entry_index,
            field_name="conversion_callback_ms",
            injection_id=injection_id,
        )
        bad = True

    sample_version = record["sample_version"]
    if not _is_int(sample_version):
        bag.add(
            Codes.TYPE_MISMATCH,
            "sample_version must be an integer",
            source=source,
            entry_index=entry_index,
            field_name="sample_version",
            injection_id=injection_id,
        )
        bad = True
    elif sample_version < 0:
        bag.add(
            Codes.VALUE_RANGE,
            "sample_version must be non-negative",
            source=source,
            entry_index=entry_index,
            field_name="sample_version",
            injection_id=injection_id,
        )
        bad = True

    event_time = record.get("event_time_ms", 0)
    if not _is_int(event_time):
        bag.add(
            Codes.TYPE_MISMATCH,
            "event_time_ms must be an integer",
            source=source,
            entry_index=entry_index,
            field_name="event_time_ms",
            injection_id=injection_id,
        )
        bad = True
    elif event_time < 0:
        bag.add(
            Codes.VALUE_RANGE,
            "event_time_ms must be non-negative",
            source=source,
            entry_index=entry_index,
            field_name="event_time_ms",
            injection_id=injection_id,
        )
        bad = True

    if bad:
        return None

    assert dense is not None and sparse is not None
    return Sample(
        request_id=request_id,
        candidate_id=candidate_id,
        dense_features=tuple(dense),
        sparse_counts=tuple(sparse),
        click=labels["click"],
        conversion=labels["conversion"],
        conversion_callback_ms=callback,
        sample_version=sample_version,
        event_time_ms=event_time,
    )


def load_samples(
    records: Sequence[Mapping[str, Any]],
    *,
    config: AppConfig,
    source: str = "<samples>",
    allow_upsert: bool = False,
    injection_id: str | None = None,
) -> SampleLoadResult:
    """Load a rolling stream while keeping exactly one active row per key."""

    bag = DiagnosticBag()
    active: dict[tuple[str, str], Sample] = {}
    events: list[Sample] = []
    for index, record in enumerate(records):
        sample = validate_sample_record(
            record,
            config=config,
            source=source,
            entry_index=index,
            bag=bag,
            injection_id=injection_id,
        )
        if sample is None:
            continue
        previous = active.get(sample.key)
        if previous is not None:
            if sample.sample_version < previous.sample_version:
                bag.add(
                    Codes.VALUE_RANGE,
                    "sample_version must be monotonic non-decreasing per request/candidate",
                    source=source,
                    entry_index=index,
                    field_name="sample_version",
                    injection_id=injection_id,
                    details={
                        "previous_version": previous.sample_version,
                        "incoming_version": sample.sample_version,
                    },
                )
                continue
            if not allow_upsert:
                bag.add(
                    Codes.DUPLICATE_KEY,
                    "request_id/candidate_id must be unique in an active sample stream",
                    source=source,
                    entry_index=index,
                    injection_id=injection_id,
                    details={"key": list(sample.key)},
                )
                continue
            active[sample.key] = sample
        else:
            active[sample.key] = sample
        events.append(sample)
    return SampleLoadResult(samples=list(active.values()), diagnostics=bag, accepted_events=events)


def parse_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    config: AppConfig,
    source: str = "<batches>",
) -> tuple[list[Batch], DiagnosticBag]:
    bag = DiagnosticBag()
    batches: list[Batch] = []
    active: dict[tuple[str, str], Sample] = {}
    for batch_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            bag.add(
                Codes.TYPE_MISMATCH,
                "batch must be an object",
                source=source,
                entry_index=batch_index,
            )
            continue
        allowed = {
            "batch_id",
            "training_event_time_ms",
            "processed_at_ms",
            "samples",
        }
        bad = False
        for key in record:
            if key not in allowed:
                bag.add(
                    Codes.FIELD_UNKNOWN,
                    f"unknown batch field {key}",
                    source=source,
                    entry_index=batch_index,
                    field_name=key,
                )
                bad = True
        batch_id = record.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            bag.add(
                Codes.TYPE_MISMATCH,
                "batch_id must be a non-empty string",
                source=source,
                entry_index=batch_index,
                field_name="batch_id",
            )
            bad = True
        training_time = record.get("training_event_time_ms")
        if not _is_int(training_time):
            bag.add(
                Codes.TYPE_MISMATCH,
                "training_event_time_ms must be an integer",
                source=source,
                entry_index=batch_index,
                field_name="training_event_time_ms",
            )
            bad = True
        elif training_time < 0:
            bag.add(
                Codes.VALUE_RANGE,
                "training_event_time_ms must be non-negative",
                source=source,
                entry_index=batch_index,
                field_name="training_event_time_ms",
            )
            bad = True
        processed_time = record.get("processed_at_ms", 0)
        if not _is_int(processed_time) or processed_time < 0:
            bag.add(
                Codes.TYPE_MISMATCH
                if not _is_int(processed_time)
                else Codes.VALUE_RANGE,
                "processed_at_ms must be a non-negative integer",
                source=source,
                entry_index=batch_index,
                field_name="processed_at_ms",
            )
            bad = True
        sample_records = record.get("samples")
        if not isinstance(sample_records, list):
            bag.add(
                Codes.TYPE_MISMATCH,
                "batch samples must be a list",
                source=source,
                entry_index=batch_index,
                field_name="samples",
            )
            bad = True
        if bad:
            continue
        batch_samples: list[Sample] = []
        local_keys: set[tuple[str, str]] = set()
        for sample_index, sample_record in enumerate(sample_records):
            sample = validate_sample_record(
                sample_record,
                config=config,
                source=source,
                entry_index=batch_index,
                bag=bag,
            )
            if sample is None:
                continue
            if sample.key in local_keys:
                bag.add(
                    Codes.DUPLICATE_KEY,
                    "duplicate request_id/candidate_id inside batch",
                    source=source,
                    entry_index=batch_index,
                    details={"sample_index": sample_index, "key": list(sample.key)},
                )
                continue
            previous = active.get(sample.key)
            if previous is not None and sample.sample_version < previous.sample_version:
                bag.add(
                    Codes.VALUE_RANGE,
                    "sample_version regressed across batches",
                    source=source,
                    entry_index=batch_index,
                    field_name="sample_version",
                    details={"sample_index": sample_index, "key": list(sample.key)},
                )
                continue
            local_keys.add(sample.key)
            active[sample.key] = sample
            batch_samples.append(sample)
        batches.append(
            Batch(
                batch_id=batch_id,
                training_event_time_ms=training_time,
                processed_at_ms=processed_time,
                samples=tuple(batch_samples),
            )
        )
    return batches, bag


COMMAND_FIELDS = {"op", "request_id", "candidate_id", "sample", "command_id"}


def load_commands(
    records: Sequence[Mapping[str, Any]], *, config: AppConfig, source: str = "<commands>"
) -> CommandLoadResult:
    bag = DiagnosticBag()
    commands: list[Command] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            bag.add(Codes.TYPE_MISMATCH, "command must be an object", source=source, entry_index=index)
            continue
        bad = False
        for key in record:
            if key not in COMMAND_FIELDS:
                bag.add(
                    Codes.FIELD_UNKNOWN,
                    f"unknown command field {key}",
                    source=source,
                    entry_index=index,
                    field_name=key,
                )
                bad = True
        operation = record.get("op")
        if operation not in {"upsert", "replace", "delete"}:
            bag.add(
                Codes.VALUE_RANGE,
                "command op must be upsert, replace or delete",
                source=source,
                entry_index=index,
                field_name="op",
            )
            bad = True
        request_id = record.get("request_id")
        candidate_id = record.get("candidate_id")
        if not isinstance(request_id, str) or not request_id:
            bag.add(
                Codes.TYPE_MISMATCH,
                "command request_id must be a non-empty string",
                source=source,
                entry_index=index,
                field_name="request_id",
            )
            bad = True
        if not isinstance(candidate_id, str) or not candidate_id:
            bag.add(
                Codes.TYPE_MISMATCH,
                "command candidate_id must be a non-empty string",
                source=source,
                entry_index=index,
                field_name="candidate_id",
            )
            bad = True
        command_id = record.get("command_id")
        if command_id is not None and not isinstance(command_id, str):
            bag.add(
                Codes.TYPE_MISMATCH,
                "command_id must be a string when present",
                source=source,
                entry_index=index,
                field_name="command_id",
                injection_id=command_id if isinstance(command_id, str) else None,
            )
            bad = True
        sample: Sample | None = None
        if operation in {"upsert", "replace"}:
            sample_record = record.get("sample")
            if not isinstance(sample_record, Mapping):
                bag.add(
                    Codes.TYPE_MISMATCH,
                    "upsert/replace command requires a sample object",
                    source=source,
                    entry_index=index,
                    field_name="sample",
                )
                bad = True
            else:
                sample = validate_sample_record(
                    sample_record,
                    config=config,
                    source=source,
                    entry_index=index,
                    bag=bag,
                )
                if sample is not None and (
                    sample.request_id != request_id or sample.candidate_id != candidate_id
                ):
                    bag.add(
                        Codes.VALUE_RANGE,
                        "command and embedded sample identifiers differ",
                        source=source,
                        entry_index=index,
                        details={"command_key": [request_id, candidate_id]},
                    )
                    bad = True
        if bad or sample is None and operation in {"upsert", "replace"}:
            continue
        commands.append(
            Command(
                operation=operation,
                request_id=request_id,
                candidate_id=candidate_id,
                sample=sample,
                command_id=command_id,
            )
        )
    return CommandLoadResult(commands=commands, diagnostics=bag)


def apply_commands(
    samples: Sequence[Sample],
    commands: Sequence[Command],
    *,
    source: str = "<commands>",
) -> SampleLoadResult:
    """Apply already-validated commands without violating active-key uniqueness."""

    bag = DiagnosticBag()
    active = {sample.key: sample for sample in samples}
    accepted_events: list[Sample] = []
    for index, command in enumerate(commands):
        current = active.get(command.key)
        if command.operation == "delete":
            if current is None:
                bag.add(
                    Codes.VALUE_RANGE,
                    "delete references an unknown active sample",
                    source=source,
                    entry_index=index,
                    details={"key": list(command.key)},
                )
                continue
            del active[command.key]
            continue
        if command.operation == "replace" and current is None:
            bag.add(
                Codes.VALUE_RANGE,
                "replace references an unknown active sample",
                source=source,
                entry_index=index,
                details={"key": list(command.key)},
            )
            continue
        if command.operation == "upsert" and current is not None:
            bag.add(
                Codes.DUPLICATE_KEY,
                "upsert would duplicate an active request/candidate key",
                source=source,
                entry_index=index,
                details={"key": list(command.key)},
            )
            continue
        assert command.sample is not None
        if current is not None and command.sample.sample_version < current.sample_version:
            bag.add(
                Codes.VALUE_RANGE,
                "replacement sample_version is lower than the active version",
                source=source,
                entry_index=index,
                field_name="sample_version",
                details={
                    "previous_version": current.sample_version,
                    "incoming_version": command.sample.sample_version,
                },
            )
            continue
        active[command.key] = command.sample
        accepted_events.append(command.sample)
    return SampleLoadResult(samples=list(active.values()), diagnostics=bag, accepted_events=accepted_events)


@dataclass(frozen=True)
class GeneratedDataset:
    samples: tuple[Sample, ...]
    batches: tuple[Batch, ...]
    commands: tuple[Command, ...]

    def sample_records(self) -> list[dict[str, Any]]:
        return [sample.to_dict() for sample in self.samples]

    def batch_records(self) -> list[dict[str, Any]]:
        return [batch.to_dict() for batch in self.batches]

    def command_records(self) -> list[dict[str, Any]]:
        return [command.to_dict() for command in self.commands]


def generate_dataset(config: AppConfig) -> GeneratedDataset:
    """Create a stable local dataset from ``config.seed`` only."""

    samples: list[Sample] = []
    for request_index in range(config.request_count):
        request_id = f"r{request_index:06d}"
        for candidate_index in range(config.candidates_per_request):
            sample_index = request_index * config.candidates_per_request + candidate_index
            candidate_id = f"c{sample_index:06d}"
            dense = tuple(
                float(stable_int(config.seed, "dense", sample_index, dim, modulo=1000))
                for dim in range(config.dense_dim)
            )
            sparse = tuple(
                stable_int(config.seed, "sparse", sample_index, dim, modulo=6)
                for dim in range(config.sparse_dim)
            )
            click_value = stable_int(config.seed, "click", sample_index, modulo=100)
            click = 1 if click_value < 48 else 0
            conversion_value = stable_int(config.seed, "conversion", sample_index, modulo=100)
            conversion = 1 if click == 1 and conversion_value < 38 else 0
            callback = (
                stable_int(config.seed, "callback", sample_index, modulo=22000)
                if click == 1
                else 0
            )
            event_time = 90_000 + stable_int(
                config.seed, "event-time", sample_index, modulo=16_000
            )
            samples.append(
                Sample(
                    request_id=request_id,
                    candidate_id=candidate_id,
                    dense_features=dense,
                    sparse_counts=sparse,
                    click=click,
                    conversion=conversion,
                    conversion_callback_ms=callback,
                    sample_version=1,
                    event_time_ms=event_time,
                )
            )

    # Two rolling batches. The first deliberately cuts off several callbacks.
    split = max(1, len(samples) * 2 // 3)
    batches = (
        Batch(
            batch_id="batch-001",
            training_event_time_ms=101_000,
            processed_at_ms=103_000,
            samples=tuple(samples[:split]),
        ),
        Batch(
            batch_id="batch-002",
            training_event_time_ms=118_000,
            processed_at_ms=121_000,
            samples=tuple(samples[split:]),
        ),
    )

    replacement_index = min(len(samples) - 1, 5)
    original = samples[replacement_index]
    replacement_dense = tuple(
        (value + 37.0 * (feature_index + 1)) % 1000
        for feature_index, value in enumerate(original.dense_features)
    )
    replacement = Sample(
        request_id=original.request_id,
        candidate_id=original.candidate_id,
        dense_features=replacement_dense,
        sparse_counts=tuple((count + 1) % 6 for count in original.sparse_counts),
        click=original.click,
        conversion=original.conversion,
        conversion_callback_ms=original.conversion_callback_ms,
        sample_version=original.sample_version + 1,
        event_time_ms=original.event_time_ms,
    )
    delete_index = min(len(samples) - 1, 10)
    deleted = samples[delete_index]
    commands = (
        Command(
            operation="replace",
            request_id=replacement.request_id,
            candidate_id=replacement.candidate_id,
            sample=replacement,
            command_id="cmd-001",
        ),
        Command(
            operation="delete",
            request_id=deleted.request_id,
            candidate_id=deleted.candidate_id,
            command_id="cmd-002",
        ),
    )
    return GeneratedDataset(
        samples=tuple(samples),
        batches=batches,
        commands=commands,
    )


def write_generated_dataset(dataset: GeneratedDataset, directory: str | Path) -> dict[str, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    samples_path = path / "samples.json"
    batches_path = path / "batches.json"
    commands_path = path / "commands.json"
    samples_path.write_text(
        json.dumps(dataset.sample_records(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    batches_path.write_text(
        json.dumps(dataset.batch_records(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    commands_path.write_text(
        json.dumps(dataset.command_records(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "samples": samples_path,
        "batches": batches_path,
        "commands": commands_path,
    }


def load_sample_json(path: str | Path, *, config: AppConfig) -> SampleLoadResult:
    source = str(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        records = data.get("samples")
        if not isinstance(records, list):
            bag = DiagnosticBag()
            bag.add(Codes.TYPE_MISMATCH, "top-level JSON must contain a samples list", source=source)
            return SampleLoadResult([], bag)
    elif isinstance(data, list):
        records = data
    else:
        bag = DiagnosticBag()
        bag.add(Codes.TYPE_MISMATCH, "JSON root must be a list or object", source=source)
        return SampleLoadResult([], bag)
    return load_samples(records, config=config, source=source)
