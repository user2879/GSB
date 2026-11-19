"""Incremental compilation and hash-pruned dependency invalidation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .compiler import (
    CompileReport,
    CompileResult,
    FrameQuantizationReport,
    compile_frame_record,
    compile_graph,
)
from .constants import PROTOCOL_V1, PROTOCOL_V2, SUPPORTED_PROTOCOLS
from .dependency import (
    CONTENT_HASH_TRIGGER,
    NODE_DIGEST,
    NODE_FRAME,
    NODE_PRIMITIVE,
    NODE_PROTOCOL,
    NODE_SLICE,
    PROTOCOL_TRIGGER,
    DependencyGraph,
    build_dependency_graph,
)
from .frames import BatchDigest, FrameRecord, batch_key_for_range, make_batch_digest
from .model import EffectGraph, PARAMETER_KEYS, Primitive
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile


@dataclass(frozen=True)
class InvalidationEntry:
    node_key: str
    node_kind: str
    reason: str
    trigger_keys: tuple[str, ...]
    protocol_version: int
    previous_stamp: str | None
    new_stamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_key": self.node_key,
            "node_kind": self.node_kind,
            "reason": self.reason,
            "trigger_keys": list(self.trigger_keys),
            "protocol_version": self.protocol_version,
            "previous_stamp": self.previous_stamp,
            "new_stamp": self.new_stamp,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvalidationEntry":
        return cls(
            node_key=str(value["node_key"]),
            node_kind=str(value["node_kind"]),
            reason=str(value["reason"]),
            trigger_keys=tuple(value["trigger_keys"]),
            protocol_version=int(value["protocol_version"]),
            previous_stamp=(
                None if value.get("previous_stamp") is None else str(value["previous_stamp"])
            ),
            new_stamp=None if value.get("new_stamp") is None else str(value["new_stamp"]),
        )


@dataclass(frozen=True)
class IgnoredChange:
    primitive_id: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    protocol_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IgnoredChange":
        return cls(
            primitive_id=str(value["primitive_id"]),
            parameter=str(value["parameter"]),
            old_value=value.get("old_value"),
            new_value=value.get("new_value"),
            reason=str(value["reason"]),
            protocol_version=int(value["protocol_version"]),
        )


@dataclass(frozen=True)
class ReuseRecord:
    node_key: str
    node_kind: str
    source: str
    version_stamp: str
    protocol_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_key": self.node_key,
            "node_kind": self.node_kind,
            "source": self.source,
            "version_stamp": self.version_stamp,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReuseRecord":
        return cls(
            node_key=str(value["node_key"]),
            node_kind=str(value["node_kind"]),
            source=str(value["source"]),
            version_stamp=str(value["version_stamp"]),
            protocol_version=int(value["protocol_version"]),
        )


@dataclass(frozen=True)
class InvalidationLedger:
    protocol_version: int
    entries: tuple[InvalidationEntry, ...] = ()
    ignored_changes: tuple[IgnoredChange, ...] = ()
    reusable: bool = True
    archived_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "ignored_changes": [change.to_dict() for change in self.ignored_changes],
            "reusable": self.reusable,
            "archived_reason": self.archived_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvalidationLedger":
        return cls(
            protocol_version=int(value["protocol_version"]),
            entries=tuple(InvalidationEntry.from_dict(item) for item in value.get("entries", [])),
            ignored_changes=tuple(
                IgnoredChange.from_dict(item) for item in value.get("ignored_changes", [])
            ),
            reusable=bool(value.get("reusable", True)),
            archived_reason=(
                None
                if value.get("archived_reason") is None
                else str(value["archived_reason"])
            ),
        )


@dataclass(frozen=True)
class UpdateResult:
    protocol_changed: bool
    previous_protocol_version: int
    protocol_version: int
    invalidated_node_keys: tuple[str, ...]
    invalidated_primitive_nodes: tuple[str, ...]
    invalidated_slices: tuple[str, ...]
    invalidated_frames: tuple[int, ...]
    invalidated_digests: tuple[str, ...]
    recompiled_frames: tuple[int, ...]
    probed_frames: tuple[int, ...]
    reused_nodes: tuple[ReuseRecord, ...]
    ignored_changes: tuple[IgnoredChange, ...]
    ledger_archived: bool

    def __post_init__(self) -> None:
        if tuple(sorted(self.invalidated_frames)) != tuple(sorted(self.recompiled_frames)):
            raise AssertionError("invalidated frame closure differs from actual recompiled frames")

    def summary(self) -> dict[str, Any]:
        return {
            "protocol_changed": self.protocol_changed,
            "previous_protocol_version": self.previous_protocol_version,
            "protocol_version": self.protocol_version,
            "invalidated_node_count": len(self.invalidated_node_keys),
            "invalidated_primitive_nodes": list(self.invalidated_primitive_nodes),
            "invalidated_slices": list(self.invalidated_slices),
            "invalidated_frames": list(self.invalidated_frames),
            "invalidated_digests": list(self.invalidated_digests),
            "recompiled_frames": list(self.recompiled_frames),
            "probed_frames": list(self.probed_frames),
            "reused_node_count": len(self.reused_nodes),
            "ignored_changes": [change.to_dict() for change in self.ignored_changes],
            "ledger_archived": self.ledger_archived,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "invalidated_node_keys": list(self.invalidated_node_keys),
            "reused_nodes": [record.to_dict() for record in self.reused_nodes],
        }


@dataclass(frozen=True)
class EngineState:
    graph: EffectGraph
    protocol_version: int
    profile: QuantProfile
    frames: tuple[FrameRecord, ...]
    digests: tuple[BatchDigest, ...]
    report: CompileReport
    dependency_graph: DependencyGraph
    ledger: InvalidationLedger
    ledger_history: tuple[InvalidationLedger, ...] = ()
    last_update: UpdateResult | None = None
    reuse_source: str = "in-memory-cache"

    @property
    def frames_by_seq(self) -> dict[int, FrameRecord]:
        return {frame.seq: frame for frame in self.frames}

    @property
    def digests_by_key(self) -> dict[str, BatchDigest]:
        return {digest.batch_key: digest for digest in self.digests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph": self.graph.to_dict(),
            "protocol_version": self.protocol_version,
            "profile": self.profile.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "digests": [digest.to_dict() for digest in self.digests],
            "report": self.report.to_dict(include_frames=True),
            "dependency_graph": self.dependency_graph.to_dict(),
            "ledger": self.ledger.to_dict(),
            "ledger_history": [ledger.to_dict() for ledger in self.ledger_history],
            "last_update": None if self.last_update is None else self.last_update.to_dict(),
            "reuse_source": self.reuse_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineState":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("unsupported engine state schema")
        protocol_version = int(value["protocol_version"])
        if protocol_version not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol version in state: {protocol_version}")
        frames = tuple(FrameRecord.from_dict(item) for item in value["frames"])
        frames = tuple(sorted(frames, key=lambda frame: frame.seq))
        if any(frame.protocol_version != protocol_version for frame in frames):
            raise ValueError("state contains frames from a different protocol version")
        expected_seqs = list(range(len(frames)))
        if [frame.seq for frame in frames] != expected_seqs:
            raise ValueError("state frames contain sequence gaps")
        digests = tuple(BatchDigest.from_dict(item) for item in value["digests"])
        for digest in digests:
            if digest.protocol_version != protocol_version:
                raise ValueError("state digest protocol stamp is inconsistent")
        ledger = InvalidationLedger.from_dict(value["ledger"])
        if ledger.protocol_version != protocol_version:
            raise ValueError("active ledger protocol version is inconsistent with state")
        history = tuple(
            InvalidationLedger.from_dict(item) for item in value.get("ledger_history", [])
        )
        if any(item.reusable for item in history):
            # Archived ledgers are forensic input only, never an active cache.
            history = tuple(replace(item, reusable=False) for item in history)
        profile_data = dict(value.get("profile", {}))
        profile = QuantProfile(
            amplitude_step=float(profile_data.get("amplitude_step", DEFAULT_QUANT_PROFILE.amplitude_step)),
            frequency_step=float(profile_data.get("frequency_step", DEFAULT_QUANT_PROFILE.frequency_step)),
            amplitude_code_max=int(profile_data.get("amplitude_code_max", DEFAULT_QUANT_PROFILE.amplitude_code_max)),
            frequency_code_max=int(profile_data.get("frequency_code_max", DEFAULT_QUANT_PROFILE.frequency_code_max)),
            max_frequency_hz=float(profile_data.get("max_frequency_hz", DEFAULT_QUANT_PROFILE.max_frequency_hz)),
        )
        graph = EffectGraph.from_dict(value["graph"])
        dependency_graph = DependencyGraph.from_dict(value["dependency_graph"])
        if dependency_graph.protocol_version != protocol_version:
            raise ValueError("dependency graph protocol version is inconsistent")
        report = CompileReport.from_dict(dict(value["report"]))
        last_update_data = value.get("last_update")
        last_update = None
        if last_update_data is not None:
            last_update = UpdateResult(
                protocol_changed=bool(last_update_data["protocol_changed"]),
                previous_protocol_version=int(last_update_data["previous_protocol_version"]),
                protocol_version=int(last_update_data["protocol_version"]),
                invalidated_node_keys=tuple(last_update_data.get("invalidated_node_keys", [])),
                invalidated_primitive_nodes=tuple(last_update_data.get("invalidated_primitive_nodes", [])),
                invalidated_slices=tuple(last_update_data.get("invalidated_slices", [])),
                invalidated_frames=tuple(int(item) for item in last_update_data.get("invalidated_frames", [])),
                invalidated_digests=tuple(last_update_data.get("invalidated_digests", [])),
                recompiled_frames=tuple(int(item) for item in last_update_data.get("recompiled_frames", [])),
                probed_frames=tuple(int(item) for item in last_update_data.get("probed_frames", [])),
                reused_nodes=tuple(
                    ReuseRecord.from_dict(item) for item in last_update_data.get("reused_nodes", [])
                ),
                ignored_changes=tuple(
                    IgnoredChange.from_dict(item)
                    for item in last_update_data.get("ignored_changes", [])
                ),
                ledger_archived=bool(last_update_data.get("ledger_archived", False)),
            )
        return cls(
            graph=graph,
            protocol_version=protocol_version,
            profile=profile,
            frames=frames,
            digests=digests,
            report=report,
            dependency_graph=dependency_graph,
            ledger=ledger,
            ledger_history=history,
            last_update=last_update,
            reuse_source="stable-commit",
        )


def _report_from_frames(
    frames: Iterable[FrameRecord],
    protocol_version: int,
    profile: QuantProfile,
) -> CompileReport:
    ordered = sorted(frames, key=lambda frame: frame.seq)
    reports: list[FrameQuantizationReport] = []
    for frame in ordered:
        channel_errors: list[dict[str, float | str | None]] = []
        for channel in frame.channels:
            channel_errors.append(
                {
                    "channel": channel.channel,
                    "amplitude": channel.amplitude,
                    "amplitude_code": channel.amplitude_code,
                    "amplitude_error": channel.amplitude_error,
                    "frequency_hz": channel.frequency_hz,
                    "frequency_code": channel.frequency_code,
                    "frequency_error": channel.frequency_error,
                }
            )
        reports.append(
            FrameQuantizationReport(
                seq=frame.seq,
                timestamp_us=frame.timestamp_us,
                channel_errors=tuple(channel_errors),
                max_amplitude_error=max(
                    (channel.amplitude_error for channel in frame.channels), default=0.0
                ),
                max_frequency_error=(
                    None
                    if protocol_version == PROTOCOL_V1
                    else max(
                        (
                            channel.frequency_error
                            for channel in frame.channels
                            if channel.frequency_error is not None
                        ),
                        default=0.0,
                    )
                ),
            )
        )
    return CompileReport(
        profile=profile.to_dict(),
        frame_count=len(ordered),
        protocol_version=protocol_version,
        max_amplitude_error=max((item.max_amplitude_error for item in reports), default=0.0),
        max_frequency_error=(
            None
            if protocol_version == PROTOCOL_V1
            else max((item.max_frequency_error or 0.0 for item in reports), default=0.0)
        ),
        amplitude_error_bound=profile.amplitude_max_error,
        frequency_error_bound=profile.frequency_max_error,
        frames=tuple(reports),
    )


def initial_state(
    graph: EffectGraph,
    protocol_version: int,
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> EngineState:
    compiled: CompileResult = compile_graph(graph, protocol_version, profile)
    dependency_graph = build_dependency_graph(
        graph, compiled.frames, compiled.digests, protocol_version
    )
    return EngineState(
        graph=graph,
        protocol_version=protocol_version,
        profile=profile,
        frames=compiled.frames,
        digests=compiled.digests,
        report=compiled.report,
        dependency_graph=dependency_graph,
        ledger=InvalidationLedger(protocol_version=protocol_version),
    )


def _diff_primitive(old: Primitive, new: Primitive) -> dict[str, tuple[Any, Any]]:
    result: dict[str, tuple[Any, Any]] = {}
    for key in PARAMETER_KEYS:
        old_value = getattr(old, key)
        new_value = getattr(new, key)
        if old_value != new_value:
            result[key] = (old_value, new_value)
    return result


def _classify_changes(
    old_graph: EffectGraph,
    new_graph: EffectGraph,
    changes: Mapping[str, Mapping[str, Any]],
    protocol_version: int,
    profile: QuantProfile,
) -> tuple[set[str], set[str], list[IgnoredChange]]:
    changed_primitive_ids: set[str] = set()
    trigger_keys: set[str] = set()
    ignored: list[IgnoredChange] = []
    old_by_id = old_graph.primitive_by_id
    new_by_id = new_graph.primitive_by_id
    for primitive_id in changes:
        old = old_by_id[primitive_id]
        new = new_by_id[primitive_id]
        for key, (old_value, new_value) in _diff_primitive(old, new).items():
            if key == "frequency_hz" and protocol_version == PROTOCOL_V1:
                ignored.append(
                    IgnoredChange(
                        primitive_id=primitive_id,
                        parameter=key,
                        old_value=old_value,
                        new_value=new_value,
                        reason="legacy_protocol_has_no_frequency_field",
                        protocol_version=protocol_version,
                    )
                )
                continue
            if key == "amplitude" and profile.same_amplitude_code(old.amplitude, new.amplitude):
                ignored.append(
                    IgnoredChange(
                        primitive_id=primitive_id,
                        parameter=key,
                        old_value=old_value,
                        new_value=new_value,
                        reason="same_quantized_amplitude_code",
                        protocol_version=protocol_version,
                    )
                )
                continue
            if key == "frequency_hz" and profile.same_frequency_code(old.frequency_hz, new.frequency_hz):
                ignored.append(
                    IgnoredChange(
                        primitive_id=primitive_id,
                        parameter=key,
                        old_value=old_value,
                        new_value=new_value,
                        reason="same_quantized_frequency_code",
                        protocol_version=protocol_version,
                    )
                )
                continue
            trigger_keys.add(key)
            changed_primitive_ids.add(primitive_id)
    return changed_primitive_ids, trigger_keys, ignored


def _seq_range(start_us: int, end_us: int, period_us: int, frame_count: int) -> range:
    if end_us <= start_us or frame_count <= 0:
        return range(0)
    first = max(0, start_us // period_us)
    last = min(frame_count - 1, (end_us - 1) // period_us)
    if last < first:
        return range(0)
    return range(first, last + 1)


def _candidate_frames(
    old_graph: EffectGraph,
    new_graph: EffectGraph,
    primitive_ids: Iterable[str],
) -> set[int]:
    candidates: set[int] = set()
    old_by_id = old_graph.primitive_by_id
    new_by_id = new_graph.primitive_by_id
    for primitive_id in primitive_ids:
        old = old_by_id[primitive_id]
        new = new_by_id[primitive_id]
        for primitive in (old, new):
            candidates.update(
                _seq_range(
                    primitive.start_us,
                    primitive.end_us,
                    new_graph.frame_period_us,
                    new_graph.frame_count,
                )
            )
    return candidates


def _batch_key_for_seq(seq: int, frame_count: int, batch_size: int) -> str:
    first = (seq // batch_size) * batch_size
    last = min(frame_count - 1, first + batch_size - 1)
    return batch_key_for_range(first, last)


def _build_reuse_records(
    old_graph: DependencyGraph,
    new_graph: DependencyGraph,
    invalidated_keys: set[str],
    source: str,
) -> tuple[ReuseRecord, ...]:
    old_nodes = old_graph.node_by_key
    records: list[ReuseRecord] = []
    for node in new_graph.nodes:
        if node.key in invalidated_keys:
            continue
        previous = old_nodes.get(node.key)
        if previous is None or previous.content_hash != node.content_hash:
            continue
        records.append(
            ReuseRecord(
                node_key=node.key,
                node_kind=node.kind,
                source=source,
                version_stamp=node.version_stamp,
                protocol_version=new_graph.protocol_version,
            )
        )
    return tuple(records)


def apply_update(
    state: EngineState,
    changes: Mapping[str, Mapping[str, Any]],
    new_protocol_version: int | None = None,
    profile: QuantProfile | None = None,
) -> tuple[EngineState, UpdateResult]:
    profile = profile or state.profile
    target_protocol = state.protocol_version if new_protocol_version is None else int(new_protocol_version)
    if target_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported protocol version: {target_protocol}")
    old_graph = state.graph
    new_graph = old_graph.with_primitive_changes(changes)
    protocol_changed = target_protocol != state.protocol_version
    changed_primitive_ids, trigger_keys, ignored = _classify_changes(
        old_graph, new_graph, changes, target_protocol, profile
    )
    if not changed_primitive_ids and not protocol_changed:
        new_dependency_graph = build_dependency_graph(
            new_graph, state.frames, state.digests, target_protocol
        )
        invalidated_keys: set[str] = set()
        reused = _build_reuse_records(
            state.dependency_graph,
            new_dependency_graph,
            invalidated_keys,
            state.reuse_source,
        )
        ledger = replace(
            state.ledger,
            ignored_changes=state.ledger.ignored_changes + tuple(ignored),
        )
        result = UpdateResult(
            protocol_changed=False,
            previous_protocol_version=state.protocol_version,
            protocol_version=target_protocol,
            invalidated_node_keys=(),
            invalidated_primitive_nodes=(),
            invalidated_slices=(),
            invalidated_frames=(),
            invalidated_digests=(),
            recompiled_frames=(),
            probed_frames=(),
            reused_nodes=reused,
            ignored_changes=tuple(ignored),
            ledger_archived=False,
        )
        return (
            replace(
                state,
                graph=new_graph,
                profile=profile,
                dependency_graph=new_dependency_graph,
                ledger=ledger,
                last_update=result,
            ),
            result,
        )

    if protocol_changed:
        candidate_seqs = set(range(new_graph.frame_count))
    else:
        candidate_seqs = _candidate_frames(old_graph, new_graph, changed_primitive_ids)

    old_frames = state.frames_by_seq
    new_frames = dict(old_frames)
    recompiled: list[int] = []
    probed: list[int] = []
    old_dependency_nodes = state.dependency_graph.node_by_key
    invalidated_entries: list[InvalidationEntry] = []
    invalidated_primitive_nodes: list[str] = []
    invalidated_slices: list[str] = []
    invalidated_frames: list[int] = []
    invalidated_digests: list[str] = []

    for primitive_id in sorted(changed_primitive_ids):
        key = f"primitive:{primitive_id}"
        new_node = None
        invalidated_primitive_nodes.append(key)
        invalidated_entries.append(
            InvalidationEntry(
                node_key=key,
                node_kind=NODE_PRIMITIVE,
                reason="source_parameter_changed",
                trigger_keys=tuple(sorted(trigger_keys)),
                protocol_version=target_protocol,
                previous_stamp=old_dependency_nodes.get(key).version_stamp
                if key in old_dependency_nodes
                else None,
                new_stamp=new_node,
            )
        )

    if protocol_changed:
        invalidated_entries.append(
            InvalidationEntry(
                node_key="protocol:current",
                node_kind=NODE_PROTOCOL,
                reason="protocol_version_changed",
                trigger_keys=(PROTOCOL_TRIGGER,),
                protocol_version=target_protocol,
                previous_stamp=old_dependency_nodes.get("protocol:current").version_stamp
                if "protocol:current" in old_dependency_nodes
                else None,
                new_stamp=None,
            )
        )

    for seq in sorted(candidate_seqs):
        probe = compile_frame_record(new_graph, seq, target_protocol, profile)
        probed.append(seq)
        old_frame = old_frames[seq]
        if probe.content_hash == old_frame.content_hash and probe.protocol_version == old_frame.protocol_version:
            continue
        new_frames[seq] = probe
        recompiled.append(seq)
        invalidated_frames.append(seq)
        key = probe.frame_key
        invalidated_entries.append(
            InvalidationEntry(
                node_key=key,
                node_kind=NODE_FRAME,
                reason=(
                    "protocol_reencode"
                    if protocol_changed
                    else "content_hash_changed"
                ),
                trigger_keys=(
                    (PROTOCOL_TRIGGER,)
                    if protocol_changed
                    else tuple(sorted(trigger_keys))
                ),
                protocol_version=target_protocol,
                previous_stamp=old_dependency_nodes.get(key).version_stamp
                if key in old_dependency_nodes
                else None,
                new_stamp=probe.version_stamp,
            )
        )

    if changed_primitive_ids:
        for time_slice in new_graph.slices:
            if set(time_slice.primitive_ids) & changed_primitive_ids:
                key = f"slice:{time_slice.id}"
                invalidated_slices.append(key)
                new_node = new_graph.slice_by_id[time_slice.id]
                new_stamp = f"src:{stable_slice_hash(new_node)[:16]}"
                invalidated_entries.append(
                    InvalidationEntry(
                        node_key=key,
                        node_kind=NODE_SLICE,
                        reason="upstream_primitive_changed",
                        trigger_keys=tuple(sorted(trigger_keys)),
                        protocol_version=target_protocol,
                        previous_stamp=old_dependency_nodes.get(key).version_stamp
                        if key in old_dependency_nodes
                        else None,
                        new_stamp=new_stamp,
                    )
                )

    touched_batches: set[str] = set()
    if protocol_changed:
        touched_batches.update(digest.batch_key for digest in state.digests)
    else:
        for seq in recompiled:
            touched_batches.add(
                _batch_key_for_seq(seq, new_graph.frame_count, new_graph.batch_size)
            )
    new_digests = {digest.batch_key: digest for digest in state.digests}
    ordered_frames = [new_frames[seq] for seq in range(new_graph.frame_count)]
    for batch_key in sorted(touched_batches):
        first_seq = int(batch_key.split(":")[1].split("-")[0])
        last_seq = int(batch_key.split(":")[1].split("-")[1])
        batch = ordered_frames[first_seq : last_seq + 1]
        new_digest = make_batch_digest(batch, batch_key)
        old_digest = state.digests_by_key.get(batch_key)
        new_digests[batch_key] = new_digest
        if old_digest is None or old_digest.digest != new_digest.digest:
            invalidated_digests.append(batch_key)
            invalidated_entries.append(
                InvalidationEntry(
                    node_key=batch_key,
                    node_kind=NODE_DIGEST,
                    reason="frame_content_changed",
                    trigger_keys=(CONTENT_HASH_TRIGGER,),
                    protocol_version=target_protocol,
                    previous_stamp=old_dependency_nodes.get(batch_key).version_stamp
                    if batch_key in old_dependency_nodes
                    else None,
                    new_stamp=f"v{new_digest.protocol_version}:{new_digest.digest[:16]}",
                )
            )

    final_frames = tuple(ordered_frames)
    final_digests = tuple(new_digests[key] for key in sorted(new_digests))
    new_dependency_graph = build_dependency_graph(
        new_graph, final_frames, final_digests, target_protocol
    )
    new_dependency_nodes = new_dependency_graph.node_by_key
    invalidated_entries = [
        replace(
            entry,
            new_stamp=(
                new_dependency_nodes[entry.node_key].version_stamp
                if entry.node_key in new_dependency_nodes
                else entry.new_stamp
            ),
        )
        for entry in invalidated_entries
    ]
    final_report = _report_from_frames(final_frames, target_protocol, profile)
    invalidated_keys = {entry.node_key for entry in invalidated_entries}
    reused = _build_reuse_records(
        state.dependency_graph,
        new_dependency_graph,
        invalidated_keys,
        state.reuse_source,
    )

    ledger_history = state.ledger_history
    ledger_archived = protocol_changed
    if protocol_changed:
        archived = replace(
            state.ledger,
            reusable=False,
            archived_reason=f"protocol_changed_to_v{target_protocol}",
        )
        ledger_history = state.ledger_history + (archived,)
        new_ledger = InvalidationLedger(protocol_version=target_protocol)
    else:
        new_ledger = state.ledger
    new_ledger = replace(
        new_ledger,
        entries=new_ledger.entries + tuple(invalidated_entries),
        ignored_changes=new_ledger.ignored_changes + tuple(ignored),
    )

    result = UpdateResult(
        protocol_changed=protocol_changed,
        previous_protocol_version=state.protocol_version,
        protocol_version=target_protocol,
        invalidated_node_keys=tuple(sorted(invalidated_keys)),
        invalidated_primitive_nodes=tuple(sorted(invalidated_primitive_nodes)),
        invalidated_slices=tuple(sorted(invalidated_slices)),
        invalidated_frames=tuple(sorted(invalidated_frames)),
        invalidated_digests=tuple(sorted(set(invalidated_digests))),
        recompiled_frames=tuple(sorted(recompiled)),
        probed_frames=tuple(sorted(probed)),
        reused_nodes=reused,
        ignored_changes=tuple(ignored),
        ledger_archived=ledger_archived,
    )
    new_state = replace(
        state,
        graph=new_graph,
        protocol_version=target_protocol,
        profile=profile,
        frames=final_frames,
        digests=final_digests,
        report=final_report,
        dependency_graph=new_dependency_graph,
        ledger=new_ledger,
        ledger_history=ledger_history,
        last_update=result,
    )
    return new_state, result


def stable_slice_hash(time_slice: Any) -> str:
    from .canonical import stable_hash

    return stable_hash(time_slice.to_dict())
