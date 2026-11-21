"""Normalized observation model shared by every input format."""

from __future__ import annotations

import ipaddress
import zlib
from dataclasses import dataclass
from typing import Any


PROTOCOL_NAMES = {6: "tcp", 17: "udp"}
PROTOCOL_NUMBERS = {name: number for number, name in PROTOCOL_NAMES.items()}

FLAG_DATA = 0x01
FLAG_RETRANSMIT = 0x02
FLAG_REORDER = 0x04
FLAG_NAMES = {
    FLAG_DATA: "DATA",
    FLAG_RETRANSMIT: "RETRANSMIT",
    FLAG_REORDER: "REORDER",
}
KNOWN_FLAGS = FLAG_DATA | FLAG_RETRANSMIT | FLAG_REORDER
LEGAL_FLAG_COMBINATIONS = {
    FLAG_DATA,
    FLAG_DATA | FLAG_RETRANSMIT,
    FLAG_DATA | FLAG_REORDER,
    FLAG_DATA | FLAG_RETRANSMIT | FLAG_REORDER,
}


@dataclass(frozen=True, order=True)
class FiveTuple:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "src_ip", str(ipaddress.IPv4Address(self.src_ip)))
            object.__setattr__(self, "dst_ip", str(ipaddress.IPv4Address(self.dst_ip)))
        except ipaddress.AddressValueError as exc:
            raise ValueError("five-tuple addresses must be canonical IPv4 addresses") from exc
        if not 0 <= self.src_port <= 65535 or not 0 <= self.dst_port <= 65535:
            raise ValueError("five-tuple ports must be in [0,65535]")
        protocol = self.protocol.lower()
        if protocol in PROTOCOL_NUMBERS:
            pass
        elif protocol not in PROTOCOL_NUMBERS:
            raise ValueError("five-tuple protocol must be tcp or udp")
        object.__setattr__(self, "protocol", protocol)

    @property
    def protocol_number(self) -> int:
        return PROTOCOL_NUMBERS[self.protocol]

    @property
    def flow_id(self) -> str:
        return (
            f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/"
            f"{self.protocol}"
        )

    @property
    def wire_flow_id(self) -> int:
        return zlib.crc32(self.flow_id.encode("utf-8")) & 0xFFFFFFFF

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FiveTuple":
        if not isinstance(value, dict):
            raise ValueError("five tuple must be an object")
        required = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"five tuple missing field: {missing[0]}")
        protocol = value["protocol"]
        if isinstance(protocol, int) and not isinstance(protocol, bool):
            if protocol not in PROTOCOL_NAMES:
                raise ValueError("five tuple protocol number is unsupported")
            protocol = PROTOCOL_NAMES[protocol]
        return cls(
            src_ip=value["src_ip"],
            dst_ip=value["dst_ip"],
            src_port=value["src_port"],
            dst_port=value["dst_port"],
            protocol=str(protocol),
        )


@dataclass(frozen=True)
class TimeBasis:
    clock: str
    unit: str
    epoch: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"clock": self.clock, "unit": self.unit, "epoch": self.epoch}
        if self.source_id is not None:
            result["source_id"] = self.source_id
        return result

    @classmethod
    def from_dict(cls, value: Any, *, field_name: str = "time_basis") -> "TimeBasis":
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        for key in ("clock", "unit", "epoch"):
            if key not in value or not isinstance(value[key], str) or not value[key]:
                raise ValueError(f"{field_name}.{key}")
        return cls(
            clock=value["clock"],
            unit=value["unit"],
            epoch=value["epoch"],
            source_id=value.get("source_id"),
        )


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    byte_offset: int
    five_tuple: FiveTuple
    seq: int
    send_ms: int
    flags: int
    arrival_ordinal: int
    payload_len: int
    source_id: str

    @property
    def flow_id(self) -> str:
        return self.five_tuple.flow_id

    @property
    def wire_flow_id(self) -> int:
        return self.five_tuple.wire_flow_id

    @property
    def flag_names(self) -> list[str]:
        return [name for bit, name in FLAG_NAMES.items() if self.flags & bit]

    @property
    def retransmit(self) -> bool:
        return bool(self.flags & FLAG_RETRANSMIT)

    @property
    def reorder(self) -> bool:
        return bool(self.flags & FLAG_REORDER)

    def summary(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "byte_offset": self.byte_offset,
            "frame_bytes": 42 + self.payload_len,
            "flow_id": self.flow_id,
            "flow_wire_id": self.wire_flow_id,
            "five_tuple": self.five_tuple.to_dict(),
            "send_seq": self.seq,
            "send_monotonic_ms": self.send_ms,
            "flags": self.flags,
            "flag_names": self.flag_names,
            "arrival_ordinal": self.arrival_ordinal,
            "payload_len": self.payload_len,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class TimeWindow:
    start_ms: int
    end_ms: int

    def to_dict(self) -> dict[str, int]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, value: Any) -> "TimeWindow":
        if not isinstance(value, dict):
            raise ValueError("time_window_ms must be an object")
        if "start_ms" not in value:
            raise ValueError("time_window_ms.start_ms")
        if "end_ms" not in value:
            raise ValueError("time_window_ms.end_ms")
        start = value["start_ms"]
        end = value["end_ms"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("time_window_ms.start_ms")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("time_window_ms.end_ms")
        if start < 0 or end < start:
            raise ValueError("time_window_ms")
        return cls(start, end)

    def translated(self, offset_ms: int) -> "TimeWindow":
        return TimeWindow(self.start_ms + offset_ms, self.end_ms + offset_ms)


@dataclass(frozen=True)
class EventRecord:
    record_number: int
    five_tuple: FiveTuple
    event_seq: int
    loss_count: int
    source_version: str
    source_id: str
    link_id: str | None
    time_window_ms: TimeWindow | None
    retransmit: bool
    reorder_depth: int
    padded_fields: tuple[str, ...]
    time_basis: TimeBasis | None
    offset_to_frame_ms: int | None
    source_digest: str

    @property
    def flow_id(self) -> str:
        return self.five_tuple.flow_id

    @property
    def interval(self) -> tuple[int, int]:
        return self.event_seq, self.event_seq + self.loss_count - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_number": self.record_number,
            "flow_id": self.flow_id,
            "five_tuple": self.five_tuple.to_dict(),
            "event_seq": self.event_seq,
            "loss_count": self.loss_count,
            "interval": {
                "start_seq": self.event_seq,
                "end_seq": self.event_seq + self.loss_count - 1,
            },
            "source_version": self.source_version,
            "source_id": self.source_id,
            "link_id": self.link_id,
            "time_window_ms": self.time_window_ms.to_dict() if self.time_window_ms else None,
            "retransmit": self.retransmit,
            "reorder_depth": self.reorder_depth,
            "padded_fields": list(self.padded_fields),
            "time_basis": self.time_basis.to_dict() if self.time_basis else None,
            "offset_to_frame_ms": self.offset_to_frame_ms,
            "source_digest": self.source_digest,
        }


@dataclass
class FrameBundle:
    source_id: str
    model_version: str
    time_basis: TimeBasis
    alignment_version: str
    frames: list[FrameObservation]
    raw_digest: str
    wire_digest: str

    def to_dict(self, include_frames: bool = True) -> dict[str, Any]:
        result = {
            "source_id": self.source_id,
            "model_version": self.model_version,
            "time_basis": self.time_basis.to_dict(),
            "alignment_version": self.alignment_version,
            "raw_digest": self.raw_digest,
            "wire_digest": self.wire_digest,
            "frame_count": len(self.frames),
        }
        if include_frames:
            result["frames"] = [frame.summary() for frame in self.frames]
        return result


@dataclass
class EventBundle:
    path: str
    source_id: str
    version: str
    five_tuple: FiveTuple | None
    time_basis: TimeBasis | None
    offset_to_frame_ms: int | None
    alignment_version: str | None
    defaults: dict[str, Any]
    records: list[EventRecord]
    raw_digest: str

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        result = {
            "path": self.path,
            "source_id": self.source_id,
            "version": self.version,
            "five_tuple": self.five_tuple.to_dict() if self.five_tuple else None,
            "time_basis": self.time_basis.to_dict() if self.time_basis else None,
            "offset_to_frame_ms": self.offset_to_frame_ms,
            "alignment_version": self.alignment_version,
            "defaults": self.defaults,
            "record_count": len(self.records),
            "raw_digest": self.raw_digest,
        }
        if include_records:
            result["records"] = [record.to_dict() for record in self.records]
        return result


@dataclass(frozen=True)
class Gap:
    flow_id: str
    start_seq: int
    end_seq: int
    classification: str
    count: int
    evidence: tuple[str, ...]
    source_versions: tuple[str, ...]
    time_window_ms: dict[str, int] | None
    time_source: str

    def interval(self) -> dict[str, int]:
        return {"start_seq": self.start_seq, "end_seq": self.end_seq}

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "gap_interval": self.interval(),
            "classification": self.classification,
            "count": self.count,
            "evidence": list(self.evidence),
            "source_versions": list(self.source_versions),
            "time_window_ms": self.time_window_ms,
            "time_source": self.time_source,
        }


@dataclass
class FlowDiagnosis:
    flow_id: str
    five_tuple: FiveTuple
    sequence_envelope: dict[str, int]
    observed_unique_count: int
    observed_frame_count: int
    gaps: list[Gap]
    true_loss_count: int
    retransmission_count: int
    reordering_count: int
    unwindowed_true_loss_count: int
    gap_digest: str
    source_versions: list[str]

    def to_dict(self, include_summary_digest: bool = True) -> dict[str, Any]:
        result = {
            "flow_id": self.flow_id,
            "five_tuple": self.five_tuple.to_dict(),
            "sequence_envelope": self.sequence_envelope,
            "observed_unique_count": self.observed_unique_count,
            "observed_frame_count": self.observed_frame_count,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "counts": {
                "true_loss": self.true_loss_count,
                "retransmission": self.retransmission_count,
                "reordering": self.reordering_count,
                "unwindowed_true_loss": self.unwindowed_true_loss_count,
            },
            "source_versions": self.source_versions,
        }
        if include_summary_digest:
            result["gap_digest"] = self.gap_digest
        return result


@dataclass
class WindowRate:
    flow_id: str
    window_start_ms: int
    window_end_ms: int
    observed_unique: int
    true_loss: int
    expected: int
    rate_ppm: int
    exact_fraction: str
    window_assignment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "observed_unique": self.observed_unique,
            "true_loss": self.true_loss,
            "expected": self.expected,
            "rate_ppm": self.rate_ppm,
            "exact_fraction": self.exact_fraction,
            "window_assignment": self.window_assignment,
        }


@dataclass
class FlowRateSummary:
    flow_id: str
    true_loss: int
    expected: int
    rate_ppm: int
    exact_fraction: str
    unwindowed_true_loss: int
    windows: list[WindowRate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "true_loss": self.true_loss,
            "expected": self.expected,
            "rate_ppm": self.rate_ppm,
            "exact_fraction": self.exact_fraction,
            "unwindowed_true_loss": self.unwindowed_true_loss,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass
class AlignmentRecord:
    flow_id: str
    frame_source_id: str | None
    event_source_ids: list[str]
    frame_clock: str | None
    event_clock: str | None
    offset_to_frame_ms: int | None
    translated_event_windows: bool
    sequence_only: bool
    aligned: bool
    missing_basis: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "frame_source_id": self.frame_source_id,
            "event_source_ids": self.event_source_ids,
            "frame_clock": self.frame_clock,
            "event_clock": self.event_clock,
            "offset_to_frame_ms": self.offset_to_frame_ms,
            "translated_event_windows": self.translated_event_windows,
            "sequence_only": self.sequence_only,
            "aligned": self.aligned,
            "missing_basis": self.missing_basis,
        }


@dataclass
class AlignmentResult:
    records: list[AlignmentRecord]
    unaligned_flow_ids: list[str]
    basis_version: str
    offset_policy: str = "translation_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_version": self.basis_version,
            "offset_policy": self.offset_policy,
            "records": [record.to_dict() for record in self.records],
            "unaligned_flow_ids": self.unaligned_flow_ids,
        }
