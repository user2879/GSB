"""Loss-event list v1/v2/v3 readers and normalization rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import format_error
from .model import EventBundle, EventRecord, FiveTuple, TimeBasis, TimeWindow
from .util import sha256_hex


SUPPORTED_VERSIONS = ("v1", "v2", "v3")
PADDED_FIELD_ORDER = ("link_id", "reorder_depth", "retransmit", "time_window_ms")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _record_error(message: str, record_number: int, field: str, **details: Any):
    return format_error(
        message,
        record_number=record_number,
        field=field,
        **details,
    )


def _validate_defaults(document: dict[str, Any], version: str) -> dict[str, Any]:
    defaults = document.get("normalization_defaults")
    if not isinstance(defaults, dict):
        raise _record_error(
            "loss-event list must explicitly declare normalization_defaults",
            record_number=0,
            field="normalization_defaults",
            version=version,
        )
    if version == "v1":
        required = ("link_id", "time_window_ms", "retransmit", "reorder_depth")
        for name in required:
            if name not in defaults:
                raise _record_error(
                    "v1 normalization default is missing",
                    record_number=0,
                    field=f"normalization_defaults.{name}",
                )
        if defaults["time_window_ms"] is not None:
            raise _record_error(
                "v1 has no measured time window; its declared default must be null",
                record_number=0,
                field="normalization_defaults.time_window_ms",
                actual=defaults["time_window_ms"],
            )
        if defaults["link_id"] is not None and not isinstance(defaults["link_id"], str):
            raise _record_error(
                "v1 link_id default must be null or a string",
                record_number=0,
                field="normalization_defaults.link_id",
            )
        if not isinstance(defaults["retransmit"], bool):
            raise _record_error(
                "v1 retransmit default must be boolean",
                record_number=0,
                field="normalization_defaults.retransmit",
            )
        if not _is_int(defaults["reorder_depth"]) or defaults["reorder_depth"] < 0:
            raise _record_error(
                "v1 reorder_depth default must be a non-negative integer",
                record_number=0,
                field="normalization_defaults.reorder_depth",
            )
    elif version == "v2":
        for name in ("retransmit", "reorder_depth"):
            if name not in defaults:
                raise _record_error(
                    "v2 normalization default is missing",
                    record_number=0,
                    field=f"normalization_defaults.{name}",
                )
        if not isinstance(defaults["retransmit"], bool):
            raise _record_error(
                "v2 retransmit default must be boolean",
                record_number=0,
                field="normalization_defaults.retransmit",
            )
        if not _is_int(defaults["reorder_depth"]) or defaults["reorder_depth"] < 0:
            raise _record_error(
                "v2 reorder_depth default must be a non-negative integer",
                record_number=0,
                field="normalization_defaults.reorder_depth",
            )
    return defaults


def _parse_time_basis(document: dict[str, Any]) -> TimeBasis | None:
    if "time_basis" not in document:
        return None
    try:
        basis = TimeBasis.from_dict(document["time_basis"], field_name="time_basis")
    except ValueError as exc:
        raise _record_error(
            "loss-event time basis is invalid",
            record_number=0,
            field=str(exc),
        ) from exc
    if basis.clock != "collector_monotonic_ms" or basis.unit != "ms":
        raise _record_error(
            "loss-event time basis must explicitly be collector_monotonic_ms in ms",
            record_number=0,
            field="time_basis",
            actual=basis.to_dict(),
        )
    return basis


def _parse_alignment(document: dict[str, Any]) -> tuple[int | None, str | None]:
    alignment = document.get("alignment")
    if alignment is None:
        return None, None
    if not isinstance(alignment, dict):
        raise _record_error(
            "alignment declaration must be an object",
            record_number=0,
            field="alignment",
        )
    if "scale" in alignment:
        raise _record_error(
            "clock alignment permits translation only; scale is forbidden",
            record_number=0,
            field="alignment.scale",
            actual=alignment["scale"],
        )
    offset = alignment.get("offset_to_frame_ms")
    if offset is None:
        return None, alignment.get("basis_version")
    if not _is_int(offset):
        raise _record_error(
            "alignment offset_to_frame_ms must be an integer",
            record_number=0,
            field="alignment.offset_to_frame_ms",
            actual=offset,
        )
    basis_version = alignment.get("basis_version")
    if not isinstance(basis_version, str) or not basis_version:
        raise _record_error(
            "alignment basis_version must be a non-empty string",
            record_number=0,
            field="alignment.basis_version",
        )
    return offset, basis_version


def _parse_five_tuple(value: Any, *, record_number: int, field: str = "five_tuple") -> FiveTuple:
    try:
        return FiveTuple.from_dict(value)
    except ValueError as exc:
        raise _record_error(
            "invalid five tuple",
            record_number=record_number,
            field=field,
            reason=str(exc),
        ) from exc


def _parse_window(value: Any, *, record_number: int) -> TimeWindow:
    try:
        return TimeWindow.from_dict(value)
    except ValueError as exc:
        field = str(exc)
        if not field.startswith("time_window_ms"):
            field = "time_window_ms"
        raise _record_error(
            "invalid event time window",
            record_number=record_number,
            field=field,
        ) from exc


def _optional_record_flow(record: dict[str, Any], default_flow: FiveTuple | None, *, record_number: int) -> FiveTuple:
    if "five_tuple" in record:
        return _parse_five_tuple(record["five_tuple"], record_number=record_number)
    if "flow" in record:
        return _parse_five_tuple(record["flow"], record_number=record_number, field="flow")
    if default_flow is None:
        raise _record_error(
            "event record has no five tuple and the file declares no default flow",
            record_number=record_number,
            field="five_tuple",
        )
    return default_flow


def _normalize_record(
    record: Any,
    *,
    record_number: int,
    version: str,
    source_id: str,
    default_flow: FiveTuple | None,
    time_basis: TimeBasis | None,
    offset_to_frame_ms: int | None,
    defaults: dict[str, Any],
    source_digest: str,
) -> EventRecord:
    if not isinstance(record, dict):
        raise _record_error(
            "event record must be an object",
            record_number=record_number,
            field="record",
        )
    for name in ("event_seq", "loss_count"):
        if name not in record:
            raise _record_error(
                "event record is missing a required field",
                record_number=record_number,
                field=name,
            )
    event_seq = record["event_seq"]
    loss_count = record["loss_count"]
    if not _is_int(event_seq) or event_seq < 0:
        raise _record_error(
            "event_seq must be a non-negative integer",
            record_number=record_number,
            field="event_seq",
            actual=event_seq,
        )
    if not _is_int(loss_count):
        raise _record_error(
            "loss_count must be an integer",
            record_number=record_number,
            field="loss_count",
            actual=loss_count,
        )
    if loss_count < 0:
        raise _record_error(
            "loss_count must not be negative",
            record_number=record_number,
            field="loss_count",
            actual=loss_count,
        )
    five_tuple = _optional_record_flow(record, default_flow, record_number=record_number)

    padded_fields: list[str] = []

    if version == "v1":
        if "link_id" in record:
            link_id = record["link_id"]
        else:
            link_id = defaults["link_id"]
            padded_fields.append("link_id")
        if link_id is not None and not isinstance(link_id, str):
            raise _record_error(
                "link_id must be a string or null",
                record_number=record_number,
                field="link_id",
            )
        if "time_window_ms" in record and record["time_window_ms"] is not None:
            window = _parse_window(record["time_window_ms"], record_number=record_number)
        else:
            window = None
            padded_fields.append("time_window_ms")
        if "retransmit" in record:
            retransmit = record["retransmit"]
        else:
            retransmit = defaults["retransmit"]
            padded_fields.append("retransmit")
        if not isinstance(retransmit, bool):
            raise _record_error(
                "retransmit must be boolean",
                record_number=record_number,
                field="retransmit",
            )
        if "reorder_depth" in record:
            reorder_depth = record["reorder_depth"]
        else:
            reorder_depth = defaults["reorder_depth"]
            padded_fields.append("reorder_depth")
        if not _is_int(reorder_depth) or reorder_depth < 0:
            raise _record_error(
                "reorder_depth must be a non-negative integer",
                record_number=record_number,
                field="reorder_depth",
            )
    else:
        if "link_id" not in record:
            raise _record_error(
                "v2/v3 event record is missing link_id",
                record_number=record_number,
                field="link_id",
            )
        link_id = record["link_id"]
        if not isinstance(link_id, str) or not link_id:
            raise _record_error(
                "link_id must be a non-empty string",
                record_number=record_number,
                field="link_id",
            )
        if "time_window_ms" not in record:
            raise _record_error(
                "v2/v3 event record is missing its measured millisecond window",
                record_number=record_number,
                field="time_window_ms",
            )
        window = _parse_window(record["time_window_ms"], record_number=record_number)

        if version == "v2":
            if "retransmit" in record:
                retransmit = record["retransmit"]
            else:
                retransmit = defaults["retransmit"]
                padded_fields.append("retransmit")
            if not isinstance(retransmit, bool):
                raise _record_error(
                    "retransmit must be boolean",
                    record_number=record_number,
                    field="retransmit",
                )
            if "reorder_depth" in record:
                reorder_depth = record["reorder_depth"]
            else:
                reorder_depth = defaults["reorder_depth"]
                padded_fields.append("reorder_depth")
            if not _is_int(reorder_depth) or reorder_depth < 0:
                raise _record_error(
                    "reorder_depth must be a non-negative integer",
                    record_number=record_number,
                    field="reorder_depth",
                )
        else:
            if "retransmit" not in record:
                raise _record_error(
                    "v3 event record is missing retransmit",
                    record_number=record_number,
                    field="retransmit",
                )
            retransmit = record["retransmit"]
            if not isinstance(retransmit, bool):
                raise _record_error(
                    "retransmit must be boolean",
                    record_number=record_number,
                    field="retransmit",
                )
            if "reorder_depth" not in record:
                raise _record_error(
                    "v3 event record is missing reorder_depth",
                    record_number=record_number,
                    field="reorder_depth",
                )
            reorder_depth = record["reorder_depth"]
            if not _is_int(reorder_depth) or reorder_depth < 0:
                raise _record_error(
                    "reorder_depth must be a non-negative integer",
                    record_number=record_number,
                    field="reorder_depth",
                )

    return EventRecord(
        record_number=record_number,
        five_tuple=five_tuple,
        event_seq=event_seq,
        loss_count=loss_count,
        source_version=version,
        source_id=source_id,
        link_id=link_id,
        time_window_ms=window,
        retransmit=retransmit,
        reorder_depth=reorder_depth,
        padded_fields=tuple(sorted(padded_fields, key=PADDED_FIELD_ORDER.index)),
        time_basis=time_basis,
        offset_to_frame_ms=offset_to_frame_ms,
        source_digest=source_digest,
    )


def normalize_event_document(
    document: Any,
    *,
    source_digest: str,
    path: str = "<memory>",
) -> EventBundle:
    if not isinstance(document, dict):
        raise _record_error(
            "loss-event list root must be an object",
            record_number=0,
            field="root",
        )
    version = document.get("version")
    if version is None:
        raise _record_error(
            "loss-event list is missing version",
            record_number=0,
            field="version",
        )
    if version not in SUPPORTED_VERSIONS:
        raise _record_error(
            "unknown loss-event list version",
            record_number=0,
            field="version",
            actual=version,
            supported=list(SUPPORTED_VERSIONS),
        )
    source_id = document.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise _record_error(
            "loss-event list is missing source_id",
            record_number=0,
            field="source_id",
        )
    defaults = _validate_defaults(document, version)

    default_flow_value = document.get("five_tuple", document.get("flow"))
    default_flow = None
    if default_flow_value is not None:
        default_flow = _parse_five_tuple(default_flow_value, record_number=0)

    time_basis = _parse_time_basis(document)
    offset_to_frame_ms, alignment_version = _parse_alignment(document)

    records_value = document.get("records")
    if not isinstance(records_value, list):
        raise _record_error(
            "loss-event list records must be an array",
            record_number=0,
            field="records",
        )
    if not records_value:
        raise _record_error(
            "loss-event list records must not be empty",
            record_number=0,
            field="records",
        )
    records = [
        _normalize_record(
            record,
            record_number=index,
            version=version,
            source_id=source_id,
            default_flow=default_flow,
            time_basis=time_basis,
            offset_to_frame_ms=offset_to_frame_ms,
            defaults=defaults,
            source_digest=source_digest,
        )
        for index, record in enumerate(records_value, start=1)
    ]
    records.sort(
        key=lambda item: (
            item.flow_id,
            item.event_seq,
            item.loss_count,
            item.link_id or "",
            item.time_window_ms.start_ms if item.time_window_ms else -1,
            item.time_window_ms.end_ms if item.time_window_ms else -1,
            item.retransmit,
            item.reorder_depth,
            item.source_version,
            item.record_number,
        )
    )
    return EventBundle(
        path=path,
        source_id=source_id,
        version=version,
        five_tuple=default_flow,
        time_basis=time_basis,
        offset_to_frame_ms=offset_to_frame_ms,
        alignment_version=alignment_version,
        defaults=defaults,
        records=records,
        raw_digest=source_digest,
    )


def load_event_bundle(
    path: str | Path,
    *,
    inject_version_conflict: bool = False,
) -> EventBundle:
    try:
        raw = Path(path).read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _record_error(
            f"cannot read loss-event JSON: {exc}",
            record_number=0,
            field="json",
        ) from exc
    digest = sha256_hex(raw)
    if inject_version_conflict and isinstance(document, dict):
        document = dict(document)
        document["version"] = "v9-injected-conflict"
    return normalize_event_document(document, source_digest=digest, path=str(path))

