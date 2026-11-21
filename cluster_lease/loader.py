"""Strict local JSON/NDJSON loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import LeaseError, spec_error
from .models import (
    EVENT_TYPE_ALIASES,
    ClusterSpec,
    HolderSpec,
    LeaseEvent,
    SpecBundle,
)


def _load_json(path: str, field: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise spec_error(field, f"file not found: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise spec_error(field, f"invalid JSON in {path}: {exc}") from None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(field, "must be a JSON object")
    return value


def _required(raw: dict[str, Any], key: str, field: str) -> Any:
    if key not in raw:
        raise spec_error(f"{field}.{key}", "required field is missing")
    return raw[key]


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise spec_error(field, "must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not _is_int(value) or value <= 0:
        raise spec_error(field, "must be a positive integer")
    return value


def _injection_code(injection: str | None, field: str) -> str:
    if injection == "missing-ref" and ".targets" in field:
        return "E_REF"
    if injection == "timing" and (
        field.endswith("lease_duration_ms") or field.endswith("renewal_interval_ms")
    ):
        return "E_TIMING"
    return "E_SPEC"


def _parse_clusters(raw_root: Any, injection: str | None) -> list[ClusterSpec]:
    raw_list = raw_root.get("clusters") if isinstance(raw_root, dict) else raw_root
    if not isinstance(raw_list, list):
        raise spec_error("clusters", "must be a list or an object containing a clusters list")
    if not raw_list:
        raise spec_error("clusters", "must contain at least one cluster")

    clusters: list[ClusterSpec] = []
    ids: set[str] = set()
    ports: set[int] = set()
    for index, raw_value in enumerate(raw_list):
        field = f"clusters[{index}]"
        raw = _require_object(raw_value, field)
        cluster_id = _nonempty_string(_required(raw, "cluster_id", field), f"{field}.cluster_id")
        if cluster_id in ids:
            raise spec_error(f"{field}.cluster_id", f"duplicate cluster_id: {cluster_id}")
        ids.add(cluster_id)

        host = _required(raw, "host", field)
        if host != "127.0.0.1":
            raise spec_error(f"{field}.host", "host must be the literal loopback address 127.0.0.1")

        port = _required(raw, "port", field)
        if not _is_int(port) or not (1 <= port <= 65535):
            raise spec_error(f"{field}.port", "port must be an integer in 1..65535")
        if port in ports:
            raise spec_error(f"{field}.port", f"duplicate port: {port}")
        ports.add(port)

        duration = _positive_int(
            _required(raw, "lease_duration_ms", field), f"{field}.lease_duration_ms"
        )
        interval = _positive_int(
            _required(raw, "renewal_interval_ms", field), f"{field}.renewal_interval_ms"
        )
        if interval >= duration:
            code = _injection_code(injection, f"{field}.lease_duration_ms")
            raise spec_error(
                f"{field}.lease_duration_ms",
                "lease_duration_ms must be greater than renewal_interval_ms",
                code=code,
            )
        capacity = _positive_int(_required(raw, "capacity", field), f"{field}.capacity")
        clusters.append(
            ClusterSpec(
                cluster_id=cluster_id,
                host=host,
                port=port,
                lease_duration_ms=duration,
                renewal_interval_ms=interval,
                capacity=capacity,
            )
        )
    return clusters


def _parse_holders(
    raw_root: Any, cluster_ids: set[str], injection: str | None
) -> list[HolderSpec]:
    raw_list = raw_root.get("holders") if isinstance(raw_root, dict) else raw_root
    if not isinstance(raw_list, list):
        raise spec_error("holders", "must be a list or an object containing a holders list")
    if not raw_list:
        raise spec_error("holders", "must contain at least one holder")

    holders: list[HolderSpec] = []
    ids: set[str] = set()
    for index, raw_value in enumerate(raw_list):
        field = f"holders[{index}]"
        raw = _require_object(raw_value, field)
        holder_id = _nonempty_string(_required(raw, "holder_id", field), f"{field}.holder_id")
        if holder_id in ids:
            raise spec_error(f"{field}.holder_id", f"duplicate holder_id: {holder_id}")
        ids.add(holder_id)
        priority = _required(raw, "priority", field)
        if not _is_int(priority):
            raise spec_error(f"{field}.priority", "must be an integer")
        targets = _required(raw, "targets", field)
        if not isinstance(targets, list) or not targets:
            raise spec_error(f"{field}.targets", "must be a non-empty list of cluster ids")
        parsed_targets: list[str] = []
        seen_targets: set[str] = set()
        for target_index, target in enumerate(targets):
            target_field = f"{field}.targets[{target_index}]"
            target_id = _nonempty_string(target, target_field)
            if target_id in seen_targets:
                raise spec_error(target_field, f"duplicate target: {target_id}")
            seen_targets.add(target_id)
            if target_id not in cluster_ids:
                code = _injection_code(injection, target_field)
                raise spec_error(target_field, f"target cluster does not exist: {target_id}", code=code)
            parsed_targets.append(target_id)
        holders.append(HolderSpec(holder_id=holder_id, priority=priority, targets=tuple(parsed_targets)))
    return holders


def _parse_events(
    path: str, cluster_ids: set[str], holder_map: dict[str, HolderSpec]
) -> list[LeaseEvent]:
    events: list[LeaseEvent] = []
    previous_timestamp: int | None = None
    previous_seq = 0
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise spec_error("events", f"file not found: {path}") from None
    except OSError as exc:
        raise spec_error("events", f"cannot open {path}: {exc}") from None

    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            field = f"events[line={line_no}]"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise spec_error(field, f"invalid NDJSON: {exc}") from None
            raw = _require_object(raw, field)
            event_seq = _required(raw, "event_seq", field)
            if not _is_int(event_seq) or event_seq <= 0:
                raise spec_error(f"{field}.event_seq", "must be a positive integer")
            if event_seq != previous_seq + 1:
                raise spec_error(
                    f"{field}.event_seq",
                    f"event_seq must be continuous from 1; expected {previous_seq + 1}",
                )
            timestamp = _required(raw, "timestamp_ms", field)
            if not _is_int(timestamp) or timestamp < 0:
                raise spec_error(f"{field}.timestamp_ms", "must be a non-negative integer")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise spec_error(f"{field}.timestamp_ms", "timestamp_ms must be monotonic non-decreasing")
            cluster_id = _nonempty_string(
                _required(raw, "cluster_id", field), f"{field}.cluster_id"
            )
            if cluster_id not in cluster_ids:
                raise spec_error(f"{field}.cluster_id", f"cluster does not exist: {cluster_id}")
            holder_id = _nonempty_string(
                _required(raw, "holder_id", field), f"{field}.holder_id"
            )
            holder = holder_map.get(holder_id)
            if holder is None:
                raise spec_error(f"{field}.holder_id", f"holder does not exist: {holder_id}")
            event_type_raw = _required(raw, "event_type", field)
            if not isinstance(event_type_raw, str) or event_type_raw not in EVENT_TYPE_ALIASES:
                raise spec_error(
                    f"{field}.event_type",
                    "must be one of apply, renew, release (Chinese labels are accepted)",
                )
            event_type = EVENT_TYPE_ALIASES[event_type_raw]
            if cluster_id not in holder.targets:
                raise spec_error(
                    f"{field}.cluster_id",
                    f"holder {holder_id} does not target cluster {cluster_id}",
                )
            events.append(
                LeaseEvent(
                    event_seq=event_seq,
                    timestamp_ms=timestamp,
                    cluster_id=cluster_id,
                    holder_id=holder_id,
                    event_type=event_type,
                    line_no=line_no,
                )
            )
            previous_seq = event_seq
            previous_timestamp = timestamp
    if not events:
        raise spec_error("events", "event stream must contain at least one event")
    return events


def _inject_raw(
    raw_clusters: Any, raw_holders: Any, injection: str | None
) -> tuple[Any, Any]:
    """Inject faults into in-memory input values, never into source code/files."""

    if injection == "missing-ref":
        if isinstance(raw_holders, dict):
            holders = raw_holders.get("holders")
        else:
            holders = raw_holders
        if not isinstance(holders, list) or not holders:
            raise LeaseError(
                code="E_REF",
                field="holders",
                message="missing-ref injection requires at least one holder",
            )
        target = holders[-1]
        if not isinstance(target, dict) or not isinstance(target.get("targets"), list):
            raise LeaseError(
                code="E_REF",
                field="holders[-1].targets",
                message="missing-ref injection requires a holder targets list",
            )
        target["targets"] = list(target["targets"]) + ["__injected_missing_cluster__"]
    elif injection == "timing":
        if isinstance(raw_clusters, dict):
            clusters = raw_clusters.get("clusters")
        else:
            clusters = raw_clusters
        if not isinstance(clusters, list) or not clusters:
            raise LeaseError(
                code="E_TIMING",
                field="clusters",
                message="timing injection requires at least one cluster",
            )
        target = clusters[0]
        if not isinstance(target, dict) or not _is_int(target.get("renewal_interval_ms")):
            raise LeaseError(
                code="E_TIMING",
                field="clusters[0].renewal_interval_ms",
                message="timing injection requires an integer renewal interval",
            )
        target["lease_duration_ms"] = target["renewal_interval_ms"]
    elif injection not in (None, "snapshot"):
        raise LeaseError(code="E_SPEC", field="injection", message=f"unknown injection: {injection}")
    return raw_clusters, raw_holders


def load_inputs(
    clusters_path: str,
    holders_path: str,
    events_path: str,
    *,
    injection: str | None = None,
) -> SpecBundle:
    """Load and validate all input.  The first violation aborts the whole load."""

    raw_clusters = _load_json(clusters_path, "clusters")
    raw_holders = _load_json(holders_path, "holders")
    raw_clusters, raw_holders = _inject_raw(raw_clusters, raw_holders, injection)
    clusters = _parse_clusters(raw_clusters, injection)
    cluster_ids = {cluster.cluster_id for cluster in clusters}

    holders = _parse_holders(raw_holders, cluster_ids, injection)

    holder_map = {holder.holder_id: holder for holder in holders}
    events = _parse_events(events_path, cluster_ids, holder_map)
    return SpecBundle(
        clusters=tuple(clusters),
        holders=tuple(holders),
        events=tuple(events),
        clusters_path=clusters_path,
        holders_path=holders_path,
        events_path=events_path,
    )


def is_ndjson_path(path: str) -> bool:
    return Path(path).suffix.lower() in (".ndjson", ".jsonl")
