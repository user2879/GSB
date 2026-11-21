"""Self-contained session snapshots, restoration and replay verification."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .canonical import SNAPSHOT_FORMAT_VERSION, sha256_value
from .errors import LeaseError
from .ledger import Ledger, utc_now
from .logging_utils import StructuredLogger
from .models import ClusterSpec, SpecBundle
from .orchestrator import ServiceOrchestrator
from .state_machine import event_to_request


def _snapshot_core(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "clusters": snapshot["clusters"],
        "event_stream_digest": snapshot["event_stream_digest"],
        "format_version": snapshot["format_version"],
        "ledger_digest": snapshot["ledger_digest"],
        "parameter_versions": snapshot["parameter_versions"],
        "stream_cursor": snapshot["stream_cursor"],
    }


def _content_digest(snapshot: dict[str, Any]) -> str:
    return sha256_value(_snapshot_core(snapshot))


def _validate_snapshot_structure(snapshot: dict[str, Any]) -> None:
    clusters = snapshot.get("clusters")
    if not isinstance(clusters, list):
        raise LeaseError("E_SNAPSHOT", "snapshot clusters must be a list", field="snapshot.clusters")
    top_cluster_fields = (
        "active_leases",
        "cluster_id",
        "decisions",
        "parameter_version",
        "parameters",
        "priorities",
        "priority_digest",
        "progress",
        "spec",
        "watermarks",
    )
    spec_fields = (
        "capacity",
        "cluster_id",
        "host",
        "lease_duration_ms",
        "port",
        "renewal_interval_ms",
    )
    decision_fields = (
        "capacity",
        "decision",
        "event_seq",
        "event_type",
        "holder_id",
        "ledger_digest",
        "payload",
        "reason",
        "timestamp_ms",
        "used_capacity",
        "watermark_fixed",
        "watermark_scale",
        "watermark_truncation",
    )
    active_fields = (
        "expires_ms",
        "holder_id",
        "last_renewed_ms",
        "renewal_count",
        "started_ms",
        "state",
    )
    watermark_fields = (
        "capacity",
        "event_seq",
        "used_capacity",
        "watermark_fixed",
        "watermark_scale",
        "watermark_truncation",
    )
    for index, cluster in enumerate(clusters):
        field = f"snapshot.clusters[{index}]"
        if not isinstance(cluster, dict):
            raise LeaseError("E_SNAPSHOT", "cluster entry must be an object", field=field)
        for key in top_cluster_fields:
            if key not in cluster:
                raise LeaseError(
                    "E_SNAPSHOT", "cluster entry is missing a required field", field=f"{field}.{key}"
                )
        if not isinstance(cluster["spec"], dict):
            raise LeaseError("E_SNAPSHOT", "spec must be an object", field=f"{field}.spec")
        for key in spec_fields:
            if key not in cluster["spec"]:
                raise LeaseError(
                    "E_SNAPSHOT",
                    "cluster spec is missing a required field",
                    field=f"{field}.spec.{key}",
                )
        if not isinstance(cluster["progress"], dict) or not {
            "last_event_seq",
            "last_timestamp_ms",
        }.issubset(cluster["progress"]):
            raise LeaseError(
                "E_SNAPSHOT", "progress must contain event cursor fields", field=f"{field}.progress"
            )
        if not isinstance(cluster["priorities"], dict):
            raise LeaseError("E_SNAPSHOT", "priorities must be an object", field=f"{field}.priorities")
        for key, required_fields in (
            ("decisions", decision_fields),
            ("active_leases", active_fields),
            ("watermarks", watermark_fields),
        ):
            entries = cluster[key]
            if not isinstance(entries, list):
                raise LeaseError("E_SNAPSHOT", f"{key} must be a list", field=f"{field}.{key}")
            for entry_index, entry in enumerate(entries):
                entry_field = f"{field}.{key}[{entry_index}]"
                if not isinstance(entry, dict):
                    raise LeaseError("E_SNAPSHOT", "entry must be an object", field=entry_field)
                for required in required_fields:
                    if required not in entry:
                        raise LeaseError(
                            "E_SNAPSHOT",
                            "snapshot entry is missing a required field",
                            field=f"{entry_field}.{required}",
                        )


def build_snapshot(ledger: Ledger, bundle: SpecBundle, path: str) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    versions: dict[str, int] = {}
    for version_row in ledger.cluster_versions():
        cluster_id = version_row["cluster_id"]
        parameter_version = int(version_row["parameter_version"])
        versions[cluster_id] = parameter_version
        progress = ledger.progress_for(cluster_id)
        decisions = []
        for row in ledger.current_decisions(cluster_id):
            decisions.append(
                {
                    "capacity": int(row["capacity"]),
                    "decision": row["decision"],
                    "event_seq": int(row["event_seq"]),
                    "event_type": row["event_type"],
                    "holder_id": row["holder_id"],
                    "ledger_digest": row["ledger_digest"],
                    "payload": row["payload"],
                    "reason": row["reason"],
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "used_capacity": int(row["used_capacity"]),
                    "watermark_fixed": int(row["watermark_fixed"]),
                    "watermark_scale": int(row["watermark_scale"]),
                    "watermark_truncation": row["watermark_truncation"],
                }
            )
        active_leases = [
            {
                "expires_ms": int(row["expires_ms"]),
                "holder_id": row["holder_id"],
                "last_event_seq": int(row["last_event_seq"]),
                "last_renewed_ms": int(row["last_renewed_ms"]),
                "priority": int(row["priority"]),
                "provenance": row["provenance"],
                "renewal_count": int(row["renewal_count"]),
                "started_ms": int(row["started_ms"]),
                "state": row["state"],
            }
            for row in ledger.current_active(cluster_id)
        ]
        ownership_intervals = [
            {
                "ended_ms": row["ended_ms"],
                "end_event_seq": row["end_event_seq"],
                "end_reason": row["end_reason"],
                "holder_id": row["holder_id"],
                "start_event_seq": row["start_event_seq"],
                "started_ms": int(row["started_ms"]),
            }
            for row in ledger.current_ownership_intervals(cluster_id)
        ]
        watermarks = [
            {
                "capacity": int(row["capacity"]),
                "event_seq": int(row["event_seq"]),
                "parameter_version": int(row["parameter_version"]),
                "used_capacity": int(row["used_capacity"]),
                "watermark_digest": row["watermark_digest"],
                "watermark_fixed": int(row["watermark_fixed"]),
                "watermark_scale": int(row["watermark_scale"]),
                "watermark_truncation": row["watermark_truncation"],
            }
            for row in ledger.current_watermarks(cluster_id)
        ]
        clusters.append(
            {
                "active_leases": active_leases,
                "cluster_id": cluster_id,
                "decisions": decisions,
                "parameter_version": parameter_version,
                "parameters": {
                    "capacity": int(version_row["capacity"]),
                    "lease_duration_ms": int(version_row["lease_duration_ms"]),
                    "priority_digest": version_row["priority_digest"],
                    "renewal_interval_ms": int(version_row["renewal_interval_ms"]),
                },
                "ownership_intervals": ownership_intervals,
                "priorities": bundle.priorities_for(cluster_id),
                "priority_digest": version_row["priority_digest"],
                "progress": {
                    "last_event_seq": int(progress["last_event_seq"]),
                    "last_timestamp_ms": int(progress["last_timestamp_ms"]),
                },
                "spec": {
                    "capacity": int(version_row["capacity"]),
                    "cluster_id": cluster_id,
                    "host": version_row["host"],
                    "lease_duration_ms": int(version_row["lease_duration_ms"]),
                    "port": int(version_row["port"]),
                    "renewal_interval_ms": int(version_row["renewal_interval_ms"]),
                },
                "watermarks": watermarks,
            }
        )
    snapshot: dict[str, Any] = {
        "clusters": clusters,
        "created_at_utc": utc_now(),
        "event_stream_digest": bundle.event_stream_digest(),
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "ledger_digest": ledger.content_digest(),
        "parameter_versions": versions,
        "snapshot_id": "",
        "stream_cursor": ledger.stream_cursor(),
    }
    digest = _content_digest(snapshot)
    snapshot["content_digest"] = digest
    snapshot["snapshot_id"] = "snap-" + digest[:16]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    ledger.record_snapshot(snapshot, str(target))
    return snapshot


def load_snapshot(path: str, *, injection: str | None = None) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except FileNotFoundError:
        raise LeaseError("E_SNAPSHOT", f"snapshot not found: {path}", field="snapshot") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise LeaseError("E_SNAPSHOT", f"invalid snapshot: {exc}", field="snapshot") from None
    if not isinstance(snapshot, dict):
        raise LeaseError("E_SNAPSHOT", "snapshot must be a JSON object", field="snapshot")
    required = (
        "clusters",
        "content_digest",
        "event_stream_digest",
        "format_version",
        "ledger_digest",
        "parameter_versions",
        "snapshot_id",
        "stream_cursor",
    )
    for field in required:
        if field not in snapshot:
            raise LeaseError(
                "E_SNAPSHOT",
                f"snapshot is missing required field: {field}",
                field=f"snapshot.{field}",
            )
    if not isinstance(snapshot["format_version"], int) or isinstance(
        snapshot["format_version"], bool
    ):
        raise LeaseError(
            "E_SNAPSHOT", "snapshot format_version must be an integer", field="format_version"
        )
    if snapshot["format_version"] != SNAPSHOT_FORMAT_VERSION:
        raise LeaseError(
            "E_SNAPSHOT",
            f"unsupported snapshot format version: {snapshot['format_version']}",
            field="format_version",
        )
    if not isinstance(snapshot["parameter_versions"], dict):
        raise LeaseError(
            "E_SNAPSHOT", "parameter_versions must be an object", field="parameter_versions"
        )
    if not isinstance(snapshot["stream_cursor"], int) or isinstance(
        snapshot["stream_cursor"], bool
    ):
        raise LeaseError(
            "E_SNAPSHOT", "stream_cursor must be an integer", field="stream_cursor"
        )
    for digest_field in ("content_digest", "event_stream_digest", "ledger_digest"):
        if not isinstance(snapshot[digest_field], str):
            raise LeaseError(
                "E_SNAPSHOT", f"{digest_field} must be a string", field=digest_field
            )
    _validate_snapshot_structure(snapshot)
    if injection == "snapshot":
        snapshot["content_digest"] = "0" * 64
    actual_digest = _content_digest(snapshot)
    if actual_digest != snapshot["content_digest"]:
        raise LeaseError(
            "E_SNAPSHOT",
            "snapshot content digest does not match its contents",
            field="content_digest",
            details={
                "actual": actual_digest,
                "declared": snapshot["content_digest"],
                "injection": injection,
            },
        )
    return snapshot


def snapshot_differences(
    snapshot: dict[str, Any], bundle: SpecBundle, ledger: Ledger
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    current_versions = ledger.versions()
    snapshot_versions = {
        str(key): int(value) for key, value in snapshot["parameter_versions"].items()
    }
    if current_versions != snapshot_versions:
        differences.append(
            {
                "current": current_versions,
                "field": "parameter_versions",
                "snapshot": snapshot_versions,
            }
        )
    current_parameters = {
        row["cluster_id"]: {
            "capacity": int(row["capacity"]),
            "host": row["host"],
            "lease_duration_ms": int(row["lease_duration_ms"]),
            "port": int(row["port"]),
            "priority_digest": row["priority_digest"],
            "renewal_interval_ms": int(row["renewal_interval_ms"]),
        }
        for row in ledger.cluster_versions()
    }
    snapshot_parameters = {
        cluster["cluster_id"]: {
            "capacity": int(cluster["spec"]["capacity"]),
            "host": cluster["spec"]["host"],
            "lease_duration_ms": int(cluster["spec"]["lease_duration_ms"]),
            "port": int(cluster["spec"]["port"]),
            "priority_digest": cluster["priority_digest"],
            "renewal_interval_ms": int(cluster["spec"]["renewal_interval_ms"]),
        }
        for cluster in snapshot["clusters"]
    }
    if current_parameters != snapshot_parameters:
        differences.append(
            {
                "current": current_parameters,
                "field": "cluster_parameters",
                "snapshot": snapshot_parameters,
            }
        )
    current_event_digest = bundle.event_stream_digest()
    if current_event_digest != snapshot["event_stream_digest"]:
        differences.append(
            {
                "current": current_event_digest,
                "field": "event_stream_digest",
                "snapshot": snapshot["event_stream_digest"],
            }
        )
    current_ledger_digest = ledger.content_digest()
    if current_ledger_digest != snapshot["ledger_digest"]:
        differences.append(
            {
                "current": current_ledger_digest,
                "field": "ledger_digest",
                "snapshot": snapshot["ledger_digest"],
            }
        )
    return differences


def _canonical_active(response_holders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "expires_ms": int(item["expires_ms"]),
                "holder_id": item["holder_id"],
                "state": item["state"],
                "started_ms": int(item["started_ms"]),
            }
            for item in response_holders
        ],
        key=lambda item: item["holder_id"],
    )


def _derive_ownership_intervals(
    responses: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    open_by_holder: dict[str, dict[str, Any]] = {}
    for event_seq in sorted(responses):
        for transition in responses[event_seq].get("transitions", []):
            holder_id = transition["holder_id"]
            after_state = transition["after_state"]
            if after_state == "granted":
                interval = {
                    "ended_ms": None,
                    "end_event_seq": None,
                    "end_reason": None,
                    "holder_id": holder_id,
                    "start_event_seq": int(transition["event_seq"]),
                    "started_ms": int(transition["timestamp_ms"]),
                }
                intervals.append(interval)
                open_by_holder[holder_id] = interval
            elif after_state in ("released", "expired"):
                interval = open_by_holder.pop(holder_id, None)
                if interval is not None:
                    interval["ended_ms"] = int(transition["timestamp_ms"])
                    interval["end_event_seq"] = int(transition["event_seq"])
                    interval["end_reason"] = transition["reason"]
    return sorted(
        intervals,
        key=lambda item: (item["started_ms"], item["holder_id"], item["start_event_seq"]),
    )


def _snapshot_specs(snapshot: dict[str, Any]) -> tuple[list[ClusterSpec], dict[str, int], dict[str, dict[str, int]]]:
    specs: list[ClusterSpec] = []
    versions: dict[str, int] = {}
    priorities: dict[str, dict[str, int]] = {}
    for cluster in sorted(snapshot["clusters"], key=lambda item: item["cluster_id"]):
        raw = cluster["spec"]
        spec = ClusterSpec(
            cluster_id=raw["cluster_id"],
            host=raw["host"],
            port=int(raw["port"]),
            lease_duration_ms=int(raw["lease_duration_ms"]),
            renewal_interval_ms=int(raw["renewal_interval_ms"]),
            capacity=int(raw["capacity"]),
        )
        specs.append(spec)
        versions[spec.cluster_id] = int(cluster["parameter_version"])
        priorities[spec.cluster_id] = {
            str(key): int(value) for key, value in cluster["priorities"].items()
        }
    return specs, versions, priorities


def replay_snapshot(
    snapshot: dict[str, Any],
    bundle: SpecBundle,
    *,
    runtime_dir: str,
    startup_timeout_ms: int,
    handshake_timeout_ms: int,
    logger: StructuredLogger,
) -> dict[str, Any]:
    specs, versions, priorities = _snapshot_specs(snapshot)
    orchestrator = ServiceOrchestrator(
        runtime_dir,
        clock_mode="logical",
        startup_timeout_ms=startup_timeout_ms,
        handshake_timeout_ms=handshake_timeout_ms,
        log_callback=logger.service_line,
    )
    try:
        orchestrator.start(specs, priorities, versions)
        actual_by_cluster: dict[str, dict[int, dict[str, Any]]] = {}
        last_response: dict[str, dict[str, Any]] = {}
        for cluster in sorted(snapshot["clusters"], key=lambda item: item["cluster_id"]):
            cluster_id = cluster["cluster_id"]
            cursor = int(cluster["progress"]["last_event_seq"])
            events = bundle.events_for(cluster_id, max_seq=cursor)
            actual_by_cluster[cluster_id] = {}
            for event in events:
                response = orchestrator.get(cluster_id).request(event_to_request(event))
                actual_by_cluster[cluster_id][event.event_seq] = response
                last_response[cluster_id] = response

        for cluster in sorted(snapshot["clusters"], key=lambda item: item["cluster_id"]):
            cluster_id = cluster["cluster_id"]
            expected_decisions = sorted(cluster["decisions"], key=lambda item: item["event_seq"])
            for expected in expected_decisions:
                event_seq = int(expected["event_seq"])
                actual = actual_by_cluster[cluster_id].get(event_seq)
                comparisons = [
                    ("decision", expected["decision"], None if actual is None else actual["decision"]),
                    ("reason", expected["reason"], None if actual is None else actual["reason"]),
                    (
                        "used_capacity",
                        int(expected["used_capacity"]),
                        None if actual is None else int(actual["used_capacity"]),
                    ),
                    (
                        "capacity",
                        int(expected["capacity"]),
                        None if actual is None else int(actual["capacity"]),
                    ),
                    (
                        "watermark_fixed",
                        int(expected["watermark_fixed"]),
                        None if actual is None else int(actual["watermark_fixed"]),
                    ),
                    (
                        "active_leases",
                        expected["payload"].get(
                            "active_leases",
                            [
                                {
                                    "expires_ms": int(item["expires_ms"]),
                                    "holder_id": item["holder_id"],
                                    "state": item["state"],
                                    "started_ms": int(item["started_ms"]),
                                }
                                for item in sorted(
                                    expected["payload"].get("active_holders", []),
                                    key=lambda item: item["holder_id"],
                                )
                            ],
                        ),
                        None if actual is None else _canonical_active(actual["active_holders"]),
                    ),
                ]
                for field, expected_value, actual_value in comparisons:
                    if expected_value != actual_value:
                        raise LeaseError(
                            "E_REPLAY",
                            "snapshot replay differs from the recorded ledger",
                            field=field,
                            details={
                                "actual": actual_value,
                                "cluster_id": cluster_id,
                                "event_seq": event_seq,
                                "expected": expected_value,
                            },
                        )
            expected_active = sorted(
                [
                    {
                        "expires_ms": int(item["expires_ms"]),
                        "holder_id": item["holder_id"],
                        "state": item["state"],
                        "started_ms": int(item["started_ms"]),
                    }
                    for item in cluster["active_leases"]
                ],
                key=lambda item: item["holder_id"],
            )
            final_response = last_response.get(cluster_id)
            actual_active = [] if final_response is None else _canonical_active(
                final_response["active_holders"]
            )
            if expected_active != actual_active:
                raise LeaseError(
                    "E_REPLAY",
                    "snapshot final active lease set differs after replay",
                    field="active_leases",
                    details={
                        "actual": actual_active,
                        "cluster_id": cluster_id,
                        "event_seq": int(cluster["progress"]["last_event_seq"]),
                        "expected": expected_active,
                    },
                )
            expected_intervals = sorted(
                [
                    {
                        "ended_ms": (
                            None
                            if item.get("ended_ms") is None
                            else int(item["ended_ms"])
                        ),
                        "end_event_seq": (
                            None
                            if item.get("end_event_seq") is None
                            else int(item["end_event_seq"])
                        ),
                        "end_reason": item.get("end_reason"),
                        "holder_id": item["holder_id"],
                        "start_event_seq": int(item["start_event_seq"]),
                        "started_ms": int(item["started_ms"]),
                    }
                    for item in cluster.get("ownership_intervals", [])
                ],
                key=lambda item: (item["started_ms"], item["holder_id"], item["start_event_seq"]),
            )
            actual_intervals = _derive_ownership_intervals(actual_by_cluster[cluster_id])
            if expected_intervals != actual_intervals:
                mismatch = next(
                    (
                        (expected_item, actual_item)
                        for expected_item, actual_item in zip(expected_intervals, actual_intervals)
                        if expected_item != actual_item
                    ),
                    (
                        expected_intervals[-1] if expected_intervals else None,
                        actual_intervals[-1] if actual_intervals else None,
                    ),
                )
                event_seq = (
                    (mismatch[0] or {}).get("start_event_seq")
                    if mismatch[0]
                    else (mismatch[1] or {}).get("start_event_seq", 0)
                )
                raise LeaseError(
                    "E_REPLAY",
                    "snapshot ownership intervals differ after replay",
                    field="ownership_intervals",
                    details={
                        "actual": mismatch[1],
                        "cluster_id": cluster_id,
                        "event_seq": event_seq,
                        "expected": mismatch[0],
                    },
                )
        logger.log(
            "SNAPSHOT_REPLAY_PASS",
            "snapshot decisions, ownership and watermarks matched replay",
            cluster_count=len(snapshot["clusters"]),
        )
        return {
            "code": "OK",
            "ok": True,
            "replay": "PASS",
            "verified_clusters": [cluster["cluster_id"] for cluster in snapshot["clusters"]],
        }
    finally:
        orchestrator.stop()


def open_snapshot_session(
    bundle: SpecBundle,
    db_path: str,
    snapshot_path: str,
    *,
    runtime_dir: str,
    logger: StructuredLogger,
    injection: str | None = None,
    startup_timeout_ms: int = 5000,
    handshake_timeout_ms: int = 2000,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path, injection=injection)
    ledger = Ledger(db_path)
    try:
        differences = snapshot_differences(snapshot, bundle, ledger)
        if differences:
            commit_id = uuid.uuid4().hex
            ledger.restore_snapshot_state(snapshot, commit_id=commit_id)
            logger.log(
                "E_SNAPSHOT",
                "snapshot differs from the current session; current view was replaced wholesale",
                differences=differences,
                restored=True,
            )
        else:
            logger.log(
                "SNAPSHOT_MATCH",
                "snapshot parameter versions and digest match the current session",
            )
        ledger.record_snapshot(snapshot, snapshot_path)
        if ledger.content_digest() != snapshot["ledger_digest"]:
            raise LeaseError(
                "E_SNAPSHOT",
                "restored ledger digest does not match the snapshot ledger digest",
                field="ledger_digest",
                details={
                    "actual": ledger.content_digest(),
                    "expected": snapshot["ledger_digest"],
                },
            )
        replay = replay_snapshot(
            snapshot,
            bundle,
            runtime_dir=runtime_dir,
            startup_timeout_ms=startup_timeout_ms,
            handshake_timeout_ms=handshake_timeout_ms,
            logger=logger,
        )
        return {
            "code": "E_SNAPSHOT" if differences else "OK",
            "differences": differences,
            "ok": not differences,
            "replay": replay,
            "restored": bool(differences),
            "snapshot_id": snapshot["snapshot_id"],
        }
    finally:
        ledger.close()
