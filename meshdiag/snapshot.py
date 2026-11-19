from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_text, canonical_json, pretty_json, read_json, sha256_json, sha256_text
from .errors import MeshError, diag
from .models import (
    AttemptTrace,
    RequestResult,
    RequestSpec,
    Service,
    instance_fingerprint,
    normalize_revision,
    parse_service_rows,
)


SCHEMA = "meshdiag.snapshot/v1"


def _result_from_dict(row: dict[str, Any]) -> RequestResult:
    traces = tuple(
        AttemptTrace(
            request_id=str(trace.get("request_id", row.get("request_id", ""))),
            attempt_no=int(trace.get("attempt_no", 0)),
            rule_id=str(trace.get("rule_id", "")),
            service_id=str(trace.get("service_id", "")),
            instance_id=str(trace.get("instance_id", "")),
            outcome=str(trace.get("outcome", "failure")),
            reason=str(trace.get("reason", "")),
            started_mono_ms=int(trace.get("started_mono_ms", 0)),
            ended_mono_ms=int(trace.get("ended_mono_ms", 0)),
            status_code=trace.get("status_code"),
            detail=str(trace.get("detail", "")),
        )
        for trace in row.get("traces", [])
        if isinstance(trace, dict)
    )
    sequence = row.get("instance_sequence", [])
    return RequestResult(
        request_id=str(row.get("request_id", "")),
        request_index=int(row.get("request_index", 0)),
        method=str(row.get("method", "GET")),
        path=str(row.get("path", "/")),
        rule_id=row.get("rule_id"),
        service_id=row.get("service_id"),
        outcome=str(row.get("outcome", "unknown")),
        reason=str(row.get("reason", "")),
        attempt_count=int(row.get("attempt_count", len(traces))),
        instance_sequence=tuple(str(item) for item in sequence),
        budget_consumed_ms=int(row.get("budget_consumed_ms", 0)),
        wall_elapsed_ms=int(row.get("wall_elapsed_ms", 0)),
        traces=traces,
        unknown=bool(row.get("unknown", False)),
        rerun_required=bool(row.get("rerun_required", False)),
        detail=str(row.get("detail", "")),
    )


class SnapshotSession:
    def __init__(self, store: "SnapshotStore", path: Path, payload: dict[str, Any]) -> None:
        self.store = store
        self.path = path
        self.payload = payload
        self._lock = threading.RLock()
        self._write_locked()

    def _write_locked(self) -> None:
        checksum = sha256_json(self.payload)
        envelope = {
            "schema": SCHEMA,
            "checksum": checksum,
            "payload": self.payload,
        }
        atomic_write_text(self.path, pretty_json(envelope) + "\n")

    def update(self, **changes: Any) -> None:
        with self._lock:
            self.payload.update(changes)
            self.payload["completed_results"] = sorted(
                self.payload.get("completed_results", []),
                key=lambda row: (int(row.get("request_index", 0)), str(row.get("request_id", ""))),
            )
            self.payload["in_progress"] = sorted(
                self.payload.get("in_progress", []),
                key=lambda row: (int(row.get("request_index", 0)), str(row.get("request_id", ""))),
            )
            self._write_locked()

    def add_attempt(self, request_id: str, trace: AttemptTrace) -> None:
        with self._lock:
            inflight = self.payload.setdefault("inflight_attempts", {})
            rows = inflight.setdefault(request_id, [])
            rows.append(trace.as_dict())
            rows.sort(key=lambda row: int(row.get("attempt_no", 0)))
            self._write_locked()

    def set_in_progress(self, rows: Iterable[dict[str, Any]]) -> None:
        self.update(in_progress=list(rows))

    def complete_request(self, result: RequestResult) -> None:
        with self._lock:
            completed = [
                row
                for row in self.payload.get("completed_results", [])
                if row.get("request_id") != result.request_id
            ]
            completed.append(result.as_dict())
            in_progress = [
                row
                for row in self.payload.get("in_progress", [])
                if row.get("request_id") != result.request_id
            ]
            inflight = self.payload.setdefault("inflight_attempts", {})
            inflight.pop(result.request_id, None)
            self.payload["completed_results"] = sorted(
                completed,
                key=lambda row: (int(row.get("request_index", 0)), str(row.get("request_id", ""))),
            )
            self.payload["in_progress"] = sorted(
                in_progress,
                key=lambda row: (int(row.get("request_index", 0)), str(row.get("request_id", ""))),
            )
            self._write_locked()

    @property
    def snapshot_id(self) -> str:
        return str(self.payload["snapshot_id"])


class SnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, snapshot_id: str) -> Path:
        safe = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in snapshot_id
        )
        return self.directory / f"{safe}.json"

    def create(self, snapshot_id: str, payload: dict[str, Any]) -> SnapshotSession:
        full_payload = dict(payload)
        full_payload["snapshot_id"] = snapshot_id
        return SnapshotSession(self, self.path_for(snapshot_id), full_payload)

    def create_for_round(
        self,
        *,
        snapshot_id: str,
        workspace_id: str,
        revision,
        services: tuple[Service, ...],
        defaults: dict[str, Any],
        global_concurrency: int,
        requests: tuple[RequestSpec, ...],
        circuit_breaker_rows: Iterable[dict[str, Any]],
        snapshot_path: str | None = None,
    ) -> SnapshotSession:
        payload = {
            "snapshot_id": snapshot_id,
            "workspace_id": workspace_id,
            "status": "running",
            "revision_version": revision.revision_version,
            "revision_digest": revision.digest,
            "canonical_text": revision.canonical_text,
            "revision_document": revision.source_document,
            "services": [service.as_dict() for service in services],
            "defaults": defaults,
            "global_concurrency": global_concurrency,
            "dependency_graph": revision.dependency_graph.as_dict(),
            "dependency_graph_version": revision.dependency_graph.version,
            "instance_fingerprint": instance_fingerprint(services),
            "requests": [request.as_dict() for request in requests],
            "completed_results": [],
            "inflight_attempts": {},
            "in_progress": [
                {
                    "request_id": request.request_id,
                    "request_index": request.index,
                    "method": request.method,
                    "path": request.path,
                    "state": "queued",
                    "attempts": 0,
                    "budget_consumed_ms": 0,
                }
                for request in requests
            ],
            "budget": {
                "global_limit": global_concurrency,
                "global_active": 0,
                "service_active": {},
                "service_limits": {
                    service.service_id: service.max_concurrency for service in services
                },
            },
            "initial_circuit_breakers": list(circuit_breaker_rows),
            "circuit_breakers": list(circuit_breaker_rows),
            "ledger_rows": {},
            "previous_snapshot_path": snapshot_path,
        }
        return self.create(snapshot_id, payload)

    def load(self, path_or_id: str | Path) -> dict[str, Any]:
        path = Path(path_or_id)
        if not path.exists():
            path = self.path_for(str(path_or_id))
        try:
            envelope = read_json(path)
        except (OSError, ValueError) as exc:
            raise MeshError(
                "E_SNAPSHOT",
                "快照无法读取",
                diagnostics=[diag("E_SNAPSHOT", str(path), str(exc))],
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA:
            raise MeshError(
                "E_SNAPSHOT",
                "未知快照格式",
                diagnostics=[diag("E_SNAPSHOT", str(path), "schema 不匹配", envelope.get("schema"))],
            )
        payload = envelope.get("payload")
        checksum = envelope.get("checksum")
        if not isinstance(payload, dict) or checksum != sha256_json(payload):
            raise MeshError(
                "E_SNAPSHOT",
                "快照校验失败，可能未完整提交",
                diagnostics=[diag("E_SNAPSHOT", str(path), "内容摘要不匹配")],
            )
        payload["_snapshot_file"] = str(path.resolve())
        return payload

    def list(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = self.load(path)
            except MeshError:
                continue
            rows.append(
                {
                    "snapshot_id": payload.get("snapshot_id"),
                    "file": str(path),
                    "status": payload.get("status"),
                    "revision_version": payload.get("revision_version"),
                    "revision_digest": payload.get("revision_digest"),
                    "dependency_graph_version": payload.get("dependency_graph_version"),
                    "completed": len(payload.get("completed_results", [])),
                    "in_progress": len(payload.get("in_progress", [])),
                }
            )
        return tuple(sorted(rows, key=lambda row: (str(row["snapshot_id"]), row["file"])))


def rebuild_revision_from_snapshot(payload: dict[str, Any]):
    try:
        services = parse_service_rows(payload.get("services"))
        revision_document = payload.get("revision_document")
        defaults = payload.get("defaults", {})
        workspace_id = payload.get("workspace_id")
        revision = normalize_revision(
            revision_document,
            services,
            workspace_id=str(workspace_id),
            defaults=defaults,
        )
    except MeshError as exc:
        raise MeshError(
            "E_SNAPSHOT",
            "快照内修订无法重建",
            diagnostics=[diag("E_SNAPSHOT", "snapshot.revision_document", exc.message)],
        ) from exc
    if revision.revision_version != payload.get("revision_version"):
        raise MeshError(
            "E_SNAPSHOT",
            "快照修订版本不一致",
            diagnostics=[
                diag(
                    "E_SNAPSHOT",
                    "snapshot.revision_version",
                    "版本字段与重建模型不一致",
                    payload.get("revision_version"),
                )
            ],
        )
    if revision.digest != payload.get("revision_digest"):
        raise MeshError(
            "E_SNAPSHOT",
            "快照修订摘要不一致",
            diagnostics=[
                diag(
                    "E_SNAPSHOT",
                    "snapshot.revision_digest",
                    "归一模型摘要不匹配",
                    payload.get("revision_digest"),
                )
            ],
        )
    if revision.dependency_graph.version != payload.get("dependency_graph_version"):
        raise MeshError(
            "E_SNAPSHOT",
            "快照依赖图版本不一致",
            diagnostics=[
                diag(
                    "E_SNAPSHOT",
                    "snapshot.dependency_graph_version",
                    "依赖图重建版本不匹配",
                    payload.get("dependency_graph_version"),
                )
            ],
        )
    return revisions_services(revision, services)


def revisions_services(revision, services):
    return revision, services


def result_rows(payload: dict[str, Any]) -> tuple[RequestResult, ...]:
    return tuple(
        sorted(
            (
                _result_from_dict(row)
                for row in payload.get("completed_results", [])
                if isinstance(row, dict)
            ),
            key=lambda item: (item.request_index, item.request_id),
        )
    )
