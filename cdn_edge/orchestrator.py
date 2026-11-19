from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import PipelineError
from .fixed import FixedGrid
from .ledger import Ledger
from .netproto import port_is_available, read_json_file, send_json_line
from .spec import RequestSpec, SourceUpdateSpec, Spec
from .stats import BatchRequestRecord, compute_batch_metrics

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class WorkerHandle:
    name: str
    kind: str
    host: str
    port: int
    process: subprocess.Popen[bytes]
    ready_file: Path
    error_file: Path
    log_file: Path
    log_handle: Any
    pid: int
    ready_payload: dict[str, Any]


class Cluster:
    def __init__(self, spec: Spec, work_dir: str | Path, *, startup_timeout: float = 12.0) -> None:
        self.spec = spec
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self.source: WorkerHandle | None = None
        self.nodes: dict[str, WorkerHandle] = {}
        self._node_policies = dict(spec.nodes)

    def _ensure_ports_free(self) -> None:
        if not port_is_available(self.spec.session.host, self.spec.session.source_port):
            raise PipelineError(
                "E_PORT",
                "source loopback port is already occupied",
                path="session.source_port",
                field="port",
                details={"host": self.spec.session.host, "port": self.spec.session.source_port},
            )
        for node_id in sorted(self.spec.nodes):
            policy = self.spec.nodes[node_id]
            if not port_is_available(self.spec.session.host, policy.port):
                raise PipelineError(
                    "E_PORT",
                    "edge loopback port is already occupied",
                    path=f"nodes[id={node_id}].port",
                    field="port",
                    details={"host": self.spec.session.host, "port": policy.port},
                )

    def check_ports(self) -> None:
        self._ensure_ports_free()

    def _start_process(self, name: str, kind: str, host: str, port: int, args: list[str]) -> WorkerHandle:
        ready_file = self.work_dir / f"{name}.ready.json"
        error_file = self.work_dir / f"{name}.error.json"
        log_file = self.work_dir / f"{name}.log"
        for path in (ready_file, error_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        log_handle = log_file.open("ab")
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        command = [sys.executable, "-m", args[0], *args[1:]]
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            **kwargs,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if error_file.exists():
                error_payload = read_json_file(error_file)
                self._terminate_process(process)
                log_handle.close()
                raise PipelineError(
                    str(error_payload.get("code", "E_PROTOCOL")),
                    str(error_payload.get("message", "worker failed to start")),
                    path=error_payload.get("path"),
                    field=error_payload.get("field"),
                    details=error_payload.get("details", {}),
                )
            if ready_file.exists():
                payload = read_json_file(ready_file)
                return WorkerHandle(
                    name=name,
                    kind=kind,
                    host=host,
                    port=port,
                    process=process,
                    ready_file=ready_file,
                    error_file=error_file,
                    log_file=log_file,
                    log_handle=log_handle,
                    pid=process.pid,
                    ready_payload=payload,
                )
            if process.poll() is not None:
                log_handle.flush()
                text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
                log_handle.close()
                raise PipelineError(
                    "E_PROTOCOL",
                    "worker exited before becoming ready",
                    path=name,
                    field="startup",
                    details={"exit_code": process.returncode, "log_tail": text[-4000:]},
                )
            time.sleep(0.02)
        self._terminate_process(process)
        log_handle.close()
        raise PipelineError(
            "E_PROTOCOL",
            "worker startup timed out",
            path=name,
            field="startup",
            details={"timeout_seconds": self.startup_timeout},
        )

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def _start_source(self, source_spec_path: str | Path, state_file: str | Path) -> WorkerHandle:
        name = "source"
        args = [
            "cdn_edge.source_worker",
            "--spec",
            str(source_spec_path),
            "--host",
            self.spec.session.host,
            "--port",
            str(self.spec.session.source_port),
            "--ready-file",
            str(self.work_dir / f"{name}.ready.json"),
            "--error-file",
            str(self.work_dir / f"{name}.error.json"),
            "--state-file",
            str(state_file),
        ]
        handle = self._start_process(name, "source", self.spec.session.host, self.spec.session.source_port, args)
        self.source = handle
        return handle

    def start_edge(
        self,
        policy_id: str,
        state_db: str | Path,
        *,
        monotonic_base_ms: int | None = None,
    ) -> WorkerHandle:
        if policy_id in self.nodes:
            raise PipelineError("E_STATE", "edge process is already running", path=policy_id, field="node_id")
        policy = self._node_policies[policy_id]
        name = f"node-{policy_id}"
        args = [
            "cdn_edge.node_worker",
            "--node-id",
            policy.node_id,
            "--host",
            self.spec.session.host,
            "--port",
            str(policy.port),
            "--source-host",
            self.spec.session.host,
            "--source-port",
            str(self.spec.session.source_port),
            "--cache-key-fields-json",
            json.dumps(list(policy.cache_key_fields)),
            "--capacity",
            str(policy.capacity),
            "--eviction",
            policy.eviction,
            "--state-db",
            str(state_db),
            "--ready-file",
            str(self.work_dir / f"{name}.ready.json"),
            "--error-file",
            str(self.work_dir / f"{name}.error.json"),
        ]
        if monotonic_base_ms is not None:
            args.extend(["--monotonic-base-ms", str(monotonic_base_ms)])
        handle = self._start_process(name, "edge", self.spec.session.host, policy.port, args)
        self.nodes[policy_id] = handle
        return handle

    def stop_edge(self, policy_id: str, *, force: bool = False) -> None:
        handle = self.nodes.pop(policy_id, None)
        if handle is None:
            return
        if not force:
            try:
                send_json_line(handle.host, handle.port, {"op": "shutdown"}, timeout=2.0)
            except PipelineError:
                pass
        self._terminate_process(handle.process)
        try:
            handle.log_handle.flush()
            handle.log_handle.close()
        except OSError:
            pass

    def restart_edge(self, policy_id: str, state_db: str | Path) -> WorkerHandle:
        self.stop_edge(policy_id, force=True)
        # A restarted edge must load all cache/counter state from its SQLite ledgers.
        return self.start_edge(policy_id, state_db)

    def start_all(self, source_spec_path: str | Path, node_db_dir: str | Path, source_state_file: str | Path) -> None:
        self._ensure_ports_free()
        self._start_source(source_spec_path, source_state_file)
        node_dir = Path(node_db_dir)
        node_dir.mkdir(parents=True, exist_ok=True)
        for node_id in sorted(self.spec.nodes):
            self.start_edge(node_id, node_dir / f"{node_id}.sqlite3")
        self.wait_source()
        for node_id in sorted(self.nodes):
            self.wait_edge(node_id)

    def wait_source(self) -> None:
        if self.source is None:
            raise PipelineError("E_STATE", "source is not running", path="source", field="process")
        self._wait_protocol(self.source)

    def wait_edge(self, policy_id: str) -> None:
        handle = self.nodes.get(policy_id)
        if handle is None:
            raise PipelineError("E_STATE", "edge is not running", path=policy_id, field="process")
        self._wait_protocol(handle)

    @staticmethod
    def _wait_protocol(handle: WorkerHandle) -> None:
        deadline = time.monotonic() + 8.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = send_json_line(handle.host, handle.port, {"op": "ping"}, timeout=1.0)
                if response.get("ok"):
                    return
            except PipelineError as exc:
                last_error = exc
            time.sleep(0.02)
        raise PipelineError(
            "E_PROTOCOL",
            "worker ready file exists but protocol ping failed",
            path=handle.name,
            field="ping",
            details={"reason": str(last_error)},
        )

    def source_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.source is None:
            raise PipelineError("E_STATE", "source is not running", path="source", field="process")
        return send_json_line(self.source.host, self.source.port, payload)

    def node_request(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        handle = self.nodes.get(node_id)
        if handle is None:
            raise PipelineError("E_STATE", "edge is not running", path=node_id, field="process")
        return send_json_line(handle.host, handle.port, payload)

    def apply_updates(self, updates: Iterable[SourceUpdateSpec]) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for update in updates:
            response = self.source_request({"op": "update", **update.as_dict()})
            if not response.get("ok"):
                raise PipelineError(
                    str(response.get("code", "E_ORIGIN")),
                    str(response.get("message", "source rejected update")),
                    path=str(response.get("path", f"source_update[path={update.path}]")),
                    field=str(response.get("field", "update")),
                    details={"response": response},
                )
            applied.append(update.as_dict())
        return applied

    def node_states(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for node_id in sorted(self.nodes):
            response = self.node_request(node_id, {"op": "dump"})
            if not response.get("ok"):
                raise PipelineError(
                    "E_STATE", "edge dump failed", path=node_id, field="dump", details={"response": response}
                )
            states[node_id] = response
        return states

    def restore_node_states(self, node_states: dict[str, Any]) -> None:
        for node_id in sorted(self.nodes):
            state = node_states.get(node_id)
            if state is None:
                continue
            response = self.node_request(
                node_id,
                {
                    "op": "restore",
                    "replace": True,
                    "entries": state.get("entries", []),
                    "counters": state.get("counters", {}),
                    "monotonic_base_ms": state.get("monotonic_base_ms"),
                },
            )
            if not response.get("ok"):
                raise PipelineError(
                    "E_STATE",
                    "edge snapshot restore failed",
                    path=node_id,
                    field="restore",
                    details={"response": response},
                )

    def close(self) -> None:
        for node_id in sorted(list(self.nodes)):
            self.stop_edge(node_id, force=False)
        if self.source is not None:
            handle = self.source
            try:
                send_json_line(handle.host, handle.port, {"op": "shutdown"}, timeout=2.0)
            except PipelineError:
                pass
            self._terminate_process(handle.process)
            try:
                handle.log_handle.flush()
                handle.log_handle.close()
            except OSError:
                pass
            self.source = None

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _batch_cursor(spec: Spec, end_index: int, next_batch_id: int | None) -> dict[str, Any]:
    request_count = len(spec.requests)
    return {
        "next_request_index": end_index + 1,
        "next_batch_id": next_batch_id,
        "request_count": request_count,
    }


def _authoritative_versions(cluster: Cluster, at_ms: int) -> dict[str, Any]:
    response = cluster.source_request({"op": "versions", "at_ms": at_ms})
    if not response.get("ok"):
        raise PipelineError(
            "E_ORIGIN", "source versions query failed", path="source", field="versions", details={"response": response}
        )
    return response


def _with_session_id(spec: Spec, session_id: str | None) -> Spec:
    if session_id is None or session_id == spec.session.session_id:
        return spec
    import copy

    raw = copy.deepcopy(spec.raw)
    raw.setdefault("session", {})["session_id"] = session_id
    session = dataclasses.replace(spec.session, session_id=session_id)
    return dataclasses.replace(spec, session=session, raw=raw)


def run_session(
    spec: Spec,
    ledger_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    session_id: str | None = None,
    record_ledger: bool = True,
    seed_node_states: dict[str, Any] | None = None,
    start_request_index: int = 0,
    source_updates: Iterable[SourceUpdateSpec] | None = None,
    preserve_work_dir: bool = False,
) -> dict[str, Any]:
    spec = _with_session_id(spec, session_id)
    updates = list(spec.source_updates if source_updates is None else source_updates)
    updates_payload = [update.as_dict() for update in updates]
    if start_request_index < 0 or start_request_index > len(spec.requests):
        raise PipelineError(
            "E_REPLAY",
            "request cursor is outside the recorded sequence",
            path=str(start_request_index),
            field="start_request_index",
            details={"request_count": len(spec.requests)},
        )
    if start_request_index > 0:
        previous_batch = spec.requests[start_request_index - 1].batch
        if start_request_index < len(spec.requests) and spec.requests[start_request_index].batch == previous_batch:
            raise PipelineError(
                "E_REPLAY",
                "snapshot cursor must point to a batch boundary",
                path=str(start_request_index),
                field="start_request_index",
                details={"batch": previous_batch},
            )
    temp_manager: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temp_manager = tempfile.TemporaryDirectory(prefix="cdn-edge-session-")
        work_dir = temp_manager.name
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    source_spec_path = work_path / "worker_spec.json"
    source_spec_path.write_text(
        json.dumps(spec.raw, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    ledger: Ledger | None = None
    cluster: Cluster | None = None
    result_batches: list[dict[str, Any]] = []
    try:
        cluster = Cluster(spec, work_path)
        # Fail with E_PORT before committing a session row.
        cluster.check_ports()
        if record_ledger:
            ledger = Ledger(ledger_path, create=True)
            first_at = spec.requests[0].at_ms if spec.requests else 0
            # Initial content metadata uses the source's full update history but
            # selects the version visible at the session's first timestamp.
            from .content import initial_versions

            initial = initial_versions(spec, updates, first_at)
            ledger.begin_session(spec, spec.grid, updates_payload, initial)
        cluster.start_all(
            source_spec_path,
            work_path / "node-ledgers",
            work_path / "source-state.json",
        )
        cluster.apply_updates(updates)
        if seed_node_states is not None:
            cluster.restore_node_states(seed_node_states)
        if record_ledger:
            if ledger is None:
                raise PipelineError("E_STATE", "ledger was not initialized", path=str(ledger_path), field="ledger")
            first_at = spec.requests[0].at_ms if spec.requests else 0
            initial_content = _authoritative_versions(cluster, first_at)
            genesis_cursor = {
                "next_request_index": 0,
                "next_batch_id": spec.requests[0].batch if spec.requests else None,
                "request_count": len(spec.requests),
            }
            ledger.commit_genesis(
                spec=spec,
                grid=spec.grid,
                source_updates=updates_payload,
                content_snapshot={
                    "at_ms": initial_content["at_ms"],
                    "content_digest": initial_content["content_digest"],
                    "content_versions": initial_content["content_versions"],
                },
                node_states=cluster.node_states(),
                cursor=genesis_cursor,
            )
        target = list(spec.requests[start_request_index:])
        batches: list[tuple[int, list[RequestSpec]]] = []
        for request in target:
            if not batches or batches[-1][0] != request.batch:
                batches.append((request.batch, []))
            batches[-1][1].append(request)
        for batch_position, (batch_id, batch_requests) in enumerate(batches):
            batch_records: list[BatchRequestRecord] = []
            for request in batch_requests:
                policy = spec.nodes[request.node_id]
                key_values = dict(request.cache_key_values(policy.cache_key_fields))
                response = cluster.node_request(
                    request.node_id,
                    {
                        "op": "fetch",
                        "seq": request.seq,
                        "request_id": request.request_id,
                        "path": request.path,
                        "at_ms": request.at_ms,
                        "key": key_values,
                        "allow_stale": True,
                    },
                )
                if not response.get("ok"):
                    raise PipelineError(
                        "E_ORIGIN",
                        "edge fetch failed",
                        path=f"request:{request.request_id}",
                        field="fetch",
                        details={"response": response},
                    )
                authoritative = _authoritative_versions(cluster, request.at_ms)
                versions = authoritative["content_versions"]
                if request.path not in versions:
                    raise PipelineError(
                        "E_STATE",
                        "authoritative version is missing for requested path",
                        path=request.path,
                        field="versions",
                    )
                authoritative_etag = str(versions[request.path]["etag"])
                returned_etag = str(response["etag"])
                stale = bool(response.get("stale", False))
                error_code = response.get("error_code")
                if stale and error_code != "E_STALE":
                    error_code = "E_STALE"
                record = BatchRequestRecord(
                    seq=request.seq,
                    request_id=request.request_id,
                    node_id=request.node_id,
                    path=request.path,
                    query=request.query,
                    at_ms=request.at_ms,
                    hit_source=str(response["hit_source"]),
                    returned_etag=returned_etag,
                    authoritative_etag=authoritative_etag,
                    version_consistent=returned_etag == authoritative_etag,
                    remaining_ttl_ms=int(response.get("remaining_ttl_ms", 0)),
                    entry_version=int(response.get("entry_version", 0)),
                    stale=stale,
                    error_code=error_code,
                    cache_key=str(response.get("cache_key", "")),
                    written_at_ms=int(response.get("written_at_ms", 0)),
                    written_mono_ms=int(response.get("written_mono_ms", 0)),
                    monotonic_base_ms=int(response.get("monotonic_base_ms", 0)),
                )
                batch_records.append(record)
            end_at = max(request.at_ms for request in batch_requests)
            content_snapshot = _authoritative_versions(cluster, end_at)
            from .content import topology_from_versions

            metrics = compute_batch_metrics(
                spec,
                spec.grid,
                batch_id,
                batch_records,
                topology_from_versions(content_snapshot["content_versions"]),
            )
            next_batch_id = batches[batch_position + 1][0] if batch_position + 1 < len(batches) else None
            last_request = batch_requests[-1]
            cursor = _batch_cursor(spec, last_request.seq, next_batch_id)
            snapshot_id = None
            if ledger is not None:
                snapshot_id = ledger.commit_batch(
                    spec=spec,
                    grid=spec.grid,
                    source_updates=updates_payload,
                    batch_id=batch_id,
                    records=batch_records,
                    metrics=metrics,
                    content_snapshot={
                        "at_ms": content_snapshot["at_ms"],
                        "content_digest": content_snapshot["content_digest"],
                        "content_versions": content_snapshot["content_versions"],
                    },
                    node_states=cluster.node_states(),
                    cursor=cursor,
                    start_seq=batch_records[0].seq,
                    end_seq=batch_records[-1].seq,
                )
            result_batches.append(
                {
                    "batch_id": batch_id,
                    "snapshot_id": snapshot_id,
                    "start_seq": batch_records[0].seq,
                    "end_seq": batch_records[-1].seq,
                    "cursor": cursor,
                    "content_digest": content_snapshot["content_digest"],
                    "metrics": metrics,
                    "requests": [record.as_dict() for record in batch_records],
                }
            )
        result = {
            "ok": True,
            "session_id": spec.session.session_id,
            "grid": spec.grid.as_dict(),
            "batch_count": len(result_batches),
            "batches": result_batches,
            "applied_source_updates": updates_payload,
            "ledger": str(ledger_path) if record_ledger else None,
        }
        return result
    finally:
        if cluster is not None:
            cluster.close()
        if ledger is not None:
            ledger.close()
        if temp_manager is not None:
            if preserve_work_dir:
                # Preserve for diagnostics only when explicitly requested.
                temp_manager.cleanup = lambda: None  # type: ignore[assignment]
            else:
                temp_manager.cleanup()
