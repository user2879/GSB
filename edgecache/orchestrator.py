"""Local subprocess orchestration and versioned round execution."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .errors import EdgeError
from .ledger import Ledger
from .model import LoadedSpec, Policy, canonical_json_bytes
from .protocol import PROTOCOL_VERSION, handshake, request
from .snapshot import LoadedSnapshot, SnapshotStore, snapshot_summary
from .spec import load_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ManagedProcess:
    role: str
    identity: str
    host: str
    port: int
    process: subprocess.Popen[bytes]
    log_path: Path
    force_killed: bool = False
    shutdown_response: dict[str, Any] | None = None

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def force_terminate(self) -> dict[str, Any]:
        if self.alive:
            self.process.kill()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        self.force_killed = True
        self.shutdown_response = None
        return {
            "role": self.role,
            "identity": self.identity,
            "pid": self.process.pid,
            "force_killed": True,
            "exit_report_available": False,
            "exit_report": None,
        }

    def graceful_stop(self, *, timeout: float) -> dict[str, Any]:
        if not self.alive:
            return self.exit_report()
        response = request(
            self.host,
            self.port,
            {"op": "shutdown", "protocol": PROTOCOL_VERSION},
            timeout=timeout,
        )
        self.shutdown_response = response
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.force_terminate()
        return self.exit_report()

    def exit_report(self) -> dict[str, Any]:
        if self.force_killed:
            return {
                "role": self.role,
                "identity": self.identity,
                "pid": self.process.pid,
                "force_killed": True,
                "exit_report_available": False,
                "exit_report": None,
            }
        return {
            "role": self.role,
            "identity": self.identity,
            "pid": self.process.pid,
            "force_killed": False,
            "exit_report_available": self.shutdown_response is not None,
            "exit_report": {
                "returncode": self.process.returncode,
                "shutdown_response": self.shutdown_response,
            }
            if self.shutdown_response is not None
            else None,
        }


class EdgeCluster:
    """A single-process coordinator for a deterministic local cluster."""

    def __init__(
        self,
        *,
        spec_path: str | Path,
        db_path: str | Path,
        runtime_dir: str | Path,
        port_base: int = 46000,
        handshake_timeout: float = 2.0,
        request_timeout: float = 5.0,
    ) -> None:
        if port_base < 1024 or port_base > 65000:
            raise EdgeError(
                code="E_PORT",
                message="port_base must be a high local port between 1024 and 65000",
                details={"port_base": port_base},
            )
        self.handshake_timeout = float(handshake_timeout)
        self.request_timeout = float(request_timeout)
        self.port_base = int(port_base)
        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.active_spec_path = self.runtime_dir / "active-spec.json"
        self.ledger = Ledger(db_path)
        self.snapshots = SnapshotStore(self.runtime_dir / "snapshots")
        self.spec = load_spec(spec_path)
        self.source: ManagedProcess | None = None
        self.nodes: dict[str, ManagedProcess] = {}
        self.node_ports: dict[str, int] = {}
        self.started = False
        self.unknown_on_open: list[int] = []
        self.termination_reports: list[dict[str, Any]] = []
        self.external_spec_ignored = False

    def __enter__(self) -> "EdgeCluster":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def source_port(self) -> int:
        return self.port_base

    def _node_ports_for(self, policy: Policy) -> dict[str, int]:
        node_ids = sorted(node.node_id for node in policy.nodes)
        highest = self.port_base + len(node_ids)
        if highest > 65535:
            raise EdgeError(
                code="E_PORT",
                message="port range exceeds the TCP port limit",
                details={"port_base": self.port_base, "nodes": len(node_ids)},
            )
        return {node_id: self.port_base + 1 + index for index, node_id in enumerate(node_ids)}

    def _check_ports(self, ports: Sequence[tuple[str, int]]) -> None:
        for role, port in ports:
            if port < 1024 or port > 65535:
                raise EdgeError(
                    code="E_PORT",
                    message="local loopback port must be a high port",
                    details={"role": role, "port": port},
                )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                except OSError:
                    pass
            elif hasattr(socket, "SO_REUSEADDR"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                sock.close()
                if os.name == "nt" and not _service_is_listening("127.0.0.1", port):
                    # A just-terminated local worker can leave TIME_WAIT
                    # sockets.  Reuse is safe here only because no process is
                    # accepting connections on the port.
                    retry = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        retry.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        retry.bind(("127.0.0.1", port))
                        retry.close()
                        continue
                    except OSError:
                        retry.close()
                raise EdgeError(
                    code="E_PORT",
                    message="loopback port is already occupied",
                    details={"role": role, "port": port, "error": str(exc)},
                ) from exc
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    def start(self) -> dict[str, Any]:
        if self.started:
            return self.summary()
        self.unknown_on_open = self.ledger.recover_pending()
        persisted = self.ledger.current_spec()
        if persisted is not None:
            self.external_spec_ignored = self.spec.digest != persisted.digest
            self.spec = persisted
        else:
            self.external_spec_ignored = False
        self._write_active_spec(self.spec)
        self._check_ports(self._ports_for(self.spec.policy))
        self._spawn_source(self.spec)
        self._spawn_nodes(self.spec.policy)
        self._wait_worker(self.source, service="content-source")
        for node_id in sorted(self.nodes):
            self._wait_worker(self.nodes[node_id], service="edge-node")
        if self.ledger.current_version() is None:
            bootstrap = self._bootstrap_round(self.spec)
        else:
            bootstrap = self._restore_current_to_nodes()
        self.started = True
        return {
            "ok": True,
            "started": True,
            "source": self._process_info(self.source),
            "nodes": [self._process_info(self.nodes[node_id]) for node_id in sorted(self.nodes)],
            "bootstrap": bootstrap,
            "recovered_unknown_rounds": self.unknown_on_open,
            "external_spec_ignored": self.external_spec_ignored,
        }

    def _ports_for(self, policy: Policy) -> list[tuple[str, int]]:
        node_ports = self._node_ports_for(policy)
        return [("content-source", self.source_port)] + [
            (f"edge-node:{node_id}", node_ports[node_id]) for node_id in sorted(node_ports)
        ]

    def _node_port_checks(self, policy: Policy) -> list[tuple[str, int]]:
        node_ports = self._node_ports_for(policy)
        return [(f"edge-node:{node_id}", node_ports[node_id]) for node_id in sorted(node_ports)]

    def _write_active_spec(self, spec: LoadedSpec) -> None:
        self.active_spec_path.write_text(
            json.dumps(spec.to_raw(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _spawn_source(self, spec: LoadedSpec) -> None:
        managed = self._spawn_worker(
            role="content-source",
            identity="content-source",
            port=self.source_port,
            args=["source", "--spec", str(self.active_spec_path), "--host", "127.0.0.1", "--port", str(self.source_port)],
        )
        self.source = managed

    def _spawn_nodes(self, policy: Policy) -> None:
        self.node_ports = self._node_ports_for(policy)
        self.nodes = {}
        for node in sorted(policy.nodes, key=lambda item: item.node_id):
            port = self.node_ports[node.node_id]
            self.nodes[node.node_id] = self._spawn_worker(
                role="edge-node",
                identity=node.node_id,
                port=port,
                args=[
                    "node",
                    "--node-id",
                    node.node_id,
                    "--source-host",
                    "127.0.0.1",
                    "--source-port",
                    str(self.source_port),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--timeout",
                    str(self.request_timeout),
                ],
            )

    def _spawn_worker(self, *, role: str, identity: str, port: int, args: list[str]) -> ManagedProcess:
        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in identity)
        log_path = self.runtime_dir / "logs" / f"{safe}-{port}.log"
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        command = [sys.executable, "-u", "-m", "edgecache.worker", *args]
        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            flags = 0
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = flags
        else:
            kwargs["start_new_session"] = True
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module, no shell
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        return ManagedProcess(role=role, identity=identity, host="127.0.0.1", port=port, process=process, log_path=log_path)

    def _wait_worker(self, managed: ManagedProcess | None, *, service: str) -> dict[str, Any]:
        if managed is None:
            raise EdgeError(code="E_PROCESS", message="worker was not started", details={"service": service})
        deadline = time.monotonic() + self.handshake_timeout
        last_error: EdgeError | None = None
        while time.monotonic() <= deadline:
            if not managed.alive:
                raise self._worker_exit_error(managed)
            remaining = max(0.05, deadline - time.monotonic())
            try:
                response = handshake(
                    managed.host,
                    managed.port,
                    client="edgecache-orchestrator",
                    timeout=remaining,
                    expected_service=service,
                )
                if service == "content-source":
                    if response.get("content_version") != self.spec.content_version:
                        raise EdgeError(
                            code="E_CONTENT",
                            message="content source started with an unexpected content version",
                            details={
                                "expected": self.spec.content_version,
                                "actual": response.get("content_version"),
                            },
                        )
                return response
            except EdgeError as exc:
                if exc.code == "E_TIMEOUT":
                    raise
                last_error = exc
                if exc.code not in {"E_PORT", "E_PROTOCOL"}:
                    raise
                time.sleep(0.02)
        if not managed.alive:
            raise self._worker_exit_error(managed)
        raise EdgeError(
            code="E_TIMEOUT",
            message="worker handshake did not complete before the socket deadline",
            details={
                "service": service,
                "port": managed.port,
                "timeout_seconds": self.handshake_timeout,
                "last_error": last_error.envelope() if last_error else None,
                "log_tail": self._tail_log(managed.log_path),
            },
        )

    def _worker_exit_error(self, managed: ManagedProcess) -> EdgeError:
        tail = self._tail_log(managed.log_path)
        for line in reversed(tail.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("code"):
                code = str(value["code"])
                if code not in {
                    "E_SPEC",
                    "E_CONTENT",
                    "E_CYCLE",
                    "E_PORT",
                    "E_TIMEOUT",
                    "E_PROTOCOL",
                    "E_PROCESS",
                    "E_STALE",
                    "E_SNAPSHOT",
                    "E_REPLAY",
                    "E_STATE",
                    "E_TK",
                }:
                    code = "E_PROCESS"
                return EdgeError(
                    code=code,
                    message=str(value.get("message") or "worker process exited during startup"),
                    path=value.get("path"),
                    field_path=value.get("field_path"),
                    details={
                        "role": managed.role,
                        "identity": managed.identity,
                        "port": managed.port,
                        "returncode": managed.process.returncode,
                        "worker_details": value.get("details"),
                        "log_tail": tail,
                    },
                )
        return EdgeError(
            code="E_PROCESS",
            message="worker process exited before completing its handshake",
            details={
                "role": managed.role,
                "identity": managed.identity,
                "port": managed.port,
                "returncode": managed.process.returncode,
                "log_tail": tail,
            },
        )

    def _tail_log(self, path: Path, limit: int = 8192) -> str:
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", errors="replace")

    def _node_request(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        managed = self.nodes.get(node_id)
        if managed is None:
            raise EdgeError(code="E_PROCESS", message="node is not managed by this cluster", details={"node_id": node_id})
        if not managed.alive:
            raise EdgeError(
                code="E_PROCESS",
                message="node process is not running; no exit report is available after forced termination",
                details={"node_id": node_id, "force_killed": managed.force_killed, "exit_report": None},
            )
        return request(managed.host, managed.port, {"protocol": PROTOCOL_VERSION, **payload}, timeout=self.request_timeout)

    def _source_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.source is None or not self.source.alive:
            raise EdgeError(
                code="E_PROCESS",
                message="content source process is not running; no exit report is available after forced termination",
                details={"force_killed": self.source.force_killed if self.source else None, "exit_report": None},
            )
        return request(self.source.host, self.source.port, {"protocol": PROTOCOL_VERSION, **payload}, timeout=self.request_timeout)

    def _configure_node(self, node_id: str, *, ledger_version: int, policy: Policy, preserve_entries: bool) -> dict[str, Any]:
        return self._node_request(
            node_id,
            {
                "op": "configure",
                "ledger_version": ledger_version,
                "policy": policy.to_dict(),
                "preserve_entries": preserve_entries,
            },
        )

    def _sync_node(self, node_id: str, *, ledger_version: int, content_version: str) -> dict[str, Any]:
        return self._node_request(
            node_id,
            {
                "op": "sync",
                "ledger_version": ledger_version,
                "content_version": content_version,
            },
        )

    def _bootstrap_round(self, spec: LoadedSpec) -> dict[str, Any]:
        ledger_version = self.ledger.begin_round(spec, round_kind="bootstrap")
        try:
            responses: list[dict[str, Any]] = []
            for node_id in sorted(self.nodes):
                self._configure_node(node_id, ledger_version=ledger_version, policy=spec.policy, preserve_entries=False)
                responses.append(self._sync_node(node_id, ledger_version=ledger_version, content_version=spec.content_version))
            entries, events = _flatten_states(responses)
            self.ledger.complete_round(
                ledger_version,
                entries=entries,
                evictions=events,
                propagation=[],
                progress_cursor=len(spec.content.records),
            )
        except EdgeError as exc:
            self.ledger.mark_unknown(ledger_version, f"bootstrap failed: {exc.code}: {exc.message}")
            raise
        return self.round_summary(ledger_version)

    def _restore_current_to_nodes(self) -> dict[str, Any]:
        current = self.ledger.current_version()
        assert current is not None
        spec = self.ledger.current_spec()
        assert spec is not None
        responses: list[dict[str, Any]] = []
        for node_id in sorted(self.nodes):
            self._configure_node(node_id, ledger_version=current, policy=spec.policy, preserve_entries=False)
            responses.append(
                self._node_request(
                    node_id,
                    {
                        "op": "restore",
                        "ledger_version": current,
                        "policy": spec.policy.to_dict(),
                        "entries": self.ledger.entries_for_version(current, node_id=node_id),
                        "normalize_snapshot_ttl": False,
                    },
                )
            )
        return {
            "ok": True,
            "recovered_ledger_version": current,
            "content_version": spec.content_version,
            "policy_version": spec.policy_version,
            "entry_count": sum(len(response.get("entries", [])) for response in responses),
            "nodes": sorted(self.nodes),
        }

    def evolve_content(self, new_spec: LoadedSpec, *, trigger_path: str) -> dict[str, Any]:
        self._require_started()
        old_spec = self._require_current_spec()
        self._validate_content_transition(old_spec, new_spec, trigger_path)
        affected = new_spec.content.descendant_order(trigger_path)
        propagation = new_spec.content.propagation_plan([trigger_path])
        ledger_version = self.ledger.begin_round(new_spec, round_kind="content_update")
        try:
            reload_response = self._source_request(
                {
                    "op": "reload",
                    "content": new_spec.content.to_declared_list(),
                    "content_version": new_spec.content_version,
                    "content_digest": new_spec.content_digest,
                }
            )
            if reload_response.get("content_version") != new_spec.content_version:
                raise EdgeError(
                    code="E_CONTENT",
                    message="content source did not adopt the requested content version",
                    details={"expected": new_spec.content_version, "actual": reload_response.get("content_version")},
                )
            responses: list[dict[str, Any]] = []
            for node_id in sorted(self.nodes):
                self._configure_node(
                    node_id,
                    ledger_version=ledger_version,
                    policy=new_spec.policy,
                    preserve_entries=True,
                )
                responses.append(
                    self._node_request(
                        node_id,
                        {
                            "op": "apply_propagation",
                            "ledger_version": ledger_version,
                            "content_version": new_spec.content_version,
                            "trigger_path": trigger_path,
                            "affected_paths": affected,
                        },
                    )
                )
            entries, events = _flatten_states(responses)
            self.ledger.complete_round(
                ledger_version,
                entries=entries,
                evictions=events,
                propagation=propagation,
                progress_cursor=len(new_spec.content.records) + len(propagation),
            )
        except EdgeError as exc:
            self.ledger.mark_unknown(ledger_version, f"content update failed: {exc.code}: {exc.message}")
            exc.details = {**exc.details, "unknown_ledger_version": ledger_version, "recover_required": True}
            raise
        self.spec = new_spec
        self._write_active_spec(new_spec)
        summary = self.round_summary(ledger_version)
        summary["trigger_path"] = trigger_path
        summary["affected_paths"] = affected
        summary["trace_digest"] = canonical_json_bytes(propagation).hex()[:16] if propagation else ""
        return summary

    def apply_policy(self, new_spec: LoadedSpec) -> dict[str, Any]:
        self._require_started()
        old_spec = self._require_current_spec()
        self._validate_policy_transition(old_spec, new_spec)
        ledger_version = self.ledger.begin_round(new_spec, round_kind="policy_update")
        old_exit_reports: list[dict[str, Any]] = []
        try:
            old_exit_reports = self._stop_nodes()
            self._check_ports(self._node_port_checks(new_spec.policy))
            self._spawn_nodes(new_spec.policy)
            for node_id in sorted(self.nodes):
                self._wait_worker(self.nodes[node_id], service="edge-node")
            responses: list[dict[str, Any]] = []
            for node_id in sorted(self.nodes):
                self._configure_node(node_id, ledger_version=ledger_version, policy=new_spec.policy, preserve_entries=False)
                responses.append(self._sync_node(node_id, ledger_version=ledger_version, content_version=new_spec.content_version))
            entries, events = _flatten_states(responses)
            self.ledger.complete_round(
                ledger_version,
                entries=entries,
                evictions=events,
                propagation=[],
                progress_cursor=len(new_spec.content.records),
            )
        except EdgeError as exc:
            self.ledger.mark_unknown(ledger_version, f"policy update failed: {exc.code}: {exc.message}")
            exc.details = {**exc.details, "unknown_ledger_version": ledger_version, "recover_required": True}
            raise
        self.spec = new_spec
        self._write_active_spec(new_spec)
        summary = self.round_summary(ledger_version)
        summary["previous_node_exit_reports"] = old_exit_reports
        return summary

    def _validate_content_transition(self, old: LoadedSpec, new: LoadedSpec, trigger_path: str) -> None:
        if old.policy.digest != new.policy.digest:
            raise EdgeError(
                code="E_SPEC",
                message="content update must not mix policy changes; use apply_policy in a separate round",
                field_path="policy.version",
                details={"old": old.policy_version, "new": new.policy_version},
            )
        old_paths = set(old.content.by_path)
        new_paths = set(new.content.by_path)
        if old_paths != new_paths:
            raise EdgeError(
                code="E_CONTENT",
                message="content update currently requires a stable path set",
                field_path="content[].path",
                details={"added": sorted(new_paths - old_paths), "removed": sorted(old_paths - new_paths)},
            )
        if trigger_path not in old.content.by_path:
            raise EdgeError(
                code="E_CONTENT",
                message="trigger path does not exist in the current content version",
                path=trigger_path,
                field_path="trigger_path",
            )
        changed: set[str] = set()
        for path in sorted(old_paths):
            old_fp = old.content.by_path[path].version_fingerprint()
            new_fp = new.content.by_path[path].version_fingerprint()
            if old_fp == new_fp:
                continue
            if old.content.by_path[path].etag == new.content.by_path[path].etag:
                raise EdgeError(
                    code="E_CONTENT",
                    message="changed content must receive a new ETag",
                    path=path,
                    field_path=f"content[{path}].etag",
                    details={"etag": new.content.by_path[path].etag},
                )
            changed.add(path)
        if trigger_path not in changed:
            raise EdgeError(
                code="E_CONTENT",
                message="trigger path content version did not change",
                path=trigger_path,
                field_path=f"content[{trigger_path}].etag",
            )
        affected = set(new.content.descendant_order(trigger_path))
        unexpected = changed - affected - {trigger_path}
        if unexpected:
            raise EdgeError(
                code="E_CONTENT",
                message="unrelated paths changed in a single-trigger content update",
                path=sorted(unexpected)[0],
                field_path=f"content[{sorted(unexpected)[0]}].etag",
                details={"unexpected_changed_paths": sorted(unexpected), "trigger_path": trigger_path},
            )

    def _validate_policy_transition(self, old: LoadedSpec, new: LoadedSpec) -> None:
        if old.content.digest != new.content.digest:
            raise EdgeError(
                code="E_SPEC",
                message="policy update must not mix content changes",
                field_path="content",
                details={"old": old.content_version, "new": new.content_version},
            )
        if new.policy.version == old.policy.version and new.policy.digest != old.policy.digest:
            raise EdgeError(
                code="E_SPEC",
                message="policy.version must change when policy fields change",
                field_path="policy.version",
                details={"version": new.policy.version},
            )
        if new.policy.digest == old.policy.digest:
            raise EdgeError(
                code="E_SPEC",
                message="policy update did not change the policy",
                field_path="policy.version",
                details={"version": new.policy.version},
            )

    def recover(self) -> dict[str, Any]:
        self._require_started(allow_unhealthy=True)
        unknown = self.ledger.recover_pending()
        spec = self.ledger.current_spec()
        if spec is None:
            raise EdgeError(code="E_STATE", message="no complete ledger version is available for recovery")
        self.spec = spec
        self._write_active_spec(spec)
        # Always reconstruct the live source and node set from the ledger.
        if self.source is not None and self.source.alive:
            try:
                self._source_request(
                    {
                        "op": "reload",
                        "content": spec.content.to_declared_list(),
                        "content_version": spec.content_version,
                        "content_digest": spec.content_digest,
                    }
                )
            except EdgeError:
                self.source.force_terminate()
                self.source = None
        if self.source is None or not self.source.alive:
            self._check_ports([("content-source", self.source_port)])
            self._spawn_source(spec)
            self._wait_worker(self.source, service="content-source")
        self._stop_nodes()
        self._check_ports(self._node_port_checks(spec.policy))
        self._spawn_nodes(spec.policy)
        for node_id in sorted(self.nodes):
            self._wait_worker(self.nodes[node_id], service="edge-node")
        restored = self._restore_current_to_nodes()
        return {
            "ok": True,
            "recovered_unknown_rounds": unknown,
            "recovered": restored,
            "source": self._process_info(self.source),
            "nodes": [self._process_info(self.nodes[node_id]) for node_id in sorted(self.nodes)],
        }

    def save_snapshot(self) -> dict[str, Any]:
        self._require_started()
        return self.snapshots.save(self.ledger)

    def load_snapshot(self, path: str | Path) -> LoadedSnapshot:
        return self.snapshots.load(path, self.ledger)

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        self._require_started(allow_unhealthy=True)
        loaded = self.snapshots.load(path, self.ledger)
        unknown = self.ledger.recover_pending()
        old_node_reports = self._stop_nodes()
        new_version = self.ledger.restore_snapshot_document(loaded.document)
        restored_spec = LoadedSpec(content=loaded.content, policy=loaded.policy, source_path=None)
        self.spec = restored_spec
        self._write_active_spec(restored_spec)
        if self.source is not None and self.source.alive:
            try:
                self._source_request(
                    {
                        "op": "reload",
                        "content": restored_spec.content.to_declared_list(),
                        "content_version": restored_spec.content_version,
                        "content_digest": restored_spec.content_digest,
                    }
                )
            except EdgeError:
                self.source.force_terminate()
                self.source = None
        if self.source is None or not self.source.alive:
            self._check_ports([("content-source", self.source_port)])
            self._spawn_source(restored_spec)
            self._wait_worker(self.source, service="content-source")
        self._check_ports(self._node_port_checks(restored_spec.policy))
        self._spawn_nodes(restored_spec.policy)
        for node_id in sorted(self.nodes):
            self._wait_worker(self.nodes[node_id], service="edge-node")
        responses: list[dict[str, Any]] = []
        for node_id in sorted(self.nodes):
            self._configure_node(node_id, ledger_version=new_version, policy=restored_spec.policy, preserve_entries=False)
            responses.append(
                self._node_request(
                    node_id,
                    {
                        "op": "restore",
                        "ledger_version": new_version,
                        "policy": restored_spec.policy.to_dict(),
                        "entries": self.ledger.entries_for_version(new_version, node_id=node_id),
                        "normalize_snapshot_ttl": False,
                    },
                )
            )
        summary = snapshot_summary(loaded)
        summary.update(
            {
                "recovered_unknown_rounds": unknown,
                "restored_ledger_version": new_version,
                "previous_node_exit_reports": old_node_reports,
                "node_entry_counts": {
                    node_id: len(self.ledger.entries_for_version(new_version, node_id=node_id)) for node_id in sorted(self.nodes)
                },
                "live_node_entry_counts": {node_id: len(response.get("entries", [])) for node_id, response in zip(sorted(self.nodes), responses)},
            }
        )
        return summary

    def terminate_node(self, node_id: str, *, force: bool = True) -> dict[str, Any]:
        managed = self.nodes.get(node_id)
        if managed is None:
            raise EdgeError(code="E_PROCESS", message="unknown node id", details={"node_id": node_id})
        if force:
            report = managed.force_terminate()
            self.termination_reports.append(report)
            return report
        report = managed.graceful_stop(timeout=self.request_timeout)
        self.termination_reports.append(report)
        return report

    def terminate_source(self, *, force: bool = True) -> dict[str, Any]:
        if self.source is None:
            raise EdgeError(code="E_PROCESS", message="content source has not been started")
        if force:
            report = self.source.force_terminate()
        else:
            report = self.source.graceful_stop(timeout=self.request_timeout)
        self.termination_reports.append(report)
        return report

    def _stop_nodes(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for node_id in sorted(self.nodes):
            managed = self.nodes[node_id]
            if not managed.alive:
                reports.append(managed.exit_report())
                continue
            try:
                reports.append(managed.graceful_stop(timeout=self.request_timeout))
            except EdgeError:
                reports.append(managed.force_terminate())
        self.nodes = {}
        self.node_ports = {}
        return reports

    def close(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        if self.nodes:
            reports.extend(self._stop_nodes())
        if self.source is not None:
            if self.source.alive:
                try:
                    reports.append(self.source.graceful_stop(timeout=self.request_timeout))
                except EdgeError:
                    reports.append(self.source.force_terminate())
            else:
                reports.append(self.source.exit_report())
            self.source = None
        self.started = False
        self.ledger.close()
        return {"ok": True, "exit_reports": reports}

    def _require_started(self, *, allow_unhealthy: bool = False) -> None:
        if not self.started:
            raise EdgeError(code="E_STATE", message="cluster has not completed start()")
        if not allow_unhealthy:
            if self.source is None or not self.source.alive:
                raise EdgeError(code="E_PROCESS", message="content source is not healthy")
            dead = [node_id for node_id, managed in self.nodes.items() if not managed.alive]
            if dead:
                raise EdgeError(
                    code="E_PROCESS",
                    message="node process is not healthy",
                    details={"node_ids": sorted(dead), "recover_required": True},
                )

    def _require_current_spec(self) -> LoadedSpec:
        spec = self.ledger.current_spec()
        if spec is None:
            raise EdgeError(code="E_STATE", message="no current complete ledger version exists")
        return spec

    def summary(self) -> dict[str, Any]:
        view = self.ledger.view_model()
        view.update(
            {
                "started": self.started,
                "source": self._process_info(self.source) if self.source else None,
                "nodes": [self._process_info(self.nodes[node_id]) for node_id in sorted(self.nodes)],
                "termination_reports": list(self.termination_reports),
            }
        )
        return view

    def round_summary(self, ledger_version: int) -> dict[str, Any]:
        row = self.ledger.version_row(ledger_version)
        entries = self.ledger.entries_for_version(ledger_version)
        evictions = self.ledger.evictions_for_version(ledger_version)
        propagation = self.ledger.propagation_for_version(ledger_version)
        by_node: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            by_node.setdefault(entry["node_id"], []).append(entry)
        return {
            "ok": True,
            "ledger_version": ledger_version,
            "status": row["status"],
            "round_kind": row["round_kind"],
            "content_version": row["content_version"],
            "policy_version": row["policy_version"],
            "progress_cursor": row["progress_cursor"],
            "entry_count": len(entries),
            "eviction_count": len(evictions),
            "propagation_count": len(propagation),
            "entries_by_node": {
                node_id: [
                    {
                        "path": entry["path"],
                        "etag": entry["etag"],
                        "remaining_ttl_ms": entry["remaining_ttl_ms"],
                        "entry_version": entry["entry_version"],
                        "stale": entry["stale"],
                    }
                    for entry in sorted(values, key=lambda item: item["cache_key"])
                ]
                for node_id, values in sorted(by_node.items())
            },
            "evictions": evictions,
            "propagation": propagation,
        }

    def _process_info(self, managed: ManagedProcess | None) -> dict[str, Any] | None:
        if managed is None:
            return None
        return {
            "role": managed.role,
            "identity": managed.identity,
            "host": managed.host,
            "port": managed.port,
            "pid": managed.process.pid,
            "alive": managed.alive,
            "returncode": managed.process.returncode,
            "force_killed": managed.force_killed,
            "exit_report_available": bool(managed.shutdown_response is not None and not managed.force_killed),
        }


def _flatten_states(responses: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for response in responses:
        node_id = str(response["node_id"])
        for entry in response.get("entries", []):
            item = dict(entry)
            item["node_id"] = node_id
            entries.append(item)
        for event in response.get("evictions", []):
            item = dict(event)
            item["node_id"] = node_id
            events.append(item)
    return entries, events


def _service_is_listening(host: str, port: int, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
