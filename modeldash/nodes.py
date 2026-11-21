"""Node subprocess supervision and the timeout-bounded dispatch client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DispatchError
from .inference import request_semantic_digest
from .protocol import PROTOCOL_VERSION, ProtocolEOF, ProtocolError, connect, is_port_free, receive_message, send_message
from .spec import ConfigRecord, NodeRecord, Spec, TaskRecord
from .util import StructuredLog, canonical_json, digest_json, ensure_directory, now_monotonic_ms, terminate_process


@dataclass
class DispatchResult:
    output_summary: str
    input_digest: str
    compute_ms: int
    items_used: list[str]
    response: dict[str, Any]
    dispatched_ms: int


class NodeSupervisor:
    """Owns all node subprocesses and retries startup using the declared backoff."""

    def __init__(self, spec: Spec, data_dir: str | Path, logger: StructuredLog | None = None) -> None:
        self.spec = spec
        self.data_dir = ensure_directory(data_dir)
        self.log_dir = ensure_directory(self.data_dir / "logs")
        self.logger = logger or StructuredLog(self.log_dir / "workbench.jsonl")
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_files: list[Any] = []

    def _node_command(self, node: NodeRecord) -> list[str]:
        return [
            sys.executable,
            "-m",
            "modeldash.node_server",
            "--node-id",
            node.node_id,
            "--port",
            str(node.port),
            "--seed",
            str(self.spec.seed),
            "--delay-ms",
            str(node.shard_delay_ms),
            "--faults-json",
            canonical_json(
                [
                    {"shard_index": fault.shard_index, "remaining": fault.remaining, "mode": fault.mode}
                    for fault in node.faults
                ]
            ),
            "--watch-parent",
        ]

    def _spawn(self) -> None:
        root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for node in self.spec.nodes:
            log_path = self.log_dir / f"node-{node.node_id}.log"
            log_handle = log_path.open("ab")
            self._log_files.append(log_handle)
            process = subprocess.Popen(
                self._node_command(node),
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self.processes[node.node_id] = process

    def _health(self, node: NodeRecord) -> tuple[bool, str, str]:
        deadline = time.monotonic() + node.startup_timeout_ms / 1000.0
        last_error = "startup timeout"
        while time.monotonic() < deadline:
            if node.node_id in self.processes and self.processes[node.node_id].poll() is not None:
                return False, "E_NODE_EXIT", f"node process exited with {self.processes[node.node_id].returncode}"
            sock: socket.socket | None = None
            try:
                sock = connect(node.host, node.port, min(250, node.startup_timeout_ms))
                send_message(sock, {"op": "hello", "protocol": PROTOCOL_VERSION})
                response = receive_message(sock)
                if response.get("op") != "hello_ok" or response.get("protocol") != PROTOCOL_VERSION:
                    return False, "E_HANDSHAKE", f"invalid handshake response: {response}"
                if response.get("node_id") != node.node_id:
                    return False, "E_HANDSHAKE", "handshake node id mismatch"
                return True, "", ""
            except socket.timeout as exc:
                return False, "E_HANDSHAKE", f"handshake timeout: {exc}"
            except (ProtocolEOF, ProtocolError, ConnectionResetError) as exc:
                return False, "E_HANDSHAKE", f"handshake failed: {exc}"
            except ConnectionRefusedError as exc:
                last_error = str(exc)
            except OSError as exc:
                last_error = str(exc)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            time.sleep(0.02)
        return False, "E_HANDSHAKE", last_error

    def _kill_all(self) -> None:
        for node_id in sorted(self.processes):
            process = self.processes[node_id]
            terminate_process(process)
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
        self.processes.clear()
        for handle in self._log_files:
            try:
                handle.close()
            except Exception:
                pass
        self._log_files.clear()

    def start_all(self) -> None:
        max_attempts = 1 + self.spec.budget.max_retries
        last_code = "E_PORT_IN_USE"
        last_node = self.spec.nodes[0].node_id if self.spec.nodes else ""
        last_detail = "node startup failed"
        for attempt in range(1, max_attempts + 1):
            occupied = [node for node in self.spec.nodes if not is_port_free(node.host, node.port)]
            if occupied:
                node = occupied[0]
                code = "E_PORT_IN_USE"
                detail = f"127.0.0.1:{node.port} is already occupied"
                self.logger.write(
                    "node_start_error",
                    code=code,
                    node_id=node.node_id,
                    port=node.port,
                    attempt=attempt,
                    detail=detail,
                )
            else:
                self._spawn()
                failures: list[tuple[str, str, str]] = []
                for node in self.spec.nodes:
                    ok, code, detail = self._health(node)
                    if not ok:
                        failures.append((node.node_id, code, detail))
                        break
                if not failures:
                    self.logger.write(
                        "nodes_started",
                        attempt=attempt,
                        nodes=[node.node_id for node in self.spec.nodes],
                        ports=[node.port for node in self.spec.nodes],
                    )
                    return
                last_node, code, detail = failures[0]
                self.logger.write(
                    "node_start_error",
                    code=code,
                    node_id=last_node,
                    attempt=attempt,
                    detail=detail,
                )
                self._kill_all()
            last_code = code
            last_node = last_node
            last_detail = detail
            if attempt <= self.spec.budget.max_retries:
                delay = self.spec.budget.retry_backoff_ms[attempt - 1]
                self.logger.write(
                    "node_startup_retry",
                    code=code,
                    node_id=last_node,
                    attempt=attempt,
                    backoff_ms=delay,
                )
                time.sleep(delay / 1000.0)
        raise DispatchError(
            last_code,
            last_node,
            f"node startup exhausted after {max_attempts} attempts: {last_detail}",
            retryable=False,
            context={"attempts": max_attempts},
        )

    def stop_all(self) -> None:
        self._kill_all()

    def __enter__(self) -> "NodeSupervisor":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()


class SocketDispatcher:
    def __init__(self, spec: Spec, logger: StructuredLog | None = None) -> None:
        self.spec = spec
        self.logger = logger

    def _payload(self, task: TaskRecord, config: ConfigRecord, shard_index: int, request_id: str) -> dict[str, Any]:
        return {
            "op": "infer",
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "task_id": task.task_id,
            "node_id": task.node_id,
            "config_version": config.version,
            "shard_index": shard_index,
            "shard_count": config.shards,
            "items": config.shard_items(shard_index),
            "input": {
                "seed": self.spec.seed,
                "size": self.spec.input_size,
                "scale": self.spec.input_scale,
            },
        }

    def dispatch(self, task: TaskRecord, shard_index: int, request_id: str) -> DispatchResult:
        config = self.spec.config(task.config_version)
        node = self.spec.node(task.node_id)
        payload = self._payload(task, config, shard_index, request_id)
        started = now_monotonic_ms()
        sock: socket.socket | None = None
        try:
            sock = connect(node.host, node.port, self.spec.budget.timeout_ms)
            send_message(sock, {"op": "hello", "protocol": PROTOCOL_VERSION})
            hello = receive_message(sock)
            if hello.get("op") != "hello_ok" or hello.get("protocol") != PROTOCOL_VERSION:
                raise DispatchError("E_HANDSHAKE", node.node_id, f"invalid handshake response: {hello}")
            if hello.get("node_id") != node.node_id:
                raise DispatchError("E_HANDSHAKE", node.node_id, "handshake node id mismatch")
            send_message(sock, payload)
            response = receive_message(sock)
            if response.get("op") == "error":
                code = str(response.get("code", "E_NODE"))
                retryable = code not in {"E_PROTOCOL", "E_IDEMPOTENCY"}
                raise DispatchError(code, node.node_id, str(response.get("detail", "node error")), retryable=retryable)
            expected_semantic = request_semantic_digest(payload)
            if response.get("op") != "result":
                raise DispatchError("E_PROTOCOL", node.node_id, f"unexpected response: {response}", retryable=False)
            if response.get("semantic_digest") != expected_semantic:
                raise DispatchError("E_PROTOCOL", node.node_id, "semantic digest mismatch", retryable=False)
            if response.get("request_id") != request_id or response.get("shard_index") != shard_index:
                raise DispatchError("E_PROTOCOL", node.node_id, "request/shard mismatch", retryable=False)
            output_summary = response.get("output_summary")
            input_digest = response.get("input_digest")
            compute_ms = response.get("compute_ms")
            items_used = response.get("items_used")
            if not isinstance(output_summary, str) or not isinstance(input_digest, str):
                raise DispatchError("E_PROTOCOL", node.node_id, "missing output or input digest", retryable=False)
            if not isinstance(compute_ms, int) or isinstance(compute_ms, bool):
                raise DispatchError("E_PROTOCOL", node.node_id, "compute_ms must be an integer", retryable=False)
            if not isinstance(items_used, list):
                raise DispatchError("E_PROTOCOL", node.node_id, "items_used must be an array", retryable=False)
            return DispatchResult(
                output_summary=output_summary,
                input_digest=input_digest,
                compute_ms=max(0, compute_ms),
                items_used=[str(item) for item in items_used],
                response=response,
                dispatched_ms=max(0, now_monotonic_ms() - started),
            )
        except socket.timeout as exc:
            raise DispatchError("E_TIMEOUT", node.node_id, f"socket timeout: {exc}") from exc
        except ConnectionRefusedError as exc:
            raise DispatchError("E_CONNECT", node.node_id, f"connection refused: {exc}") from exc
        except (ProtocolEOF, ProtocolError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise DispatchError("E_TRANSPORT", node.node_id, f"transport failure: {exc}") from exc
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            if self.logger is not None:
                self.logger.write(
                    "dispatch_finished",
                    node_id=node.node_id,
                    task_id=task.task_id,
                    shard_index=shard_index,
                    request_id=request_id,
                )


def request_id_for(session_id: str, task: TaskRecord, shard_index: int) -> str:
    material = {"session": session_id, "task": task.task_id, "config_version": task.config_version, "shard": shard_index}
    return "req-" + digest_json(material)[:32]
