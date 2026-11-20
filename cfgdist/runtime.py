"""Node declaration loading, port checks and subprocess node pool."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import BizError, ErrorCode
from .logging_setup import StructuredLogger
from .protocol import is_port_in_use, request


def load_node_declaration(path: str | Path) -> list[dict]:
    declaration_path = Path(path)
    try:
        payload = json.loads(declaration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BizError(
            ErrorCode.E_FORMAT,
            f"cannot parse node declaration {declaration_path}: {exc}",
            {"field_path": str(declaration_path)},
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        raise BizError(
            ErrorCode.E_FORMAT,
            "node declaration must be an object with a 'nodes' array",
            {"field_path": "$"},
        )
    nodes = payload["nodes"]
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        base = f"$.nodes[{index}]"
        if not isinstance(node, dict):
            raise BizError(ErrorCode.E_FORMAT, "node entry must be an object", {"field_path": base})
        for field in ("id", "host", "port"):
            if field not in node:
                raise BizError(
                    ErrorCode.E_FORMAT,
                    f"required field '{field}' missing",
                    {"field_path": f"{base}.{field}"},
                )
        if not isinstance(node["id"], str) or node["id"] in seen:
            raise BizError(
                ErrorCode.E_FORMAT,
                "node id must be a unique non-empty string",
                {"field_path": f"{base}.id"},
            )
        seen.add(node["id"])
        if node["host"] not in ("127.0.0.1", "localhost"):
            raise BizError(
                ErrorCode.E_FORMAT,
                "only loopback hosts (127.0.0.1/localhost) are permitted",
                {"field_path": f"{base}.host"},
            )
        if isinstance(node["port"], bool) or not isinstance(node["port"], int):
            raise BizError(
                ErrorCode.E_FORMAT,
                "port must be an integer",
                {"field_path": f"{base}.port"},
            )
        if not 1024 <= node["port"] <= 65535:
            raise BizError(
                ErrorCode.E_FORMAT,
                "port must be a high-order port in [1024, 65535]",
                {"field_path": f"{base}.port"},
            )
        node.setdefault("faults", {})
        if not isinstance(node["faults"], dict):
            raise BizError(
                ErrorCode.E_FORMAT,
                "faults must be an object",
                {"field_path": f"{base}.faults"},
            )
    ports = [node["port"] for node in nodes]
    if len(set(ports)) != len(ports):
        raise BizError(
            ErrorCode.E_FORMAT,
            "duplicate ports in node declaration",
            {"field_path": "$.nodes[*].port"},
        )
    return nodes


def check_ports_free(nodes: list[dict]) -> None:
    busy = [
        {"node": node["id"], "host": node["host"], "port": node["port"]}
        for node in nodes
        if is_port_in_use(node["host"], node["port"])
    ]
    if busy:
        raise BizError(
            ErrorCode.E_PORT,
            "one or more loopback ports are already in use; choose other high-order ports",
            {"busy": busy},
        )


@dataclass
class ManagedNode:
    spec: dict
    process: subprocess.Popen

    @property
    def id(self) -> str:
        return self.spec["id"]


class NodePool:
    """Start/stop node simulator subprocesses and wait for readiness."""

    def __init__(self, nodes: list[dict], logger: StructuredLogger | None = None):
        self.nodes = nodes
        self.logger = logger
        self._managed: list[ManagedNode] = []

    def start(self, startup_timeout_ms: int = 5000) -> None:
        check_ports_free(self.nodes)
        for node in self.nodes:
            command = [
                sys.executable,
                "-m",
                "cfgdist.node_simulator",
                node["id"],
                node["host"],
                str(node["port"]),
                json.dumps(node.get("faults", {})),
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._managed.append(ManagedNode(spec=node, process=process))
            if self.logger is not None:
                self.logger.log("node.start", node=node["id"], port=node["port"])
        deadline = time.monotonic() + startup_timeout_ms / 1000.0
        for node in self.nodes:
            while time.monotonic() < deadline:
                try:
                    response = request(
                        node["host"],
                        node["port"],
                        {"kind": "health"},
                        timeout_ms=500,
                    )
                    if response.get("ok"):
                        break
                except (OSError, BizError):
                    time.sleep(0.05)
            else:
                raise BizError(
                    ErrorCode.E_NODE,
                    f"node {node['id']} did not become ready in time",
                    {"node": node["id"]},
                )

    def stop(self) -> None:
        for managed in self._managed:
            if managed.process.poll() is None:
                managed.process.terminate()
        for managed in self._managed:
            try:
                managed.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                managed.process.kill()
                managed.process.wait(timeout=5)
            if self.logger is not None:
                self.logger.log("node.stop", node=managed.id)
        self._managed.clear()

    def __enter__(self) -> "NodePool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def wait_for_port_release(nodes: list[dict], timeout_ms: int = 3000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        if not any(is_port_in_use(node["host"], node["port"]) for node in nodes):
            return
        time.sleep(0.05)
    raise BizError(ErrorCode.E_PORT, "ports did not release after node shutdown", {})
