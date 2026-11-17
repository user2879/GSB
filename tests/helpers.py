from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

from tunnelvpn.config import TunnelConfig, parse_config
from tunnelvpn.peer import PeerServer


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return ports


def raw_config(
    tmp_path: Path,
    *,
    route_version: str = "test-v1",
    route_max_sessions: int | None = None,
    route_bandwidth_kbps: int = 16,
    global_bandwidth_kbps: int = 64,
    global_max_sessions: int = 4,
    peer_timeout_ms: int = 200,
    connect_timeout_ms: int = 100,
    fragment_timeout_ms: int = 120,
    peer_behaviors: tuple[str, ...] = ("normal", "digest-error"),
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    ports = free_ports(len(peer_behaviors) * 2 + 2)
    listen_ports = ports[: len(peer_behaviors)]
    target_ports = ports[len(peer_behaviors) : len(peer_behaviors) * 2]
    script_path = tmp_path / "peer-script.json"
    script_path.write_text(
        json.dumps(
            {
                "by_sequence": {
                    "slow:0": 500,
                    "slow:1": 500,
                    "slow:2": 500,
                }
            }
        ),
        encoding="utf-8",
    )
    peers: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    for index, behavior in enumerate(peer_behaviors):
        peer_id = f"peer-{behavior}-{index}"
        route_id = (
            behavior.replace("-", "_")
            if behavior not in {"normal", "digest-error"}
            else ("alpha" if behavior == "normal" else "corrupt")
        )
        if behavior == "delay-script":
            route_id = "slow"
        if behavior == "unreachable":
            route_id = "dead"
        if any(route["id"] == route_id for route in routes):
            route_id = f"{route_id}-{index}"
        peers.append(
            {
                "id": peer_id,
                "host": "127.0.0.1",
                "port": target_ports[index],
                "behavior": behavior,
                "delay_ms": 500 if behavior == "delay-script" else 0,
                **(
                    {"script": str(script_path)}
                    if behavior == "delay-script"
                    else {}
                ),
            }
        )
        routes.append(
            {
                "id": route_id,
                "listen": {"host": "127.0.0.1", "port": listen_ports[index]},
                "target": {"host": "127.0.0.1", "port": target_ports[index]},
                "peer_id": peer_id,
                "mtu": 192,
                "cipher": "AES-256-GCM",
                "bandwidth_kbps": route_bandwidth_kbps,
                "max_sessions": (
                    route_max_sessions
                    if route_max_sessions is not None
                    else 2
                ),
            }
        )
    raw = {
        "schema_version": 1,
        "route_version": route_version,
        "global": {
            "bandwidth_kbps": global_bandwidth_kbps,
            "max_sessions": global_max_sessions,
            "fragment_timeout_ms": fragment_timeout_ms,
            "peer_timeout_ms": peer_timeout_ms,
            "connect_timeout_ms": connect_timeout_ms,
        },
        "peers": peers,
        "routes": routes,
    }
    return raw, {"listen": listen_ports, "target": target_ports}


def config(tmp_path: Path, **kwargs: Any) -> tuple[TunnelConfig, dict[str, Any]]:
    raw, ports = raw_config(tmp_path, **kwargs)
    return parse_config(copy.deepcopy(raw)), {"raw": raw, "ports": ports}


class PeerGroup:
    def __init__(self, config: TunnelConfig, log_dir: Path) -> None:
        self.config = config
        self.log_dir = log_dir
        self.servers: list[PeerServer] = []

    def __enter__(self) -> "PeerGroup":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for peer in self.config.peers:
            if peer.behavior == "unreachable":
                continue
            server = PeerServer(
                self.config,
                peer,
                log_path=self.log_dir / f"peer-{peer.id}.jsonl",
            )
            server.start()
            self.servers.append(server)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for server in reversed(self.servers):
            server.stop()
