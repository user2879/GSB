from __future__ import annotations

import socket
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def zone_v1(root: Path) -> Path:
    return root / "data" / "zones" / "lab.v1.json"


@pytest.fixture
def zone_v2(root: Path) -> Path:
    return root / "data" / "zones" / "lab.v2.json"


@pytest.fixture
def upstream_v1(root: Path) -> Path:
    return root / "data" / "upstream" / "v1.json"


@pytest.fixture
def upstream_v2(root: Path) -> Path:
    return root / "data" / "upstream" / "v2.json"


def free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def free_dual_port() -> int:
    for _ in range(100):
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(("127.0.0.1", 0))
        port = tcp.getsockname()[1]
        tcp.close()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(("127.0.0.1", port))
        except OSError:
            udp.close()
            continue
        udp.close()
        return port
    raise RuntimeError("could not find a free UDP/TCP loopback port")

