from __future__ import annotations

import socket
from pathlib import Path
from typing import Iterable

from quota_ledger.injection import inject_snapshot_corruption  # re-export for tests
from quota_ledger.samplegen import write_sample


def free_port_base(count: int = 4) -> int:
    for base in range(44000, 60000):
        sockets: list[socket.socket] = []
        try:
            for offset in range(count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    except OSError:
                        pass
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", base + offset))
                sockets.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no free high loopback port range")


def make_sample(root: Path, count: int = 4) -> tuple[Path, int]:
    base = free_port_base(count)
    sample = root / "sample"
    write_sample(sample, base_port=base)
    return sample, base


def wait_ports_free(ports: Iterable[int], timeout_s: float = 6.0) -> bool:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        available = True
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            except OSError:
                available = False
                break
            finally:
                sock.close()
        if available:
            return True
        time.sleep(0.1)
    return False
