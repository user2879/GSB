from __future__ import annotations

import socket
from typing import Any

from .constants import Code


def check_loopback_port(host: str, port: int, timeout: float = 0.2) -> dict[str, Any]:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    address = "::1" if host == "::1" else ("127.0.0.1" if host == "localhost" else host)
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.bind((address, port))
    except OSError as exc:
        return {
            "host": host,
            "port": port,
            "free": False,
            "code": Code.PORT_IN_USE,
            "message": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return {"host": host, "port": port, "free": True}


def check_worker_ports(workers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for worker in sorted(workers, key=lambda w: w["worker_id"]):
        result = check_loopback_port(worker.get("host", "127.0.0.1"), int(worker["port"]))
        result["worker_id"] = worker["worker_id"]
        checks.append(result)
        if not result["free"]:
            failures.append({
                "code": Code.PORT_IN_USE,
                "source": "loopback-port-check",
                "item_id": worker["worker_id"],
                "field": "port",
                "message": result["message"],
            })
    return checks, failures
