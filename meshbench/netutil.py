"""Loopback-only networking helpers and preflight checks."""

import socket

from .config import LOOPBACK
from .errors import UpstreamError


def is_port_free(port: int, host: str = LOOPBACK, timeout: float = 0.5) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(timeout)
    try:
        probe.connect((host, port))
    except OSError:
        return True
    else:
        return False
    finally:
        probe.close()


def assert_ports_free(ports, host: str = LOOPBACK):
    """Raise/return a stable, human-readable list of already-bound ports."""
    busy = sorted({port for port in ports if not is_port_free(port, host)})
    if busy:
        raise UpstreamError(
            "high ports already in use on 127.0.0.1; aborting before binding",
            busy_ports=busy,
        )
    return busy


def preflight_upstreams(instances, host: str = LOOPBACK, timeout: float = 1.0):
    """Verify every simulated upstream is reachable on the loopback interface."""
    unreachable = []
    for inst in sorted(instances, key=lambda item: (item.port, item.instance_id)):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(timeout)
        try:
            probe.connect((host, inst.port))
        except OSError:
            unreachable.append({"instance_id": inst.instance_id, "port": inst.port})
        finally:
            probe.close()
    if unreachable:
        raise UpstreamError(
            "simulated upstream unreachable on 127.0.0.1",
            unreachable=unreachable,
        )
    return unreachable
