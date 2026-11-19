"""Local-loopback socket helpers and E_PORT preflight checks."""

from __future__ import annotations

import ipaddress
import socket

from .errors import BusinessError


def require_loopback_host(host: str) -> None:
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise BusinessError(
            "E_ARGUMENT",
            "host must be an IPv4 loopback literal (for example 127.0.0.1)",
            details={"host": host},
        ) from exc
    if not address.is_loopback:
        raise BusinessError("E_ARGUMENT", "non-loopback hosts are not permitted", details={"host": host})


def check_tcp_port(host: str, port: int) -> None:
    require_loopback_host(host)
    if not 0 < port <= 65535:
        raise BusinessError("E_PORT", "TCP port must be between 1 and 65535", details={"port": port})
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise BusinessError(
            "E_PORT",
            f"TCP port is unavailable: {exc}",
            details={"host": host, "port": port, "transport": "tcp"},
        ) from exc
    finally:
        sock.close()


def check_udp_port(host: str, port: int) -> None:
    require_loopback_host(host)
    if not 0 < port <= 65535:
        raise BusinessError("E_PORT", "UDP port must be between 1 and 65535", details={"port": port})
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise BusinessError(
            "E_PORT",
            f"UDP port is unavailable: {exc}",
            details={"host": host, "port": port, "transport": "udp"},
        ) from exc
    finally:
        sock.close()


def check_loopback_ports(host: str, ports: list[tuple[str, int]]) -> None:
    for transport, port in ports:
        if transport == "udp":
            check_udp_port(host, port)
        elif transport == "tcp":
            check_tcp_port(host, port)
        else:
            raise BusinessError("E_PORT", f"unknown transport: {transport}")


def query_udp(host: str, port: int, payload: bytes, *, timeout: float = 2.0) -> bytes:
    require_loopback_host(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(payload, (host, port))
        response, _ = sock.recvfrom(65535)
        return response
    except OSError as exc:
        raise BusinessError(
            "E_UPSTREAM",
            f"local UDP query failed: {exc}",
            details={"host": host, "port": port},
        ) from exc
    finally:
        sock.close()


def query_tcp(host: str, port: int, payload: bytes, *, timeout: float = 2.0) -> bytes:
    from .wire import frame_tcp, receive_tcp_message

    require_loopback_host(host)
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.sendall(frame_tcp(payload))
        return receive_tcp_message(sock)
    except (OSError, ConnectionError) as exc:
        raise BusinessError(
            "E_UPSTREAM",
            f"local TCP query failed: {exc}",
            details={"host": host, "port": port},
        ) from exc
    finally:
        sock.close()
