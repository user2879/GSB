#!/usr/bin/env python3
"""Offline environment preflight for Windows or WSL."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunnelvpn.config import load_config  # noqa: E402
from tunnelvpn.errors import TunnelError  # noqa: E402


def _port_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local tunnel demo prerequisites")
    parser.add_argument("--config", default="data/demo_routes.json")
    parser.add_argument("--require-tk", action="store_true")
    parser.add_argument("--require-numpy", action="store_true")
    args = parser.parse_args(argv)
    checks: list[dict[str, object]] = []
    fatal = False

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "name": "python>=3.11",
            "ok": python_ok,
            "actual": sys.version.split()[0],
            "executable": sys.executable,
        }
    )
    fatal |= not python_ok

    try:
        tkinter = importlib.import_module("tkinter")
        checks.append(
            {"name": "tkinter", "ok": True, "actual": tkinter.TkVersion, "required": args.require_tk}
        )
    except ImportError as exc:
        checks.append(
            {"name": "tkinter", "ok": False, "error": str(exc), "required": args.require_tk}
        )
        fatal |= args.require_tk

    try:
        numpy = importlib.import_module("numpy")
        checks.append(
            {"name": "numpy", "ok": True, "actual": numpy.__version__, "required": args.require_numpy}
        )
    except ImportError as exc:
        checks.append(
            {"name": "numpy", "ok": False, "error": str(exc), "required": args.require_numpy}
        )
        fatal |= args.require_numpy

    try:
        config = load_config(args.config)
        checks.append(
            {
                "name": "config",
                "ok": True,
                "route_version": config.route_version,
                "routes": len(config.routes),
            }
        )
        for route in config.routes:
            free = _port_free(route.listen.host, route.listen.port)
            checks.append(
                {
                    "name": f"listen-port:{route.id}",
                    "ok": free,
                    "endpoint": f"{route.listen.host}:{route.listen.port}",
                }
            )
            fatal |= not free
        for peer in config.peers:
            if peer.behavior == "unreachable":
                continue
            free = _port_free(peer.endpoint.host, peer.endpoint.port)
            checks.append(
                {
                    "name": f"peer-port:{peer.id}",
                    "ok": free,
                    "endpoint": f"{peer.endpoint.host}:{peer.endpoint.port}",
                }
            )
            fatal |= not free
    except TunnelError as exc:
        checks.append({"name": "config", "ok": False, "error": exc.to_dict()})
        fatal = True

    result = {"code": "OK" if not fatal else "E_CONFIG", "checks": checks}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
