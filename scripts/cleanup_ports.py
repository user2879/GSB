#!/usr/bin/env python3
"""Terminate recorded demo children and inspect configured loopback ports."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunnelvpn.config import load_config  # noqa: E402
from tunnelvpn.demo import DemoHarness  # noqa: E402
from tunnelvpn.errors import TunnelError  # noqa: E402


def _free(host: str, port: int) -> bool:
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
    parser = argparse.ArgumentParser(description="Clean demo loopback ports")
    parser.add_argument("--config", default="data/demo_routes.json")
    parser.add_argument("--state-dir", default="artifacts/demo-state")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        harness = DemoHarness(config, args.state_dir)
        terminated = harness.cleanup_recorded_processes()
        time.sleep(0.25)
        endpoints: list[dict[str, object]] = []
        for route in config.routes:
            endpoints.append(
                {
                    "kind": "listen",
                    "route_id": route.id,
                    "host": route.listen.host,
                    "port": route.listen.port,
                    "free": _free(route.listen.host, route.listen.port),
                }
            )
        for peer in config.peers:
            if peer.behavior == "unreachable":
                continue
            endpoints.append(
                {
                    "kind": "peer-target",
                    "peer_id": peer.id,
                    "host": peer.endpoint.host,
                    "port": peer.endpoint.port,
                    "free": _free(peer.endpoint.host, peer.endpoint.port),
                }
            )
        result = {
            "code": "OK" if all(item["free"] for item in endpoints) else "E_CONFIG",
            "terminated_pids": terminated,
            "ports": endpoints,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["code"] == "OK" else 1
    except TunnelError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
