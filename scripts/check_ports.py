from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshgov.upstream import load_upstream_config


def probe(port: int) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        return True, "free"
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Preflight deterministic upstream ports")
    parser.add_argument("--config", default="samples/upstreams.json")
    args = parser.parse_args(argv)
    try:
        specs = load_upstream_config(args.config)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "code": "E_UPSTREAM",
                    "message": str(exc),
                    "ports": [],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3
    rows = []
    failed = False
    for spec in specs:
        free, detail = probe(spec.port)
        failed = failed or not free
        rows.append(
            {
                "instance": spec.name,
                "host": "127.0.0.1",
                "port": spec.port,
                "status": "free" if free else "occupied",
                "detail": detail,
            }
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "fail" if failed else "pass",
                "code": "E_UPSTREAM" if failed else None,
                "ports": rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

