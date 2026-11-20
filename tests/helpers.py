from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_V1 = ROOT / "samples" / "demo_spec.json"
SAMPLE_V2 = ROOT / "samples" / "demo_spec_v2.json"
SAMPLE_V3 = ROOT / "samples" / "demo_spec_v3.json"


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_raw(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def free_port_base(count: int = 4, start: int = 44000) -> int:
    for base in range(start, 65000 - count):
        sockets: list[socket.socket] = []
        try:
            for port in range(base, base + count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no free port range")
