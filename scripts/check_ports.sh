#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
python - <<'PY'
import json
import socket
import sys

ports = [48101, 48102, 48103, 48191]
occupied = []
for port in ports:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            occupied.append(port)
if occupied:
    print(json.dumps({
        "ok": False,
        "code": "E_PORT",
        "field": "port",
        "message": "declared loopback port(s) are occupied",
        "details": {"occupied_ports": occupied},
    }))
    sys.exit(2)
print(json.dumps({"ok": True, "code": "E_OK", "free_ports": ports}))
PY
