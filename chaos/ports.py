"""Runtime loopback port allocation.

Declaration entries may use port 0; the CLI allocates free high loopback
ports and writes a resolved declaration next to the ledger. The resolved
file carries a digest of its source declaration (ports normalized to 0) so
it is reused across crash/resume sessions but regenerated whenever the
source declaration itself changes.
"""

import json
import os
import socket

from .timeline import digest as _digest

SOURCE_KEY = "_source_digest"


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def normalized_digest(spec_dict):
    doc = json.loads(json.dumps(spec_dict))
    doc.pop(SOURCE_KEY, None)
    for svc in doc.get("services", []):
        svc["port"] = 0
        svc["control_port"] = 0
    return _digest(doc)


def resolve_ports(spec_dict, existing_path=None):
    """Return a declaration with port 0 replaced by free ports, reusing a
    previously resolved file when it matches the source declaration."""
    want = normalized_digest(spec_dict)
    if existing_path and os.path.exists(existing_path):
        try:
            with open(existing_path, encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get(SOURCE_KEY) == want:
                return doc
        except (OSError, json.JSONDecodeError):
            pass
    doc = json.loads(json.dumps(spec_dict))
    used = set()
    for svc in doc["services"]:
        for key in ("port", "control_port"):
            if svc.get(key, 0) == 0:
                port = free_port()
                while port in used:
                    port = free_port()
                svc[key] = port
                used.add(port)
    doc[SOURCE_KEY] = want
    if existing_path:
        with open(existing_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
    return doc
