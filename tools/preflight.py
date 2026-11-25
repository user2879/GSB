"""Confirm every declared loopback port can be bound; suggest a rewrite if busy."""

import argparse
import json
import socket

from sharding_diag.model import load_declaration
from sharding_diag.errors import DiagError


def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("declaration")
    args = parser.parse_args()
    try:
        decl = load_declaration(args.declaration)
    except DiagError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        return 2
    busy = []
    for node in decl.nodes:
        ok = check_port(node.host, node.port)
        print(f"{node.node_id} {node.host}:{node.port} -> {'BINDABLE' if ok else 'BUSY'}")
        if not ok:
            busy.append(node.node_id)
    if busy:
        print("Ports busy; edit config/declaration.json to select free high ports for:", busy)
        return 1
    print("all loopback ports bindable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
