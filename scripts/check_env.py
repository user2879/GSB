"""Check the local Windows/Python capabilities required by the workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sqlite3
import ssl
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-ui", action="store_true", help="fail when tkinter is unavailable")
    args = parser.parse_args(argv)
    result = {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "openssl_version": ssl.OPENSSL_VERSION,
        "ssl_ok": bool(ssl.OPENSSL_VERSION),
        "sqlite3_version": sqlite3.sqlite_version,
        "hashlib_sha256_ok": hashlib.sha256(b"tlstrust").hexdigest()
        == "22cb8cb0bfba15585d6f2f0ad5454b759d4cd0687b995c8888bb0a88f2761d6f",
    }
    try:
        import tkinter  # noqa: F401

        result["tkinter_available"] = True
        result["tkinter_version"] = tkinter.TkVersion
    except Exception as exc:
        result["tkinter_available"] = False
        result["tkinter_error"] = str(exc)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        result["loopback_ephemeral_port"] = port
        result["loopback_port_ok"] = 1024 <= port <= 65535 or port == 0
    finally:
        sock.close()
    result["ok"] = bool(
        result["python_ok"]
        and result["ssl_ok"]
        and result["sqlite3_version"]
        and result["hashlib_sha256_ok"]
        and result["loopback_port_ok"]
        and (result["tkinter_available"] or not args.strict_ui)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
