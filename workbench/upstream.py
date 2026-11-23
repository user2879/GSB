"""Deterministic upstream service, launched as a child process.

Protocol (one request per connection, LF-terminated lines):
gateway -> upstream:  "REQ <seq> <service_id> <client_key>"
upstream -> gateway:  "RCPT <seq> <service_id> <cksum16>"
where cksum16 = sha256(request_line_without_LF).hexdigest()[:16].

Fault-injection client keys (deterministic, for diagnostics):
"__timeout__"    : accept and stay silent for 30s (real socket timeout)
"__badreceipt__" : reply with a field-missing receipt "RCPT"
"__wrongsvc__"   : reply with a mismatched service id in the receipt
"""

import argparse
import hashlib
import socket
import threading
import time

SILENT_SECONDS = 30.0
MAX_LINE = 4096


def _handle(conn, service_id):
    try:
        fh = conn.makefile("rb")
        line = fh.readline(MAX_LINE)
        if not line:
            return
        try:
            text = line.decode("utf-8").rstrip("\n")
        except UnicodeDecodeError:
            conn.sendall(b"RCPT\n")
            return
        parts = text.split(" ")
        if len(parts) != 4 or parts[0] != "REQ":
            conn.sendall(b"RCPT\n")
            return
        _, seq, svc, client_key = parts
        if client_key == "__timeout__":
            time.sleep(SILENT_SECONDS)
            return
        if client_key == "__badreceipt__":
            conn.sendall(b"RCPT\n")
            return
        cksum = hashlib.sha256(line.rstrip(b"\n")).hexdigest()[:16]
        if client_key == "__wrongsvc__":
            conn.sendall(f"RCPT {seq} svc-WRONG {cksum}\n".encode("utf-8"))
            return
        reply = f"RCPT {seq} {svc} {cksum}\n"
        conn.sendall(reply.encode("utf-8"))
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def serve(port, service_id):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    print(f"READY {service_id} {port}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_handle, args=(conn, service_id),
                             daemon=True).start()
    finally:
        srv.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="workbench.upstream")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args(argv)
    serve(args.port, args.service)


if __name__ == "__main__":
    main()
