"""Deterministic subscriber processor hosted in an independent subprocess.

Run as ``python -m gsb.processor --subscriber-id ... --port ...``. The process
binds a real TCP listener on 127.0.0.1, performs a HELLO/READY handshake under a
real socket timeout, and serves line-delimited JSON control messages plus
BATCH work that runs NumPy integer fixed-point aggregation.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading

from .aggregation import CALIBER_VERSION, aggregate
from .clock import now_ms, sleep_ms
from .receipts import encode_x1, encode_x2, encode_x3

READY_MARKER = "READY"


class ProcessorServer:
    def __init__(self, subscriber_id, host, port, seed, budget_ms, log_path=None):
        self.subscriber_id = subscriber_id
        self.host = host
        self.port = port
        self.seed = seed
        self.budget_ms = budget_ms
        self.log_path = log_path
        self._leases: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._server = None
        self._stop = threading.Event()

    def log(self, message):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            self.log(f"BIND_ERROR {exc}")
            print(f"BIND_ERROR:{exc}", flush=True)
            raise
        server.listen(16)
        server.settimeout(0.25)
        self._server = server
        print(READY_MARKER, flush=True)
        while not self._stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def stop(self):
        self._stop.set()
        if self._server:
            self._server.close()

    def _handle(self, conn):
        conn.settimeout(5.0)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    reply = self._dispatch(line.decode("utf-8").strip())
                    if reply is not None:
                        conn.sendall((reply + "\n").encode("utf-8"))
                    if reply == "BYE":
                        return
        finally:
            conn.close()

    def _dispatch(self, line):
        if line == "HELLO":
            return "READY"
        if line == "PING":
            return "PONG"
        if line == "BYE":
            return "BYE"
        message = json.loads(line)
        kind = message["kind"]
        if kind == "LEASE_GRANT":
            return self._lease_grant(message)
        if kind == "LEASE_RENEW":
            return self._lease_renew(message)
        if kind == "BATCH":
            return self._batch(message)
        return json.dumps({"ok": False, "error": f"unknown kind {kind}"})

    def _lease_grant(self, message):
        lease_id = message["lease_id"]
        with self._lock:
            self._leases[lease_id] = {
                "subscriber_id": message["subscriber_id"],
                "version": int(message["version"]),
                "expires_at_ms": int(message["expires_at_ms"]),
            }
        return json.dumps(
            {
                "ok": True,
                "kind": "LEASE_GRANTED",
                "lease_id": lease_id,
                "version": message["version"],
            }
        )

    def _lease_renew(self, message):
        lease_id = message["lease_id"]
        expected = int(message["expected_version"])
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "E_LEASE",
                        "reason": "unknown lease",
                        "lease_id": lease_id,
                    }
                )
            if lease["version"] != expected:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "E_LEASE",
                        "lease_id": lease_id,
                        "reason": f"version conflict current={lease['version']} expected={expected}",
                        "current_version": lease["version"],
                    }
                )
            lease["version"] = expected + 1
            lease["expires_at_ms"] = int(message["new_expires_at_ms"])
        return json.dumps(
            {
                "ok": True,
                "kind": "LEASE_RENEWED",
                "lease_id": lease_id,
                "version": expected + 1,
            }
        )

    def _batch(self, message):
        lease_id = message["lease_id"]
        with self._lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            return json.dumps(
                {"ok": False, "code": "E_LEASE", "reason": "batch has no active lease"}
            )
        seqs = list(message["event_seqs"])
        payloads = list(message["payloads"])
        work_ms = int(message.get("work_ms", 0))
        if work_ms > 0:
            sleep_ms(work_ms)  # real elapsed processing time, deterministic amount
        result = aggregate(self.seed, seqs, payloads)
        # One completion instant shared by every format of this batch so the
        # three receipts normalize to a byte-identical core.
        completion = now_ms()
        requested = message.get("formats", ["x3"])
        batch_id = message["batch_id"]
        covered = list(message.get("covered_refs", []))
        receipts = {}
        for fmt in requested:
            if fmt == "x1":
                receipts[fmt] = encode_x1(batch_id, result["fixed_point_sum"])
            elif fmt == "x2":
                receipts[fmt] = encode_x2(
                    batch_id, self.subscriber_id, completion, result["fixed_point_sum"]
                )
            elif fmt == "x3":
                receipts[fmt] = encode_x3(
                    batch_id,
                    self.subscriber_id,
                    completion,
                    result["fixed_point_sum"],
                    seqs,
                    covered,
                    CALIBER_VERSION,
                )
        return json.dumps(
            {
                "ok": True,
                "kind": "BATCH_RECEIPT",
                "batch_id": batch_id,
                "completion_ms": completion,
                "aggregate": result,
                "receipts": receipts,
            }
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscriber-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-ms", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)
    server = ProcessorServer(
        args.subscriber_id,
        args.host,
        args.port,
        args.seed,
        args.budget_ms,
        args.log,
    )
    try:
        server.serve()
    except OSError:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
