"""Node worker subprocess.

Each node is a standalone OS process hosting a threaded loopback TCP server.
The wire protocol is one JSON object per line::

    request  {"type": "compute", "request_id", "node_id", "shard_id",
              "rows", "node_seed", "fixed_point_scale", "capacity_version"}
    response {"type": "compute_result", ..., "summary", "elapsed_ms"}

The process also answers ``ping``/``pong`` and exits on ``shutdown``.
Compute time is measured with the OS monotonic clock.
"""

import argparse
import json
import socket
import sys
import threading
import time

from .sampling import compute_summary, generate_sample


class NodeWorker:
    def __init__(self, node_id, host, port, concurrency):
        self.node_id = node_id
        self.host = host
        self.port = port
        self._slots = threading.Semaphore(concurrency)
        self._stop = threading.Event()
        self._server = None
        self._threads = []

    def serve(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(64)
        self._server.settimeout(0.25)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            thread.start()
            self._threads.append(thread)
        self._server.close()

    def _handle(self, conn):
        with conn:
            buffer = b""
            while not self._stop.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        request = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        response = {"type": "error", "reason": "bad_json"}
                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                        continue
                    response = self._dispatch(request)
                    conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    if request.get("type") == "shutdown":
                        self._stop.set()
                        return

    def _dispatch(self, request):
        kind = request.get("type")
        if kind == "ping":
            return {"type": "pong", "node_id": self.node_id}
        if kind == "shutdown":
            return {"type": "bye", "node_id": self.node_id}
        if kind != "compute":
            return {"type": "error", "node_id": self.node_id, "reason": f"unknown_type:{kind}"}

        if not self._slots.acquire(blocking=False):
            return {"type": "error", "node_id": self.node_id,
                    "shard_id": request.get("shard_id"), "reason": "node_busy"}
        start = time.monotonic_ns()
        try:
            sample = generate_sample(
                int(request["node_seed"]),
                request["shard_id"],
                int(request["rows"]),
            )
            summary = compute_summary(
                sample,
                self.node_id,
                request["shard_id"],
                int(request["fixed_point_scale"]),
            )
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return {
                "type": "compute_result",
                "request_id": request.get("request_id"),
                "node_id": self.node_id,
                "shard_id": request["shard_id"],
                "capacity_version": int(request["capacity_version"]),
                "summary": summary,
                "elapsed_ms": int(elapsed_ms),
            }
        except Exception as exc:  # surface, do not crash the node
            return {"type": "error", "node_id": self.node_id,
                    "shard_id": request.get("shard_id"), "reason": repr(exc)}
        finally:
            self._slots.release()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json")
    args = parser.parse_args(argv)
    params = json.loads(args.params_json)
    worker = NodeWorker(
        params["node_id"], params["host"], int(params["port"]), int(params["concurrency"])
    )
    worker.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
