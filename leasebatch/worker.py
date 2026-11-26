"""Real loopback TCP worker subprocesses speaking JSON-lines.

Each worker runs as an independent OS child process bound to a fixed
127.0.0.1 port.  A request/response is one newline-terminated JSON object per
direction.  Workers compute the integer fixed-point forward pass with NumPy
and answer in one of the historical receipt formats (r1/r2/r3), allowing the
run to mix parameter versions and formats.
"""

import multiprocessing as mp
import socket
import time

from .fixedpoint import forward_batch, parameter_digest
from .receipt import R1_FIELDS, R2_FIELDS, R3_FIELDS

HOST = "127.0.0.1"


def _read_line(conn_file):
    raw = conn_file.readline()
    if not raw:
        return None
    return raw.strip()


def _serve_worker(worker_id, port, model_dim, hidden_dim, ready_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, port))
    listener.listen(16)
    listener.settimeout(0.5)
    with open(ready_path, "w", encoding="utf-8") as handle:
        handle.write("ready")
    while True:
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        with conn, conn.makefile("rwb") as stream:
            line = _read_line(stream)
            if line is None:
                continue
            import json

            request = json.loads(line)
            if request.get("op") == "shutdown":
                stream.write(b"{\"ok\":true}\n")
                stream.flush()
                listener.close()
                return
            response = _handle_batch(request, worker_id, model_dim,
                                     hidden_dim)
            stream.write(json.dumps(response).encode("utf-8") + b"\n")
            stream.flush()


def _handle_batch(request, worker_id, model_dim, hidden_dim):
    started = time.monotonic_ns()
    batch_id = request["batch_id"]
    request_ids = request["request_ids"]
    param_version = int(request["param_version"])
    seed = int(request["seed"])
    fmt = request.get("format", "r3")

    result = forward_batch(
        request_ids, worker_id, model_dim, hidden_dim, param_version, seed
    )
    elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    param_digest = parameter_digest(
        worker_id, model_dim, hidden_dim, param_version, seed
    )
    superseded = list(request.get("superseded_refs", []))

    full = {
        "batch_id": batch_id,
        "request_ids": result["request_ids"],
        "output_digest": result["output_digest"],
        "param_version": param_version,
        "elapsed_ms": int(elapsed_ms),
        "worker_id": worker_id,
        "ranks": list(range(len(result["request_ids"]))),
        "superseded_refs": superseded,
        "param_digest": param_digest,
    }
    field_set = {"r1": R1_FIELDS, "r2": R2_FIELDS, "r3": R3_FIELDS}[fmt]
    return {key: full[key] for key in field_set}


class WorkerPool:
    """Spawns, addresses and tears down the worker subprocesses."""

    def __init__(self, spec, tmp_dir):
        self.spec = spec
        self.tmp_dir = tmp_dir
        self.processes = {}
        self._ctx = mp.get_context("spawn")

    def start(self):
        for worker in self.spec.workers:
            ready_path = f"{self.tmp_dir}/ready-{worker.worker_id}"
            process = self._ctx.Process(
                target=_serve_worker,
                args=(
                    worker.worker_id,
                    worker.port,
                    worker.model_dim,
                    worker.hidden_dim,
                    ready_path,
                ),
                name=f"worker-{worker.worker_id}",
            )
            process.daemon = True
            process.start()
            self.processes[worker.worker_id] = process
        self._await_ready()

    def _await_ready(self):
        deadline = time.monotonic() + 10.0
        for worker in self.spec.workers:
            ready_path = f"{self.tmp_dir}/ready-{worker.worker_id}"
            while time.monotonic() < deadline:
                try:
                    with open(ready_path, "r", encoding="utf-8"):
                        break
                except FileNotFoundError:
                    time.sleep(0.01)
            else:
                raise RuntimeError(
                    f"worker {worker.worker_id} did not become ready"
                )

    def _port_of(self, worker_id):
        return self.spec.worker_by_id[worker_id].port

    def send_batch(self, worker_id, payload, timeout=10.0):
        import json

        with socket.create_connection(
            (HOST, self._port_of(worker_id)), timeout=timeout
        ) as conn:
            with conn.makefile("rwb") as stream:
                stream.write(json.dumps(payload).encode("utf-8") + b"\n")
                stream.flush()
                raw = stream.readline()
        return json.loads(raw.decode("utf-8"))

    def shutdown(self):
        for worker_id, process in list(self.processes.items()):
            try:
                self.send_batch(worker_id, {"op": "shutdown"}, timeout=2.0)
            except (OSError, EOFError):
                pass
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self.processes.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
