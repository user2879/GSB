"""Inference backend subprocess and its gateway-side process controller.

The worker is a plain Python process (``python hotgray/backend.py``) that owns
a loopback TCP server, keeps all parameter versions in memory and performs
NumPy forward passes. Hot swapping replaces the active parameters on the same
connection while retaining old versions read-only for merge traceability.
"""

import argparse
import json
import os
import socket
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hotgray import determinism as det
    from hotgray.errors import HandshakeError, PortError, ReplyError
    from hotgray.protocol import recv_message, send_message, validate_reply
else:
    from . import determinism as det
    from .errors import HandshakeError, PortError, ReplyError
    from .protocol import recv_message, send_message, validate_reply

CONNECT_TIMEOUT = 5.0
IO_TIMEOUT = 8.0


def _build_version(backend_id, model_version, quant_version, model_dim,
                   hidden_dim, base_seed, seq):
    seed = det.next_hotswap_seed(base_seed, seq) if seq else base_seed
    params = det.build_parameters(seed, model_dim, hidden_dim)
    digest = det.parameter_digest(
        backend_id, model_version, quant_version, model_dim,
        hidden_dim, seed, params,
    )
    return {"seq": seq, "seed": seed, "params": params, "digest": digest}


def worker_main(argv=None):  # pragma: no cover - exercised via subprocess
    parser = argparse.ArgumentParser(description="hotgray inference worker")
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quant-version", type=int, default=det.QUANT_VERSION)
    args = parser.parse_args(argv)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((args.host, args.port))
    except OSError as exc:
        print(json.dumps({"fatal": "E_PORT", "message": str(exc)}),
              flush=True)
        return 2
    server.listen(1)
    server.settimeout(30.0)
    print(json.dumps({"ready": True, "backend_id": args.backend_id,
                      "port": args.port}), flush=True)

    try:
        conn, _ = server.accept()
    except socket.timeout:
        return 0
    conn.settimeout(IO_TIMEOUT)

    versions = {
        0: _build_version(args.backend_id, args.model_version,
                          args.quant_version, args.model_dim,
                          args.hidden_dim, args.seed, 0)
    }
    active_seq = 0

    while True:
        try:
            message = recv_message(conn, IO_TIMEOUT)
        except HandshakeError:
            return 0
        mtype = message.get("type")
        if mtype == "shutdown":
            send_message(conn, {"type": "shutdown_ack",
                                "backend_id": args.backend_id})
            return 0
        if mtype == "ping":
            send_message(conn, {"type": "hello", "backend_id": args.backend_id,
                                "seq": active_seq,
                                "param_digest": versions[active_seq]["digest"],
                                "sample_ids": [], "output_digests": [],
                                "batch_digest": None, "elapsed_ms": 0})
            continue
        if mtype == "hotswap":
            seq = int(message["new_seq"])
            if seq not in versions:
                versions[seq] = _build_version(
                    args.backend_id, message.get("model_version",
                                                 args.model_version),
                    args.quant_version, args.model_dim, args.hidden_dim,
                    args.seed, seq,
                )
            active_seq = seq
            send_message(conn, {"type": "hotswap_ack",
                                "backend_id": args.backend_id, "seq": seq,
                                "param_digest": versions[seq]["digest"],
                                "sample_ids": [], "output_digests": [],
                                "batch_digest": None, "elapsed_ms": 0})
            continue
        if mtype == "drop_version":
            versions.pop(int(message["seq"]), None)
            send_message(conn, {"type": "drop_ack",
                                "backend_id": args.backend_id,
                                "seq": active_seq,
                                "param_digest": versions[active_seq]["digest"],
                                "sample_ids": [], "output_digests": [],
                                "batch_digest": None, "elapsed_ms": 0})
            continue
        if mtype == "batch":
            seq = int(message["seq"])
            if seq not in versions:
                send_message(conn, {"type": "error",
                                    "backend_id": args.backend_id, "seq": seq,
                                    "param_digest": None, "sample_ids": [],
                                    "output_digests": [], "batch_digest": None,
                                    "elapsed_ms": 0,
                                    "error": "unknown_param_version"})
                continue
            version = versions[seq]
            import numpy as np

            inputs = np.asarray(message["inputs"], dtype=np.float64)
            started = time.perf_counter()
            outputs = det.forward(version["params"], inputs)
            elapsed_ms = max(1, int(round((time.perf_counter() - started) * 1000)))
            per_sample = []
            for idx, sample_id in enumerate(message["sample_ids"]):
                per_sample.append({
                    "sample_id": sample_id,
                    "output_digest": det.output_digest(
                        message["rule_id"], [sample_id],
                        version["digest"], outputs[idx:idx + 1]),
                })
            batch_digest = det.output_digest(
                message["rule_id"], message["sample_ids"],
                version["digest"], outputs)
            send_message(conn, {
                "type": "batch_ack",
                "backend_id": args.backend_id,
                "seq": seq,
                "param_digest": version["digest"],
                "sample_ids": list(message["sample_ids"]),
                "output_digests": per_sample,
                "batch_digest": batch_digest,
                "elapsed_ms": elapsed_ms,
            })
            continue
        send_message(conn, {"type": "error", "backend_id": args.backend_id,
                            "seq": active_seq, "error": f"unknown type {mtype}"})


class BackendProcess:
    """Gateway-side controller for one worker subprocess."""

    def __init__(self, backend_decl, python_executable=None,
                 connect_timeout=CONNECT_TIMEOUT):
        self.decl = backend_decl
        self.backend_id = backend_decl["backend_id"]
        self.host = backend_decl["host"]
        self.port = backend_decl["port"]
        self.model_version = backend_decl["model_version"]
        self.python_executable = python_executable or sys.executable
        self.connect_timeout = connect_timeout
        self.process = None
        self.sock = None
        self.active_seq = 0
        self.version_digests = {}

    def start(self):
        worker_path = os.path.abspath(__file__)
        cmd = [
            self.python_executable, worker_path,
            "--backend-id", self.backend_id,
            "--model-version", self.model_version,
            "--host", self.host, "--port", str(self.port),
            "--model-dim", str(self.decl["model_dim"]),
            "--hidden-dim", str(self.decl["hidden_dim"]),
            "--seed", str(self.decl["seed"]),
            "--quant-version", str(self.decl["quant_version"]),
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(__import__("subprocess"),
                                    "CREATE_NO_WINDOW", 0)
        import subprocess

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        ready_line = self._wait_ready()
        if isinstance(ready_line, dict) and ready_line.get("fatal") == "E_PORT":
            raise PortError(
                f"port {self.host}:{self.port} could not be bound: "
                f"{ready_line.get('message', '')}",
                field=f"backends.{self.backend_id}.port",
            )
        self._connect()
        self.handshake()

    def _wait_ready(self):
        deadline = time.time() + self.connect_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                err = self.process.stderr.read().decode("utf-8", "replace")
                raise PortError(
                    f"worker {self.backend_id} exited early: {err.strip()}",
                    field=f"backends.{self.backend_id}",
                )
            line = self.process.stdout.readline()
            if not line:
                time.sleep(0.02)
                continue
            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
        raise PortError(
            f"worker {self.backend_id} did not become ready in time",
            field=f"backends.{self.backend_id}",
        )

    def _connect(self):
        deadline = time.time() + self.connect_timeout
        last_error = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            try:
                sock.connect((self.host, self.port))
                self.sock = sock
                return
            except OSError as exc:
                last_error = exc
                sock.close()
                time.sleep(0.05)
        raise PortError(
            f"cannot connect to {self.host}:{self.port}: {last_error}",
            field=f"backends.{self.backend_id}.port",
        )

    def handshake(self):
        send_message(self.sock, {"type": "ping"})
        try:
            reply = recv_message(self.sock, self.connect_timeout)
        except HandshakeError as exc:
            raise HandshakeError(
                f"handshake timeout for {self.backend_id}: {exc}",
                field=f"backends.{self.backend_id}",
            ) from exc
        validate_reply(reply, expected_type="hello", expected_seq=self.active_seq)
        self.version_digests[reply["seq"]] = reply["param_digest"]
        return reply

    def hotswap(self, new_seq):
        if new_seq in self.version_digests:
            return self.version_digests[new_seq]
        send_message(self.sock, {"type": "hotswap", "new_seq": new_seq,
                                 "model_version": self.model_version})
        try:
            reply = recv_message(self.sock, self.connect_timeout)
        except HandshakeError as exc:
            raise HandshakeError(
                f"hot-swap switch timeout on {self.backend_id}: {exc}",
                field=f"backends.{self.backend_id}.seq",
            ) from exc
        validate_reply(reply, expected_type="hotswap_ack",
                       expected_seq=new_seq)
        self.active_seq = new_seq
        self.version_digests[new_seq] = reply["param_digest"]
        return reply["param_digest"]

    def drop_version(self, seq):
        send_message(self.sock, {"type": "drop_version", "seq": seq})
        reply = recv_message(self.sock, self.connect_timeout)
        validate_reply(reply, expected_type="drop_ack")
        self.version_digests.pop(seq, None)

    def infer_batch(self, rule_id, seq, sample_ids, inputs):
        message = {"type": "batch", "rule_id": rule_id, "seq": seq,
                   "sample_ids": list(sample_ids),
                   "inputs": [[float(v) for v in row] for row in inputs]}
        send_message(self.sock, message)
        try:
            reply = recv_message(self.sock, IO_TIMEOUT)
        except HandshakeError as exc:
            raise ReplyError(f"batch reply timeout on {self.backend_id}: {exc}",
                             field="output_digest") from exc
        validate_reply(reply, expected_type="batch_ack", expected_seq=seq)
        if reply["param_digest"] != self.version_digests.get(seq):
            raise ReplyError(
                f"backend {self.backend_id} returned unexpected param digest",
                field="param_digest",
            )
        if set(reply["sample_ids"]) != set(sample_ids):
            raise ReplyError("backend reply sample set mismatch",
                             field="sample_ids")
        replied = {item["sample_id"]: item["output_digest"]
                   for item in reply["output_digests"]}
        if set(replied) != set(sample_ids):
            raise ReplyError("per-sample digest coverage mismatch",
                             field="output_digests")
        reply["digest_by_sample"] = replied
        return reply

    def stop(self):
        try:
            if self.sock is not None:
                send_message(self.sock, {"type": "shutdown"})
                try:
                    recv_message(self.sock, 2.0)
                except HandshakeError:
                    pass
                self.sock.close()
        finally:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except Exception:
                    self.process.kill()
            self.sock = None


if __name__ == "__main__":
    sys.exit(worker_main())
