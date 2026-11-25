"""工作进程：独立子进程在 127.0.0.1 上监听 JSON 行请求。"""

import json
import socket
import sys
import threading
import time

from .inference import compute_request_digests
from .protocol import recv_line, send_line


class WorkerServer:
    """子进程内的真实 TCP 服务。"""

    def __init__(self, spec_dict):
        self.spec = spec_dict
        self.worker = next(
            w for w in spec_dict["workers"] if w["worker_id"] == spec_dict["_this_worker"]
        )
        self.lease_versions = {}
        self.param_version = self.worker["param_version"]
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", int(self.worker["port"])))
        except OSError as exc:
            print(json.dumps({"type": "fatal", "code": "E_PORT",
                              "message": str(exc)}), flush=True)
            return 2
        server.listen(16)
        server.settimeout(0.25)
        print(json.dumps({"type": "ready", "host": "127.0.0.1",
                          "port": self.worker["port"],
                          "worker_id": self.worker["worker_id"]}), flush=True)

        while not self._stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        server.close()
        return 0

    def _handle(self, conn):
        try:
            send_line(conn, {"type": "hello", "host": "127.0.0.1",
                             "worker_id": self.worker["worker_id"]})
            while True:
                try:
                    message = recv_line(conn, self.spec["limits"]["read_timeout_ms"],
                                        location="worker")
                except ConnectionError:
                    return
                reply = self._dispatch(message)
                send_line(conn, reply)
                if message.get("type") == "shutdown":
                    return
        except Exception as exc:  # 子进程内兜底，避免线程静默死亡
            try:
                send_line(conn, {"type": "error", "code": "E_WORKER",
                                 "message": str(exc)})
            except OSError:
                pass
        finally:
            conn.close()

    def _dispatch(self, message):
        mtype = message.get("type")
        if mtype == "ping":
            return {"type": "pong", "worker_id": self.worker["worker_id"],
                    "param_version": self.param_version}
        if mtype == "param_update":
            with self._lock:
                self.param_version = int(message["param_version"])
            return {"type": "param_updated", "worker_id": self.worker["worker_id"],
                    "param_version": self.param_version}
        if mtype == "lease_grant":
            with self._lock:
                self.lease_versions[message["lease_id"]] = int(message["lease_version"])
            return {"type": "lease_ack", "lease_id": message["lease_id"],
                    "lease_version": int(message["lease_version"])}
        if mtype == "lease_renew":
            return self._renew(message)
        if mtype == "batch":
            return self._infer(message)
        if mtype == "shutdown":
            self._stop.set()
            return {"type": "bye"}
        return {"type": "error", "code": "E_WORKER", "message": f"未知消息类型 {mtype!r}"}

    def _renew(self, message):
        lease_id = message["lease_id"]
        with self._lock:
            current = self.lease_versions.get(lease_id)
            if current is None:
                return {"type": "error", "code": "E_LEASE",
                        "message": f"工作进程无租约 {lease_id!r} 记录",
                        "lease_id": lease_id}
            if current != int(message["expected_version"]):
                return {"type": "error", "code": "E_LEASE",
                        "message": (f"租约版本乐观校验失败: 期望 {current}, "
                                    f"收到 {message['expected_version']}"),
                        "lease_id": lease_id,
                        "expected_actual": current}
            new_version = int(message["new_version"])
            self.lease_versions[lease_id] = new_version
        return {"type": "lease_renewed", "lease_id": lease_id,
                "lease_version": new_version}

    def _infer(self, message):
        started = time.perf_counter_ns()
        batch = [(item["request_id"], int(item["value"])) for item in message["items"]]
        model = self.spec["model"]
        param_version = int(message.get("param_version", self.param_version))
        digests = compute_request_digests(
            batch, model["seed"], param_version,
            model["input_dim"], model["hidden_dim"])
        elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
        return {
            "type": "batch_ack",
            "batch_id": message["batch_id"],
            "worker_id": self.worker["worker_id"],
            "param_version": param_version,
            "elapsed_ms": int(elapsed_ms),
            "results": [
                {"request_id": request_id,
                 "output_digest": digests[request_id]["digest"],
                 "output": digests[request_id]["output"]}
                for request_id, _ in batch
            ],
        }


def worker_main(spec_dict):
    server = WorkerServer(spec_dict)
    return server.serve()


def _entrypoint():
    spec_dict = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    sys.exit(worker_main(spec_dict))


if __name__ == "__main__":
    _entrypoint()
