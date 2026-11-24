"""回环子进程确定性工作器与真实 TCP 客户端。

请求信封: b"RLQ1" | 4 字节大端长度 | 规范化 JSON 载荷。
响应信封: ``receipt_ledger.codec`` 的 r1/r2/r3 回执。
工作器按 (param_version) 惰性构建并缓存只读参数，可并发处理连接。
"""

import argparse
import json
import os
import socketserver
import sys
import time

import numpy as np

from . import codec
from .deterministic import build_parameters, forward, output_digest

REQUEST_MAGIC = b"RLQ1"


def _read_exact(handle, size):
    chunks = []
    remaining = size
    while remaining:
        block = handle.recv(remaining)
        if not block:
            raise ConnectionError("连接在读取完成前关闭")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _read_request(handle):
    header = _read_exact(handle, 8)
    if header[:4] != REQUEST_MAGIC:
        raise ValueError("请求魔数不匹配")
    length = int.from_bytes(header[4:8], "big")
    return json.loads(_read_exact(handle, length).decode("utf-8"))


class WorkerState:
    """单工作器进程内的只读参数缓存与故障注入计数。"""

    def __init__(self, worker):
        self.worker = worker
        self._cache = {}
        faults = os.environ.get("RL_FAULT_BATCHES", "")
        self.fault_plan = {}
        for token in faults.split(","):
            if ":" in token:
                bid, count = token.split(":", 1)
                self.fault_plan[bid] = int(count)

    def parameters(self, param_version):
        if param_version not in self._cache:
            self._cache[param_version] = build_parameters(
                self.worker, param_version
            )
        return self._cache[param_version]


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, state):
        self.state = state
        super().__init__(address, handler)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            req = _read_request(self.request)
        except (ConnectionError, ValueError, json.JSONDecodeError):
            return
        state = self.server.state
        bid = req["batch_id"]
        if state.fault_plan.get(bid, 0) > 0:
            # 测试用故障注入：直接关闭连接模拟超时/断连，不产生任何人为延迟。
            state.fault_plan[bid] -= 1
            return
        weight, bias, param_digest = state.parameters(req["param_version"])
        per_request, batch_material = [], []
        started = time.monotonic_ns()
        for item in req["requests"]:
            hidden = forward(weight, bias, item["vector"])
            digest = output_digest(hidden)
            per_request.append((item["seq"], digest))
            batch_material.append([item["seq"], digest])
        elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        batch_digest = output_digest(
            np.frombuffer(
                b"".join(d.encode("ascii") for _, d in sorted(per_request)),
                dtype=np.uint8,
            )
        )
        fields = {
            "batch_id": bid,
            "worker_id": state.worker.worker_id,
            "destination": req["destination"],
            "output_digest": batch_digest,
            "param_digest": param_digest,
            "quant": state.worker.quant,
            "elapsed_ms": int(elapsed_ms),
            "request_seqs": sorted(item["seq"] for item in req["requests"]),
            "param_version": req["param_version"],
        }
        self.request.sendall(codec.encode_receipt(req["format"], fields))


def run_worker(workers_path: str, worker_id: str, port: int):
    from .spec import _validate_workers

    # 父进程被强制终止（含 os._exit / kill -9）后，内核向本工作器发送 SIGTERM，
    # 避免孤儿工作器继续占用回环端口；非 Linux 平台忽略该可选设置。
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, 15, 0, 0, 0)
    except (OSError, AttributeError):
        pass

    with open(workers_path, "r", encoding="utf-8") as handle:
        workers = _validate_workers(json.load(handle))
    worker = next(w for w in workers if w.worker_id == worker_id)
    state = WorkerState(worker)
    with ThreadingServer(("127.0.0.1", port), Handler, state) as server:
        sys.stdout.write(f"READY {worker_id} 127.0.0.1:{port}\n")
        sys.stdout.flush()
        server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="回执账本确定性工作器")
    parser.add_argument("--workers", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args(argv)
    run_worker(args.workers, args.worker_id, args.port)


if __name__ == "__main__":
    main()


def _frame_request(payload: dict) -> bytes:
    body = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return REQUEST_MAGIC + len(body).to_bytes(4, "big") + body


def call_worker(worker, task, receipt_format: str, merge_seq: int,
                ):
    """通过真实回环套接字调用工作器，按声明策略超时重试。

    请求幂等（同批次同参数重算结果一致），重试不改变批次归属语义。
    返回原始回执信封字节。
    """
    import socket

    payload = {
        "batch_id": task.batch_id,
        "param_version": task.param_version,
        "format": receipt_format,
        "merge_seq": merge_seq,
        "destination": task.worker_id,
        "requests": [
            {
                "seq": req.seq,
                "client_key": req.client_key,
                "vector": list(req.vector),
                "destination": task.destinations[req.seq],
            }
            for req in task.requests
        ],
    }
    frame = _frame_request(payload)
    attempts = 0
    last_error = None
    while attempts <= task.max_retries:
        sock = socket.create_connection((worker.host, worker.port),
                                        timeout=task.socket_timeout_ms / 1000.0)
        try:
            sock.settimeout(task.socket_timeout_ms / 1000.0)
            sock.sendall(frame)
            header = _read_exact(sock, 9)
            if header[:4] != codec.MAGIC:
                raise ConnectionError("回执魔数不匹配")
            length = int.from_bytes(header[5:9], "big")
            body = _read_exact(sock, length)
            return header + body, attempts
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            attempts += 1
            if task.retry_policy == "bounded" and attempts > task.max_retries:
                break
        finally:
            sock.close()
    raise ConnectionError(
        f"批次 {task.batch_id} 工作器 {worker.worker_id} "
        f"重试 {task.max_retries} 次后仍失败: {last_error}"
    )
