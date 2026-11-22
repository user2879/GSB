"""本地模拟组件节点：独立子进程 + 127.0.0.1 JSON 行协议。

启动方式（平台通过 ``sys.executable -m configreg.node`` 拉起，Windows /
Linux 行为一致）::

    python -m configreg.node --node-id node-a --host 127.0.0.1 --port 19001 \
        --seed 20261122 --fail-rate 0.0

协议（每行一个 JSON 对象，``\\n`` 分帧）：

* 请求 ``{"type":"slice", "request_id":..., "node_id":..., "version":...,
  "item":..., "value":..., "targets":[...], "sample_seed":..., "work_repeat":...}``
* 响应 ``{"type":"result", "request_id":..., "ok":true, "node_id":...,
  "version":..., "item":..., "summary":{...}, "elapsed_ms":...}``
* ``{"type":"shutdown"}`` 让节点优雅退出。

``--fail-rate`` 注入真实网络故障：按比例丢弃请求（不回包），用于验证
客户端真实套接字超时与重试隔离，不改变正常采集语义。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import socketserver
import time
import threading
from typing import Any

import numpy as np


def _stable_int_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_summary(
    node_id: str,
    item: str,
    value: float,
    sample_seed: int,
    work_repeat: int,
    extra_sleep_ms: int = 0,
) -> dict[str, float]:
    """对固定种子样本执行分组聚合。

    同一会话中两版清单使用同一个 ``sample_seed``，且节点/配置项参与
    种子派生，因此两版在同一节点、同一配置项上看到的样本逐元素相同，
    差异只来自下发的配置值 ``value``。聚合结果对样本顺序不敏感。
    """
    seed = _stable_int_seed(sample_seed, node_id, item)
    rng = np.random.default_rng(seed)
    sample_count = 256
    samples = rng.normal(loc=float(value), scale=1.0, size=sample_count)
    groups = rng.integers(low=0, high=4, size=sample_count)
    weights = rng.random(sample_count) + 0.5

    started = time.perf_counter_ns()
    accumulator = np.zeros(4, dtype=np.float64)
    weight_sum = np.zeros(4, dtype=np.float64)
    for _ in range(max(1, work_repeat)):
        accumulator.fill(0.0)
        weight_sum.fill(0.0)
        np.add.at(accumulator, groups, samples * weights)
        np.add.at(weight_sum, groups, weights)
    if extra_sleep_ms > 0:
        # 确定性的额外执行耗时：只影响整数毫秒耗时，不改变聚合摘要。
        time.sleep(extra_sleep_ms / 1000.0)
    group_mean = accumulator / weight_sum
    elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
    return {
        "sample_count": int(sample_count),
        "group_mean": [float(x) for x in group_mean],
        "grand_mean": float(np.average(samples, weights=weights)),
        "stddev": float(np.std(samples, dtype=np.float64)),
        "checksum": float(np.sum(group_mean)),
        "work_repeat": int(work_repeat),
        "extra_sleep_ms": int(extra_sleep_ms),
        "elapsed_ms_observed": int(elapsed_ms),
    }


class ThreadedLineServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_cls, node_meta: dict[str, Any]):
        self.node_meta = node_meta
        self.fault_rng = random.Random(
            _stable_int_seed(node_meta["seed"], node_meta["node_id"], "fault")
        )
        self.fault_lock = threading.Lock()
        super().__init__(server_address, handler_cls)


class LineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: ThreadedLineServer = self.server  # type: ignore[assignment]
        for raw_line in self.rfile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if request.get("type") == "shutdown":
                self.wfile.write(b'{"type":"bye","ok":true}\n')
                self.wfile.flush()
                server.shutdown_signal = True  # type: ignore[attr-defined]
                return
            if request.get("type") != "slice":
                continue
            fail_rate = float(server.node_meta["fail_rate"])
            with server.fault_lock:
                should_drop = fail_rate > 0.0 and server.fault_rng.random() < fail_rate
            if should_drop:
                # 真实故障注入：丢弃请求不回包，由客户端套接字超时触发重试。
                continue
            summary = deterministic_summary(
                node_id=server.node_meta["node_id"],
                item=request["item"],
                value=float(request["value"]),
                sample_seed=int(request["sample_seed"]),
                work_repeat=int(request.get("work_repeat", 1)),
                extra_sleep_ms=int(request.get("extra_sleep_ms", 0)),
            )
            response = {
                "type": "result",
                "request_id": request.get("request_id"),
                "ok": True,
                "node_id": server.node_meta["node_id"],
                "version": request.get("version"),
                "item": request.get("item"),
                "summary": summary,
                "elapsed_ms": int(summary["elapsed_ms_observed"]),
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            self.wfile.flush()


def serve_forever(node_id: str, host: str, port: int, seed: int, fail_rate: float) -> None:
    server = ThreadedLineServer(
        (host, port),
        LineHandler,
        {"node_id": node_id, "seed": seed, "fail_rate": fail_rate},
    )
    server.shutdown_signal = False  # type: ignore[attr-defined]

    def _watchdog() -> None:
        while not getattr(server, "shutdown_signal", False):
            time.sleep(0.05)
        time.sleep(0.05)
        server.shutdown()

    threading.Thread(target=_watchdog, daemon=True).start()
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="配置分发模拟组件节点")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    args = parser.parse_args(argv)
    serve_forever(args.node_id, args.host, args.port, args.seed, args.fail_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
