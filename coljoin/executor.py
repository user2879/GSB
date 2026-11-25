"""执行器子进程：在 127.0.0.1 真实监听并处理分片请求。

启动方式：``python -m coljoin.executor --data-dir DIR --port P --id ID``。
执行器只做确定性本地求值，单调时钟毫秒由 time.monotonic_ns 换算。
"""

import argparse
import json
import socket
import time
from typing import Any, Dict

from .errors import ColjoinError
from .executor_logic import execute_plan_on_batch
from .loader import load_query
from .protocol import HELLO, recv_frame, send_frame
from .storage import load_generated_specs


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def handle_request(data_dir: str, request: Dict[str, Any]) -> Dict[str, Any]:
    groups = load_generated_specs(data_dir)
    query = load_query(request["query_path"], groups)
    plan = request["plan"]
    batch = request["batch"]
    clock_start = now_ms()
    aggregate = execute_plan_on_batch(data_dir, plan, query, batch)
    clock_end = now_ms()
    return {
        "type": "receipt",
        "shard_id": batch["shard_id"],
        "aggregate": aggregate["groups"],
        "plan_version": int(request["plan_version"]),
        "elapsed_ms": max(0, clock_end - clock_start),
        "executor_id": request["executor_id"],
        "cost_vector": request["cost_vector"],
        "clock_start_ms": clock_start,
        "clock_end_ms": clock_end,
    }


def serve(data_dir: str, port: int, executor_id: str, timeout: float = 10.0) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(8)
    server.settimeout(timeout)
    print(json.dumps({"event": "listening", "executor_id": executor_id, "port": port}), flush=True)
    try:
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            conn.settimeout(timeout)
            with conn:
                try:
                    send_frame(conn, HELLO)
                    handshake = recv_frame(conn)
                except ColjoinError:
                    # 端口探测或失败握手会提前断开，不能使执行器进程退出。
                    continue
                if handshake.get("type") != "ready":
                    continue
                while True:
                    request = recv_frame(conn)
                    if request.get("type") == "shutdown":
                        send_frame(conn, {"type": "bye"})
                        return
                    try:
                        response = handle_request(data_dir, request)
                    except ColjoinError as exc:
                        response = {"type": "error", "code": exc.code, "message": str(exc)}
                    send_frame(conn, response)
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--id", required=True)
    parser.add_argument("--timeout", default=10.0, type=float)
    args = parser.parse_args()
    serve(args.data_dir, args.port, args.id, args.timeout)


if __name__ == "__main__":
    main()
