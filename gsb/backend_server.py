"""推理后端子进程：在 127.0.0.1 上动态分配端口，按声明格式版本产出回执。

协议（每连接一帧请求、一帧响应）：
    请求:  "GSB-REQ <n>\n" + n 字节 JSON
    响应:  "GSB-RESP <n>\n" + n 字节 JSON {"ok": bool, ...}

命令：
    ping     -> 存活与当前参数版本
    update   -> {"seed": int} 整体替换参数，版本序号递增
    exec     -> {"batch_id","route","sample_ids","param_version","delay_ms"?}
    shutdown -> 退出
启动后向 stdout 打印 "PORT <实际端口>" 供父进程读取。
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

from .hotupdate import compute_param_digest
from .receipts import compute_output_digest, encode_frame


def _read_frame(conn: socket.socket) -> dict:
    header = b""
    while not header.endswith(b"\n"):
        chunk = conn.recv(1)
        if not chunk:
            raise ConnectionError("连接在头部读取前关闭")
        header += chunk
    parts = header.decode("utf-8").strip().split(" ")
    if len(parts) != 2 or parts[0] != "GSB-REQ":
        raise ValueError("非法请求帧头")
    length = int(parts[1])
    payload = b""
    while len(payload) < length:
        chunk = conn.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("连接在负载读取前关闭")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


def _write_frame(conn: socket.socket, data: dict) -> None:
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    conn.sendall(f"GSB-RESP {len(payload)}\n".encode("utf-8") + payload)


class BackendState:
    def __init__(self, decl: dict):
        self.decl = decl
        self.lock = threading.Lock()
        self.seed = decl["seed"]
        self.param_version = 1
        self.param_digest = compute_param_digest(
            decl["seed"], decl["model_dim"], decl["hidden_dim"], decl["quant_profile"]
        )

    def update(self, new_seed: int) -> dict:
        with self.lock:
            self.seed = new_seed
            self.param_version += 1
            self.param_digest = compute_param_digest(
                new_seed,
                self.decl["model_dim"],
                self.decl["hidden_dim"],
                self.decl["quant_profile"],
            )
            return {
                "param_version": self.param_version,
                "param_digest": self.param_digest,
            }

    def execute(self, request: dict) -> dict:
        delay_ms = int(request.get("delay_ms", 0))
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        with self.lock:
            current_version = self.param_version
            param_digest = self.param_digest
        bound = request["param_version"]
        if bound != current_version:
            return {
                "ok": False,
                "error": "STALE_VERSION",
                "bound_version": bound,
                "current_version": current_version,
            }
        started = time.monotonic()
        batch_id = request["batch_id"]
        route = request["route"]
        sample_ids = sorted(request["sample_ids"])
        output_digest = compute_output_digest(
            batch_id, route, param_digest, sample_ids
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        data = {
            "batch_id": batch_id,
            "route": route,
            "output_digest": output_digest,
        }
        fmt = self.decl["format_version"]
        if fmt in ("v2", "v3"):
            data["param_digest"] = param_digest
            data["quant_profile"] = self.decl["quant_profile"]
            data["elapsed_ms"] = elapsed_ms
        if fmt == "v3":
            data["sample_ids"] = sample_ids
            data["param_version"] = current_version
        return {"ok": True, "frame": encode_frame(fmt, data)}


def serve(decl: dict) -> None:
    state = BackendState(decl)
    stop_event = threading.Event()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    server.settimeout(0.2)
    port = server.getsockname()[1]
    print(f"PORT {port}", flush=True)

    def handle(conn: socket.socket) -> None:
        try:
            request = _read_frame(conn)
            cmd = request.get("cmd")
            if cmd == "ping":
                response = {"ok": True, "param_version": state.param_version}
            elif cmd == "update":
                response = {"ok": True, **state.update(int(request["seed"]))}
            elif cmd == "exec":
                response = state.execute(request)
            elif cmd == "shutdown":
                response = {"ok": True}
                _write_frame(conn, response)
                conn.close()
                stop_event.set()
                return
            else:
                response = {"ok": False, "error": f"未知命令: {cmd}"}
            _write_frame(conn, response)
        except Exception as exc:  # noqa: BLE001 - 子进程边界，回报错误
            try:
                _write_frame(conn, {"ok": False, "error": str(exc)})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while not stop_event.is_set():
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        except OSError:
            return
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
    server.close()


def main() -> None:
    decl = json.loads(sys.argv[1])
    serve(decl)


if __name__ == "__main__":
    main()
