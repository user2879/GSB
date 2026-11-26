"""工作进程子进程：在 127.0.0.1 回环端口上以行协议服务批次推理。

协议（TCP，换行分隔）：
  C->S  HELLO <worker_id> <param_version>
  S->C  READY <worker_id> <param_version> <param_digest>
  C->S  BATCH <batch_id> <rid1,rid2,...>
  S->C  若干 r3 回执行，随后一行 END
  C->S  QUIT      关闭当前连接
  C->S  SHUTDOWN  停止服务进程
端口被占用时以 E_PORT 报告并以退出码 3 退出。
"""

import argparse
import os
import socket
import sys
import threading

from .core import compute_receipt_lines, generate_params, param_digest, param_spec_dict, load_config
from .errors import PortError


def serve(worker_id, host, port, cfg, delay_ms=0):
    threading.Thread(target=_stdin_watchdog, daemon=True).start()
    spec = param_spec_dict(worker_id, cfg["workers_by_id"][worker_id]["seed"], cfg)
    p_digest = param_digest(spec, generate_params(spec))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
    except OSError as exc:
        print(str(PortError(f"端口被占用: {host}:{port}", port=port, detail=str(exc))),
              file=sys.stderr, flush=True)
        sys.exit(3)
    srv.listen(64)
    try:
        while True:
            conn, _ = srv.accept()
            try:
                _handle(conn, cfg, worker_id, p_digest, delay_ms)
            except (ConnectionError, OSError):
                pass
            finally:
                conn.close()
    except _Shutdown:
        pass
    finally:
        srv.close()


class _Shutdown(Exception):
    pass


def _stdin_watchdog():
    """托管父进程被强制终止（kill -9）后，stdin 管道 EOF，子进程自行退出。"""
    try:
        while sys.stdin.buffer.read(1) != b"":
            pass
    except (OSError, ValueError):
        pass
    os._exit(0)


def _handle(conn, cfg, worker_id, p_digest, delay_ms):
    buf = conn.makefile("r", encoding="utf-8", newline="\n")
    while True:
        line = buf.readline()
        if not line:
            return
        parts = line.rstrip("\n").split(" ")
        cmd = parts[0]
        if cmd == "HELLO":
            param_version = parts[2] if len(parts) > 2 else "0"
            conn.sendall(f"READY {worker_id} {param_version} {p_digest}\n".encode("utf-8"))
        elif cmd == "BATCH":
            batch_id = parts[1]
            rids = parts[2].split(",") if len(parts) > 2 and parts[2] else []
            param_version = int(parts[3]) if len(parts) > 3 else 0
            lines = compute_receipt_lines(cfg, worker_id, param_version, batch_id, rids, delay_ms)
            conn.sendall(("\n".join(lines) + "\nEND\n").encode("utf-8"))
        elif cmd == "QUIT":
            return
        elif cmd == "SHUTDOWN":
            raise _Shutdown()
        else:
            conn.sendall(b"ERR unknown-command\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--compute-delay-ms", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    serve(args.worker_id, args.host, args.port, cfg, args.compute_delay_ms)


if __name__ == "__main__":
    main()
