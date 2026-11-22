"""确定性模拟服务子进程(仅依赖标准库, 可独立运行)。

协议(127.0.0.1 回环 TCP, JSON 行):
  服务端 -> {"hello": <id>, "version": 1}
  客户端 -> {"cmd_seq": n, "action": "isolate"|"recover", "instance": <id>}
  服务端 -> {"instance": <id>, "cmd_seq": n, "isolated": bool, "elapsed_ms": int}

模式: normal | nohello(不应答握手) | badreply(回执缺 isolated 字段)
处理耗时确定: elapsed_ms = (cmd_seq * 7 + len(id)) % 5。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys


def serve(instance_id: str, host: str, port: int, mode: str) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
    except OSError as exc:
        print(json.dumps({"error": "E_PORT", "port": port, "reason": str(exc)}))
        return 3
    srv.listen(1)
    isolated = False
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return 0
        with conn:
            if mode == "nohello":
                # 不发送握手, 读至客户端放弃(客户端将以真实套接字超时判 E_HANDSHAKE)
                conn.makefile("r", encoding="utf-8").readline()
                continue
            conn.sendall((json.dumps({"hello": instance_id, "version": 1}) + "\n").encode())
            fh = conn.makefile("r", encoding="utf-8")
            while True:
                line = fh.readline()
                if not line:
                    break
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = cmd.get("action")
                if action == "isolate":
                    isolated = True
                elif action == "recover":
                    isolated = False
                elapsed = (int(cmd.get("cmd_seq", 0)) * 7 + len(instance_id)) % 5
                reply = {"instance": instance_id, "cmd_seq": cmd.get("cmd_seq"),
                         "isolated": isolated, "elapsed_ms": elapsed}
                if mode == "badreply":
                    reply.pop("isolated")
                conn.sendall((json.dumps(reply) + "\n").encode())
            fh.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="faultdiag deterministic mock service")
    parser.add_argument("--id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="normal",
                        choices=["normal", "nohello", "badreply"])
    args = parser.parse_args(argv)
    return serve(args.id, args.host, args.port, args.mode)


if __name__ == "__main__":
    sys.exit(main())
