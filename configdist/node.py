"""组件节点子进程：绑定唯一回环端口，逐行 JSON 协议应答。

协议（每行一个 JSON 对象）：
  <- {"op":"hello"}
  -> {"op":"hello_ack","node",...,"state":{...}}        依据持久化状态恢复
  <- {"op":"apply","batch_id","manifest_version","key","key_version",
      "value","context","derived"}
  -> s1: {"op":"receipt","view":"s1","node","batch_id","key",
          "manifest_version","digest","elapsed_ms"}
     s2: 同上 + {"summary","key_version","profile"}
  <- {"op":"shutdown"}  -> {"op":"bye"} 后退出
端口被占用时输出 {"error":"E_PORT",...} 并以退出码 3 退出。
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

from .canon import canon
from .digest import compute_summary, digest_of, effective_value
from .samples import load_data

EXIT_PORT_BUSY = 3


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class NodeServer:
    def __init__(self, node_id, port, view, profile, data_path, state_dir,
                 fudge=0):
        self.id = node_id
        self.port = port
        self.view = view
        self.profile = profile
        self.fudge = fudge
        self.series = load_data(data_path)
        self.state_path = os.path.join(state_dir, f"node-{node_id}.state.json")
        self.state = self._load_state()
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"manifest_version": 0, "key_versions": {}, "applied": 0}

    def _persist_state(self):
        _atomic_write(self.state_path, canon(self.state))

    def _handle_apply(self, msg):
        started = time.monotonic_ns()
        mv = msg["manifest_version"]
        key = msg["key"]
        key_version = msg["key_version"]
        value = effective_value(key, msg.get("value"), msg.get("context", {}),
                                msg.get("derived", []))
        summary = compute_summary(mv, self.profile,
                                  [[key, value, key_version]], self.series)
        if self.fudge:
            obj = json.loads(summary)
            obj["agg"] += self.fudge
            summary = canon(obj)
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        with self._lock:
            self.state["manifest_version"] = mv
            self.state["key_versions"][key] = key_version
            self.state["applied"] += 1
            self._persist_state()
        receipt = {"op": "receipt", "view": self.view, "node": self.id,
                   "batch_id": msg.get("batch_id", 0), "key": key,
                   "manifest_version": mv, "digest": digest_of(summary),
                   "elapsed_ms": int(elapsed_ms)}
        if self.view == "s2":
            receipt["summary"] = summary
            receipt["key_version"] = key_version
            receipt["profile"] = self.profile
        return receipt

    def _handle(self, msg):
        op = msg.get("op")
        if op == "hello":
            return {"op": "hello_ack", "node": self.id, "view": self.view,
                    "profile": self.profile, "state": self.state}
        if op == "apply":
            return self._handle_apply(msg)
        if op == "shutdown":
            self._shutdown.set()
            return {"op": "bye", "node": self.id}
        return {"op": "error", "code": "E_PROTO", "detail": f"未知 op: {op!r}"}

    def _serve_conn(self, conn):
        with conn:
            f = conn.makefile("rwb")
            for raw in f:
                try:
                    msg = json.loads(raw.decode("utf-8"))
                    reply = self._handle(msg)
                except Exception as exc:  # 节点侧容错：非法输入回 E_PROTO
                    reply = {"op": "error", "code": "E_PROTO",
                             "detail": str(exc)}
                f.write((canon(reply) + "\n").encode("utf-8"))
                f.flush()
                if self._shutdown.is_set():
                    return

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", self.port))
        except OSError:
            print(canon({"error": "E_PORT", "node": self.id,
                         "port": self.port}), flush=True)
            return EXIT_PORT_BUSY
        srv.listen(16)
        srv.settimeout(0.2)
        print(canon({"ready": True, "node": self.id, "port": self.port}),
              flush=True)
        while not self._shutdown.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._serve_conn, args=(conn,),
                             daemon=True).start()
        srv.close()
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="configdist.node")
    parser.add_argument("--id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--view", required=True, choices=("s1", "s2"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--fudge", type=int, default=0)
    args = parser.parse_args(argv)
    os.makedirs(args.state_dir, exist_ok=True)
    server = NodeServer(args.id, args.port, args.view, args.profile,
                        args.data, args.state_dir, args.fudge)
    return server.serve()


if __name__ == "__main__":
    sys.exit(main())
