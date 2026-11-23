"""模拟组件节点：独立子进程，127.0.0.1 回环 TCP 上的 JSON 行协议。

协议（每行一个 JSON 对象）：
- {"type":"dispatch","batch_id":..,"scale_exp":..,"groups":..,
   "upsert":[item...],"remove":[key...]}
  -> 合并切片后重算视图，回执：
     {"type":"receipt","node_id":..,"batch_id":..,"digest":..,
      "key_digests":{..},"config_version":..,"elapsed_ms":..,"scale_exp":..}
- {"type":"ping"} -> {"type":"pong","node_id":..}
- {"type":"shutdown"} -> {"type":"bye","node_id":..} 后退出
"""

import argparse
import json
import os
import socket
import sys
import time

from . import sample


def _mono_ms():
    return time.monotonic_ns() // 1_000_000


class NodeServer:
    def __init__(self, node_id, host, port, seed, sample_size, log_path):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.seed = seed
        self.sample_size = sample_size
        self.items = {}
        self.log = open(log_path, "a", encoding="utf-8")

    def _log(self, event):
        event = dict(event)
        event["node_id"] = self.node_id
        event["mono_ms"] = _mono_ms()
        self.log.write(json.dumps(event, sort_keys=True) + "\n")
        self.log.flush()

    def _handle(self, msg):
        mtype = msg.get("type")
        if mtype == "ping":
            return {"type": "pong", "node_id": self.node_id}, False
        if mtype == "shutdown":
            self._log({"event": "shutdown"})
            return {"type": "bye", "node_id": self.node_id}, True
        if mtype == "dispatch":
            start = _mono_ms()
            for item in msg.get("upsert", []):
                self.items[item["key"]] = item
            for key in msg.get("remove", []):
                self.items.pop(key, None)
            scale_exp = int(msg["scale_exp"])
            groups = int(msg["groups"])
            digest = sample.view_digest(
                self.items, self.seed, self.sample_size, scale_exp, groups)
            kd = sample.key_digests(
                self.items, self.seed, self.sample_size, scale_exp, groups)
            elapsed = _mono_ms() - start
            self._log({
                "event": "dispatch",
                "batch_id": msg.get("batch_id"),
                "item_count": len(self.items),
                "digest": digest,
                "elapsed_ms": elapsed,
            })
            return {
                "type": "receipt",
                "node_id": self.node_id,
                "batch_id": msg.get("batch_id"),
                "digest": digest,
                "key_digests": kd,
                "config_version": sample.config_version_of(self.items),
                "elapsed_ms": elapsed,
                "scale_exp": scale_exp,
            }, False
        return {"type": "error", "node_id": self.node_id,
                "error": "unknown message type"}, False

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        self._log({"event": "listen", "host": self.host, "port": self.port})
        try:
            while True:
                conn, _ = srv.accept()
                with conn:
                    buf = b""
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            msg = json.loads(line.decode("utf-8"))
                            resp, done = self._handle(msg)
                            conn.sendall(
                                json.dumps(resp, sort_keys=True).encode("utf-8")
                                + b"\n")
                            if done:
                                return
        finally:
            srv.close()
            self.log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cfgpipe.node")
    parser.add_argument("--id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    NodeServer(args.id, args.host, args.port, args.seed,
               args.sample_size, args.log).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
