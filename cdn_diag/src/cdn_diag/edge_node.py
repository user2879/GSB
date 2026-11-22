"""边缘节点子进程：维护带版本缓存条目，按 JSON 行协议回报状态。

启动：python -m cdn_diag.edge_node --port P --node-id ID --capacity N \
        --eviction lru --key-fields node_id,path
节点被强制终止后重启只能依据编排器下发的已持久化条目恢复。
"""

import argparse
import base64
import socket
import threading

from .cache import Entry, NodeCache
from .errors import ProtoError
from .proto import recv_obj, send_obj, serve

PROTOCOL_VERSION = 1


class EdgeNode:
    def __init__(self, node_id, capacity, eviction, key_fields):
        self.node_id = node_id
        self.cache = NodeCache(node_id, capacity, eviction, key_fields)
        self._lock = threading.Lock()

    def handle(self, req):
        t = req.get("type")
        if t == "hello":
            return {
                "type": "hello_ack",
                "protocol": PROTOCOL_VERSION,
                "server_id": self.node_id,
            }
        if t == "restore":
            entries = [Entry.from_dict(e) for e in req.get("entries", [])]
            with self._lock:
                self.cache.restore(entries)
            return {"type": "restored", "count": len(entries)}
        if t == "request":
            return self._request(req)
        if t == "fetch":
            return self._fetch(req)
        if t == "store_replay":
            return self._store_replay(req)
        if t == "invalidate_replace":
            return self._invalidate_replace(req)
        if t == "dump":
            with self._lock:
                return {
                    "type": "entries",
                    "entries": [e.to_dict() for e in self.cache.dump()],
                }
        if t == "shutdown":
            return None
        raise ProtoError("未知命令", command=t)

    def _request(self, req):
        with self._lock:
            status, entry, remaining, evicted = self.cache.lookup(
                req["path"], int(req["tick_ms"])
            )
        reply = {
            "type": "outcome",
            "result": status,
            "etag": entry.etag if entry else None,
            "ttl_remaining_ms": remaining,
            "source": entry.source if entry else None,
            "entry_version": entry.entry_version if entry else None,
            "expired_evicted": evicted.to_dict() if evicted else None,
        }
        return reply

    def _fetch(self, req):
        """凭租约向内容源拉取分片并写入缓存（真实套接字 + 超时）。"""
        origin_port = int(req["origin_port"])
        timeout = float(req["pull_timeout_ms"]) / 1000.0
        with socket.create_connection(("127.0.0.1", origin_port), timeout=timeout) as s:
            s.settimeout(timeout)
            f = s.makefile("rb")
            send_obj(
                s,
                {
                    "type": "pull",
                    "path": req["path"],
                    "lease_id": req["lease_id"],
                    "lease_version": req["lease_version"],
                },
            )
            reply = recv_obj(f)
        if reply["type"] == "error":
            return reply
        if reply["type"] != "content":
            raise ProtoError("拉取回执类型错误", got=reply["type"])
        body = base64.b64decode(reply["body_base64"])
        with self._lock:
            entry, evicted = self.cache.store(
                req["path"],
                reply["etag"],
                body,
                int(reply["ttl_ms"]),
                int(req["tick_ms"]),
                req["source"],
            )
        return {
            "type": "stored",
            "path": req["path"],
            "etag": entry.etag,
            "entry_version": entry.entry_version,
            "ttl_ms": entry.ttl_ms,
            "evictions": [e.to_dict() for e in evicted],
        }

    def _store_replay(self, req):
        """重放模式：不下发网络拉取，按账本记录的内容版本直接写入。"""
        with self._lock:
            entry, evicted = self.cache.store(
                req["path"],
                req["etag"],
                base64.b64decode(req.get("body_base64") or ""),
                int(req["ttl_ms"]),
                int(req["tick_ms"]),
                req["source"],
            )
        return {
            "type": "stored",
            "path": req["path"],
            "etag": entry.etag,
            "entry_version": entry.entry_version,
            "ttl_ms": entry.ttl_ms,
            "evictions": [e.to_dict() for e in evicted],
        }

    def _invalidate_replace(self, req):
        with self._lock:
            result = self.cache.replace(
                req["path"],
                req["etag"],
                base64.b64decode(req["body_base64"]),
                int(req["ttl_ms"]),
                int(req["tick_ms"]),
            )
        if result is None:
            return {
                "type": "replaced",
                "path": req["path"],
                "present": False,
                "old_entry_version": None,
                "new_entry_version": None,
            }
        old_v, new_v = result
        return {
            "type": "replaced",
            "path": req["path"],
            "present": True,
            "old_entry_version": old_v,
            "new_entry_version": new_v,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--eviction", required=True)
    parser.add_argument("--key-fields", required=True)
    args = parser.parse_args()

    node = EdgeNode(
        args.node_id, args.capacity, args.eviction, args.key_fields.split(",")
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", args.port))
    except OSError:
        print("E_PORT", flush=True)
        raise SystemExit(3)
    sock.listen(8)
    print(f"READY {args.port}", flush=True)
    try:
        while True:
            conn, _ = sock.accept()
            threading.Thread(
                target=serve, args=(conn, node.handle), daemon=True
            ).start()
    finally:
        sock.close()


if __name__ == "__main__":
    main()
