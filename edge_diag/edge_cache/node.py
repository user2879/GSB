"""缓存节点子进程。

职责（真实独立进程，127.0.0.1 回环 TCP，JSON 行协议）：
- 按声明端口真实监听，支持主进程线程池的多连接并发（节点内加锁保护租约表）
- 按节点种子与内容键生成确定性分片字节，NumPy 完成整数定点校验并回传摘要
- 维护租约版本号，续约做乐观校验：版本不符回 E_LEASE 且不延长到期
- 到期判定只比较真实单调时钟时间戳
启动成功向 stdout 写一行 ready JSON；绑定失败写 error 行（E_PORT）后退出。
"""

import json
import socket
import threading
import time

from . import determinism
from .errors import LeaseError


def _mono_ms():
    return time.monotonic_ns() // 1_000_000


class CacheNode:
    def __init__(self, node_spec, contents, session_scale):
        self.spec = node_spec
        self.contents = {c["content_key"]: c for c in contents}
        self.session_scale = session_scale
        self._leases = {}
        self._lock = threading.Lock()
        self._server = None
        self._stop = threading.Event()

    # ---- 租约表（节点侧，版本号权威由节点维护）----
    def _install_lease(self, lease_id, shard_id, quota_version, grant_ms,
                       expire_ms, renew_count, version):
        self._leases[lease_id] = {
            "lease_id": lease_id,
            "shard_id": shard_id,
            "quota_version": quota_version,
            "grant_ms": grant_ms,
            "expire_ms": expire_ms,
            "renew_count": renew_count,
            "version": version,
        }

    def _handle_renew(self, item):
        lease_id = item["lease_id"]
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return self._error_receipt(item, "E_LEASE",
                                           "节点不存在该租约（可能已被回收）")
            if lease["version"] != item["expected_version"]:
                return self._error_receipt(
                    item, "E_LEASE",
                    f"租约版本冲突 expected={item['expected_version']} "
                    f"actual={lease['version']}")
            now = _mono_ms()
            if now >= lease["expire_ms"]:
                return self._error_receipt(item, "E_LEASE",
                                           "租约已到期，续约被拒绝")
            lease["version"] += 1
            lease["renew_count"] += 1
            lease["expire_ms"] = now + int(item["ttl_ms"])
            return {
                "ok": True,
                "op": "renew",
                "lease_id": lease_id,
                "shard_id": lease["shard_id"],
                "node_id": self.spec["node_id"],
                "quota_version": lease["quota_version"],
                "renew_count": lease["renew_count"],
                "lease_version": lease["version"],
                "grant_ms": lease["grant_ms"],
                "expire_ms": lease["expire_ms"],
            }

    def _error_receipt(self, item, code, message):
        return {
            "ok": False,
            "code": code,
            "message": message,
            "op": item.get("op"),
            "lease_id": item.get("lease_id"),
            "node_id": self.spec["node_id"],
            "ref": item.get("content_key"),
        }

    def _payload_for(self, content_key, shard_index):
        content = self.contents[content_key]
        return determinism.generate_shard_bytes(
            content_key, content["content_seed"], shard_index,
            content["shard_byte_size"])

    def _digest_receipt(self, item, op):
        start = _mono_ms()
        content_key = item["content_key"]
        shard_index = item["shard_index"]
        payload = self._payload_for(content_key, shard_index)
        digest = determinism.compute_digest(
            self.spec["node_id"], self.spec["node_seed"], content_key,
            shard_index, payload, self.session_scale)
        elapsed = _mono_ms() - start
        return {
            "ok": True,
            "op": op,
            "node_id": self.spec["node_id"],
            "lease_id": item.get("lease_id"),
            "shard_id": digest["shard_id"],
            "content_key": content_key,
            "shard_index": shard_index,
            "quota_version": item["quota_version"],
            "digest": digest["digest"],
            "byte_hash": digest["byte_hash"],
            "fp_sum": digest["fp_sum"],
            "fp_sqsum": digest["fp_sqsum"],
            "sample_count": digest["sample_count"],
            "session_scale": digest["session_scale"],
            "elapsed_ms": elapsed,
        }

    def handle_message(self, message):
        mtype = message.get("type")
        if mtype == "hello":
            return {
                "type": "hello_ack",
                "node_id": self.spec["node_id"],
                "port": self.spec["port"],
                "quota_version": self.spec["quota_version"],
                "lease_count": len(self._leases),
            }
        if mtype == "batch":
            receipts = []
            for item in message["items"]:
                op = item["op"]
                if op in ("write", "read"):
                    receipts.append(self._digest_receipt(item, op))
                elif op == "renew":
                    receipts.append(self._handle_renew(item))
                else:
                    receipts.append(self._error_receipt(
                        item, "E_PROTOCOL", f"未知批次操作 {op}"))
            return {"type": "batch_ack", "batch_id": message["batch_id"],
                    "receipts": receipts}
        if mtype == "grant":
            with self._lock:
                self._install_lease(
                    message["lease_id"],
                    determinism.shard_id(message["content_key"],
                                         message["shard_index"]),
                    message["quota_version"], message["grant_ms"],
                    message["expire_ms"], message["renew_count"],
                    message["lease_version"])
            return {"type": "grant_ack", "ok": True,
                    "lease_id": message["lease_id"],
                    "node_id": self.spec["node_id"]}
        if mtype == "reclaim":
            with self._lock:
                self._leases.pop(message["lease_id"], None)
            return {"type": "reclaim_ack", "ok": True,
                    "lease_id": message["lease_id"],
                    "reason": message.get("reason"),
                    "node_id": self.spec["node_id"]}
        if mtype == "restore":
            with self._lock:
                for lease in message["leases"]:
                    self._install_lease(
                        lease["lease_id"], lease["shard_id"],
                        lease["quota_version"], lease["grant_ms"],
                        lease["expire_ms"], lease["renew_count"],
                        lease["lease_version"])
            return {"type": "restore_ack", "ok": True,
                    "restored": len(message["leases"]),
                    "node_id": self.spec["node_id"]}
        if mtype == "ping":
            return {"type": "pong", "node_id": self.spec["node_id"],
                    "mono_ms": _mono_ms()}
        if mtype == "shutdown":
            self._stop.set()
            return {"type": "shutdown_ack", "ok": True,
                    "node_id": self.spec["node_id"]}
        return {"type": "error", "ok": False, "code": "E_PROTOCOL",
                "message": f"未知消息类型 {mtype}"}

    def _serve_client(self, conn):
        from .protocol import LineConnection
        line_conn = LineConnection(conn)
        try:
            while not self._stop.is_set():
                message = line_conn.read_message()
                reply = self.handle_message(message)
                line_conn.send_message(reply)
                if message.get("type") == "shutdown":
                    return
        except (OSError, EOFError):
            return
        except Exception as exc:  # 节点不崩溃，回错误行
            try:
                line_conn.send_message(
                    {"type": "error", "ok": False, "code": "E_NODE",
                     "message": str(exc)})
            except OSError:
                pass
        finally:
            line_conn.close()

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.spec["host"], self.spec["port"]))
        except OSError as exc:
            print(json.dumps({"event": "error", "code": "E_PORT",
                              "message": str(exc),
                              "port": self.spec["port"]}),
                  flush=True)
            raise
        server.listen(64)
        self._server = server
        server.settimeout(0.5)
        print(json.dumps({"event": "ready",
                          "node_id": self.spec["node_id"],
                          "port": self.spec["port"]}), flush=True)
        while not self._stop.is_set():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._serve_client,
                                      args=(conn,), daemon=True)
            thread.start()
        server.close()


def run_node(node_spec, contents, session_scale):
    node = CacheNode(node_spec, contents, session_scale)
    node.serve()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args()
    from .spec import load_spec
    spec = load_spec(args.spec)
    node_spec = next(n for n in spec["nodes"]
                     if n["node_id"] == args.node_id)
    run_node(node_spec, spec["contents"],
             spec["session"]["session_scale"])


if __name__ == "__main__":
    main()
