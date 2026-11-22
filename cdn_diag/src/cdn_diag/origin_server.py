"""内容源子进程：持有内容声明，签发租约，响应节点分片拉取。

启动：python -m cdn_diag.origin_server --port P --spec content_spec.json
"""

import argparse
import base64
import socket
import threading

from .derive import downstream_of, recompute_derived
from .errors import GraphError, ProtoError
from .lease import LeaseRegistry
from .proto import send_obj, serve
from .spec import content_digest, load_content_spec

PROTOCOL_VERSION = 1


class OriginServer:
    def __init__(self, spec_path):
        self.spec = load_content_spec(spec_path)
        self.leases = LeaseRegistry()
        self._lock = threading.Lock()

    # ---- 命令处理 ----
    def handle(self, req):
        t = req.get("type")
        if t == "hello":
            return {
                "type": "hello_ack",
                "protocol": PROTOCOL_VERSION,
                "server_id": "origin",
                "content_digest": self.spec.digest,
            }
        if t == "issue_lease":
            return self._issue_lease(req)
        if t == "renew_lease":
            return self._renew_lease(req)
        if t == "pull":
            return self._pull(req)
        if t == "update_content":
            return self._update_content(req)
        if t == "get":
            return self._get(req)
        if t == "reclaim_leases":
            return {"type": "reclaimed", "lease_ids": self.leases.reclaim_expired()}
        if t == "shutdown":
            return None
        raise ProtoError("未知命令", command=t)

    def _issue_lease(self, req):
        with self._lock:
            lease, conflicts = self.leases.issue(
                req["node_id"], req["content_versions"], int(req["ttl_ms"])
            )
        return {"type": "lease", "lease": lease, "conflicts": conflicts}

    def _renew_lease(self, req):
        with self._lock:
            lease = self.leases.renew(
                req["lease_id"], int(req["lease_version"]), int(req["ttl_ms"])
            )
        return {"type": "lease", "lease": lease, "conflicts": []}

    def _pull(self, req):
        path = req["path"]
        with self._lock:
            content = self.spec.contents.get(path)
            if content is None:
                raise ProtoError("路径不存在", path=path)
            self.leases.validate_for_pull(
                req["lease_id"], int(req["lease_version"]), path, content.etag
            )
            return {
                "type": "content",
                "path": path,
                "etag": content.etag,
                "body_base64": base64.b64encode(content.body).decode("ascii"),
                "ttl_ms": content.ttl_ms,
            }

    def _update_content(self, req):
        """内容版本演进：更新基础路径，重算派生，返回稳定拓扑序的受影响集合。"""
        path = req["path"]
        with self._lock:
            content = self.spec.contents.get(path)
            if content is None:
                raise ProtoError("路径不存在", path=path)
            if "derived_from" in req and req["derived_from"] is not None:
                # 运行期派生关系变更（异常注入可在此造环 -> E_GRAPH）
                content.derived_from = tuple(req["derived_from"])
            if content.derived_from and "derived_from" not in req:
                raise ProtoError("派生路径不能直接更新内容", path=path)
            old_etag = content.etag
            content.etag = req["etag"]
            content.body = base64.b64decode(req["body_base64"])
            try:
                recompute_derived(self.spec.contents)
            except GraphError:
                raise
            affected_paths = downstream_of(self.spec.contents, [path])
            affected = [
                {
                    "path": p,
                    "etag": self.spec.contents[p].etag,
                    "body_base64": base64.b64encode(self.spec.contents[p].body).decode("ascii"),
                    "ttl_ms": self.spec.contents[p].ttl_ms,
                }
                for p in affected_paths
            ]
            self.spec.digest = content_digest(self.spec.contents)
            return {
                "type": "content_updated",
                "trigger_path": path,
                "old_etag": old_etag,
                "new_etag": content.etag,
                "affected": affected,
            }

    def _get(self, req):
        content = self.spec.contents.get(req["path"])
        if content is None:
            raise ProtoError("路径不存在", path=req["path"])
        return {
            "type": "content",
            "path": content.path,
            "etag": content.etag,
            "body_base64": base64.b64encode(content.body).decode("ascii"),
            "ttl_ms": content.ttl_ms,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    server = OriginServer(args.spec)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", args.port))
    except OSError:
        print("E_PORT", flush=True)
        raise SystemExit(3)
    sock.listen(16)
    print(f"READY {args.port}", flush=True)
    try:
        while True:
            conn, _ = sock.accept()
            threading.Thread(
                target=serve, args=(conn, server.handle), daemon=True
            ).start()
    finally:
        sock.close()


if __name__ == "__main__":
    main()
