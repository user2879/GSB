"""编排器：子进程生命周期、握手、请求流、派生失效传播、快照与重放。"""

import base64
import os
import socket
import subprocess
import sys
import time

from .cache import Entry, SOURCE_MISS_FETCH, SOURCE_REVALIDATED
from . import errors
from .errors import DiagError, HandshakeError, PortError, ProtoError
from .ledger import Ledger, restore_snapshot
from .proto import recv_obj, send_obj, validate_hello_ack
from .scheduler import PullScheduler, PullTask
from .spec import Content, ContentSpec, content_digest, load_content_spec

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_ready(proc, timeout_s=10.0):
    """读取子进程 READY 行；绑定失败退出码映射为 E_PORT。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            code = proc.wait()
            if code == 3:
                raise PortError("子进程绑定端口失败，端口被占用", exit_code=code)
            raise PortError("子进程启动失败", exit_code=code)
        line = line.strip()
        if line.startswith("READY"):
            return int(line.split()[1])
        if line == "E_PORT":
            raise PortError("子进程报告端口被占用")
    raise PortError("等待子进程 READY 超时")


class ServerHandle:
    def __init__(self, name, proc, port):
        self.name = name
        self.proc = proc
        self.port = port

    def call(self, obj, timeout_ms):
        with socket.create_connection(
            ("127.0.0.1", self.port), timeout=timeout_ms / 1000.0
        ) as s:
            s.settimeout(timeout_ms / 1000.0)
            f = s.makefile("rb")
            send_obj(s, obj)
            reply = recv_obj(f)
        if reply.get("type") == "error":
            code = reply.get("code", "E_INTERNAL")
            exc_cls = next(
                (c for c in vars(errors).values()
                 if isinstance(c, type) and issubclass(c, DiagError)
                 and getattr(c, "code", None) == code),
                DiagError,
            )
            raise exc_cls(reply.get("message", "远端错误"), **reply.get("details", {}))
        return reply


class Orchestrator:
    def __init__(self, workspace, policy, logger):
        self.workspace = workspace
        self.policy = policy
        self.logger = logger
        self.ledger = None
        self.origin = None
        self.nodes = {}  # node_id -> ServerHandle
        self._used_ports = set()
        self.sample_spec_path = os.path.join(workspace, "samples", "content_spec.json")
        self.spec_path = os.path.join(workspace, "ledger", "working_spec.json")
        self.ledger_path = os.path.join(workspace, "ledger", "ledger.db")
        self.snapshot_path = os.path.join(workspace, "ledger", "snapshot.db")
        self.scheduler = PullScheduler(
            policy.global_inflight, policy.per_node_inflight, policy.max_retries
        )
        self._pull_seq = 0

    # ---- 端口与子进程 ----
    def _alloc_port(self):
        lo, hi = self.policy.port_range
        for port in range(lo, hi + 1):
            if port in self._used_ports:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                s.close()
                continue
            s.close()
            self._used_ports.add(port)
            return port
        raise PortError("高位端口范围内无可用端口", port_range=[lo, hi])

    def _spawn(self, module, extra_args, name):
        port = self._alloc_port()
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        stderr_path = os.path.join(self.workspace, "logs", f"{name}.stderr.log")
        stderr_fh = open(stderr_path, "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", f"cdn_diag.{module}", "--port", str(port)]
            + extra_args,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            env=env,
        )
        try:
            bound = _wait_ready(proc)
        except PortError as exc:
            self.logger.log("port_error", code="E_PORT", server=name, port=port)
            raise
        return ServerHandle(name, proc, bound)

    def _handshake(self, handle, expected_server_id):
        timeout_ms = self.policy.handshake_timeout_ms
        try:
            with socket.create_connection(
                ("127.0.0.1", handle.port), timeout=timeout_ms / 1000.0
            ) as s:
                s.settimeout(timeout_ms / 1000.0)
                f = s.makefile("rb")
                send_obj(s, {"type": "hello", "protocol": 1})
                reply = recv_obj(f)
        except (socket.timeout, TimeoutError) as exc:
            self.logger.log("handshake_timeout", code="E_HANDSHAKE", server=handle.name)
            raise HandshakeError("握手超时", server=handle.name) from exc
        except OSError as exc:
            raise HandshakeError(
                "握手连接失败", server=handle.name, error=str(exc)
            ) from exc
        try:
            validate_hello_ack(reply, expected_server_id)
        except ProtoError as exc:
            self.logger.log(
                "handshake_proto_error", code="E_PROTO", server=handle.name, **exc.details
            )
            raise
        self.logger.log("handshake_ok", server=handle.name, port=handle.port)

    # ---- 生命周期 ----
    def _spawn_node(self, node_policy):
        return self._spawn(
            "edge_node",
            [
                "--node-id", node_policy.node_id,
                "--capacity", str(node_policy.capacity),
                "--eviction", node_policy.eviction,
                "--key-fields", ",".join(self.policy.cache_key_fields),
            ],
            node_policy.node_id,
        )

    def start(self):
        # 工作副本：演进写回此处，样本声明保持只读
        import shutil

        os.makedirs(os.path.dirname(self.spec_path), exist_ok=True)
        shutil.copyfile(self.sample_spec_path, self.spec_path)
        spec = load_content_spec(self.spec_path)
        self.ledger = Ledger(self.ledger_path)
        self.ledger.initialize(spec, self.policy.policy_version)
        self.origin = self._spawn("origin_server", ["--spec", self.spec_path], "origin")
        self._handshake(self.origin, "origin")
        for node in self.policy.nodes:
            handle = self._spawn_node(node)
            self._handshake(handle, node.node_id)
            self.nodes[node.node_id] = handle

    def stop(self):
        for handle in [self.origin, *self.nodes.values()]:
            if handle is None:
                continue
            try:
                handle.call({"type": "shutdown"}, 1000)
            except Exception:
                pass
            try:
                handle.proc.wait(timeout=5)
            except Exception:
                handle.proc.kill()
            try:
                if handle.proc.stdout:
                    handle.proc.stdout.close()
            except Exception:
                pass
        if self.ledger:
            self.ledger.close()

    def kill_nodes(self):
        """强制终止全部节点（模拟崩溃）。"""
        for handle in self.nodes.values():
            handle.proc.kill()
            handle.proc.wait()
            try:
                if handle.proc.stdout:
                    handle.proc.stdout.close()
            except Exception:
                pass
        self.nodes = {}

    def restart_nodes(self, restore_entries=True):
        """仅依据账本持久化条目恢复节点，不携带任何内存状态。"""
        for node in self.policy.nodes:
            handle = self._spawn_node(node)
            self._handshake(handle, node.node_id)
            entries = self.ledger.entries_for(node.node_id) if restore_entries else []
            handle.call(
                {"type": "restore", "entries": [e.to_dict() for e in entries]},
                self.policy.pull_timeout_ms,
            )
            self.nodes[node.node_id] = handle

    # ---- 请求流（批次两阶段：集中查找 -> 统一并发回源）----
    def run_batches(self, batches):
        outcomes = []
        for batch in batches:
            pendings = [self._lookup(batch.batch, item) for item in batch.requests]
            tasks = []
            for p in pendings:
                if p["needs_pull"]:
                    tasks.append(self._make_pull_task(p))
            if tasks:
                records = self.scheduler.run(tasks)
                for rec in records:
                    self.ledger.record_event(
                        "pull_attempt",
                        {
                            "task_seq": rec.task_seq,
                            "node": rec.node_id,
                            "path": rec.path,
                            "lease_version": rec.lease_version,
                            "attempt_no": rec.attempt_no,
                            "outcome": rec.outcome,
                            "detail": rec.detail,
                            "wait_start_mono_ms": rec.wait_start_mono_ms,
                            "wait_end_mono_ms": rec.wait_end_mono_ms,
                        },
                    )
            for p in pendings:
                outcomes.append(self._finalize(p))
        return outcomes

    def _lookup(self, batch_no, item):
        handle = self.nodes[item.node]
        reply = handle.call(
            {"type": "request", "path": item.path, "tick_ms": item.tick_ms},
            self.policy.pull_timeout_ms,
        )
        result = reply["result"]
        evictions = []
        if reply.get("expired_evicted"):
            evictions.append(
                {"cache_key": reply["expired_evicted"]["cache_key"], "reason": "expired"}
            )
            self.ledger.remove_entry(item.node, reply["expired_evicted"]["cache_key"])
        return {
            "batch": batch_no,
            "node": item.node,
            "path": item.path,
            "tick_ms": item.tick_ms,
            "result": result,
            "source": reply.get("source"),
            "etag": reply.get("etag"),
            "ttl_remaining_ms": reply.get("ttl_remaining_ms", 0),
            "evictions": evictions,
            "needs_pull": result in ("miss", "expired"),
            "stored": None,
            "pull_failed": None,
        }

    def _make_pull_task(self, pending):
        self._pull_seq += 1
        seq = self._pull_seq
        node_id = pending["node"]
        path = pending["path"]
        tick_ms = pending["tick_ms"]
        source = SOURCE_MISS_FETCH if pending["result"] == "miss" else SOURCE_REVALIDATED
        pending["fetch_source"] = source
        origin = self.origin
        policy = self.policy
        ledger = self.ledger
        logger = self.logger
        node_handle = self.nodes[node_id]

        def execute(attempt_no):
            content = origin.call({"type": "get", "path": path}, policy.pull_timeout_ms)
            lease_reply = origin.call(
                {
                    "type": "issue_lease",
                    "node_id": node_id,
                    "content_versions": {path: content["etag"]},
                    "ttl_ms": policy.lease_ttl_ms,
                },
                policy.pull_timeout_ms,
            )
            lease = lease_reply["lease"]
            ledger.record_lease(lease)
            for conflict in lease_reply.get("conflicts", []):
                ledger.record_event("lease_conflict", conflict)
                logger.log("lease_conflict", **conflict)
            reply = node_handle.call(
                {
                    "type": "fetch",
                    "path": path,
                    "lease_id": lease["lease_id"],
                    "lease_version": lease["lease_version"],
                    "origin_port": origin.port,
                    "pull_timeout_ms": policy.pull_timeout_ms,
                    "tick_ms": tick_ms,
                    "source": source,
                },
                policy.pull_timeout_ms + 1000,
            )
            if reply.get("type") == "error":
                return "error", reply.get("message", ""), lease["lease_version"]
            pending["stored"] = reply
            return "ok", "", lease["lease_version"]

        return PullTask(seq=seq, node_id=node_id, path=path, execute=execute)

    def _finalize(self, p):
        if p["needs_pull"]:
            if p["stored"] is None:
                raise ProtoError("拉取失败", node=p["node"], path=p["path"])
            stored = p["stored"]
            p["source"] = p["fetch_source"]
            p["etag"] = stored["etag"]
            p["ttl_remaining_ms"] = stored["ttl_ms"]
            for ev in stored["evictions"]:
                p["evictions"].append({"cache_key": ev["cache_key"], "reason": "capacity"})
                self.ledger.remove_entry(p["node"], ev["cache_key"])
            entry = Entry(
                cache_key=self._cache_key(p["node"], p["path"]),
                path=p["path"],
                etag=stored["etag"],
                body=b"",
                ttl_ms=stored["ttl_ms"],
                write_tick_ms=p["tick_ms"],
                entry_version=stored["entry_version"],
                source=p["fetch_source"],
                last_access_tick_ms=p["tick_ms"],
            )
            self.ledger.upsert_entry(p["node"], entry)
        outcome = {
            "batch": p["batch"],
            "node": p["node"],
            "path": p["path"],
            "tick_ms": p["tick_ms"],
            "result": p["result"],
            "source": p["source"],
            "etag": p["etag"],
            "ttl_remaining_ms": p["ttl_remaining_ms"],
            "evictions": p["evictions"],
            "stored": p["stored"],
        }
        self.ledger.record_event("request", outcome)
        return outcome

    def _cache_key(self, node_id, path):
        record = {"node_id": node_id, "path": path}
        return "|".join(str(record[f]) for f in self.policy.cache_key_fields)

    # ---- 内容版本演进与派生失效传播 ----
    def evolve_content(self, path, new_etag, new_body, tick_ms, derived_from=None):
        reply = self.origin.call(
            {
                "type": "update_content",
                "path": path,
                "etag": new_etag,
                "body_base64": base64.b64encode(new_body).decode("ascii"),
                "derived_from": derived_from,
            },
            self.policy.pull_timeout_ms,
        )
        spec_now = self._fetch_origin_spec()
        self.ledger.replace_contents(spec_now)
        self.ledger.set_meta("content_digest", spec_now.digest)
        self._persist_spec(spec_now)
        propagation_id = self.ledger.record_event(
            "propagation_start",
            {
                "trigger_path": path,
                "old_etag": reply["old_etag"],
                "new_etag": reply["new_etag"],
            },
        )
        trace = []
        # 受影响路径 = 触发路径自身 + 稳定拓扑序的全部下游；同一路径仅替换一次
        affected = [
            {
                "path": path,
                "etag": reply["new_etag"],
                "body_base64": base64.b64encode(new_body).decode("ascii"),
                "ttl_ms": spec_now.contents[path].ttl_ms,
            }
        ] + reply["affected"]
        for item in affected:
            for node_id in sorted(self.nodes):
                rep = self.nodes[node_id].call(
                    {
                        "type": "invalidate_replace",
                        "path": item["path"],
                        "etag": item["etag"],
                        "body_base64": item["body_base64"],
                        "ttl_ms": item["ttl_ms"],
                        "tick_ms": tick_ms,
                    },
                    self.policy.pull_timeout_ms,
                )
                event = {
                    "propagation_id": propagation_id,
                    "trigger_path": path,
                    "affected_path": item["path"],
                    "node": node_id,
                    "present": rep["present"],
                    "old_entry_version": rep["old_entry_version"],
                    "new_entry_version": rep["new_entry_version"],
                    "new_etag": item["etag"],
                }
                self.ledger.record_event("propagation", event)
                if rep["present"]:
                    entry = Entry(
                        cache_key=self._cache_key(node_id, item["path"]),
                        path=item["path"],
                        etag=item["etag"],
                        body=base64.b64decode(item["body_base64"]),
                        ttl_ms=item["ttl_ms"],
                        write_tick_ms=tick_ms,
                        entry_version=rep["new_entry_version"],
                        source="authoritative",
                        last_access_tick_ms=tick_ms,
                    )
                    self.ledger.upsert_entry(node_id, entry)
                trace.append(event)
        return trace

    def _persist_spec(self, spec):
        """内容版本演进后，将源站最新声明写回磁盘（派生路径仅保留派生关系）。"""
        import base64 as _b64
        import json as _json

        doc = {"spec_version": spec.spec_version, "contents": []}
        for c in spec.contents.values():
            item = {"path": c.path, "ttl_ms": c.ttl_ms}
            if c.derived_from:
                item["derived_from"] = list(c.derived_from)
                item["combine"] = c.combine
            else:
                item["etag"] = c.etag
                item["body_base64"] = _b64.b64encode(c.body).decode("ascii")
            doc["contents"].append(item)
        with open(self.spec_path, "w", encoding="utf-8") as fh:
            _json.dump(doc, fh, ensure_ascii=False, indent=2)

    def _fetch_origin_spec(self):
        contents = {}
        ledger_spec = self.ledger.load_contents()
        for path in ledger_spec.contents:
            c = self.origin.call({"type": "get", "path": path}, self.policy.pull_timeout_ms)
            old = ledger_spec.contents[path]
            contents[path] = Content(
                path=path,
                etag=c["etag"],
                body=base64.b64decode(c["body_base64"]),
                ttl_ms=c["ttl_ms"],
                derived_from=old.derived_from,
                combine=old.combine,
            )
        return ContentSpec(spec_version=1, contents=contents, digest=content_digest(contents))

    # ---- 快照与恢复 ----
    def save_snapshot(self):
        self.ledger.save_snapshot(self.snapshot_path)
        self.logger.log("snapshot_saved", snapshot=self.snapshot_path)
        return self.snapshot_path

    def restore(self):
        current_spec = load_content_spec(self.spec_path)
        self.ledger.close()
        self.ledger, diff, replaced = restore_snapshot(
            self.snapshot_path, self.ledger_path, current_spec, logger=self.logger
        )
        self.logger.log("snapshot_restored", diff=diff, full_replace=replaced)
        return diff, replaced

    # ---- 重放校验 ----
    def replay(self):
        from .replay import Replayer

        return Replayer(self).run()
