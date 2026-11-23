"""管线编排：加载声明 -> 启动节点子进程 -> 初始全量下发 -> 变更集增量下发。

每个批次：失效推导 -> 仅编译受影响项 -> 调度下发 -> 收敛判定 ->
同一 SQLite 事务写账本 -> 落盘快照（可独立复核的证据链）。

恢复语义：账本内容摘要仅绑定声明文件（变更集进度由游标跟踪）；
重新打开时摘要或格式版本不一致则整体替换账本。强制终止后重启，
依据已完整提交的记录恢复，未提交批次中的节点标记为 unknown 并要求重跑。
"""

import json
import os
import socket
import subprocess
import sys
import time

from . import sample
from .compiler import compile_all, dispatch_payload, sha256_hex
from .converge import check_convergence, expected_view
from .errors import DispatchError
from .graph import (Invalidation, compute_invalidation, node_subscription_map,
                    topo_order)
from .ledger import Ledger
from .replay import compute_error_bounds
from .scheduler import Dispatcher, DispatchTask, mono_ms
from .spec import (apply_changeset, initial_state, load_changeset,
                   load_declaration)


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_node(host, port, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5) as sock:
                sock.sendall(b'{"type":"ping"}\n')
                if sock.recv(1024):
                    return True
        except OSError:
            time.sleep(0.1)
    return False


class Pipeline:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(workdir)
        self.decl_path = os.path.join(self.workdir, "declaration.json")
        self.changesets_dir = os.path.join(self.workdir, "changesets")
        self.state_path = os.path.join(self.workdir, "state.json")
        self.ledger_path = os.path.join(self.workdir, "ledger.db")
        self.snapshots_dir = os.path.join(self.workdir, "snapshots")
        self.logs_dir = os.path.join(self.workdir, "logs")
        self.decl = None
        self.changesets = []
        self.ledger = None
        self.state = None
        self.topo = []
        self.procs = {}
        self.artifact_history = {}   # batch_id -> {key: (bytes, version)}
        self.dispatch_log = []       # 每批次实际下发的键集合（证据）

    # -- 加载与校验 ---------------------------------------------------------

    def load(self):
        self.decl = load_declaration(self.decl_path)
        self.changesets = []
        prev = None
        if os.path.isdir(self.changesets_dir):
            for name in sorted(os.listdir(self.changesets_dir)):
                if not name.endswith(".json"):
                    continue
                cs = load_changeset(os.path.join(self.changesets_dir, name),
                                    self.decl, prev)
                self.changesets.append(cs)
                prev = cs.version
        self.state = initial_state(self.decl)
        self.topo = topo_order(
            list(self.state.items),
            {k: it.derived_from for k, it in self.state.items.items()})
        return self.decl, self.changesets

    def content_digest(self):
        """账本内容摘要：声明文件字节的 SHA-256（变更集进度由游标跟踪）。"""
        with open(self.decl_path, "rb") as fh:
            return sha256_hex(fh.read())

    def open_ledger(self):
        self.ledger = Ledger(self.ledger_path, self.content_digest())
        return self.ledger

    # -- 节点子进程 ---------------------------------------------------------

    def start_nodes(self):
        os.makedirs(self.logs_dir, exist_ok=True)
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for node in self.decl.nodes:
            log_path = os.path.join(self.logs_dir, "node_%s.log" % node.node_id)
            proc = subprocess.Popen([
                sys.executable, "-m", "cfgpipe.node",
                "--id", node.node_id, "--host", node.host,
                "--port", str(node.port), "--seed", str(node.seed),
                "--sample-size", str(node.sample_size), "--log", log_path,
            ], cwd=pkg_root, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.procs[node.node_id] = proc
        for node in self.decl.nodes:
            if not wait_for_node(node.host, node.port):
                raise DispatchError("node failed to start", node=node.node_id,
                                    field="port")

    def stop_nodes(self):
        for node in self.decl.nodes:
            try:
                with socket.create_connection((node.host, node.port),
                                              timeout=2) as sock:
                    sock.sendall(b'{"type":"shutdown"}\n')
                    sock.recv(1024)
            except OSError:
                pass
        for proc in self.procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs = {}

    # -- 批次执行 -----------------------------------------------------------

    def _dispatcher(self):
        s = self.decl.session
        return Dispatcher(s.global_concurrency, s.node_concurrency,
                          s.dispatch_timeout_ms, s.backoff_ms)

    def _snapshot(self, batch_id, artifacts, keys):
        out_dir = os.path.join(self.snapshots_dir, batch_id)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {}
        for key in sorted(keys):
            blob, version = artifacts[key]
            with open(os.path.join(out_dir, "item_%s.json" % key), "wb") as fh:
                fh.write(blob)
            manifest[key] = version
        with open(os.path.join(out_dir, "manifest.json"), "wb") as fh:
            fh.write(json.dumps(manifest, sort_keys=True,
                                indent=2).encode("utf-8"))
        return manifest

    def _dispatch_batch(self, batch_id, seq, changeset_version, state,
                        artifacts, upsert_keys_per_node, remove_keys_per_node,
                        invalidation, error_bounds, affected_nodes):
        """执行一次下发批次并提交账本。返回收敛后的回执记录。"""
        s = self.decl.session
        node_by_id = self.decl.node_map()
        self.ledger.begin_batch(batch_id, sorted(affected_nodes), mono_ms())
        tasks = []
        for nid in sorted(affected_nodes):
            node = node_by_id[nid]
            payload = dispatch_payload(
                batch_id, s.scale_exp, s.groups,
                [artifacts[k][0] for k in sorted(upsert_keys_per_node.get(nid, []))],
                remove_keys_per_node.get(nid, []))
            tasks.append(DispatchTask(nid, node.host, node.port,
                                      payload, batch_id))
        results = self._dispatcher().run(tasks)
        receipts = {}
        for res in results:
            if not res.ok:
                raise DispatchError("dispatch failed after retries: %s" % res.error,
                                    node=res.node_id, field="dispatch")
            rec = dict(res.receipt)
            rec["_attempts"] = res.attempts
            rec["_wait_ms"] = res.wait_end_ms - res.wait_start_ms
            receipts[res.node_id] = rec
        node_subs = node_subscription_map(state)
        expected_views = {
            nd.node_id: expected_view(state.items, node_subs.get(nd.node_id, []),
                                      artifacts, s.scale_exp)
            for nd in self.decl.nodes
        }
        carried_basis = self.ledger.last_receipts()
        records = check_convergence(
            list(self.decl.nodes), set(affected_nodes), receipts,
            expected_views, carried_basis, s.scale_exp, s.groups)
        self.ledger.commit_batch(
            batch_id, seq, changeset_version, mono_ms(),
            [records[nid] for nid in sorted(records)],
            list(invalidation.steps), error_bounds)
        return records

    def _initial_batch(self, report, progress):
        s = self.decl.session
        artifacts = compile_all(self.state, s.scale_exp, self.topo)
        self.artifact_history["batch_0000"] = artifacts
        node_subs = node_subscription_map(self.state)
        upsert = {nd.node_id: list(node_subs.get(nd.node_id, []))
                  for nd in self.decl.nodes}
        bounds = compute_error_bounds(self.state, self.decl, self.topo,
                                      s.scale_exp, s.groups)
        empty_inv = Invalidation(version=0, items=(), nodes=(), steps=())
        manifest = self._snapshot("batch_0000", artifacts, artifacts.keys())
        self._dispatch_batch(
            "batch_0000", 0, 0, self.state, artifacts, upsert, {}, empty_inv,
            bounds, {nd.node_id for nd in self.decl.nodes})
        self.dispatch_log.append({"batch_id": "batch_0000",
                                  "upsert": {k: sorted(v)
                                             for k, v in upsert.items()}})
        report["batches"].append({"batch_id": "batch_0000", "seq": 0,
                                  "affected_items": list(self.topo),
                                  "affected_nodes": sorted(upsert),
                                  "manifest": manifest})
        progress("batch_0000 committed: full dispatch to %d nodes"
                 % len(self.decl.nodes))
        return artifacts

    def _changeset_batches(self, start_idx, artifacts, report, progress):
        """应用 changesets[start_idx-1:]，start_idx 为起始批次序号（>=1）。"""
        s = self.decl.session
        for idx in range(start_idx, len(self.changesets) + 1):
            cs = self.changesets[idx - 1]
            batch_id = "batch_%04d" % idx
            prev_artifacts = artifacts
            next_state = apply_changeset(self.state, cs)
            next_subs = node_subscription_map(next_state)
            inv = compute_invalidation(self.state, cs, next_subs)
            artifacts = compile_all(next_state, s.scale_exp, self.topo)
            self.artifact_history[batch_id] = artifacts

            affected = set(inv.items)
            stable = []
            for key in self.topo:
                if key not in affected:
                    stable.append({
                        "key": key,
                        "identical": prev_artifacts[key][0] == artifacts[key][0]
                        and prev_artifacts[key][1] == artifacts[key][1],
                    })
            report["byte_stability"].append(
                {"batch_id": batch_id, "unchanged_items": stable})

            upsert = {}
            remove = {}
            for nid in inv.nodes:
                keys = [k for k in inv.items if k in next_subs.get(nid, [])]
                for ch in cs.changes:
                    if nid in ch.subscribers_add and ch.key not in keys:
                        keys.append(ch.key)
                if keys:
                    upsert[nid] = keys
                rm = [ch.key for ch in cs.changes
                      if nid in ch.subscribers_remove]
                if rm:
                    remove[nid] = rm
            bounds = compute_error_bounds(next_state, self.decl, self.topo,
                                          s.scale_exp, s.groups)
            manifest = self._snapshot(batch_id, artifacts, inv.items)
            records = self._dispatch_batch(
                batch_id, idx, cs.version, next_state, artifacts, upsert,
                remove, inv, bounds, set(inv.nodes))
            self.dispatch_log.append({
                "batch_id": batch_id,
                "upsert": {k: sorted(v) for k, v in upsert.items()},
                "remove": remove})
            self.state = next_state
            report["batches"].append({
                "batch_id": batch_id, "seq": idx,
                "changeset_version": cs.version,
                "affected_items": list(inv.items),
                "affected_nodes": list(inv.nodes),
                "invalidation": json.loads(inv.canonical().decode("utf-8")),
                "receipts": {nid: records[nid]["source"]
                             for nid in sorted(records)},
                "manifest": manifest,
            })
            progress("%s committed: %d items, %d nodes invalidated"
                     % (batch_id, len(inv.items), len(inv.nodes)))
        return artifacts

    def run(self, upto=None, progress=print):
        """从头完整执行：初始全量批次 + 依序应用前 upto 个变更集。"""
        report = {"batches": [], "byte_stability": []}
        artifacts = self._initial_batch(report, progress)
        saved = self.changesets
        if upto is not None:
            self.changesets = saved[:upto]
        try:
            self._changeset_batches(1, artifacts, report, progress)
        finally:
            self.changesets = saved
        with open(self.state_path, "wb") as fh:
            fh.write(self.state.canonical())
        return report

    def resume(self, progress=print):
        """强制终止后的恢复：依据已提交记录续跑剩余变更集。

        返回 (report, recovery_info)。未提交批次中的节点标记为 unknown。
        """
        info = self.ledger.recovery_scan()
        if info.unknown_nodes:
            progress("recovery: uncommitted batch nodes marked UNKNOWN: %s"
                     % ",".join(info.unknown_nodes))
            self.ledger.clear_inflight()
        cursor = info.cursor  # 最近已提交批次序号
        self.state = initial_state(self.decl)
        for cs in self.changesets[:cursor]:
            self.state = apply_changeset(self.state, cs)
        s = self.decl.session
        artifacts = compile_all(self.state, s.scale_exp, self.topo)
        self.artifact_history["batch_%04d" % max(cursor, 0)] = artifacts
        report = {"batches": [], "byte_stability": []}
        if cursor < 0:
            artifacts = self._initial_batch(report, progress)
            cursor = 0
        self._changeset_batches(cursor + 1, artifacts, report, progress)
        with open(self.state_path, "wb") as fh:
            fh.write(self.state.canonical())
        return report, info

    # -- 独立校验 -----------------------------------------------------------

    def full_rebuild_compare(self):
        """从零全量重建最终状态，逐项比对产物字节与逐节点期望摘要。"""
        state = initial_state(self.decl)
        for cs in self.changesets:
            state = apply_changeset(state, cs)
        s = self.decl.session
        topo = topo_order(list(state.items),
                          {k: it.derived_from for k, it in state.items.items()})
        rebuilt = compile_all(state, s.scale_exp, topo)
        final = self.artifact_history["batch_%04d" % len(self.changesets)]
        diffs = [key for key in topo if rebuilt[key][0] != final[key][0]]
        node_digests = {}
        node_subs = node_subscription_map(state)
        for node in self.decl.nodes:
            view = expected_view(state.items, node_subs.get(node.node_id, []),
                                 rebuilt, s.scale_exp)
            node_digests[node.node_id] = sample.view_digest(
                view, node.seed, node.sample_size, s.scale_exp, s.groups)
        return {"item_diffs": diffs, "node_digests": node_digests}
