"""下发编排引擎。

关键不变量：
1. 三重约束：全局活跃租约数 <= global_lease_limit；单节点活跃租约数 <=
   per_node_lease_limit 且占用槽位 <= capacity_slots；累计下发字节 <=
   global_byte_budget（预算授予时预占、不返还）。
2. 槽位唯一归属：slot_key=(node_id, slot_index) 同一时刻至多被一个活跃
   租约持有；到期回收与配额回收互斥，先到者记录唯一裁决依据。
3. 到期只由真实单调时钟判定（now >= expire_ms），不使用 sleep/线程超时。
4. 工作真实投射到线程池；每个事件的账本变更在主连接单事务原子提交。
"""

import itertools
import json
import subprocess
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from . import determinism
from .clock import MonoClock
from .errors import (BudgetError, ChecksumError, DiagError, LeaseError,
                     PortError, TimeoutError)
from .ledger import Ledger, LedgerWriter
from .ledger_queue import LedgerThread
from .spec import spec_digest as determinism_id
from .protocol import LineConnection, connect_loopback

LEASE_ID_SEQ = itertools.count(1)
BATCH_ID_SEQ = itertools.count(1)


def _numeric_suffix(value, prefix):
    return int(value[len(prefix):]) if value.startswith(prefix) else 0


class NodeProcess:
    def __init__(self, spec, contents, session_scale, python_exe=None):
        self.spec = spec
        self.contents = contents
        self.session_scale = session_scale
        self.python_exe = python_exe or sys.executable
        self.proc = None
        self._conn_lock = threading.Lock()

    def start(self, spec_path):
        cmd = [self.python_exe, "-m", "edge_cache.node",
               "--spec", str(spec_path), "--node-id", self.spec["node_id"]]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise PortError(
                f"节点 {self.spec['node_id']} 启动失败: {err}",
                f"{self.spec['host']}:{self.spec['port']}")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PortError(f"节点启动行无法解析: {line!r}",
                            self.spec["node_id"]) from exc
        if msg.get("event") == "error":
            raise PortError(
                f"节点端口绑定失败: {msg.get('message')}",
                f"{self.spec['host']}:{self.spec['port']}")
        if msg.get("event") != "ready":
            raise PortError(f"节点未就绪: {msg}", self.spec["node_id"])

    def connect(self):
        sock = connect_loopback(self.spec["host"], self.spec["port"])
        return LineConnection(sock)

    def request(self, message):
        with self._conn_lock:
            conn = self.connect()
            try:
                conn.send_message(message)
                return conn.read_message()
            finally:
                conn.close()

    def force_kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.request({"type": "shutdown"})
            except (PortError, TimeoutError):
                self.force_kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.force_kill()


class Scheduler:
    """三重约束闸门。等待队列记录真实等待区间，按约束放行。"""

    def __init__(self, spec, clock):
        self.spec = spec
        self.clock = clock
        self.global_limit = spec["session"]["global_lease_limit"]
        self.byte_budget = spec["session"]["global_byte_budget"]
        self._cond = threading.Condition()
        self._waiters = deque()
        self._active_global = 0
        self._node_active = {n["node_id"]: 0 for n in spec["nodes"]}
        self._budget_used = 0

    def restore(self, snapshot):
        with self._cond:
            self._active_global = len(snapshot["leases"])
            for lease in snapshot["leases"]:
                self._node_active[lease["node_id"]] += 1
            self._budget_used = snapshot["budget_used"]

    def budget_used(self):
        with self._cond:
            return self._budget_used

    def try_reserve_budget(self, byte_size):
        with self._cond:
            if self._budget_used + byte_size > self.byte_budget:
                return False
            self._budget_used += byte_size
            return True

    def release_active(self, node_id):
        with self._cond:
            self._active_global -= 1
            self._node_active[node_id] -= 1
            self._pump_locked()

    def acquire(self, node_id, node_limit):
        """阻塞获取并发名额；等待区间真实发生并由调用方记账。"""
        gate = threading.Event()
        with self._cond:
            self._waiters.append((node_id, node_limit, gate))
            self._pump_locked()
        gate.wait()

    def _pump_locked(self):
        # 只按三重约束裁决，不做队头阻塞：前面的等待者不满足时，
        # 后续满足条件者仍可放行，避免持名额任务被排队任务反向饿死。
        idx = 0
        while idx < len(self._waiters):
            node_id, node_limit, gate = self._waiters[idx]
            if (self._active_global < self.global_limit
                    and self._node_active[node_id] < node_limit):
                self._active_global += 1
                self._node_active[node_id] += 1
                del self._waiters[idx]
                gate.set()
            else:
                idx += 1


class Engine:
    def __init__(self, spec, spec_path, db_path, clock=None,
                 workers=None, python_exe=None):
        self.spec = spec
        self.spec_path = str(spec_path)
        self.clock = clock or MonoClock()
        self.session_scale = spec["session"]["session_scale"]
        self.content_map = {c["content_key"]: c for c in spec["contents"]}
        self.node_map = {n["node_id"]: n for n in spec["nodes"]}
        self.db_path = db_path
        self.spec_digest_value = determinism_id(spec)
        self.ledger_thread = LedgerThread(
            lambda: Ledger(db_path, spec, self.spec_digest_value))
        self.ledger = self.ledger_thread.ledger
        self.writer = LedgerWriter(self.ledger)
        self.scheduler = Scheduler(spec, self.clock)
        # 池容量须容纳全部在途写入并保留余量，避免排队任务占满线程后
        # 持名额任务无法推进而线程饥饿
        self.workers = workers or max(
            8, len(spec["events"]) + 2, len(spec["nodes"]) * 4)
        self.pool = ThreadPoolExecutor(max_workers=self.workers,
                                       thread_name_prefix="dispatch")
        self.python_exe = python_exe
        self.nodes = {}
        self._state_lock = threading.RLock()
        # 内存态：与账本保持一致，仅缓存已提交事实
        self.leases = {}
        self.node_leases = {n["node_id"]: set() for n in spec["nodes"]}
        self.free_slots = {n["node_id"]: deque(range(n["capacity_slots"]))
                           for n in spec["nodes"]}
        self.held_slots = {n["node_id"]: {} for n in spec["nodes"]}
        self.quota_version = {n["node_id"]: n["quota_version"]
                              for n in spec["nodes"]}
        self.results = []
        self.unknown_batches = []


    def _tx(self, fn):
        """在专用账本线程内运行单事务 fn(conn)，成功提交、异常回滚。"""
        def job():
            conn = self.ledger.conn
            self.writer.begin()
            try:
                value = fn(conn)
                self.writer.commit()
                return value
            except Exception:
                self.writer.rollback()
                raise
        return self.ledger_thread.submit(job)

    # ---------- 进程编排 ----------
    def start_nodes(self, skip=None):
        skip = skip or set()
        for node_spec in self.spec["nodes"]:
            if node_spec["node_id"] in skip:
                continue
            node = NodeProcess(node_spec, self.spec["contents"],
                               self.session_scale, self.python_exe)
            node.start(self.spec_path)
            self.nodes[node_spec["node_id"]] = node

    def run_until(self, seq):
        """运行并汇合 seq 及之前的全部写入，停在清晰提交边界。"""
        inflight = {}
        ordered = deque()
        for event in self.spec["events"]:
            if event["seq"] > seq:
                break
            if event["type"] != "write":
                continue
            future = self.handle_write(event)
            inflight[event["seq"]] = future
            ordered.append(event)
        self._drain_all(inflight, ordered)
        return {"crashed_at": seq,
                "committed": [r["seq"] for r in self.results
                              if r.get("action") == "grant"],
                "last_committed_seq": self.ledger_thread.submit(
                    self.ledger.last_committed_seq)}

    def stop_nodes(self):
        for node in self.nodes.values():
            node.stop()

    def force_kill_node(self, node_id):
        self.nodes[node_id].force_kill()

    def restart_node(self, node_id):
        node = NodeProcess(self.node_map[node_id], self.spec["contents"],
                           self.session_scale, self.python_exe)
        node.start(self.spec_path)
        self.nodes[node_id] = node
        return node

    # ---------- 恢复 ----------
    def recover(self):
        """依据完整提交记录重建内存态并向节点回放租约表。"""
        self._sync_id_counters()
        snap = self.ledger_thread.submit(self.ledger.snapshot)
        self.scheduler.restore(snap)
        for lease in snap["leases"]:
            self.leases[lease["lease_id"]] = dict(lease)
            self.node_leases[lease["node_id"]].add(lease["lease_id"])
            self.free_slots[lease["node_id"]] = deque(
                s for s in self.free_slots[lease["node_id"]]
                if s != lease["slot_index"])
            self.held_slots[lease["node_id"]][lease["lease_id"]] = \
                lease["slot_index"]
        self.unknown_batches = list(snap["unknown"])
        last_seq = self.ledger_thread.submit(self.ledger.last_committed_seq)
        # 恢复点真实时钟裁决：跨进程停机期间已到期的租约显式回收入账，
        # 使账本、内存与重放三者收敛到同一结论。使用保留负 seq 段，避免
        # 与正常事件的 (seq,sub_index) 主键冲突。
        recovery_seq = -1000 - last_seq
        recovered_expired = self.sweep_expired(recovery_seq)
        # 未确认批次：先回收其可能占用，再要求重跑
        for item in snap["unknown"]:
            if item["resolved"] != "unknown":
                continue
            self._recover_unknown_occupancy(item)
        # 节点租约表回放（崩溃后节点为空）
        by_node = {}
        for lease in snap["leases"]:
            by_node.setdefault(lease["node_id"], []).append({
                "lease_id": lease["lease_id"],
                "shard_id": lease["shard_id"],
                "quota_version": lease["quota_version"],
                "grant_ms": lease["grant_ms"],
                "expire_ms": lease["expire_ms"],
                "renew_count": lease["renew_count"],
                "lease_version": lease["lease_version"],
            })
        self._recovered_expired = recovered_expired
        for node_id, leases in by_node.items():
            if node_id in self.nodes:
                self.nodes[node_id].request(
                    {"type": "restore", "leases": leases})
        return last_seq, snap

    def _recover_unknown_occupancy(self, item):
        node_id = item["node_id"]
        slot_index = item.get("slot_index")

        def tx(conn):
            if slot_index is not None and slot_index not in \
                    self.held_slots[node_id].values():
                self.free_slots[node_id].append(slot_index)
            self.writer.record_event(
                conn, item["seq"], 900, "unknown_reclaim", item,
                node_id=node_id, shard_id=item["shard_id"],
                reason="unknown_batch_preempt",
                basis="recovery: 未完整提交批次，先回收占用后要求重跑")
            self.writer.resolve_unknown(conn, item["batch_id"],
                                        "reclaimed_rerun")
            self.writer.advance(conn, item["seq"])
        self._tx(tx)

    def _expired_leases(self, now_ms):
        return [lid for lid, lease in self.leases.items()
                if lease["state"] == "active" and now_ms >= lease["expire_ms"]]

    def sweep_expired(self, seq, sub_offset=0):
        """真实单调时钟裁决到期：now >= expire_ms 即回收，无任何 sleep。"""
        now_ms = self.clock.now_ms()
        expired = self._expired_leases(now_ms)
        for sub, lease_id in enumerate(expired):
            self._reclaim(lease_id, reason="expired",
                          basis=f"monotonic now_ms={now_ms} >= "
                                f"expire_ms={self.leases[lease_id]['expire_ms']}",
                          seq=seq, sub_index=sub + sub_offset,
                          change_ms=now_ms)
        return len(expired)

    def _reclaim(self, lease_id, reason, basis, seq, sub_index=0,
                 change_ms=None, new_quota_version=None):
        """唯一归属回收：同一槽位只记录一次归属变化，先到先得。

        幂等：若租约已被到期/配额路回收，直接返回 None，不重复记账、
        不重复释放名额，从而保证双路回收的唯一归属。
        """
        with self._state_lock:
            lease = self.leases.get(lease_id)
            if lease is None or lease["state"] != "active":
                return None
            node_id = lease["node_id"]
            slot_index = lease["slot_index"]
        change_ms = change_ms if change_ms is not None else self.clock.now_ms()

        def tx(conn):
            with self._state_lock:
                if self.leases.get(lease_id, {}).get("state") != "active":
                    return False
                lease["state"] = "reclaimed:" + reason
                self.writer.upsert_lease(conn, lease)
                slot_key = f"{node_id}:{slot_index}"
                self.writer.insert_ownership(
                    conn, slot_key, None, node_id, slot_index, None,
                    reason, basis, seq,
                    new_quota_version if new_quota_version is not None
                    else lease["quota_version"], change_ms)
                self.writer.record_event(
                    conn, seq, sub_index, "reclaim",
                    {"lease_id": lease_id, "reason": reason, "basis": basis},
                    node_id=node_id, shard_id=lease["shard_id"],
                    lease_id=lease_id, reason=reason, basis=basis)
                self.held_slots[node_id].pop(lease_id, None)
                self.free_slots[node_id].append(slot_index)
                self.node_leases[node_id].discard(lease_id)
                water = self._water(node_id)
                if new_quota_version is not None:
                    self.writer.update_water(conn, node_id,
                                             water["used_slots"],
                                             water["active_leases"],
                                             new_quota_version)
                else:
                    self.writer.update_water(conn, node_id,
                                             water["used_slots"],
                                             water["active_leases"])
                self.writer.advance(conn, seq)
                return True
        reclaimed = self._tx(tx)
        if not reclaimed:
            return None
        node = self.nodes.get(node_id)
        if node is not None:
            try:
                node.request(
                    {"type": "reclaim", "lease_id": lease_id,
                     "reason": reason})
            except (PortError, TimeoutError):
                pass
        self.scheduler.release_active(node_id)
        return lease

    def _water(self, node_id):
        active = len(self.node_leases[node_id])
        used = len(self.held_slots[node_id])
        return {"used_slots": used, "active_leases": active}

    def _next_lease_id(self):
        return f"lease-{next(LEASE_ID_SEQ):06d}"

    def _next_batch_id(self):
        return f"batch-{next(BATCH_ID_SEQ):06d}"

    def _sync_id_counters(self):
        """恢复后把租约/批次计数器推进到账本最大值之后，避免复用 id。"""
        def query(conn):
            max_lease = conn.execute(
                "SELECT MAX(lease_id) FROM leases").fetchone()[0]
            max_batch = conn.execute(
                "SELECT MAX(batch_id) FROM unknown_batches").fetchone()[0]
            return (max_lease, max_batch)
        max_lease, max_batch = self.ledger_thread.submit(
            lambda: query(self.ledger.conn))
        lease_n = _numeric_suffix(max_lease, "lease-") if max_lease else 0
        batch_n = _numeric_suffix(max_batch, "batch-") if max_batch else 0
        for _ in range(lease_n):
            next(LEASE_ID_SEQ)
        for _ in range(batch_n):
            next(BATCH_ID_SEQ)

    def handle_write(self, event):
        seq = event["seq"]
        node_id = event["node_id"]
        node_spec = self.node_map[node_id]
        content = self.content_map[event["content_key"]]
        shard_index = event["shard_index"]
        shard_id = determinism.shard_id(event["content_key"], shard_index)
        byte_size = content["shard_byte_size"]

        # 预算闸门：耗尽直接 E_BUDGET 拒绝，不绕过限制
        if not self.scheduler.try_reserve_budget(byte_size):
            raise BudgetError(
                f"全局字节预算不足: 需 {byte_size}，"
                f"已用 {self.scheduler.budget_used()} / "
                f"{self.spec['session']['global_byte_budget']}",
                f"event#{seq}")

        # 真实线程池：工作线程内先排队获取并发名额（等待区间真实记账），
        # 再做账本/TCP I/O，避免“持账本等名额、持名额等账本”的循环等待
        future = self.pool.submit(self._dispatch_worker, event, node_spec,
                                  content, shard_id, byte_size)
        return future

    def _dispatch_worker(self, event, node_spec, content, shard_id, byte_size):
        seq = event["seq"]
        node_id = event["node_id"]
        wait_start = self.clock.now_ms()
        self.scheduler.acquire(node_id, node_spec["per_node_lease_limit"])
        wait_end = self.clock.now_ms()
        try:
            result = self._dispatch_grant(event, node_spec, content, shard_id,
                                          byte_size, wait_start, wait_end)
        except Exception:
            # 授予失败：立即释放并发名额
            self.scheduler.release_active(node_id)
            raise
        return result

    def _dispatch_grant(self, event, node_spec, content, shard_id,
                        byte_size, wait_start, wait_end):
        seq = event["seq"]
        node_id = event["node_id"]
        batch_id = self._next_batch_id()
        grant_ms = self.clock.now_ms()
        expire_ms = grant_ms + event["ttl_ms"]
        lease_id = self._next_lease_id()
        quota_version = self.quota_version[node_id]

        with self._state_lock:
            # 槽位分配（节点容量硬约束），锁内裁决避免多线程双重分配
            if not self.free_slots[node_id]:
                self.scheduler.release_active(node_id)
                raise LeaseError("节点容量槽位耗尽，无法授予",
                                 f"event#{seq}.{node_id}")
            slot_index = self.free_slots[node_id].popleft()

        # 未确认批次先落账：崩溃后据此识别 unknown 并先回收
        try:
            self._tx(lambda conn: (
                self.writer.insert_wait(
                    conn, seq, node_id, shard_id, wait_start),
                self.writer.insert_unknown(
                    conn, batch_id, seq, node_id, shard_id, slot_index,
                    grant_ms)))
        except Exception:
            with self._state_lock:
                self.free_slots[node_id].appendleft(slot_index)
            self.scheduler.release_active(node_id)
            raise

        # 真实 TCP 批次下发（节点内 NumPy 定点校验）
        message = {
            "type": "batch", "batch_id": batch_id,
            "items": [{
                "op": "write",
                "lease_id": lease_id,
                "content_key": event["content_key"],
                "shard_index": event["shard_index"],
                "quota_version": quota_version,
            }]}
        try:
            ack = self.nodes[node_id].request(message)
            receipt = self._validate_receipt(ack, event, node_id)
            grant_ack = self.nodes[node_id].request({
                "type": "grant", "lease_id": lease_id,
                "content_key": event["content_key"],
                "shard_index": event["shard_index"],
                "quota_version": quota_version, "grant_ms": grant_ms,
                "expire_ms": expire_ms, "renew_count": 0,
                "lease_version": 1})
        except Exception:
            with self._state_lock:
                if slot_index not in self.held_slots[node_id].values():
                    self.free_slots[node_id].append(slot_index)
            self.scheduler.release_active(node_id)
            raise
        if not grant_ack.get("ok"):
            with self._state_lock:
                self.free_slots[node_id].append(slot_index)
            self.scheduler.release_active(node_id)
            raise LeaseError(f"节点授予失败: {grant_ack}", f"event#{seq}")

        lease = {
            "lease_id": lease_id, "node_id": node_id,
            "shard_id": shard_id, "content_key": event["content_key"],
            "shard_index": event["shard_index"], "grant_ms": grant_ms,
            "expire_ms": expire_ms, "renew_count": 0, "lease_version": 1,
            "quota_version": quota_version, "slot_index": slot_index,
            "state": "active",
        }

        def tx(conn):
            self.writer.finish_wait(conn, seq, node_id, shard_id, wait_end)
            self.writer.insert_receipt(conn, seq, receipt)
            self.writer.upsert_lease(conn, lease)
            with self._state_lock:
                self.held_slots[node_id][lease_id] = slot_index
                self.node_leases[node_id].add(lease_id)
                water = self._water(node_id)
            self.writer.update_water(conn, node_id, water["used_slots"],
                                     water["active_leases"])
            self.writer.insert_ownership(
                conn, f"{node_id}:{slot_index}", lease_id, node_id,
                slot_index, shard_id, "grant",
                f"event#{seq} 授予，配额版本 {quota_version}", seq,
                quota_version, grant_ms)
            self.writer.record_event(
                conn, seq, 0, "grant", lease, node_id=node_id,
                shard_id=shard_id, lease_id=lease_id,
                quota_version=quota_version, reason="grant",
                basis="scheduler 三重约束通过后授予")
            self.ledger.set_budget_used(conn, self.scheduler.budget_used())
            self.writer.resolve_unknown(conn, batch_id, "committed")
            self.writer.advance(conn, seq)
        self._tx(tx)
        self.leases[lease_id] = lease
        return {"seq": seq, "action": "grant", "lease": lease,
                "receipt": receipt, "wait_ms": wait_end - wait_start}

    def _validate_receipt(self, ack, event, node_id):
        if ack.get("type") != "batch_ack" or not ack.get("receipts"):
            raise ChecksumError(f"节点批次回执异常: {ack}",
                                f"event#{event['seq']}")
        receipt = ack["receipts"][0]
        if not receipt.get("ok"):
            raise ChecksumError(f"节点处理失败: {receipt}",
                                f"event#{event['seq']}")
        # 主进程独立推导同一摘要，逐字段比对，防止节点口径漂移
        content = self.content_map[event["content_key"]]
        payload = determinism.generate_shard_bytes(
            event["content_key"], content["content_seed"],
            event["shard_index"], content["shard_byte_size"])
        expected = determinism.compute_digest(
            node_id, self.node_map[node_id]["node_seed"],
            event["content_key"], event["shard_index"], payload,
            self.session_scale)
        for field in ("digest", "fp_sum", "fp_sqsum", "byte_hash",
                      "sample_count", "session_scale"):
            if receipt.get(field) != expected[field]:
                raise ChecksumError(
                    f"分片摘要字段 {field} 不一致: "
                    f"node={receipt.get(field)} expect={expected[field]}",
                    f"event#{event['seq']}.{node_id}."
                    f"{receipt.get('shard_id')}.{field}")
        return receipt

    def _find_active_lease(self, node_id, shard_id):
        with self._state_lock:
            for lease_id in list(self.node_leases[node_id]):
                lease = self.leases.get(lease_id)
                if lease is None:
                    self.node_leases[node_id].discard(lease_id)
                    continue
                if lease["shard_id"] == shard_id and lease["state"] == "active":
                    return lease
            return None

    def handle_read(self, event):
        seq = event["seq"]
        node_id = event["node_id"]
        shard_id = determinism.shard_id(event["content_key"],
                                        event["shard_index"])
        lease = self._find_active_lease(node_id, shard_id)
        if lease is None:
            raise LeaseError("读取目标不存在活跃租约",
                             f"event#{seq}.{node_id}.{shard_id}")
        ack = self.nodes[node_id].request({
            "type": "batch", "batch_id": self._next_batch_id(),
            "items": [{"op": "read", "lease_id": lease["lease_id"],
                       "content_key": event["content_key"],
                       "shard_index": event["shard_index"],
                       "quota_version": lease["quota_version"]}]})
        receipt = self._validate_receipt(ack, event, node_id)

        def tx_read(conn):
            self.writer.insert_receipt(conn, seq, receipt)
            self.writer.record_event(
                conn, seq, 0, "read",
                {"lease_id": lease["lease_id"], "digest": receipt["digest"]},
                node_id=node_id, shard_id=shard_id,
                lease_id=lease["lease_id"],
                quota_version=lease["quota_version"], reason="read",
                basis="只读校验，不改变归属")
            self.writer.advance(conn, seq)
        self._tx(tx_read)
        return {"seq": seq, "action": "read", "receipt": receipt}

    def handle_renew(self, event):
        seq = event["seq"]
        node_id = event["node_id"]
        shard_id = determinism.shard_id(event["content_key"],
                                        event["shard_index"])
        lease = self._find_active_lease(node_id, shard_id)
        if lease is None:
            raise LeaseError("续约目标不存在活跃租约",
                             f"event#{seq}.{node_id}.{shard_id}")
        expected_version = lease["lease_version"]
        ack = self.nodes[node_id].request({
            "type": "batch", "batch_id": self._next_batch_id(),
            "items": [{"op": "renew", "lease_id": lease["lease_id"],
                       "expected_version": expected_version,
                       "ttl_ms": event["ttl_ms"],
                       "content_key": event["content_key"],
                       "shard_index": event["shard_index"]}]})
        receipt = ack["receipts"][0]
        if not receipt.get("ok"):
            # E_LEASE：版本不符或已到期，拒绝且不延长到期
            raise LeaseError(
                f"续约被节点拒绝: {receipt.get('message')}",
                f"event#{seq}.{node_id}.{shard_id}")
        def tx_renew(conn):
            lease["lease_version"] = receipt["lease_version"]
            lease["renew_count"] = receipt["renew_count"]
            lease["expire_ms"] = receipt["expire_ms"]
            self.writer.upsert_lease(conn, lease)
            self.writer.record_event(
                conn, seq, 0, "renew",
                {"expected_version": expected_version,
                 "new_version": lease["lease_version"],
                 "renew_count": lease["renew_count"],
                 "expire_ms": lease["expire_ms"]},
                node_id=node_id, shard_id=shard_id,
                lease_id=lease["lease_id"],
                quota_version=lease["quota_version"], reason="renew",
                basis=f"乐观版本校验通过 {expected_version}->"
                      f"{lease['lease_version']}")
            self.writer.advance(conn, seq)
        self._tx(tx_renew)
        return {"seq": seq, "action": "renew", "lease": dict(lease)}

    def handle_release_or_cancel(self, event, reason):
        seq = event["seq"]
        node_id = event["node_id"]
        shard_id = determinism.shard_id(event["content_key"],
                                        event["shard_index"])
        lease = self._find_active_lease(node_id, shard_id)
        if lease is None:
            raise LeaseError(f"{reason} 目标不存在活跃租约",
                             f"event#{seq}.{node_id}.{shard_id}")
        result = self._reclaim(
            lease["lease_id"], reason=reason,
            basis=f"event#{seq} 显式 {reason}，归属释放回空闲池", seq=seq)
        return {"seq": seq, "action": reason, "lease_id": lease["lease_id"]}

    def handle_quota_change(self, event):
        seq = event["seq"]
        node_id = event["node_id"]
        new_quota = event["new_quota_version"]
        old_quota = self.quota_version[node_id]
        if new_quota == old_quota:
            raise LeaseError("配额版本必须实际更替",
                             f"event#{seq}.new_quota_version")
        now_ms = self.clock.now_ms()
        # 先做真实到期裁决：到期归属优先落账，避免同槽位被重复计入
        self.sweep_expired(seq, sub_offset=100)
        affected = [lid for lid in list(self.node_leases[node_id])
                    if self.leases[lid]["state"] == "active"
                    and self.leases[lid]["quota_version"] == old_quota]
        for sub, lease_id in enumerate(affected):
            self._reclaim(
                lease_id, reason="quota_recycle",
                basis=f"配额版本换代 {old_quota}->{new_quota}，"
                      f"旧版本活跃租约全部回收",
                seq=seq, sub_index=sub + 1, change_ms=now_ms,
                new_quota_version=new_quota)
        def tx_quota(conn):
            self.quota_version[node_id] = new_quota
            water = self._water(node_id)
            self.writer.update_water(
                conn, node_id, water["used_slots"],
                water["active_leases"], new_quota)
            self.writer.record_event(
                conn, seq, 0, "quota_change",
                {"old": old_quota, "new": new_quota,
                 "recycled": len(affected)}, node_id=node_id,
                quota_version=new_quota, reason="quota_change",
                basis=f"配额版本更替 {old_quota}->{new_quota}")
            self.writer.advance(conn, seq)
        self._tx(tx_quota)
        return {"seq": seq, "action": "quota_change",
                "recycled": len(affected)}

    def run(self, start_after_seq=0, crash_plan=None):
        """顺序消费事件，写入真实并行且不阻塞事件流。

        写请求在三重约束不满足时于工作线程真实排队；主线程继续推进后续
        事件，使释放/配额事件能够解锁排队者。非写事件按目标租约做最小
        汇合，保证它观察到的是不晚于自身 seq 的已提交事实。
        """
        crash_plan = crash_plan or {}
        inflight = {}
        ordered = deque()

        def drain_ready():
            while ordered and inflight[ordered[0]["seq"]].done():
                event = ordered.popleft()
                future = inflight.pop(event["seq"])
                try:
                    self.results.append(future.result())
                except Exception as exc:
                    self.results.append(
                        {"seq": event["seq"], "action": "error",
                         "error": str(exc),
                         "code": getattr(exc, "code", "E_INTERNAL")})

        def wait_seq(seq):
            if seq not in inflight:
                return
            future = inflight.pop(seq)
            ordered.remove(next(e for e in ordered if e["seq"] == seq))
            try:
                self.results.append(future.result())
            except Exception as exc:
                self.results.append(
                    {"seq": seq, "action": "error", "error": str(exc),
                     "code": getattr(exc, "code", "E_INTERNAL")})

        for event in self.spec["events"]:
            if event["seq"] <= start_after_seq:
                continue
            etype = event["type"]
            if etype == "write":
                try:
                    future = self.handle_write(event)
                except DiagError as exc:
                    # 预算/容量类同步拒绝：落账为 rejected，不中断事件流
                    self._record_rejection(event, exc)
                    continue
                inflight[event["seq"]] = future
                ordered.append(event)
                if event["seq"] in crash_plan:
                    self._drain_all(inflight, ordered)
                    self.force_kill_node(crash_plan[event["seq"]])
                    return {"crashed_at": event["seq"],
                            "node_id": crash_plan[event["seq"]]}
                continue

            # 非写事件：先落地所有已完成的在途写入
            drain_ready()
            # 每个事件处理前先做真实到期裁决
            self.sweep_expired(event["seq"], sub_offset=100)
            if etype in ("read", "renew"):
                # 读/续约需要目标租约已就绪：仅汇合同节点同分片在途写入，
                # 不等待无关排队者，避免释放事件被反向阻塞
                target_seq = self._inflight_for(event, inflight)
                if target_seq is not None:
                    wait_seq(target_seq)
                    drain_ready()
                if etype == "read":
                    self._append_guarded(event, self.handle_read)
                else:
                    self._append_guarded(event, self.handle_renew)
            elif etype in ("release", "cancel"):
                # 释放/取消优先执行以解锁排队写入；若目标分片仍在途则先汇合
                target_seq = self._inflight_for(event, inflight)
                if target_seq is not None:
                    wait_seq(target_seq)
                    drain_ready()
                self._append_guarded(
                    event, self.handle_release_or_cancel,
                    "released" if etype == "release" else "cancelled")
                drain_ready()
            elif etype == "quota_change":
                # 配额回收需在该节点全部在途写入落地后裁决
                for seq in self._inflight_for_node(event["node_id"], inflight):
                    wait_seq(seq)
                drain_ready()
                self._append_guarded(event, self.handle_quota_change)
            else:
                raise LeaseError(f"未支持的事件类型 {etype}",
                                 f"event#{event['seq']}")
        self._drain_all(inflight, ordered)
        last_seq = self.spec["events"][-1]["seq"]
        self.sweep_expired(-(last_seq + 1), sub_offset=100)
        return {"crashed_at": None, "results": list(self.results)}

    def _record_rejection(self, event, exc):
        payload = {"seq": event["seq"], "type": event["type"],
                   "code": exc.code, "error": str(exc),
                   "location": exc.location}

        def tx(conn):
            self.writer.record_event(
                conn, event["seq"], 0, "rejected", payload,
                node_id=event.get("node_id"),
                shard_id=determinism.shard_id(
                    event["content_key"], event["shard_index"])
                if "content_key" in event else None,
                reason=exc.code, basis=exc.message)
            self.writer.advance(conn, event["seq"])
        self._tx(tx)
        self.results.append({"seq": event["seq"], "action": "rejected",
                             "type": event["type"], "code": exc.code,
                             "error": str(exc), "location": exc.location})

    def _append_guarded(self, event, handler, *args):
        """业务拒绝(E_LEASE/E_BUDGET 等)转成带码结果并落账，不中断事件流。"""
        try:
            self.results.append(handler(event, *args) if args
                                else handler(event))
        except DiagError as exc:
            self._record_rejection(event, exc)

    def _inflight_for(self, event, inflight):
        want_shard = determinism.shard_id(event["content_key"],
                                          event["shard_index"])
        for seq, future in inflight.items():
            ev = next(e for e in self.spec["events"] if e["seq"] == seq)
            if (ev["node_id"] == event["node_id"]
                    and determinism.shard_id(ev["content_key"],
                                             ev["shard_index"]) == want_shard):
                return seq
        return None

    def _inflight_for_node(self, node_id, inflight):
        return [seq for seq in inflight
                if next(e for e in self.spec["events"]
                        if e["seq"] == seq)["node_id"] == node_id]

    def _drain_all(self, inflight, ordered):
        while ordered:
            event = ordered.popleft()
            future = inflight.pop(event["seq"])
            try:
                self.results.append(future.result())
            except Exception as exc:
                self.results.append(
                    {"seq": event["seq"], "action": "error",
                     "error": str(exc),
                     "code": getattr(exc, "code", "E_INTERNAL")})

    def _await(self, pending):
        for event, future in pending:
            try:
                result = future.result()
            except Exception as exc:
                self.results.append({"seq": event["seq"], "action": "error",
                                     "error": str(exc),
                                     "code": getattr(exc, "code", "E_INTERNAL")})
                continue
            self.results.append(result)

    def water_report(self):
        report = []
        for node in self.spec["nodes"]:
            node_id = node["node_id"]
            active = sorted(
                (self.leases[lid] for lid in self.node_leases[node_id]),
                key=lambda x: x["slot_index"])
            report.append({
                "node_id": node_id,
                "quota_version": self.quota_version[node_id],
                "capacity_slots": node["capacity_slots"],
                "used_slots": self._water(node_id)["used_slots"],
                "per_node_limit": node["per_node_lease_limit"],
                "active_leases": self._water(node_id)["active_leases"],
                "active": [{"lease_id": a["lease_id"],
                            "shard_id": a["shard_id"],
                            "slot_index": a["slot_index"],
                            "expire_ms": a["expire_ms"],
                            "renew_count": a["renew_count"],
                            "lease_version": a["lease_version"]}
                           for a in active],
            })
        return {
            "global_active": sum(self._water(n["node_id"])["active_leases"]
                                 for n in self.spec["nodes"]),
            "global_limit": self.spec["session"]["global_lease_limit"],
            "budget_used": self.scheduler.budget_used(),
            "byte_budget": self.spec["session"]["global_byte_budget"],
            "nodes": report,
            "last_committed_seq": self.ledger_thread.submit(
                self.ledger.last_committed_seq),
        }

    def close(self):
        self.pool.shutdown(wait=True)
        # 连接在账本线程内创建，关闭也必须在该线程执行
        self.ledger_thread.submit(self.ledger.close)
        self.ledger_thread.shutdown()
