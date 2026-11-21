"""连接槽位资源管理：建立、复用、借用、空闲回收与水位统计。

连接状态机：established -> active <-> idle -> released。
所有状态迁移在内部锁保护下完成，可被多线程调用；时间戳全部来自
注入的 Clock（生产为真实单调时钟，测试为虚拟时钟）。
"""

from dataclasses import dataclass, field
from threading import RLock

from .errors import BorrowCycleError, DeadlineError, EngineError

ST_ESTABLISHED = "established"
ST_ACTIVE = "active"
ST_IDLE = "idle"
ST_RELEASED = "released"


@dataclass
class Connection:
    conn_id: str
    owner_partition: int
    home_partition: int
    is_borrowed: bool
    key_cap: int
    state: str = ST_ESTABLISHED
    keys: set = field(default_factory=set)
    established_ms: int = 0
    last_active_ms: int = 0
    peak_keys: int = 0
    batch_count: int = 0
    borrow_deadline_ms: int = None
    lender: int = None
    unknown: bool = False
    subpartition: int = -1

    def to_ledger(self):
        return {
            "conn_id": self.conn_id,
            "owner_partition": self.owner_partition,
            "home_partition": self.home_partition,
            "is_borrowed": int(self.is_borrowed),
            "key_cap": self.key_cap,
            "state": self.state,
            "key_count": len(self.keys),
            "peak_keys": self.peak_keys,
            "established_ms": self.established_ms,
            "last_active_ms": self.last_active_ms,
            "borrow_deadline_ms": self.borrow_deadline_ms,
            "lender": self.lender,
            "unknown": int(self.unknown),
            "subpartition": self.subpartition,
        }


class SlotManager:
    def __init__(self, topology, clock, inject_cycle=False, inject_deadline=False,
                 release_on_key_cap=False, max_wait_ms=60000):
        self.topology = topology
        self.clock = clock
        self.release_on_key_cap = release_on_key_cap
        self.max_wait_ms = max_wait_ms
        self.lock = RLock()
        self.connections = {}
        self.partition_conns = {p: [] for p in range(topology.partition_count)}
        self.borrows = {b.borrower: b for b in topology.borrows}
        self.seq = 0
        self.slot_seq = {}
        self.peak_active = 0
        self.id_offset = 0
        self.wait_events = []
        self.inject_cycle = inject_cycle
        self.inject_deadline = inject_deadline
        if inject_cycle:
            self._raise_cycle()

    def _raise_cycle(self):
        edges = sorted((b.borrower, b.lender) for b in self.topology.borrows)
        raise BorrowCycleError(
            "注入异常：借用关系成环",
            "topology.borrows",
            {"edges": edges},
        )

    def _new_id(self, partition, sub):
        self.seq += 1
        global_index = self.seq + self.id_offset
        suffix = f"-s{sub}" if sub >= 0 else "-s-"
        return f"conn-p{partition}{suffix}slot{global_index:04d}"

    def _active_count_locked(self):
        return sum(
            1 for c in self.connections.values()
            if c.state in (ST_ACTIVE, ST_IDLE)
        )

    def _lender_load_locked(self, lender):
        return sum(
            len(c.keys) for c in self.connections.values()
            if c.home_partition == lender and c.state != ST_RELEASED
        )

    def _establish_locked(self, owner, home, cap, borrowed, deadline, sub):
        conn_id = self._new_id(owner, sub)
        now = self.clock.now_ms()
        conn = Connection(
            conn_id=conn_id, owner_partition=owner, home_partition=home,
            is_borrowed=borrowed, key_cap=cap, state=ST_ESTABLISHED,
            established_ms=now, last_active_ms=now,
            borrow_deadline_ms=deadline,
            lender=home if borrowed else None,
            subpartition=sub,
        )
        self.connections[conn_id] = conn
        self.partition_conns[owner].append(conn)
        return conn

    def _try_establish_local(self, partition, pspec, sub):
        with self.lock:
            owned_active = sum(
                1 for c in self.partition_conns[partition]
                if c.state != ST_RELEASED and not c.is_borrowed
            )
            if owned_active >= pspec.max_connections:
                return None
            if self._active_count_locked() >= self.topology.global_max_connections:
                return None
            return self._establish_locked(
                partition, partition, pspec.key_cap_per_connection,
                False, None, sub,
            )

    def _try_borrow(self, partition, sub, global_full=False):
        borrow = self.borrows.get(partition)
        if borrow is None:
            return None
        with self.lock:
            lender_spec = self.topology.partitions[borrow.lender]
            lender_home_active = sum(
                1 for c in self.partition_conns[borrow.lender]
                if c.state != ST_RELEASED and not c.is_borrowed
            )
            lender_borrowed_out = sum(
                1 for c in self.connections.values()
                if c.state != ST_RELEASED and c.is_borrowed and c.lender == borrow.lender
            )
            borrower_already = any(
                c.state != ST_RELEASED and c.is_borrowed
                for c in self.partition_conns[partition]
            )
            if borrower_already:
                return None
            if lender_home_active + lender_borrowed_out >= lender_spec.max_connections:
                return None
            if not global_full and self._active_count_locked() >= self.topology.global_max_connections:
                return None
            now = self.clock.now_ms()
            deadline = now + borrow.deadline_ms
            if self.inject_deadline:
                deadline = now - 1
            if deadline <= now:
                raise DeadlineError(
                    "注入异常：归还截止早于借用时刻",
                    f"topology.borrows[partition={partition}]",
                    {"borrow_ms": now, "deadline_ms": deadline},
                )
            return self._establish_locked(
                partition, borrow.lender, borrow.borrowed_key_cap,
                True, deadline, sub,
            )

    def acquire(self, partition, key, sub=-1):
        """为 (分区, 子分区, 分组键) 获取可承载该键的连接。

        同键复用同一连接；键达上限或资源不足时建立新连接；
        全局并发耗尽且无法借用时排队等待，记录单调时钟等待区间。
        """
        pspec = self.topology.partitions[partition]
        start_wait = None
        while True:
            with self.lock:
                self.recycle_locked()
                for conn in self.partition_conns[partition]:
                    if conn.state == ST_RELEASED or conn.unknown:
                        continue
                    if conn.subpartition != sub:
                        continue
                    if key in conn.keys:
                        self._activate_locked(conn)
                        return conn
                candidate = None
                for conn in self.partition_conns[partition]:
                    if conn.state == ST_RELEASED or conn.unknown:
                        continue
                    if conn.subpartition != sub:
                        continue
                    if len(conn.keys) < conn.key_cap:
                        candidate = conn
                        break
                if candidate is not None:
                    candidate.keys.add(key)
                    candidate.peak_keys = max(candidate.peak_keys, len(candidate.keys))
                    self._activate_locked(candidate)
                    return candidate
            established = self._try_establish_local(partition, pspec, sub)
            if established is None:
                global_full = self._active_count_locked() >= self.topology.global_max_connections
                established = self._try_borrow(partition, sub, global_full=global_full)
            if established is not None:
                with self.lock:
                    established.keys.add(key)
                    established.peak_keys = 1
                    self._activate_locked(established)
                    return established
            if start_wait is None:
                start_wait = self.clock.now_ms()
            self.clock.sleep_ms(1)
            end_wait = self.clock.now_ms()
            if end_wait - start_wait > self.max_wait_ms:
                raise EngineError(
                    "E_WAIT_TIMEOUT", "连接等待超过最大时长，槽位拓扑可能不可行",
                    f"partitions[{partition}]",
                    {"waited_ms": end_wait - start_wait})
            with self.lock:
                self.wait_events.append({
                    "partition": partition,
                    "subpartition": sub,
                    "wait_start_ms": start_wait,
                    "wait_end_ms": end_wait,
                })

    def _activate_locked(self, conn):
        conn.state = ST_ACTIVE
        conn.last_active_ms = self.clock.now_ms()
        active = self._active_count_locked()
        self.peak_active = max(self.peak_active, active)

    def mark_idle(self, conn_id):
        with self.lock:
            conn = self.connections[conn_id]
            if conn.state == ST_ACTIVE:
                conn.state = ST_IDLE
                conn.last_active_ms = self.clock.now_ms()

    def release(self, conn_id):
        with self.lock:
            conn = self.connections[conn_id]
            if conn.state != ST_RELEASED:
                conn.state = ST_RELEASED
                conn.keys.clear()

    def recycle_locked(self):
        """空闲超时或键达上限的连接真实释放（active 计数随之减少）。"""
        now = self.clock.now_ms()
        for conn in list(self.connections.values()):
            if conn.state == ST_RELEASED:
                continue
            if conn.is_borrowed and conn.borrow_deadline_ms is not None \
                    and now >= conn.borrow_deadline_ms:
                self.release(conn.conn_id)
                continue
            if len(conn.keys) >= conn.key_cap:
                self.release(conn.conn_id)
                continue
            if conn.state == ST_IDLE:
                idle_limit = self.topology.partitions[conn.home_partition].idle_recycle_ms
                if now - conn.last_active_ms >= idle_limit:
                    self.release(conn.conn_id)

    def mark_unknown(self):
        """强制终止后：运行中/借用中连接标记未知，要求重跑。"""
        with self.lock:
            for conn in self.connections.values():
                if conn.state != ST_RELEASED:
                    conn.unknown = True

    def watermark_snapshot(self):
        with self.lock:
            self.recycle_locked()
            return {
                "peak_active_connections": self.peak_active,
                "global_max_connections": self.topology.global_max_connections,
                "active_connections": self._active_count_locked(),
                "connections": [
                    c.to_ledger() for c in sorted(
                        self.connections.values(), key=lambda x: x.conn_id
                    )
                ],
                "wait_events": list(self.wait_events),
            }
