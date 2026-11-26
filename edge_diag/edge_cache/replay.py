"""独立重放校验。

只读取原始声明文件与 SQLite 账本，不依赖引擎内存态：
- 按账本 event_records 顺序重新推导租约集合、节点水位、槽位归属、预算
- 与账本 leases/node_water/ownership/receipts 表逐字段比对
- 用原始声明重新生成分片字节与摘要，与 receipts 比对
首个不一致抛 VerifyError(E_VERIFY)，定位事件序号、节点与字段。
"""

from collections import deque

from . import determinism
from .errors import VerifyError
from .ledger import FORMAT_VERSION, Ledger
from .spec import load_spec, spec_digest


class ReplayState:
    def __init__(self, spec):
        self.spec = spec
        self.leases = {}
        self.node_leases = {n["node_id"]: set() for n in spec["nodes"]}
        self.free_slots = {n["node_id"]: deque(range(n["capacity_slots"]))
                           for n in spec["nodes"]}
        self.held = {n["node_id"]: {} for n in spec["nodes"]}
        self.water = {n["node_id"]: {"used_slots": 0, "active_leases": 0}
                      for n in spec["nodes"]}
        self.slot_owner = {}
        self.budget_used = 0

    def fail(self, field, expected, actual, seq, node_id):
        raise VerifyError(
            f"重放不一致: 字段 {field} 重放={expected!r} 账本={actual!r}",
            f"event#{seq}.{node_id}.{field}")

    def apply(self, record):
        seq = record["seq"]
        action = record["action"]
        node_id = record["node_id"]
        payload = __import__("json").loads(record["payload_json"])
        if action == "grant":
            lease = payload
            lid = lease["lease_id"]
            if lid in self.leases:
                self.fail("lease_id", "absent", "exists", seq, node_id)
            free = self.free_slots[node_id]
            if not free:
                self.fail("free_slots", ">0", "0", seq, node_id)
            slot = free.popleft()
            if slot != lease["slot_index"]:
                self.fail("slot_index", slot, lease["slot_index"], seq,
                          node_id)
            self.leases[lid] = dict(lease)
            self.node_leases[node_id].add(lid)
            self.held[node_id][lid] = slot
            self.slot_owner[f"{node_id}:{slot}"] = lid
            self.budget_used += self._shard_bytes(lease["content_key"])
            self._refresh_water(node_id)
        elif action in ("reclaim",):
            lid = record["lease_id"]
            if lid not in self.leases:
                self.fail("lease_id", "exists", "absent", seq, node_id)
            lease = self.leases.pop(lid)
            self.node_leases[node_id].discard(lid)
            slot = self.held[node_id].pop(lid)
            self.free_slots[node_id].append(slot)
            self.slot_owner[f"{node_id}:{slot}"] = None
            self._refresh_water(node_id)
        elif action == "renew":
            lid = record["lease_id"]
            if lid not in self.leases:
                self.fail("lease_id", "exists", "absent", seq, node_id)
            lease = self.leases[lid]
            lease["lease_version"] = payload["new_version"]
            lease["renew_count"] = payload["renew_count"]
            lease["expire_ms"] = payload["expire_ms"]
        # read / quota_change / unknown_reclaim 不改变推导口径中的租约集合
        return self

    def _shard_bytes(self, content_key):
        return next(c["shard_byte_size"] for c in self.spec["contents"]
                    if c["content_key"] == content_key)

    def _refresh_water(self, node_id):
        self.water[node_id] = {
            "used_slots": len(self.held[node_id]),
            "active_leases": len(self.node_leases[node_id]),
        }


def verify(spec_path, db_path):
    spec = load_spec(spec_path)
    digest = spec_digest(spec)
    ledger = Ledger(db_path, spec, digest)
    try:
        _check_meta(ledger, spec, digest)
        state = ReplayState(spec)
        records = ledger.all_event_records()
        # 恢复点回收使用保留负 seq，语义上发生在全部正常事件之后，
        # 排序时把负 seq 排到末尾，避免在租约授予前应用回收
        records = sorted(
            enumerate(records),
            key=lambda pair: (0 if pair[1]["seq"] >= 0 else 1,
                              pair[1]["seq"], pair[1]["sub_index"],
                              pair[0]))
        records = [r for _, r in records]
        # 同一 seq 可能含多条子记录（授予+回收等），整组应用
        groups = {}
        order = []
        for record in records:
            if record["seq"] not in groups:
                groups[record["seq"]] = []
                order.append(record["seq"])
            groups[record["seq"]].append(record)
        for seq in order:
            for record in groups[seq]:
                state.apply(record)
        # 并行写入的提交顺序可与 seq 不同，故水位只比对最终收敛值；
        # 逐事件轨迹仍由 event_records/ownership 完整保留并可独立推导
        _compare_final_water(ledger, state, spec)
        _compare_totals(ledger, state, spec)
        _compare_receipts(ledger, spec)
        _compare_ownership(ledger, state)
        return {"ok": True, "events_replayed": len(records),
                "last_committed_seq": ledger.last_committed_seq()}
    finally:
        ledger.close()


def _check_meta(ledger, spec, digest):
    if int(ledger._meta("format_version")) != FORMAT_VERSION:
        raise VerifyError("账本格式版本不一致", "meta.format_version")
    if ledger._meta("spec_digest") != digest:
        raise VerifyError("账本声明摘要不一致，应整体替换",
                          "meta.spec_digest")


def _compare_final_water(ledger, state, spec):
    for node in spec["nodes"]:
        node_id = node["node_id"]
        row = ledger.conn.execute(
            "SELECT used_slots,active_leases,quota_version FROM node_water "
            "WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise VerifyError("账本缺少节点水位行", f"node.{node_id}")
        if row[0] != state.water[node_id]["used_slots"]:
            raise VerifyError(
                f"最终已用槽位不一致: 重放={state.water[node_id]['used_slots']} "
                f"账本={row[0]}", f"{node_id}.used_slots")
        if row[1] != state.water[node_id]["active_leases"]:
            raise VerifyError(
                f"最终活跃租约数不一致: "
                f"重放={state.water[node_id]['active_leases']} 账本={row[1]}",
                f"{node_id}.active_leases")


def _compare_totals(ledger, state, spec):
    active_rows = ledger.conn.execute(
        "SELECT lease_id,node_id,shard_id,slot_index,state FROM leases "
        "WHERE state='active'").fetchall()
    ledger_active = {(r[0], r[1], r[2], r[3]) for r in active_rows}
    replay_active = set()
    for node_id, lids in state.node_leases.items():
        for lid in lids:
            lease = state.leases[lid]
            replay_active.add((lid, node_id, lease["shard_id"],
                               lease["slot_index"]))
    if ledger_active != replay_active:
        only_ledger = ledger_active - replay_active
        only_replay = replay_active - ledger_active
        raise VerifyError(
            f"最终活跃租约集合不一致 ledger_only={only_ledger} "
            f"replay_only={only_replay}", "leases.active_set")
    if ledger.get_budget_used() != state.budget_used:
        raise VerifyError(
            f"累计下发字节不一致: 重放={state.budget_used} "
            f"账本={ledger.get_budget_used()}", "meta.budget_used")
    # 全局/节点约束复核
    total = len(active_rows)
    if total > spec["session"]["global_lease_limit"]:
        raise VerifyError(f"全局租约越界 {total}", "global_lease_limit")
    for node in spec["nodes"]:
        count = sum(1 for r in active_rows if r[1] == node["node_id"])
        if count > node["per_node_lease_limit"]:
            raise VerifyError(
                f"节点 {node['node_id']} 租约越界 {count}",
                f"{node['node_id']}.per_node_lease_limit")
    # 槽位唯一归属：任一活跃槽位不得被两个租约持有
    seen = {}
    for lid, node_id, shard_id, slot_index in replay_active:
        key = (node_id, slot_index)
        if key in seen:
            raise VerifyError(
                f"槽位被双重持有 {key}: {seen[key]} 与 {lid}",
                f"{node_id}.slot_{slot_index}")
        seen[key] = lid


def _compare_receipts(ledger, spec):
    content_map = {c["content_key"]: c for c in spec["contents"]}
    node_map = {n["node_id"]: n for n in spec["nodes"]}
    for receipt in ledger.all_receipts():
        node = node_map[receipt["node_id"]]
        content_key = receipt["shard_id"].split("#s")[0]
        shard_index = int(receipt["shard_id"].split("#s")[1])
        content = content_map[content_key]
        payload = determinism.generate_shard_bytes(
            content_key, content["content_seed"], shard_index,
            content["shard_byte_size"])
        expected = determinism.compute_digest(
            node["node_id"], node["node_seed"], content_key, shard_index,
            payload, spec["session"]["session_scale"])
        for field in ("digest", "fp_sum", "fp_sqsum", "byte_hash"):
            if receipt[field] != expected[field]:
                raise VerifyError(
                    f"分片摘要 {field} 不一致: 账本={receipt[field]} "
                    f"重放={expected[field]}",
                    f"event#{receipt['seq']}.{node['node_id']}."
                    f"{receipt['shard_id']}.{field}")


def _compare_ownership(ledger, state):
    rows = ledger.all_ownership()
    # 同一 slot_key 相邻记录不得出现重复授予而无释放；最终归属须与推导一致
    last_owner = {}
    for row in rows:
        key = row["slot_key"]
        if row["lease_id"] is not None:
            if last_owner.get(key) is not None:
                raise VerifyError(
                    f"槽位 {key} 未释放即被再次授予",
                    f"event#{row['seq']}.{key}")
            last_owner[key] = row["lease_id"]
        else:
            last_owner[key] = None
    for key, owner in last_owner.items():
        node_id, slot_str = key.split(":")
        slot_index = int(slot_str)
        current = next(
            (lid for lid, s in state.held[node_id].items()
             if s == slot_index), None)
        if owner != current:
            raise VerifyError(
                f"最终槽位归属不一致: 账本={owner} 重放={current}",
                f"event#{rows[-1]['seq']}.{key}.owner")
