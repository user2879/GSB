"""确定性因果排序、快照合成、同 tick 并发写裁决、版本链与压缩。"""

from __future__ import annotations

import os

from .errors import E_BASE_VERSION, E_PEER_GAP, PipelineError
from .model import AdjudicationRecord, Event, Snapshot, content_hash
from .store import Store

COMPACT_THRESHOLD = 8   # 链上快照数超过该阈值触发压缩
COMPACT_KEEP = 4        # 压缩后保留的最新快照数


def apply_op(state: dict, event: Event) -> list[tuple[str, str, object]]:
    """把单个事件应用到状态，返回 [(entity, field, value)] 写入列表。"""
    p = event.payload
    writes = []
    if event.op_type == "spawn":
        entity = p["entity"]
        fields = {"x": p["x"], "y": p["y"]}
        state[entity] = {"kind": p["kind"], "fields": dict(fields)}
        writes.append((entity, "kind", p["kind"]))
        writes.extend((entity, name, value) for name, value in fields.items())
    elif event.op_type == "move":
        entity = p["entity"]
        if entity in state:
            fields = state[entity]["fields"]
            fields["x"] = fields.get("x", 0) + p["dx"]
            fields["y"] = fields.get("y", 0) + p["dy"]
            writes.append((entity, "x", fields["x"]))
            writes.append((entity, "y", fields["y"]))
    elif event.op_type == "destroy":
        entity = p["entity"]
        if entity in state:
            for name in sorted(state[entity]["fields"]):
                writes.append((entity, name, None))
            writes.append((entity, "kind", None))
            del state[entity]
    elif event.op_type == "set_attr":
        entity = p["entity"]
        if entity in state:
            state[entity]["fields"][p["field"]] = p["value"]
            writes.append((entity, p["field"], p["value"]))
    return writes


class Engine:
    def __init__(self, store: Store, *, die_after_commits: int | None = None):
        self.store = store
        self.state: dict = {}
        self.version = 0
        self.expected_seq: dict[str, int] = {}
        self.known_versions = {0}
        self.manifest = {"head": 0, "versions": [
            {"version": 0, "parent": -1, "hash": self._genesis_hash()}]}
        self.adjudications: list[dict] = []
        self.hashes: list[dict] = [{"version": 0, "state_hash": self._genesis_hash()}]
        self.die_after_commits = die_after_commits
        self._commits = 0
        if not store.has("manifest.json"):
            genesis = Snapshot(0, -1, -1, {}).finalize()
            store.write_json("snapshots/v0.json", genesis.to_dict())
            store.write_json("logs/v0.jsonl", [])
            store.write_json("manifest.json", self.manifest)
            store.write_json("online_hashes.json", self.hashes)

    def _genesis_hash(self) -> str:
        return Snapshot(0, -1, -1, {}).finalize().state_hash

    # ---- 校验 ----
    def _validate(self, events: list[Event]) -> None:
        for event in events:
            expected = self.expected_seq.get(event.peer_id, 0)
            if event.local_seq != expected:
                raise PipelineError(
                    E_PEER_GAP,
                    f"peer {event.peer_id} 本地序号跳变：期望 {expected}，实际 {event.local_seq}",
                    field="local_seq",
                )
            self.expected_seq[event.peer_id] = expected + 1
            if event.base_snapshot_version not in self.known_versions:
                raise PipelineError(
                    E_BASE_VERSION,
                    f"事件 {event.event_id} 指向未知基版本 {event.base_snapshot_version}",
                    field="base_snapshot_version",
                )

    # ---- 主流程 ----
    def run(self, events: list[Event]) -> None:
        by_tick: dict[int, list[Event]] = {}
        for event in events:
            by_tick.setdefault(event.tick, []).append(event)
        for tick in sorted(by_tick):
            # 同一 tick 内按 (peer_id 字典序, 本地序号) 形成提交序
            ordered = sorted(by_tick[tick], key=lambda e: (e.peer_id, e.local_seq))
            self._validate(ordered)
            self._commit_tick(tick, ordered)
            if self.die_after_commits is not None:
                self._commits += 1
                if self._commits >= self.die_after_commits:
                    # 模拟下一次提交进行到一半被强制终止：
                    # 留下未提交快照与临时文件，不落任何退出标记
                    nxt = self.version + 1
                    with open(os.path.join(self.store.root, "snapshots",
                                           f"v{nxt}.json"), "w") as fh:
                        fh.write("{}")
                    with open(os.path.join(self.store.root, "logs",
                                           f"v{nxt}.jsonl.tmp"), "w") as fh:
                        fh.write("[")
                    os._exit(137)
        self.store.mark_clean_shutdown()

    @classmethod
    def resume(cls, store: Store) -> "Engine":
        """强制终止后：只依据已提交清单、快照与日志恢复内存状态。"""
        engine = cls(store)
        manifest = store.load_manifest()
        head = manifest["head"]
        snapshot = store.load_snapshot(head)
        engine.state = {k: {"kind": v["kind"], "fields": dict(v["fields"])}
                        for k, v in snapshot.state.items()}
        engine.version = head
        engine.known_versions = {entry["version"] for entry in manifest["versions"]}
        engine.manifest = manifest
        for entry in manifest["versions"][1:]:
            for op in store.load_ops(entry["version"]):
                peer = op["peer_id"]
                engine.expected_seq[peer] = engine.expected_seq.get(peer, 0) + 1
        if store.has("online_hashes.json"):
            committed = engine.known_versions
            engine.hashes = [h for h in store.read_json("online_hashes.json")
                             if h["version"] in committed]
        else:
            engine.hashes = [{"version": e["version"], "state_hash": e["hash"]}
                             for e in manifest["versions"]]
        return engine

    def _commit_tick(self, tick: int, ordered: list[Event]) -> None:
        writes_this_tick: dict[tuple[str, str], tuple[Event, object]] = {}
        ops_log = []
        for event in ordered:
            writes = apply_op(self.state, event)
            for entity, field, value in writes:
                key = (entity, field)
                if key in writes_this_tick:
                    loser, _loser_value = writes_this_tick[key]
                    record = AdjudicationRecord(
                        tick=tick, entity_id=entity, field=field,
                        winner_event_id=event.event_id, winner_peer=event.peer_id,
                        winner_format=event.source_format,
                        loser_event_id=loser.event_id, loser_peer=loser.peer_id,
                        loser_format=loser.source_format, final_value=value,
                    )
                    self.adjudications.append(record.__dict__)
                    self.store.append_ledger("adjudication.jsonl", record.__dict__)
                writes_this_tick[key] = (event, value)
            ops_log.append(event.to_dict())

        parent = self.version
        self.version += 1
        snapshot = Snapshot(self.version, parent, tick,
                            {k: {"kind": v["kind"], "fields": dict(v["fields"])}
                             for k, v in self.state.items()}).finalize()
        self.known_versions.add(self.version)
        self.manifest["head"] = self.version
        self.manifest["versions"].append(
            {"version": self.version, "parent": parent, "hash": snapshot.state_hash})
        self.store.commit_version(snapshot, ops_log, self.manifest)
        self.hashes.append({"version": self.version, "state_hash": snapshot.state_hash})
        # 每次提交后原子刷新哈希清单；崩溃恢复时按清单截断未提交尾部
        self.store.write_json("online_hashes.json", self.hashes)
        self._maybe_compact()

    # ---- 压缩：超阈值时把最旧若干版本折叠进新基快照，压缩记录进版本链账本 ----
    def _maybe_compact(self) -> None:
        versions = [e["version"] for e in self.manifest["versions"]]
        if len(versions) <= COMPACT_THRESHOLD:
            return
        new_base = versions[-COMPACT_KEEP]
        merged = versions[:-COMPACT_KEEP]
        base_snapshot = self.store.load_snapshot(new_base)
        base_snapshot.parent_version = -1  # 折叠后的新基
        self.store.write_json(f"snapshots/v{new_base}.json", base_snapshot.to_dict())
        for v in merged:
            for rel in (f"snapshots/v{v}.json", f"logs/v{v}.jsonl"):
                path = os.path.join(self.store.root, rel)
                if os.path.exists(path):
                    os.remove(path)
        record = {
            "action": "compact",
            "rule": f"len(chain)>{COMPACT_THRESHOLD} -> keep last {COMPACT_KEEP}",
            "merged_versions": merged,
            "new_base": new_base,
            "new_base_hash": base_snapshot.state_hash,
        }
        self.store.append_ledger("chain_ledger.jsonl", record)
        kept = self.manifest["versions"][-COMPACT_KEEP:]
        kept[0] = {"version": new_base, "parent": -1, "hash": base_snapshot.state_hash}
        self.manifest["versions"] = kept
        self.manifest["compacted_base"] = new_base
        self.store.write_json("manifest.json", self.manifest)


# ---- 跨版本追赶与全量重建 ----

def replay_ops(state: dict, ops: list[dict]) -> dict:
    for op in ops:
        apply_op(state, Event.from_dict(op))
    return state


def catch_up(store: Store, from_version: int, to_version: int) -> dict:
    """落后端只拉取缺失操作日志追赶，结果须与目标版本快照逐字段一致。"""
    snapshot = store.load_snapshot(from_version)
    state = {k: {"kind": v["kind"], "fields": dict(v["fields"])}
             for k, v in snapshot.state.items()}
    for v in range(from_version + 1, to_version + 1):
        replay_ops(state, store.load_ops(v))
    target = store.load_snapshot(to_version)
    caught_hash = content_hash({"version": to_version, "state": state})
    return {
        "from_version": from_version,
        "to_version": to_version,
        "caught_up_hash": caught_hash,
        "target_hash": target.state_hash,
        "match": caught_hash == target.state_hash and state == target.state,
    }


def rebuild(store: Store, to_version: int | None = None) -> dict:
    """自链基（创世或压缩基）全量重建。"""
    manifest = store.load_manifest()
    head = manifest["head"] if to_version is None else to_version
    base_version = manifest["versions"][0]["version"]
    base = store.load_snapshot(base_version)
    state = {k: {"kind": v["kind"], "fields": dict(v["fields"])}
             for k, v in base.state.items()}
    per_version = [{"version": base_version, "state_hash": base.state_hash}]
    for v in range(base_version + 1, head + 1):
        replay_ops(state, store.load_ops(v))
        snap_hash = content_hash({"version": v, "state": state})
        per_version.append({"version": v, "state_hash": snap_hash})
    return {"head": head, "base": base_version, "per_version": per_version}
