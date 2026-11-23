"""会话编排：加载、调度、演进、恢复、重放、异常注入的门面。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from .content import BlockRegistry
from .errors import (
    E_OVERSUB,
    E_SNAPSHOT,
    StructuredLog,
    WorkbenchError,
)
from .evolution import StaleArchive, impact_analysis
from .fixedpoint import ErrorLedger, FixedPoint
from .ledger import FORMAT_VERSION, Ledger, Snapshot
from .scheduler import Scheduler
from .spec import GroupSpec, load_spec, validate_group
from .verdict import Candidate, UrgencyRanker


@dataclass
class GroupContext:
    spec: GroupSpec
    version: int = 1
    err_ledger: ErrorLedger | None = None
    last_run: dict | None = None
    last_ranking: dict | None = None
    stale: StaleArchive = field(default_factory=StaleArchive)


class Session:
    """诊断工作台会话。一个会话管理若干副本组，共享账本与快照。"""

    def __init__(self, workdir: str):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.log = StructuredLog(workdir)
        self.groups: dict[str, GroupContext] = {}
        self.ledger: Ledger | None = None
        self.snapshot = Snapshot(workdir)
        self._db_path = os.path.join(workdir, "ledger.db")

    # ---- 内容摘要（规格指纹 + 块登记摘要） ----
    def _content_digest(self, block_digests: dict | None = None) -> str:
        h = hashlib.sha256()
        for gid in sorted(self.groups):
            h.update(self.groups[gid].spec.content_fingerprint().encode())
        if block_digests:
            for bid in sorted(block_digests):
                h.update(bid.encode())
                h.update(block_digests[bid].encode())
        return h.hexdigest()

    # ---- 加载（含快照一致性检查） ----
    def load(self, spec_path: str, recover: bool = False) -> dict:
        specs = load_spec(spec_path)
        for spec in specs:
            self.groups[spec.group_id] = GroupContext(
                spec=spec, err_ledger=ErrorLedger(FixedPoint(spec.params.scale_exponent))
            )
        # 块内容生成与修复前摘要登记（确定性、逐位一致）。
        block_digests: dict[str, str] = {}
        for spec in specs:
            registry = self._registry_for(spec.group_id)
            for block in spec.blocks:
                block_digests[block.block_id] = registry.materialize(block)
        digest = self._content_digest(block_digests)
        status = "fresh"
        snapshot_error = None
        try:
            status = self.snapshot.check_or_replace(digest, self._db_path)
        except WorkbenchError as err:
            if err.code == E_SNAPSHOT:
                self.log.emit_error(err)
                status = "replaced"
                snapshot_error = err.to_dict()
            else:
                raise
        self.ledger = Ledger(self._db_path, digest, spec_version=1)
        unknown = []
        if recover or status == "consistent":
            unknown = self.ledger.mark_pending_unknown()
            if unknown:
                self.log.emit(
                    "RECOVER",
                    "回收未提交批次占用配额并标记 UNKNOWN",
                    unknown_batches=[b["batch_id"] for b in unknown],
                )
        self._block_digests = block_digests
        return {
            "status": status,
            "groups": sorted(self.groups),
            "unknown_batches": [b["batch_id"] for b in unknown],
            "content_digest": digest,
            "snapshot_error": snapshot_error,
        }

    def _registry_for(self, group_id: str) -> BlockRegistry:
        return BlockRegistry(os.path.join(self.workdir, "blocks", group_id))

    # ---- 超订检查（E_OVERSUB） ----
    def check_oversubscription(self) -> None:
        for ctx in self.groups.values():
            spec = ctx.spec
            total_kbps = sum(n.bandwidth_kbps for n in spec.nodes)
            if total_kbps * 1024 > spec.params.global_budget_bytes:
                err = WorkbenchError(
                    E_OVERSUB,
                    f"节点配额之和超出全局预算: {spec.group_id}",
                    group_id=spec.group_id,
                    field="bandwidth_kbps",
                    quota_sum_kbps=total_kbps,
                    budget_bytes=spec.params.global_budget_bytes,
                )
                self.log.emit_error(err)
                raise err

    # ---- 调度 ----
    def run_group(self, group_id: str, crash_after_commits: int | None = None) -> dict:
        ctx = self.groups[group_id]
        spec = ctx.spec
        ranker = UrgencyRanker(ctx.err_ledger.fp)
        candidates = [Candidate(b, spec.node(b.node_id)) for b in spec.blocks]
        ranking = ranker.adjudicate_all(candidates)
        ctx.last_ranking = ranking
        scheduler = Scheduler(
            spec,
            self._registry_for(group_id),
            ctx.err_ledger,
            ledger=self.ledger,
            log=self.log,
            spec_version=ctx.version,
            ranked_order=ranking["order"],
            crash_after_commits=crash_after_commits,
        )
        summary = scheduler.run()
        ctx.last_run = summary
        progress = self.ledger.progress() if self.ledger else {"cursor": 0}
        self.snapshot.write(
            self._content_digest(self._block_digests), ctx.version, progress["cursor"]
        )
        return {
            "summary": summary,
            "ranking": ranking,
            "error_report": ctx.err_ledger.report(),
        }

    def run_all(self, crash_after_commits: int | None = None) -> dict:
        self.check_oversubscription()
        return {gid: self.run_group(gid, crash_after_commits) for gid in sorted(self.groups)}

    # ---- 演进（整体替换 + 陈旧防护） ----
    def evolve(self, group_id: str, new_spec_dict: dict) -> dict:
        ctx = self.groups[group_id]
        old = ctx.spec
        new_spec = validate_group(new_spec_dict, entry=f"evolve({group_id})")
        if new_spec.group_id != group_id:
            raise WorkbenchError(
                "E_SPEC", "演进不得变更副本组标识", entry=group_id, field="group_id"
            )
        impact = impact_analysis(old, new_spec, ctx.last_run)
        # 旧版本结论整体归档为陈旧。
        ctx.stale.archive(
            ctx.version,
            {
                "run": ctx.last_run,
                "ranking": ctx.last_ranking,
                "error_report": ctx.err_ledger.report() if ctx.err_ledger else None,
            },
        )
        # 当前视图整体替换。
        ctx.version += 1
        ctx.spec = new_spec
        ctx.err_ledger = ErrorLedger(FixedPoint(new_spec.params.scale_exponent))
        ctx.last_run = None
        ctx.last_ranking = None
        registry = self._registry_for(group_id)
        for block in new_spec.blocks:
            self._block_digests[block.block_id] = registry.materialize(block)
        digest = self._content_digest(self._block_digests)
        if self.ledger is not None:
            self.ledger._init_meta(digest, ctx.version)
        self.snapshot.write(digest, ctx.version, 0)
        result = {
            "group_id": group_id,
            "version": ctx.version,
            "impact": impact,
            "stale_versions": [e["version"] for e in ctx.stale.entries],
        }
        self.log.emit("EVOLVE", "预算参数/节点集合演进，视图整体替换", **result)
        return result

    # ---- 当前视图（仅当前版本，陈旧数据不参与展示） ----
    def current_view(self, group_id: str) -> dict:
        ctx = self.groups[group_id]
        return {
            "group_id": group_id,
            "version": ctx.version,
            "spec": {
                "params": ctx.spec.params.__dict__,
                "nodes": [n.__dict__ for n in ctx.spec.nodes],
                "blocks": [b.__dict__ for b in ctx.spec.blocks],
            },
            "run": ctx.last_run,
            "ranking": ctx.last_ranking,
            "error_report": ctx.err_ledger.report() if ctx.err_ledger else None,
        }

    # ---- 异常注入 ----
    def inject_corrupt_block(self, block_id: str) -> None:
        for gid in sorted(self.groups):
            path = self._registry_for(gid).path_for(block_id)
            if os.path.exists(path):
                with open(path, "r+b") as fh:
                    first = fh.read(1)
                    fh.seek(0)
                    fh.write(bytes([first[0] ^ 0xFF]))
                self.log.emit("INJECT", "注入块内容损坏", block_id=block_id)
                return
        raise WorkbenchError("E_SPEC", f"注入目标块不存在: {block_id}", entry=block_id, field="block_id")

    def inject_oversubscription(self) -> None:
        """注入配额超订：内存中放大配额后触发 E_OVERSUB 检查。"""
        for ctx in self.groups.values():
            spec = ctx.spec
            inflated = [
                type(n)(n.node_id, n.host, n.port, n.bandwidth_kbps * 1024 + 1, n.priority_weight)
                for n in spec.nodes
            ]
            ctx.spec = GroupSpec(spec.group_id, spec.params, inflated, spec.blocks)
        self.log.emit("INJECT", "注入配额之和超出全局预算")
        self.check_oversubscription()

    def inject_snapshot_mismatch(self) -> None:
        snap = self.snapshot.read() or {}
        snap["content_digest"] = "0" * 64
        with open(self.snapshot.path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh)
        self.log.emit("INJECT", "注入账本内容摘要不符")
