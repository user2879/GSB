"""会话引擎：状态机 + 编译/分发/收敛/快照/恢复的端到端编排。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compiler import Compiler
from .convergence import check_convergence
from .dispatcher import Dispatcher
from .fixedpoint import GridSpec
from .manifest import load_manifest, load_nodes
from .model import ManifestVersion, StateTransition
from .orchestrator import NodeOrchestrator
from .replay import replay_snapshot
from .snapshot import SnapshotStore, content_digest
from .structured_log import StructuredLogger

VALID_TRANSITIONS = {
    "INIT": {"LOADED"},
    "LOADED": {"COMPILED", "RESTORED"},
    "COMPILED": {"DISPATCHING"},
    "DISPATCHING": {"CONVERGED", "FAILED"},
    "CONVERGED": {"PERSISTED"},
    "FAILED": {"PERSISTED"},
    "RESTORED": {"COMPILED", "REPLAYED"},
    "REPLAYED": {"COMPILED"},
    "PERSISTED": {"COMPILED"},
}


class SessionEngine:
    def __init__(
        self,
        workdir: str | Path,
        manifest_path: str | Path,
        nodes_path: str | Path,
        db_path: str | Path,
        log_path: str | Path,
        exponent: int = 3,
        int_bits: int = 63,
        max_concurrency: int = 2,
    ):
        self.workdir = Path(workdir)
        self.manifest_path = Path(manifest_path)
        self.nodes_path = Path(nodes_path)
        self.db_path = Path(db_path)
        self.logger = StructuredLogger(log_path)
        self.grid = GridSpec(exponent=exponent, int_bits=int_bits)
        self.grid.validate_power_of_ten()
        self.max_concurrency = max_concurrency
        self.nodes = load_nodes(self.nodes_path)
        self.versions = load_manifest(self.manifest_path, set(self.nodes))
        self.compiler = Compiler(self.grid)
        self.store = SnapshotStore(self.db_path)
        self.store.set_meta("grid", json.dumps(self.grid.to_json()))
        self.store.set_meta("manifest", str(self.manifest_path))
        self.state = "INIT"
        self._transition_seq = 0
        self._orchestrator: Optional[NodeOrchestrator] = None
        self.last_result = None
        self.last_outcome = None
        self.last_peak_inflight = 0
        self.last_report: Dict[str, Any] = {}
        self._transition("LOADED", "manifest+nodes validated")

    # ------------------------------------------------------------ state machine
    def _transition(self, to_state: str, reason: str) -> StateTransition:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if to_state not in allowed:
            raise RuntimeError(
                f"非法状态迁移 {self.state} -> {to_state}（允许: {sorted(allowed)}）"
            )
        self._transition_seq += 1
        event = StateTransition(
            seq=self._transition_seq,
            from_state=self.state,
            to_state=to_state,
            reason=reason,
            monotonic_ms=self.logger.monotonic_ms(),
            at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.state = to_state
        self.logger.log(
            "state.transition",
            seq=event.seq,
            **{
                "from": event.from_state,
                "to": event.to_state,
                "reason": event.reason,
            },
        )
        return event

    def _ready_for_compile(self, reason: str) -> StateTransition:
        """把持久化/重放后的状态收敛到可编译入口。"""
        if self.state == "PERSISTED" or self.state == "REPLAYED":
            return self._transition("COMPILED", reason)
        if self.state == "RESTORED":
            return self._transition("COMPILED", reason)
        return self._transition("COMPILED", reason)

    # ------------------------------------------------------------------- run all
    def process_versions(self, up_to: Optional[int] = None) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        targets = [v for v in self.versions if up_to is None or v.version <= up_to]
        for manifest in targets:
            summaries.append(self.process_one(manifest))
        return summaries

    def process_one(self, manifest: ManifestVersion) -> Dict[str, Any]:
        transitions: List[StateTransition] = []
        transitions.append(self._ready_for_compile(f"v{manifest.version} compile entry"))

        result = self.compiler.compile_version(manifest)
        self.last_result = result

        # 编译期先落一版快照（分发失败也保留版本状态），收敛后用 upsert 刷新
        digest = content_digest(
            result.version, self.grid, result.compiled, result.plan, result.graph
        )
        self.store.save_version(
            version=result.version,
            grid=self.grid,
            compiled=result.compiled,
            plan=result.plan,
            graph=result.graph,
            fingerprints=result.fingerprints,
            manifest_json=json.dumps(manifest.to_json(), ensure_ascii=True),
            digest=digest,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        transitions.append(self._transition("DISPATCHING", f"v{manifest.version} dispatch begin"))
        self._orchestrator = NodeOrchestrator(
            self.nodes, self.grid.scale, self.logger, workdir=self.workdir
        )
        try:
            handles = self._orchestrator.start_all()
            dispatcher = Dispatcher(handles, self.logger, self.max_concurrency)
            outcome = dispatcher.dispatch(result.plan, result.compiled)
            self.last_outcome = outcome
            self.last_peak_inflight = dispatcher.peak_inflight
            self.store.append_records(result.version, outcome.records)

            if outcome.node_failures:
                transitions.append(self._transition("FAILED", "node dispatch failures"))
                self.store.append_transitions(result.version, transitions)
                fail_event = self._transition("PERSISTED", "failure state persisted")
                self.store.append_transitions(result.version, [fail_event])
                from .errors import ConvergenceError

                raise ConvergenceError(
                    "存在节点下发失败",
                    {"node_failures": outcome.node_failures},
                )

            report = check_convergence(
                result.plan, result.compiled, result.graph, outcome, self.grid.scale
            )
            self.last_report = report
            transitions.append(self._transition("CONVERGED", f"v{manifest.version} digests matched"))
            # 最终回执在落库前显式标记 accepted；中间重试回执保持 accepted=false
            accepted_keys = set(outcome.receipts)
            for receipt in outcome.all_receipts:
                if (
                    receipt.node_id in accepted_keys
                    and receipt.seq == outcome.receipts[receipt.node_id].seq
                ):
                    receipt.accepted = True
            self.store.append_receipts(result.version, outcome.all_receipts)
            self.store.append_transitions(result.version, transitions)
            persisted = self._transition("PERSISTED", "snapshot written")
            self.store.append_transitions(result.version, [persisted])
        finally:
            if self._orchestrator:
                self._orchestrator.shutdown_all()
                self._orchestrator = None

        return {
            "version": result.version,
            "recompiled": result.recompiled,
            "unchanged": result.unchanged,
            "invalidated": result.invalidated,
            "deleted": result.deleted,
            "digest": digest,
            "peak_inflight": self.last_peak_inflight,
            "concurrency_budget": self.max_concurrency,
            "convergence": self.last_report,
            "final_attempts": outcome.final_attempts,
            "error_bounds": {
                key: result.compiled[key].error_bound for key in sorted(result.compiled)
            },
        }

    # ------------------------------------------------------------------ restore
    def restore_latest(self) -> Dict[str, Any]:
        version = self.store.latest_version()
        if version is None:
            return {"restored": False, "reason": "empty snapshot"}
        return self.restore_version(version)

    def restore_version(self, version: int) -> Dict[str, Any]:
        snapshot = self.store.load_version(version)
        recomputed = content_digest(
            snapshot.version,
            snapshot.grid,
            snapshot.compiled,
            snapshot.plan,
            snapshot.graph,
        )
        consistent = recomputed == snapshot.content_digest
        # 整体替换当前配置项、派生图与编译器记忆，绝不增量合并
        self.grid = snapshot.grid
        self.compiler.restore_state(
            fingerprints=snapshot.fingerprints,
            compiled=snapshot.compiled,
            graph=snapshot.graph,
        )
        self.state = "LOADED"
        self._transition("RESTORED", f"snapshot v{version} whole-state replacement")
        self.logger.log(
            "snapshot.restored",
            version=version,
            digest_consistent=consistent,
            snapshot_digest=snapshot.content_digest,
            recomputed_digest=recomputed,
            mode="whole_replace",
        )
        return {
            "restored": True,
            "version": version,
            "digest_consistent": consistent,
            "mode": "whole_replace",
            "snapshot": snapshot,
        }

    def replay(self) -> Dict[str, Any]:
        version = self.store.latest_version()
        snapshot = self.store.load_version(version)
        result = replay_snapshot(snapshot)
        self.state = "RESTORED"
        self._transition("REPLAYED", f"v{version} replay verified")
        return result

    def close(self) -> None:
        if self._orchestrator:
            self._orchestrator.shutdown_all()
        self.logger.close()
        self.store.close()
