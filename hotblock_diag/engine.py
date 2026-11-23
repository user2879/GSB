"""End-to-end diagnostic engine.

The engine is deterministic: given identical tensors, configs, edit streams,
budgets and injection plans it produces byte-identical reports. Incremental
runs reuse cached block-pair operators through content-addressed keys; only
artifacts reachable from an edited upstream object are invalidated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .blocks import all_block_fingerprints, build_blocks
from .cache import (PersistentCache, bitmap_cache_key,
                    block_pair_cache_key, hot_table_cache_key)
from .config import (GEAR_FRAC, Command, FixedPointConfig, LayoutInfo,
                     PrefilterConfig, config_blob_hash)
from .errors import (E_ACC_OVERFLOW, E_INJECTION, E_NONFINITE_SCALE,
                     E_PACK_TRUNCATED, E_SELECT_UNCONVERGED,
                     E_SNAPSHOT_STALE, FailureList)
from .fixedpoint import BlockPairScore, score_block_pair
from .hottable import HotTable
from .injection import InjectionPlan
from .snapshot import STAGE_NAMES, Manifest, SnapshotStore, SignalFlag
from .scheduler import ScoreTask, Scheduler
from .tensorpack import TensorPack, read_pack
from .util import dump_json, short_hash


@dataclass
class EngineState:
    pack: TensorPack | None = None
    layout: LayoutInfo | None = None
    pre: PrefilterConfig | None = None
    fp: FixedPointConfig | None = None
    blocks: dict[str, list[np.ndarray]] = field(default_factory=dict)
    q_fps: list[str] = field(default_factory=list)
    k_fps: list[str] = field(default_factory=list)
    block_scores: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    bitmaps: list[dict[int, list[int]]] = field(default_factory=list)
    bitmap_trajectory: list[dict[str, list[int]]] = field(default_factory=list)
    hot_table: HotTable | None = None
    hot_round_inputs: list[tuple[int, dict[int, list[int]],
                                  dict[tuple[int, int], float]]] = field(default_factory=list)
    rounds_completed: int = 0
    converged: bool = False
    pending_tasks: list[ScoreTask] = field(default_factory=list)
    score_plan: list[dict[str, int]] = field(default_factory=list)
    budget_failures: list[dict[str, Any]] = field(default_factory=list)


class DiagnosticEngine:
    def __init__(self, workspace: str, pre: PrefilterConfig,
                 fixed: FixedPointConfig, injections: InjectionPlan,
                 failures: FailureList, fresh: bool = False):
        self.workspace = workspace
        os.makedirs(workspace, exist_ok=True)
        self.pre = pre
        self.fp = fixed
        self.injections = injections
        self.failures = failures
        self.fresh = fresh
        self.cache = PersistentCache(os.path.join(workspace, "cache"))
        self.snapshots = SnapshotStore(os.path.join(workspace, "snapshots"), failures)
        if injections.has("tamper_snapshot") and not fresh:
            tamper = injections.of_kind("tamper_snapshot")[0]
            if self.snapshots.load_manifest() is not None:
                self.snapshots.tamper()
                failures.add(E_SNAPSHOT_STALE, "snapshot_restore",
                             os.path.join(workspace, "snapshots"),
                             injection_id=tamper.injection_id,
                             detail="manifest version tampered by injection")
        self.signal_flag = SignalFlag()
        self.scheduler: Scheduler | None = None
        self.state = EngineState(pre=pre, fp=fixed)
        if fresh:
            self.cache.reset()
            self.snapshots.reset()
            self.state = EngineState(pre=pre, fp=fixed)

    # ------------------------------------------------------------------ inputs
    def load_pack(self, pack_path: str) -> bool:
        if self.injections.has("truncate_pack") and os.path.basename(pack_path) != "truncated_pack.bin":
            self._apply_truncation(pack_path)
            pack_path = self.resolve_pack_path(pack_path)
        pack = read_pack(pack_path, self.failures)
        if pack is None:
            return False
        self.state.pack = pack
        return True

    def _apply_truncation(self, source: str) -> None:
        injection = self.injections.of_kind("truncate_pack")[0]
        target = os.path.join(self.workspace, "truncated_pack.bin")
        with open(source, "rb") as handle:
            blob = handle.read()
        cut = int(injection.target.get("drop_bytes", max(1, len(blob) // 8)))
        with open(target, "wb") as handle:
            handle.write(blob[:-cut])
        self.failures.add(E_PACK_TRUNCATED, "tensor_parse", target,
                          injection_id=injection.injection_id,
                          detail=f"dropped {cut} trailing bytes")

    def resolve_pack_path(self, original: str) -> str:
        if self.injections.has("truncate_pack"):
            return os.path.join(self.workspace, "truncated_pack.bin")
        return original

    def build_layout(self) -> bool:
        from .config import validate_layout
        pack = self.state.pack
        assert pack is not None
        q, k, v = pack.get("query"), pack.get("key"), pack.get("value")
        layout = validate_layout(q.shape[0], k.shape[0], v.shape[0],
                                 q.shape[1], k.shape[1], v.shape[1],
                                 self.pre, self.failures)
        if layout is None:
            return False
        self.state.layout = layout
        self.state.blocks = build_blocks(pack.tensors, layout, self.pre)
        self.state.q_fps = all_block_fingerprints(self.state.blocks["query"])
        self.state.k_fps = all_block_fingerprints(self.state.blocks["key"])
        return True

    def tensor_fingerprint(self) -> str:
        assert self.state.pack is not None
        return self.state.pack.raw_fingerprint

    def config_hash(self) -> str:
        return config_blob_hash(self.pre, self.fp)

    # ------------------------------------------------------------- score cache
    def _bp_key(self, qb: int, kb: int) -> str:
        return block_pair_cache_key(
            self.state.q_fps[qb], self.state.k_fps[kb],
            self.fp.quant_gear, self.fp.acc_bits, self.fp.saturation,
        )

    def score_task(self, task: ScoreTask) -> dict[str, Any]:
        qb, kb = task.qb, task.kb
        key = self._bp_key(qb, kb)
        cached = self.cache.get("blockpair", key)
        if cached is not None:
            cached = dict(cached)
            cached["cache"] = "reused"
            return cached
        q_block = self.state.blocks["query"][qb]
        k_block = self.state.blocks["key"][kb]
        injected_scale = None
        force_overflow = False
        fired: list[str] = []
        for injection in self.injections.block_injections(qb, kb):
            if injection.kind == "nonfinite_scale":
                injected_scale = float(injection.target.get("value", "nan"))
                fired.append(injection.injection_id)
            elif injection.kind == "force_overflow":
                force_overflow = True
                fired.append(injection.injection_id)
        try:
            result: BlockPairScore = score_block_pair(
                qb=qb, kb=kb, query_block=q_block, key_block=k_block,
                frac_bits=self.fp.frac_bits, acc_bits=self.fp.acc_bits,
                saturation=self.fp.saturation, rounding=self.fp.rounding,
                temperature=self.pre.temperature,
                injected_scale=injected_scale, force_overflow=force_overflow,
            )
        except FloatingPointError as exc:
            self.failures.add(E_NONFINITE_SCALE, "block_scoring",
                              f"qb={qb};kb={kb}", exc,
                              injection_id=",".join(fired), detail=str(exc))
            return {"qb": qb, "kb": kb, "failed": True,
                    "code": E_NONFINITE_SCALE, "cache_key": key,
                    "injection_ids": fired}
        payload = result.to_dict()
        if result.overflows:
            self.failures.add(E_ACC_OVERFLOW, "block_scoring",
                              f"qb={qb};kb={kb}",
                              OverflowError("accumulator saturation"),
                              injection_id=",".join(fired) if fired else None,
                              detail=f"{len(result.overflows)} overflow position(s)")
        if fired:
            self.failures.add(E_INJECTION, "block_scoring", f"qb={qb};kb={kb}",
                              injection_id=",".join(fired),
                              detail="injection fired during scoring")
        payload["cache_key"] = key
        payload["cache"] = "recomputed"
        payload["injection_ids"] = fired
        refs = [f"qblock:{qb}:{self.state.q_fps[qb]}",
                f"kblock:{kb}:{self.state.k_fps[kb]}"]
        self.cache.put("blockpair", key, payload, refs=refs)
        return payload

    # --------------------------------------------------------------- selection
    def _eligible_score_map(self, round_idx: int) -> dict[int, list[tuple[int, float, str]]]:
        rows: dict[int, list[tuple[int, float, str]]] = {}
        for qb in range(self.state.layout.q_blocks):
            entries = []
            for kb in range(self.state.layout.k_blocks):
                payload = self.state.block_scores.get((qb, kb))
                if payload is None or payload.get("failed"):
                    continue
                entries.append((kb, float(payload["block_score"]), payload["cache_key"]))
            rows[qb] = entries
        return rows

    def select_round(self, round_idx: int) -> dict[int, list[int]]:
        rows = self._eligible_score_map(round_idx)
        bitmaps: dict[int, list[int]] = {}
        for qb in range(self.state.layout.q_blocks):
            entries = sorted(rows.get(qb, []), key=lambda item: item[0])
            if not entries:
                bitmaps[qb] = []
                continue
            max_score = max(score for _, score, _ in entries)
            floor = self.pre.score_floor_ratio * max_score
            qualified = [(kb, score, key) for kb, score, key in entries
                         if score >= floor - self.fp.tolerance]
            qualified.sort(key=lambda item: (-item[1], item[0]))
            chosen = [kb for kb, _, _ in qualified[:self.pre.candidates]]
            bitmaps[qb] = chosen
            row_key = bitmap_cache_key(self.state.q_fps[qb],
                                       self.pre.fingerprint(),
                                       self.fp.quant_gear, round_idx)
            refs = [self._bp_key(qb, kb) for kb in chosen]
            self.cache.put("bitmap", row_key,
                           {"qb": qb, "round": round_idx, "kb_list": chosen},
                           refs=refs)
        return bitmaps

    def run_selection(self, scheduler: Scheduler,
                       max_batches: int | None = None) -> str:
        state = self.state
        q_blocks, k_blocks = state.layout.q_blocks, state.layout.k_blocks
        all_pairs = [(qb, kb) for qb in range(q_blocks) for kb in range(k_blocks)]
        initial_tasks = [ScoreTask(qb, kb, {}) for qb, kb in all_pairs]
        state.score_plan = [{"qb": qb, "kb": kb} for qb, kb in all_pairs]

        pending = sorted(state.pending_tasks, key=lambda item: item.key)
        state.pending_tasks = []
        round_idx = state.rounds_completed
        while True:
            if pending:
                tasks = sorted(pending, key=lambda item: item.key)
                pending = []
            elif round_idx == 0:
                tasks = initial_tasks
            else:
                tasks = self._tasks_for_bitmap(state.bitmaps[-1])
            tasks = self._dedupe_tasks(tasks, state.block_scores)
            while tasks:
                batch = scheduler.run_batch(tasks, self.score_task)
                for key, payload in batch.results.items():
                    if payload.get("failed"):
                        continue
                    state.block_scores[key] = payload
                if batch.global_exhausted or batch.exhausted_q:
                    ordered_pending = sorted(batch.pending, key=lambda item: item.key)
                    state.pending_tasks = ordered_pending
                    state.budget_failures = [
                        item.to_dict() for item in self.failures.failures
                        if item.code in ("E_SCORE_BUDGET_EXHAUSTED",
                                         "E_QBLOCK_BUDGET_EXHAUSTED")
                    ]
                    if (self.signal_flag.requested
                            or (max_batches is not None
                                and scheduler.batches_executed >= max_batches)):
                        return "budget_paused"
                    tasks = ordered_pending
                    state.pending_tasks = []
                    continue
                tasks = []
            bitmaps = self.select_round(round_idx)
            state.bitmaps.append(bitmaps)
            state.bitmap_trajectory.append({str(qb): list(kb_list)
                                            for qb, kb_list in bitmaps.items()})
            score_lookup = {key: float(payload["block_score"])
                            for key, payload in state.block_scores.items()
                            if not payload.get("failed")}
            state.hot_round_inputs.append((round_idx, bitmaps, score_lookup))
            if state.hot_table is None:
                state.hot_table = HotTable(self.fp.hot_capacity, self.fp.hot_keep)
            state.hot_table.capacity = self.fp.hot_capacity
            state.hot_table.keep = self.fp.hot_keep
            hot_snapshot = state.hot_table.apply_round(round_idx, bitmaps, score_lookup)
            bitmap_row_keys = [
                bitmap_cache_key(self.state.q_fps[qb], self.pre.fingerprint(),
                                 self.fp.quant_gear, round_idx)
                for qb in sorted(bitmaps)
            ]
            hot_key = hot_table_cache_key(
                bitmap_row_keys, self.fp.hot_capacity, self.fp.hot_keep,
                self.fp.quant_gear, round_idx,
            )
            self.cache.put("hottable", hot_key, hot_snapshot,
                           refs=bitmap_row_keys)
            state.rounds_completed = round_idx + 1
            if round_idx > 0 and bitmaps == state.bitmaps[-2]:
                state.converged = True
                return "converged"
            if round_idx + 1 >= self.fp.max_rounds:
                self.failures.add(
                    E_SELECT_UNCONVERGED, "selection", "bitmap",
                    detail=f"max_rounds={self.fp.max_rounds} reached without stable bitmap",
                )
                return "unconverged"
            round_idx += 1

    def _tasks_for_bitmap(self, bitmaps: dict[int, list[int]]) -> list[ScoreTask]:
        tasks: list[ScoreTask] = []
        for qb in sorted(bitmaps):
            for kb in bitmaps[qb]:
                tasks.append(ScoreTask(qb, kb, {}))
        return tasks

    @staticmethod
    def _dedupe_tasks(tasks: list[ScoreTask],
                      known: dict[tuple[int, int], dict[str, Any]]) -> list[ScoreTask]:
        seen: set[tuple[int, int]] = set()
        unique: list[ScoreTask] = []
        for task in tasks:
            if task.key in seen or task.key in known:
                continue
            seen.add(task.key)
            unique.append(task)
        return unique

    # ------------------------------------------------------------- snapshotting
    def stage_commit(self, reason: str) -> Manifest:
        state = self.state
        self.snapshots.stage_stage("tensor_parse", {
            "tensor_fingerprint": self.tensor_fingerprint(),
            "q_fingerprints": state.q_fps,
            "k_fingerprints": state.k_fps,
            "layout": state.layout.__dict__,
        })
        self.snapshots.stage_stage("block_scoring", {
            "block_scores": {f"{qb}:{kb}": payload
                             for (qb, kb), payload in
                             sorted(state.block_scores.items())},
        })
        self.snapshots.stage_stage("selection", {
            "trajectory": state.bitmap_trajectory,
            "rounds_completed": state.rounds_completed,
            "converged": state.converged,
        })
        self.snapshots.stage_stage(
            "hot_summary",
            state.hot_table.state_dict() if state.hot_table is not None else {},
        )
        manifest = Manifest(
            global_version=state.rounds_completed + 1,
            config_hash=self.config_hash(),
            tensor_fingerprint=self.tensor_fingerprint(),
            completed_rounds=state.rounds_completed,
            stage_versions={stage: state.rounds_completed + 1
                            for stage in STAGE_NAMES},
            snapshot_interval=self.fp.snapshot_interval,
        )
        self.snapshots.commit(manifest)
        return manifest

    # ------------------------------------------------------------------- edits
    def apply_edits(self, commands: list[Command]) -> None:
        """Apply a validated command stream and propagate invalidation."""
        structural_reset = any(
            command.cmd == "update_prefilter"
            and ("q_block" in command.payload.get("params", {})
                 or "k_block" in command.payload.get("params", {}))
            for command in commands
        )
        if structural_reset:
            self.cache.reset()
            self.snapshots.reset()
            self._reset_iteration_state()
            self.state.q_fps = []
            self.state.k_fps = []
            self.state.blocks = {}
            self.state.layout = None
            self.state.pack = None
            return

        for command in commands:
            if command.cmd == "replace_qblock":
                self._replace_block("query", command.index, command.payload["data"],
                                    reason="replace_qblock")
            elif command.cmd == "replace_kblock":
                self._replace_block("key", command.index, command.payload["data"],
                                    reason="replace_kblock")
            elif command.cmd == "update_prefilter":
                patch = command.payload.get("params", {})
                for key, value in patch.items():
                    setattr(self.pre, key, value)
                self.cache.invalidate_descendants(
                    self.cache.family_keys("bitmap")
                    + self.cache.family_keys("hottable"),
                    reason="update_prefilter",
                )
                self.state.bitmaps = []
                self.state.bitmap_trajectory = []
                self.state.hot_table = None
                self.state.hot_round_inputs = []
                self.state.rounds_completed = 0
                self.state.converged = False
            elif command.cmd == "set_quant":
                gear = command.payload["quant_gear"]
                self.fp.quant_gear = gear
                self.fp.frac_bits = GEAR_FRAC[gear]
                self.cache.reset()
                self._reset_iteration_state()
            elif command.cmd == "set_hot_capacity":
                self.fp.hot_capacity = int(command.payload["capacity"])
                self.cache.invalidate_descendants(
                    self.cache.family_keys("hottable"),
                    reason="set_hot_capacity",
                )
                self.state.hot_table = None
                self.state.hot_round_inputs = []
        self.state.q_fps = all_block_fingerprints(self.state.blocks["query"])
        self.state.k_fps = all_block_fingerprints(self.state.blocks["key"])

    def _replace_block(self, family: str, index: int, flat_data: list[float],
                       reason: str) -> None:
        layout = self.state.layout
        block_len = self.pre.q_block if family == "query" else self.pre.k_block
        affected_keys: list[str] = []
        if family == "query":
            for kb in range(layout.k_blocks):
                affected_keys.append(self._bp_key(index, kb))
        else:
            for qb in range(layout.q_blocks):
                affected_keys.append(self._bp_key(qb, index))
        self.cache.invalidate_descendants(affected_keys, reason=reason)

        array = np.asarray(flat_data, dtype=np.float32).reshape(block_len, layout.dim)
        self.state.blocks[family][index] = np.ascontiguousarray(array)
        fp_attr = "q_fps" if family == "query" else "k_fps"
        new_fp = short_hash(np.ascontiguousarray(array, dtype="<f4").tobytes())
        setattr(self.state, fp_attr,
                [new_fp if i == index else value
                 for i, value in enumerate(getattr(self.state, fp_attr))])
        if family == "query":
            for key in list(self.state.block_scores):
                if key[0] == index:
                    self.state.block_scores.pop(key)
            self.state.bitmaps = []
            self.state.bitmap_trajectory = []
            self.state.rounds_completed = 0
            self.state.converged = False
            self.state.hot_table = None
            self.state.hot_round_inputs = []
        else:
            for key in list(self.state.block_scores):
                if key[1] == index:
                    self.state.block_scores.pop(key)
            self.state.bitmaps = []
            self.state.bitmap_trajectory = []
            self.state.rounds_completed = 0
            self.state.converged = False
            self.state.hot_table = None
            self.state.hot_round_inputs = []

    def _reset_iteration_state(self) -> None:
        self.state.block_scores = {}
        self.state.bitmaps = []
        self.state.bitmap_trajectory = []
        self.state.hot_table = None
        self.state.hot_round_inputs = []
        self.state.rounds_completed = 0
        self.state.converged = False

    # -------------------------------------------------------------------- run
    def run_to_completion(self, max_batches: int | None = None) -> dict[str, Any]:
        self.signal_flag.install()
        if self.scheduler is None:
            self.scheduler = Scheduler(
                threads=self.fp.threads, batch_budget=self.fp.batch_budget,
                qblock_quota=self.fp.qblock_quota, failures=self.failures,
            )
        scheduler = self.scheduler
        status = "budget_paused"
        while True:
            status = self.run_selection(scheduler, max_batches=max_batches)
            if status != "budget_paused":
                break
            self.stage_commit("batch")
            if self.signal_flag.requested:
                status = "signal_paused"
                break
            if max_batches is not None:
                break
        manifest = self.stage_commit("final")
        self.signal_flag.restore()
        report = self.build_report(status, scheduler, manifest)
        report_path = os.path.join(self.workspace, "diagnostic_report.json")
        dump_json(report_path, report)
        return report

    def build_report(self, status: str, scheduler: Scheduler,
                     manifest: Manifest) -> dict[str, Any]:
        state = self.state
        bitmap_summary = {}
        if state.bitmaps:
            final_bitmap = state.bitmaps[-1]
            for qb, kb_list in final_bitmap.items():
                bitmap_summary[str(qb)] = {
                    "selected": kb_list,
                    "count": len(kb_list),
                    "fingerprint": short_hash(repr((qb, kb_list)).encode()),
                }
        block_pairs = {
            f"{qb}:{kb}": payload
            for (qb, kb), payload in sorted(state.block_scores.items())
        }
        return {
            "status": status,
            "converged": state.converged,
            "rounds_completed": state.rounds_completed,
            "config_hash": self.config_hash(),
            "tensor_fingerprint": self.tensor_fingerprint(),
            "prefilter": self.pre.__dict__,
            "fixedpoint": self.fp.__dict__,
            "block_pair_count": len(block_pairs),
            "block_pairs": block_pairs,
            "bitmap_trajectory": state.bitmap_trajectory,
            "bitmap_summary": bitmap_summary,
            "hot_table": state.hot_table.state_dict() if state.hot_table else None,
            "score_plan": state.score_plan,
            "budget": {
                "batch_budget": self.fp.batch_budget,
                "qblock_quota": self.fp.qblock_quota,
                "total_consumed": scheduler.total_consumed,
                "deduction_sequence": scheduler.total_deductions,
                "pending_tasks": [dict(task.payload, qb=task.qb, kb=task.kb)
                                  for task in state.pending_tasks],
            },
            "cache_audit": self.cache.audit.to_dict(),
            "snapshot_manifest": manifest.to_dict(),
            "failures": self.failures.to_list(),
        }
