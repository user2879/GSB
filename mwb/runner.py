"""分块重算编排：有界线程池、内存预算、增量失效与中断恢复（业务要求 4-7）。"""
from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .atomicio import digest_of
from .chunkstate import compute_chunk
from .curves import pedal_track, tempo_at_ticks
from .editengine import apply_edit, edit_dependency_keys, edit_tick_range, validate_targets
from .errors import E_BUDGET, E_DIGEST, WorkbenchError
from .expand import expand_score
from .models import Edit, ExpressionParams, Score
from .persistence import ChunkStore, version_id

EXIT_OK = 0
EXIT_GRACEFUL_TERMINATION = 2
EXIT_BUSINESS_ERROR = 1


class GracefulStop:
    """可处理终止信号：只在 tick 块边界生效。"""

    def __init__(self):
        self.event = threading.Event()

    def request(self, *_args):
        self.event.set()

    @property
    def requested(self) -> bool:
        return self.event.is_set()

    def install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self.request)
            except (ValueError, OSError):
                pass


class WorkbenchRunner:
    def __init__(
        self,
        score: Score,
        base_params: ExpressionParams,
        edits: list[Edit],
        state_root: str,
        chunk_size: int,
        concurrency: int,
        memory_budget_chunks: int,
        score_dict: dict,
        params_dict: dict,
        edit_script_dict: dict,
        tolerances: dict | None = None,
    ):
        self.score = score
        self.base_params = base_params
        self.edits = edits
        self.state_root = state_root
        self.chunk_size = chunk_size
        self.concurrency = max(1, concurrency)
        self.memory_budget = max(1, memory_budget_chunks)
        if self.memory_budget < 1:
            raise WorkbenchError(E_BUDGET, "内存预算至少容纳 1 个块")
        self.tolerances = tolerances or {"duration": 0, "velocity": 0}
        self.version = version_id(score_dict, params_dict, edit_script_dict)
        self.store = ChunkStore.open(state_root, self.version)
        self.placed, self.total_ticks, self.cross_voice_overlaps = expand_score(score)
        self.chunk_count = (self.total_ticks + chunk_size - 1) // chunk_size
        self.impact_log: list[dict] = []
        self.stop = GracefulStop()
        # 干跑语义校验（引用目标存在性），非法时不产生持久化输出
        validate_targets(edits, base_params, self.placed)

    def _params_at(self, step_cursor: int) -> tuple[ExpressionParams, list]:
        """重放到第 step_cursor 步后的参数；placed 从干净展开重建以保证可重复。"""
        params = self.base_params.clone()
        placed, _, _ = expand_score(self.score)
        for edit in self.edits[:step_cursor]:
            apply_edit(params, edit, placed)
        return params, placed

    def _invalidate_for_edit(self, manifest: dict, edit: Edit, placed) -> list[int]:
        keys = set(edit_dependency_keys(edit, placed))
        tick_range = edit_tick_range(edit, self.score.ticks_per_beat)
        invalidated = []
        for entry in manifest["chunks"]:
            dep_hit = bool(keys & set(entry["dependency_keys"]))
            range_hit = (
                tick_range is not None
                and tick_range[0] < entry["tick_hi"]
                and tick_range[1] > entry["tick_lo"]
            )
            if dep_hit or range_hit:
                invalidated.append(entry["chunk_index"])
        return sorted(set(invalidated))

    def _guarded_compute(self, index: int, params, placed, tempo_full, pedal_full) -> dict:
        try:
            return compute_chunk(
                index,
                self.chunk_size,
                self.total_ticks,
                placed,
                params,
                self.score.ticks_per_beat,
                tempo_full=tempo_full,
                pedal_full=pedal_full,
            )
        except WorkbenchError:
            raise
        except Exception as exc:  # pragma: no cover - 防御性包装
            raise WorkbenchError(E_DIGEST, f"块 {index} 计算异常: {exc}") from exc

    @staticmethod
    def _upsert_manifest_entry(manifest: dict, chunk: dict) -> None:
        entry = {
            "chunk_index": chunk["chunk_index"],
            "tick_lo": chunk["tick_lo"],
            "tick_hi": chunk["tick_hi"],
            "declared_ticks": chunk["declared_ticks"],
            "dependency_keys": chunk["dependency_keys"],
            "digest": digest_of({k: v for k, v in chunk.items() if k != "digest"}),
        }
        manifest["chunks"] = [c for c in manifest["chunks"] if c["chunk_index"] != chunk["chunk_index"]]
        manifest["chunks"].append(entry)
        manifest["chunks"].sort(key=lambda c: c["chunk_index"])

    def _run_chunk_pool(self, pending: list[int], params, placed, manifest: dict) -> list[int]:
        """有界线程池 + 内存预算门控；按块索引序发放预算，首错取消其余待处理块。"""
        if not pending:
            return []
        tempo_full = tempo_at_ticks(params, self.score.ticks_per_beat, self.total_ticks).tolist()
        pedal_full = pedal_track(params, self.score.ticks_per_beat, self.total_ticks).tolist()
        order = sorted(pending)
        budget = threading.BoundedSemaphore(self.memory_budget)
        in_flight: dict = {}
        completed: list[int] = []
        next_submit = 0
        first_failure: WorkbenchError | None = None

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while next_submit < len(order) or in_flight:
                # 预算门控：内存预算或并发度占满时，后续（更大索引）块等待
                while (
                    next_submit < len(order)
                    and len(in_flight) < self.concurrency
                    and self._budget_available(budget)
                ):
                    index = order[next_submit]
                    budget.acquire()
                    future = pool.submit(self._guarded_compute, index, params, placed, tempo_full, pedal_full)
                    in_flight[future] = index
                    next_submit += 1
                if not in_flight:
                    break
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    budget.release()
                    try:
                        chunk = future.result()
                    except WorkbenchError as exc:
                        first_failure = exc
                        continue
                    self.store.write_chunk(chunk)
                    self._upsert_manifest_entry(manifest, chunk)
                    completed.append(index)
                if first_failure is not None:
                    break
            if first_failure is not None:
                # 取消尚未开始的待处理块；已在执行的块结果丢弃
                for future in in_flight:
                    future.cancel()
                raise first_failure
        return sorted(completed)

    @staticmethod
    def _budget_available(budget: threading.BoundedSemaphore) -> bool:
        # 非阻塞探测：拿不到立即返回 False，让后续块等待
        acquired = budget.acquire(blocking=False)
        if acquired:
            budget.release()
        return acquired

    def run(self) -> dict:
        manifest = self.store.read_manifest()
        resume_step = 0
        if manifest is not None:
            if manifest.get("version_id") != self.version:
                raise WorkbenchError(E_DIGEST, "状态目录版本与输入不一致")
            resume_step = int(manifest.get("step_cursor", 0))
            valid = {c["chunk_index"] for c in manifest["chunks"]}
            self.store.quarantine_orphans(valid)
            for entry in list(manifest["chunks"]):
                try:
                    self.store.verify_chunk(entry["chunk_index"])
                except WorkbenchError:
                    manifest["chunks"] = [c for c in manifest["chunks"] if c["chunk_index"] != entry["chunk_index"]]
        else:
            manifest = {
                "version_id": self.version,
                "chunk_size": self.chunk_size,
                "total_ticks": self.total_ticks,
                "chunk_count": self.chunk_count,
                "step_cursor": 0,
                "chunks": [],
            }
            self.store.write_manifest(manifest)

        params, placed = self._params_at(resume_step)
        missing = [i for i in range(self.chunk_count) if i not in {c["chunk_index"] for c in manifest["chunks"]}]
        if missing:
            self._run_chunk_pool(missing, params, placed, manifest)
            manifest["step_cursor"] = resume_step
            self.store.write_manifest(manifest)

        for edit in self.edits[resume_step:]:
            if self.stop.requested:
                self._handle_stop(manifest, resume_step)
                return {"status": "interrupted", "exit_code": EXIT_GRACEFUL_TERMINATION, "step_cursor": resume_step}
            params, placed = self._params_at(resume_step)
            invalidated = self._invalidate_for_edit(manifest, edit, placed)
            apply_edit(params, edit, placed)
            resume_step += 1
            self._run_chunk_pool(invalidated, params, placed, manifest)
            manifest["step_cursor"] = resume_step
            self.store.write_manifest(manifest)
            self.impact_log.append(
                {
                    "step": edit.step,
                    "op": edit.op,
                    "args": edit.args,
                    "invalidated_chunks": invalidated,
                    "dependency_keys": edit_dependency_keys(edit, self.placed),
                    "tick_range": edit_tick_range(edit, self.score.ticks_per_beat),
                }
            )
            if self.stop.requested:
                self._handle_stop(manifest, resume_step)
                return {"status": "interrupted", "exit_code": EXIT_GRACEFUL_TERMINATION, "step_cursor": resume_step}

        # 最终摘要自检：发现坏块则剔除并回退补算，再重新登记 manifest
        self._heal_bad_chunks(manifest)
        self.store.clear_interrupt_marker()
        return {
            "status": "completed",
            "exit_code": EXIT_OK,
            "step_cursor": resume_step,
            "manifest": manifest,
            "params_final": params,
            "placed_final": placed,
        }


    def _heal_bad_chunks(self, manifest: dict) -> None:
        """逐块摘要校验；坏块/缺块删除后按当前步参数补算（E_DIGEST 回退补算）。"""
        good: list[dict] = []
        bad_indices: list[int] = []
        for entry in list(manifest["chunks"]):
            try:
                self.store.verify_chunk(entry["chunk_index"])
                good.append(entry)
            except WorkbenchError as exc:
                if exc.code != "E_DIGEST":
                    raise
                bad_indices.append(entry["chunk_index"])
                bad_path = self.store.chunk_path(entry["chunk_index"])
                if os.path.exists(bad_path):
                    os.unlink(bad_path)
        present = {c["chunk_index"] for c in good}
        missing = [i for i in range(manifest["chunk_count"]) if i not in present]
        if not bad_indices and not missing:
            return
        manifest["chunks"] = good
        params, placed = self._params_at(manifest["step_cursor"])
        self._run_chunk_pool(missing, params, placed, manifest)
        self.store.write_manifest(manifest)

    def _handle_stop(self, manifest: dict, step_cursor: int) -> None:
        # 仅在块边界落盘已完成块；未完成块没有 manifest 条目，不会被误认为完成
        manifest["step_cursor"] = step_cursor
        self.store.write_manifest(manifest)
        self.store.write_interrupt_marker({"step_cursor": step_cursor, "chunk_size": self.chunk_size})
