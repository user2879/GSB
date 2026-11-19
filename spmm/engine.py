"""增量 SpMM 引擎：行块依赖图、精确失效、预算调度、并发执行与快照。"""

from __future__ import annotations

import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RESULT_PARAMS, EngineConfig
from .csr import check_finite, fingerprint_inputs, validate_csr, validate_shapes
from .errors import (
    E_BUDGET_ZERO,
    E_RECOMPUTE_PENDING,
    E_SHAPE_MISMATCH,
    E_SNAPSHOT_STALE,
    SpmmError,
)
from .kernel import b_row_abs_keys, compute_block
from .snapshot import load_snapshot, save_snapshot


@dataclass
class RunResult:
    report: dict
    C: np.ndarray
    row_metas: list


def _num_blocks(m, bs):
    return (int(m) + int(bs) - 1) // int(bs)


class IncrementalEngine:
    """维护行块级依赖关系，在有限重算预算下调度增量计算。"""

    def __init__(self, state_dir, block_sleep_ms=0, install_signals=False):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.block_sleep_ms = int(block_sleep_ms)
        self._stop_requested = False
        if install_signals:
            self.install_signal_handlers()

    def install_signal_handlers(self):
        def _handler(signum, frame):
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    # ------------------------------------------------------------------ 校验
    def _validate(self, a, b, c, config):
        validate_csr(a)
        check_finite("A.values", a.values)
        check_finite("B", b)
        check_finite("C", c)
        validate_shapes(a, b, c)
        config.validate(a.n_rows, a.n_cols)

    # ------------------------------------------------------------------ 全量
    def full_build(self, a, b, c, config, report_path=None, extra=None):
        self._validate(a, b, c, config)
        num_blocks = _num_blocks(a.n_rows, config.row_block_size)
        return self._execute(
            a, b, c, config,
            old=None, old_c=None,
            invalid=list(range(num_blocks)), reused=[], pending_prev=[],
            budget=num_blocks, mode="full",
            report_path=report_path, extra=extra,
        )

    # ------------------------------------------------------------------ 增量
    def incremental_run(self, a, b, c, config, edits=None, base_input_fp=None,
                        base_config_fp=None, report_path=None, extra=None):
        edits = edits or {}
        snap = load_snapshot(self.state_dir)
        if snap is None:
            raise SpmmError(E_SNAPSHOT_STALE, "状态目录中不存在完整快照，拒绝增量续跑")
        state, old_c = snap
        new_input_fp = fingerprint_inputs(a, b, c)
        new_config_fp = config.fingerprint()
        resume = (
            bool(state.get("run_in_progress"))
            and state["input_fp"] == new_input_fp
            and state["config_fp"] == new_config_fp
        )
        if not resume:
            if state["input_fp"] != base_input_fp or state["config_fp"] != base_config_fp:
                raise SpmmError(
                    E_SNAPSHOT_STALE,
                    "快照版本/配置指纹/输入指纹与当前基准不一致，拒绝续跑",
                    [{
                        "field": "snapshot",
                        "detail": "input_fp 或 config_fp 不匹配",
                        "snapshot_version": state.get("version"),
                    }],
                )
        self._validate(a, b, c, config)
        if resume:
            invalid = list(state["invalid_blocks"])
            reused = list(state["completed_blocks"])
            pending_prev = list(state["pending_queue"])
        else:
            invalid, reused, pending_prev = self._invalidate(state, a, config, edits)
        return self._execute(
            a, b, c, config,
            old=state, old_c=old_c,
            invalid=invalid, reused=reused, pending_prev=pending_prev,
            budget=int(config.recompute_budget), mode="incremental",
            report_path=report_path, extra=extra,
        )

    # ------------------------------------------------------------------ 失效判定
    def _invalidate(self, state, a, config, edits):
        old_cfg = state["config"]
        new_cfg = config.to_dict()
        changed = [k for k in RESULT_PARAMS if old_cfg.get(k) != new_cfg.get(k)]
        deleted = sorted(int(r) for r in edits.get("deleted_rows", []))
        replaced = {int(r) for r in edits.get("replaced_rows", [])}
        appended = int(edits.get("appended_rows", 0))
        old_m = int(state["M"])
        new_m = a.n_rows
        if new_m != old_m - len(deleted) + appended:
            raise SpmmError(
                E_SHAPE_MISMATCH,
                "编辑序列与矩阵行数不一致",
                [{"field": "edits",
                  "detail": f"old_M={old_m}, deleted={len(deleted)}, "
                            f"appended={appended}, new_M={new_m}"}],
            )
        bs = int(config.row_block_size)
        num_blocks = _num_blocks(new_m, bs)
        completed = set(int(x) for x in state["completed_blocks"])
        pending_prev = [int(x) for x in state["pending_queue"]]
        if changed:
            # 块大小/alpha/beta/精度档位/累加顺序变更：全部输出行块失效。
            return list(range(num_blocks)), [], pending_prev
        first_deleted = deleted[0] if deleted else None
        invalid = []
        for blk in range(num_blocks):
            lo = blk * bs
            hi = min((blk + 1) * bs, new_m)
            ok = blk in completed
            if ok:
                for r in range(lo, hi):
                    if r in replaced:
                        ok = False
                        break
                    if first_deleted is not None and r >= first_deleted:
                        ok = False
                        break
                    if r >= old_m:
                        ok = False
                        break
            if not ok:
                invalid.append(blk)
        reused = sorted(x for x in completed if x not in set(invalid) and x < num_blocks)
        return invalid, reused, pending_prev

    # ------------------------------------------------------------------ 执行
    def _execute(self, a, b, c, config, old, old_c, invalid, reused,
                 pending_prev, budget, mode, report_path, extra):
        t0 = time.perf_counter()
        bs = int(config.row_block_size)
        m, n = a.n_rows, int(b.shape[1])
        invalid = sorted({int(x) for x in invalid})
        reused = sorted({int(x) for x in reused})
        recompute_set = sorted((set(invalid) | {int(x) for x in pending_prev}) - set(reused))
        to_compute = recompute_set[:budget] if budget > 0 else []
        pending = recompute_set[len(to_compute):]
        codes = []
        if pending:
            codes.append(E_RECOMPUTE_PENDING)
        if budget == 0 and recompute_set:
            codes.append(E_BUDGET_ZERO)

        blocks_meta = {}
        if old is not None:
            for blk in reused:
                blocks_meta[blk] = old["blocks"][str(blk)]
        new_c = np.full((m, n), np.nan, dtype=np.float32)
        if old_c is not None:
            for blk in reused:
                lo, hi = blk * bs, min((blk + 1) * bs, m)
                new_c[lo:hi] = old_c[lo:hi]

        abs_keys = b_row_abs_keys(b)
        version = int(old["version"]) if old is not None else 0
        completed_set = set(reused)
        deduction = []
        timings = []
        failures = []
        remaining = budget
        idx = 0
        while idx < len(to_compute):
            chunk = to_compute[idx: idx + int(config.snapshot_commit_interval)]
            try:
                results = self._compute_chunk(chunk, a, b, c, config, abs_keys)
            except SpmmError:
                raise
            except Exception as exc:  # pragma: no cover - 防御性记录
                failures.append({"chunk": chunk, "error": str(exc)})
                raise
            for res in results:
                blk = res["block_id"]
                blocks_meta[blk] = {
                    "fingerprint": res["fingerprint"],
                    "deps": res["deps"],
                    "row_metas": res["row_metas"],
                }
                new_c[res["row0"]:res["row1"]] = res["C"]
                completed_set.add(blk)
                remaining -= 1
                deduction.append({"block": blk, "remaining": remaining})
                timings.append({"block": blk, "elapsed_ms": res["elapsed_ms"]})
            idx += len(chunk)
            version += 1
            self._commit(
                version, config, a, b, c, new_c, blocks_meta, completed_set,
                pending_queue=sorted(pending + to_compute[idx:]),
                invalid=invalid, reused=sorted(completed_set),
                in_progress=bool(pending) or idx < len(to_compute),
            )
            if self._stop_requested:
                # 当前行块快照已提交，按可处理退出信号语义以 0 退出。
                sys.exit(0)
        if not to_compute:
            version += 1
            self._commit(
                version, config, a, b, c, new_c, blocks_meta, completed_set,
                pending_queue=sorted(pending), invalid=invalid,
                reused=sorted(completed_set), in_progress=bool(pending),
            )

        row_metas = self._assemble_row_metas(m, bs, blocks_meta, completed_set)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        report = {
            "mode": mode,
            "invalidated_blocks": invalid,
            "reused_blocks": reused,
            "recomputed_blocks": to_compute,
            "pending_blocks": pending,
            "previous_pending_blocks": sorted(int(x) for x in pending_prev),
            "recomputed_count": len(to_compute),
            "budget": {
                "limit": int(budget),
                "consumed": len(to_compute),
                "deduction_sequence": deduction,
            },
            "codes": codes,
            "precision": config.precision,
            "accumulation_order": config.accumulation_order,
            "snapshot_version": version,
            "failures": failures,
            "block_timings": sorted(timings, key=lambda t: t["block"]),
            "timing_summary": _timing_summary(timings),
            "elapsed_ms": elapsed_ms,
        }
        if extra:
            report.update(extra)
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return RunResult(report=report, C=new_c, row_metas=row_metas)

    def _compute_chunk(self, chunk, a, b, c, config, abs_keys):
        bs = int(config.row_block_size)
        m = a.n_rows
        tasks = [(blk, blk * bs, min((blk + 1) * bs, m)) for blk in chunk]
        results = {}
        workers = max(1, int(config.max_workers))
        if workers == 1 or len(tasks) <= 1:
            for blk, lo, hi in tasks:
                results[blk] = compute_block(
                    blk, lo, hi, a, b, c, config, abs_keys, self.block_sleep_ms)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(compute_block, blk, lo, hi, a, b, c, config,
                                abs_keys, self.block_sleep_ms): blk
                    for blk, lo, hi in tasks
                }
                for fut in as_completed(futures):
                    blk = futures[fut]
                    results[blk] = fut.result()
        # 稳定排序键 = 行块序号，保证与单线程按行块序号顺序执行一致。
        return [results[blk] for blk in sorted(results)]

    def _commit(self, version, config, a, b, c, new_c, blocks_meta,
                completed_set, pending_queue, invalid, reused, in_progress):
        state = {
            "version": int(version),
            "run_in_progress": bool(in_progress),
            "config": config.to_dict(),
            "config_fp": config.fingerprint(),
            "input_fp": fingerprint_inputs(a, b, c),
            "M": int(a.n_rows),
            "num_blocks": _num_blocks(a.n_rows, config.row_block_size),
            "completed_blocks": sorted(int(x) for x in completed_set),
            "pending_queue": sorted(int(x) for x in pending_queue),
            "invalid_blocks": sorted(int(x) for x in invalid),
            "reused_blocks": sorted(int(x) for x in reused),
            "blocks": {str(blk): blocks_meta[blk] for blk in sorted(blocks_meta)},
        }
        save_snapshot(self.state_dir, state, new_c)

    @staticmethod
    def _assemble_row_metas(m, bs, blocks_meta, completed_set):
        metas = [None] * m
        for blk in sorted(completed_set):
            info = blocks_meta[blk]
            for meta in info["row_metas"]:
                metas[int(meta["row"])] = meta
        for r in range(m):
            if metas[r] is None:
                metas[r] = {"row": r, "pending": True, "marker": "PENDING_ROW"}
        return metas


def _timing_summary(timings):
    if not timings:
        return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "total_ms": 0.0}
    vals = [t["elapsed_ms"] for t in timings]
    return {
        "count": len(vals),
        "min_ms": min(vals),
        "max_ms": max(vals),
        "mean_ms": sum(vals) / len(vals),
        "total_ms": sum(vals),
    }
