"""流水线编排：加载校验 -> 块对打分 -> 在线 softmax 归约 -> 输出汇总。

每批产出可复现 JSON 报告；支持快照续跑、异常注入隔离与优雅退出。
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import threading

import numpy as np

from .cache import BlockCache, tensor_fp
from .compare import CompareResult, compare_query_block
from .errors import Code, OperatorError
from .fixedpoint import quantize
from .injection import apply_load_time, block_fault_for, raise_block_fault
from .kernel import compute_score_block, online_softmax
from .reference import reference_query_block, split_blocks
from .scheduler import BudgetScheduler
from .tensor_pack import validate_pack

STAGE_LOAD = "load_validate"
STAGE_SCORE = "block_score"
STAGE_SOFTMAX = "softmax_reduce"
STAGE_OUTPUT = "output_summary"


class Pipeline:
    def __init__(self, cfg, dataset, injections=(), snapshot_mgr=None,
                 reports_dir=None, tensor_pack_fp=""):
        self.cfg = cfg
        self.dataset = dataset
        self.injections = list(injections)
        self.snapshot = snapshot_mgr
        self.reports_dir = reports_dir
        self.tensor_pack_fp = tensor_pack_fp
        self.cache = BlockCache(cfg)
        self.stop_requested = threading.Event()
        self.batch_reports = []
        self._carried_pending = []
        self._last_stage_artifacts = {}
        if reports_dir:
            os.makedirs(reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def install_signal_handlers(self):
        """注册退出信号：置位停止标记，当前批次落盘后才停止。"""
        def handler(signum, frame):
            self.stop_requested.set()
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:
            pass  # 非主线程（如测试环境）不注册

    def request_stop(self):
        self.stop_requested.set()

    # ------------------------------------------------------------------
    def run(self, num_batches: int, on_batch_complete=None) -> list:
        cursor, resume_issues = (self.snapshot.check_resume()
                                 if self.snapshot else (0, []))
        for issue in resume_issues:
            self.batch_reports.append({"batch": None, "resume_issues": [issue]})
        for batch in range(cursor, num_batches):
            report = self.run_batch(batch)
            self.batch_reports.append(report)
            if self.reports_dir:
                path = os.path.join(self.reports_dir,
                                    f"batch_{batch:04d}.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, ensure_ascii=False, indent=2)
            due = ((batch + 1) % self.cfg.snapshot_commit_interval_batches == 0)
            if self.snapshot and (due or self.stop_requested.is_set()):
                self.snapshot.commit(batch + 1, self._last_stage_artifacts)
            if on_batch_complete:
                on_batch_complete(batch, report)
            if self.stop_requested.is_set():
                break
        return self.batch_reports

    # ------------------------------------------------------------------
    def run_batch(self, batch: int) -> dict:
        hits_start = len(self.cache.hits)
        misses_start = len(self.cache.misses)
        tensors = self.dataset.tensors_for_batch(batch)
        meta = dict(self.dataset.meta)
        tensors, meta, cfg_b, hits = apply_load_time(
            tensors, meta, self.cfg, self.injections)

        issues = [i.to_dict() for i in cfg_b.validate()]
        pack_issues = validate_pack(tensors, meta, cfg_b)
        issues += [i.to_dict() for i in pack_issues]
        failures = []
        affected_q, affected_k = set(), set()
        for issue in pack_issues:
            ctx = issue.context
            if issue.code == Code.E_NONFINITE:
                inj_id = self._match_injection(hits, "nonfinite",
                                               ctx.get("tensor"))
                for qi in ctx.get("query_blocks", []):
                    affected_q.add(qi)
                    failures.append(self._failure(
                        STAGE_LOAD, qi, None, issue.code, issue.message, inj_id))
                for ki in ctx.get("key_blocks", []):
                    affected_k.add(ki)
                    failures.append(self._failure(
                        STAGE_LOAD, None, ki, issue.code, issue.message, inj_id))

        fatal_codes = {Code.E_TENSOR_MISSING, Code.E_HEAD_DIM_MISMATCH,
                       Code.E_SHAPE_DATA_MISMATCH, Code.E_FRAC_BITS_OUT_OF_RANGE,
                       Code.E_QUANT_LEVEL_INVALID}
        fatal = any(i["code"] in fatal_codes for i in issues)

        report = {
            "batch": batch,
            "config_hash": cfg_b.config_hash(),
            "tensor_pack_fp": self.tensor_pack_fp,
            "seed": self.dataset.seed,
            "issues": issues,
            "injection_hits": [
                {"id": h.id, "type": h.type, "code": h.code} for h in hits],
            "failures": failures,
            "budget_events": [],
            "pending_tasks": [],
            "cache": {},
            "overflows": [],
            "compare": {},
            "peak_concurrency": 0,
            "completed_query_blocks": [],
            "excluded_query_blocks": sorted(affected_q),
        }

        if fatal:
            report["fatal"] = True
            self._last_stage_artifacts = self._artifacts(report, {}, {}, {})
            return report

        q, k, v = tensors["Q"], tensors["K"], tensors["V"]
        # 非有限元素所在块已被隔离；量化前清洗以避免无关告警
        q_clean = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        k_clean = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
        v_clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        frac, level = cfg_b.fixed_point_frac_bits, cfg_b.quantization_level_bits
        q_int = quantize(q_clean, frac, level)
        k_int = quantize(k_clean, frac, level)
        v_int = quantize(v_clean, frac, level)
        q_blocks = split_blocks(q_int, cfg_b.query_block_len)
        k_blocks = split_blocks(k_int, cfg_b.key_block_len)
        v_blocks = split_blocks(v_int, cfg_b.key_block_len)
        num_q, num_k = len(q_blocks), len(k_blocks)
        valid_qis = [i for i in range(num_q) if i not in affected_q]
        valid_kis = [i for i in range(num_k) if i not in affected_k]

        # ---- 阶段：块对打分（缓存 + 预算调度 + 线程池） ----------------
        score_keys, score_results, needed = {}, {}, []
        for qi in valid_qis:
            q_fp = tensor_fp(q_blocks[qi])
            for ki in valid_kis:
                key = self.cache.score_key(qi, ki, q_fp,
                                           tensor_fp(k_blocks[ki]))
                score_keys[(qi, ki)] = key
                cached = self.cache.lookup("score", (qi, ki), key)
                if cached is not None:
                    score_results[(qi, ki)] = cached
                else:
                    needed.append((qi, ki))

        scheduler = BudgetScheduler(cfg_b, num_q, num_k)
        scheduler.pending = list(self._carried_pending)
        runnable = scheduler.plan(needed)

        def score_fn(task):
            inj = block_fault_for(self.injections, "matmul",
                                  task.query_block, task.key_block)
            if inj:
                raise_block_fault(inj, STAGE_SCORE, task.query_block,
                                  task.key_block)
            return compute_score_block(
                q_blocks[task.query_block], k_blocks[task.key_block],
                cfg_b, task.query_block, task.key_block)

        executed = scheduler.execute(runnable, score_fn)
        for pair, result in executed["results"].items():
            score_results[pair] = result
            self.cache.store(score_keys[pair], result)
        for (qi, ki), exc in executed["errors"].items():
            failures.append(self._failure(
                STAGE_SCORE, qi, ki, exc.code, exc.message,
                getattr(exc, "injection_id", None)))
        report["budget_events"] = [e.to_dict() for e in scheduler.events]
        report["pending_tasks"] = [
            {"query_block": t.query_block, "key_block": t.key_block}
            for t in scheduler.pending]
        report["peak_concurrency"] = executed["peak_concurrency"]
        self._carried_pending = list(scheduler.pending)

        # ---- 阶段：在线 softmax 归约（状态缓存） -----------------------
        traces = {}
        for qi in valid_qis:
            kis = [ki for ki in valid_kis if (qi, ki) in score_results]
            if len(kis) < len(valid_kis):
                continue  # 块对失败或待办：本批该查询块不产出
            skey_map = {ki: score_keys[(qi, ki)] for ki in kis}
            state_key = self.cache.state_key(qi, skey_map)
            cached = self.cache.lookup("state", qi, state_key)
            if cached is not None:
                traces[qi] = cached
                continue
            inj = next((block_fault_for(self.injections, "softmax", qi, ki)
                        for ki in kis
                        if block_fault_for(self.injections, "softmax", qi, ki)),
                       None)
            if inj:
                failures.append(self._failure(
                    STAGE_SOFTMAX, qi, inj.target.get("key_block"),
                    Code.E_INJECTED_FAULT,
                    f"注入故障 {inj.id} 命中在线 softmax 归约", inj.id))
                continue
            trace = online_softmax(
                [score_results[(qi, ki)] for ki in kis],
                [v_blocks[ki] for ki in kis], cfg_b, qi)
            self.cache.store(state_key, trace)
            traces[qi] = trace

        # ---- 阶段：输出汇总（输出缓存 + float64 比对） -----------------
        outputs = {}
        for qi, trace in traces.items():
            v_fps = {ki: tensor_fp(v_blocks[ki]) for ki in valid_kis}
            out_key = self.cache.out_key(qi, {qi: self.cache.state_key(
                qi, {ki: score_keys[(qi, ki)] for ki in valid_kis})}, v_fps)
            cached = self.cache.lookup("out", qi, out_key)
            if cached is not None:
                outputs[qi] = cached
            else:
                outputs[qi] = trace.out_fp
                self.cache.store(out_key, trace.out_fp)
                self.cache.record_out_deps(
                    out_key, valid_kis,
                    [self.cache.state_key(
                        qi, {ki: score_keys[(qi, ki)] for ki in valid_kis})])

        # float64 参考路径：与主路径完全相同的有效键块集合，保证逐块对齐
        kf_blocks = [b for ki, b in enumerate(
            split_blocks(k, cfg_b.key_block_len)) if ki in set(valid_kis)]
        vf_blocks = [b for ki, b in enumerate(
            split_blocks(v, cfg_b.key_block_len)) if ki in set(valid_kis)]
        qf_blocks = split_blocks(q, cfg_b.query_block_len)
        compare = CompareResult()
        for qi, trace in traces.items():
            sblks = [score_results[(qi, ki)] for ki in valid_kis]
            ref_trace = reference_query_block(
                qf_blocks[qi].astype(np.float64),
                [b.astype(np.float64) for b in kf_blocks],
                [b.astype(np.float64) for b in vf_blocks],
                cfg_b.stability_offset, qi)
            compare_query_block(trace, sblks, ref_trace, cfg_b, compare)
        report["compare"] = compare.stats()
        report["compare"]["final_out_max_abs_diff"] = {
            str(qi): float(np.max(np.abs(d))) if d.size else 0.0
            for qi, d in compare.final_out_diffs.items()}
        report["overflows"] = self._collect_overflows(score_results, traces)
        report["cache"] = {
            "reuse": [{"kind": k, "block": b}
                      for k, b in self.cache.hits[hits_start:]],
            "recompute": [{"kind": k, "block": b}
                          for k, b in self.cache.misses[misses_start:]],
        }
        report["cache"]["deps"] = {
            k[:12]: v for k, v in self.cache.deps.items()}
        report["completed_query_blocks"] = sorted(traces)
        report["excluded_query_blocks"] = sorted(
            set(report["excluded_query_blocks"])
            | (set(valid_qis) - set(traces)))
        report["outputs_fp"] = {str(qi): outputs[qi] for qi in outputs}

        self._last_stage_artifacts = self._artifacts(
            report, score_results, traces, outputs)
        return report

    # ------------------------------------------------------------------
    def _match_injection(self, hits, inj_type, tensor_name):
        for h in hits:
            if h.type == inj_type and h.target.get("tensor") == tensor_name:
                return h.id
        return None

    @staticmethod
    def _failure(stage, qi, ki, code, message, injection_id=None):
        return {"stage": stage, "query_block": qi, "key_block": ki,
                "code": code, "message": message,
                "injection_id": injection_id}

    @staticmethod
    def _collect_overflows(score_results, traces):
        events = []
        for (qi, ki), res in sorted(score_results.items()):
            events.extend(e.to_dict() for e in res.overflows)
        for qi, trace in sorted(traces.items()):
            events.extend(e.to_dict() for e in trace.overflows)
        return events

    @staticmethod
    def _artifacts(report, score_results, traces, outputs):
        slim = {k: v for k, v in report.items() if k != "outputs_fp"}
        return {
            STAGE_LOAD: pickle.dumps({"issues": report["issues"],
                                      "failures": report["failures"]}),
            STAGE_SCORE: pickle.dumps(score_results),
            STAGE_SOFTMAX: pickle.dumps(traces),
            STAGE_OUTPUT: pickle.dumps({"outputs": outputs, "report": slim}),
        }
