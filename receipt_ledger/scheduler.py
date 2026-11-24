"""受全局上限与每工作器配额双重约束的并行批次调度。"""

import os
import signal
import subprocess
import sys
import threading
import time

from . import codec
from .deterministic import build_parameters
from .merge import merge_receipts, synthesize_conclusion
from .worker import call_worker


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class WorkerPool:
    """以子进程在 127.0.0.1 高位端口托管确定性工作器。"""

    def __init__(self, spec):
        self.spec = spec
        self.processes = {}

    def start(self):
        workers_path = self.spec.source_paths["workers"]
        for worker in self.spec.workers:
            proc = subprocess.Popen(
                [sys.executable, "-m", "receipt_ledger.worker",
                 "--workers", workers_path,
                 "--worker-id", worker.worker_id,
                 "--port", str(worker.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            ready = proc.stdout.readline().strip()
            expected = f"READY {worker.worker_id} 127.0.0.1:{worker.port}"
            if ready != expected:
                err = proc.stderr.read() if proc.poll() is not None else ""
                raise RuntimeError(
                    f"工作器 {worker.worker_id} 启动失败: {ready} {err}"
                )
            self.processes[worker.worker_id] = proc

    def stop(self):
        for proc in self.processes.values():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        proc.kill()
                    proc.wait(timeout=5)
        self.processes.clear()
        # 给内核留出口 TIME_WAIT/端口回收窗口，避免紧接的测试绑定冲突。
        time.sleep(0.05)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class Scheduler:
    def __init__(self, spec, ledger, pool, *, crash_after_commits=None):
        self.spec = spec
        self.ledger = ledger
        self.pool = pool
        self.global_slots = threading.BoundedSemaphore(spec.global_concurrency)
        self.worker_slots = {
            worker.worker_id: threading.BoundedSemaphore(
                spec.per_worker_concurrency
            )
            for worker in spec.workers
        }
        self.commit_lock = threading.Lock()
        self.committed = 0
        self.crash_after_commits = crash_after_commits
        self._param_cache = {}

    def _parameter(self, worker, param_version):
        key = (worker.worker_id, param_version)
        if key not in self._param_cache:
            _, _, summary = build_parameters(worker, param_version)
            self._param_cache[key] = summary
        return self._param_cache[key]

    def execute_batch(self, task):
        """获取双重配额 -> 真实套接字调用 -> 三格式归一 -> 归并 -> 合成。"""
        worker = self.spec.worker(task.worker_id)
        param_digest = self._parameter(worker, task.param_version)
        wait_start = now_ms()
        self.global_slots.acquire()
        try:
            self.worker_slots[task.worker_id].acquire()
            try:
                wait_end = now_ms()
                run_start = now_ms()
                receipts, retry_count = [], 0
                for merge_seq, fmt in enumerate(task.receipt_formats):
                    raw, attempts = self._call_with_attempts(
                        worker, task, fmt, merge_seq
                    )
                    retry_count += attempts
                    source_fmt, obj, raw_len = codec.decode_receipt(raw)
                    context = task.context(param_digest, worker.quant)
                    normalized = codec.normalize_receipt(
                        source_fmt, obj, context, raw_len
                    )
                    normalized["merge_seq"] = merge_seq
                    receipts.append(
                        {"merge_seq": merge_seq, "raw": raw,
                         "normalized": normalized, "adopted": False,
                         "reject_reason": None}
                    )
                run_ms = now_ms() - run_start
                winner_view, rejected = merge_receipts(
                    [item["normalized"] for item in receipts]
                )
                rejected_map = {
                    (item["merge_seq"]): item["reason"] for item in rejected
                }
                for item in receipts:
                    item["adopted"] = (
                        item["merge_seq"] == winner_view["merge_seq"]
                    )
                    item["reject_reason"] = rejected_map.get(item["merge_seq"])
                per_request = self._per_request_outputs(
                    worker, task, winner_view
                )
                conclusion = synthesize_conclusion(
                    task, winner_view, per_request
                )
                timing = {
                    "wait_start_ms": wait_start,
                    "wait_end_ms": wait_end,
                    "wait_ms": wait_end - wait_start,
                    "run_ms": run_ms,
                    "retry_count": retry_count,
                }
                return conclusion, receipts, timing
            finally:
                self.worker_slots[task.worker_id].release()
        finally:
            self.global_slots.release()

    def _call_with_attempts(self, worker, task, fmt, merge_seq):
        return call_worker(worker, task, fmt, merge_seq)

    def _per_request_outputs(self, worker, task, winner_view):
        # 权威输出摘要来自与工作器相同的确定性前向规则，逐位复算。
        from .deterministic import forward, output_digest
        weight, bias, _ = build_parameters(worker, task.param_version)
        outputs = []
        for req in task.requests:
            digest = output_digest(forward(weight, bias, req.vector))
            outputs.append((req.seq, digest))
        return outputs

    def run(self, plan, start_seq: int = 0):
        """并发执行但按规划 seq 顺序提交，保证账本证据链确定。"""
        results = {}
        done = threading.Event()

        def worker_thread(task):
            try:
                results[task.batch_id] = self.execute_batch(task)
            except Exception as exc:  # 记录异常由主线程抛出
                results[task.batch_id] = exc
            finally:
                done.set()

        threads = []
        for task in plan[start_seq:]:
            thread = threading.Thread(
                target=worker_thread, args=(task,),
                name=f"batch-{task.batch_id}", daemon=True,
            )
            thread.start()
            threads.append(thread)

        next_seq = start_seq
        while next_seq < len(plan):
            task = plan[next_seq]
            while task.batch_id not in results:
                done.wait(0.005)
                done.clear()
            outcome = results.pop(task.batch_id)
            if isinstance(outcome, Exception):
                for thread in threads:
                    thread.join(timeout=0.1)
                raise outcome
            conclusion, receipts, timing = outcome
            prev = self.ledger.last_chain_digest(self.spec.spec_version)
            self.ledger.commit_batch(
                self.spec.spec_version, next_seq, task, conclusion,
                receipts, timing, prev,
            )
            self.committed += 1
            if (self.crash_after_commits is not None
                    and self.committed >= self.crash_after_commits):
                self._hard_crash()
            next_seq += 1
        for thread in threads:
            thread.join()

    @staticmethod
    def _hard_crash():
        # 强制终止：os._exit 不执行 finally / 缓冲刷新，模拟 kill -9。
        os._exit(137)
