"""主流水线：加载规格 -> 打开账本 -> 托管后端 -> 分流 -> 调度执行 ->
回执归一 -> 归并 -> 单事务落账。支持热更新场景与崩溃恢复重跑。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from .backend_host import BackendHost, BackendStaleError, BackendTimeoutError
from .errors import e_mixed
from .hotupdate import ParamRegistry
from .ledger import Ledger
from .merge import GrayConclusion, MergeCandidate, merge_receipts
from .receipts import NormContext, decode_and_normalize
from .replay import verify_replay
from .routing import HitRecord, route_request
from .scheduler import BatchJob, RetryableError, Scheduler
from .spec import Spec, input_content_digest, load_spec

GLOBAL_LIMIT = 4
PER_BACKEND_LIMIT = 2
EXEC_TIMEOUT_MS = 2000
MAX_RETRIES = 2


class SimulatedCrash(Exception):
    """测试用：模拟进程被强制终止。"""


@dataclass
class BatchResult:
    batch_id: str
    conclusion: GrayConclusion
    hits: list[HitRecord]


@dataclass
class SessionReport:
    completed: list[BatchResult] = field(default_factory=list)
    unknown_batches: list[str] = field(default_factory=list)
    snapshot_diffs: list[str] = field(default_factory=list)
    replay_verified: int = 0
    exit_report_available: bool = True


class _Hosts:
    def __init__(self, spec: Spec):
        self._hosts = {bid: BackendHost(decl) for bid, decl in spec.backends.items()}

    def start(self) -> None:
        for host in self._hosts.values():
            host.start()

    def stop(self) -> None:
        for host in self._hosts.values():
            host.stop()

    def kill(self) -> None:
        for host in self._hosts.values():
            host.kill()

    def get(self, backend_id: str) -> BackendHost:
        return self._hosts[backend_id]


def _execute(host: BackendHost, payload: dict, timeout_ms: int) -> dict:
    try:
        return host.request(payload, timeout_ms)
    except BackendTimeoutError as exc:
        raise RetryableError(str(exc)) from exc


def _apply_hot_updates(index, hot_updates, hosts, registry) -> None:
    for backend_id, new_seed in hot_updates.get(index, []):
        hosts.get(backend_id).hot_update(new_seed)
        registry.hot_update(backend_id, new_seed)


def _run_batches(
    spec: Spec,
    ledger: Ledger,
    hosts: _Hosts,
    registry: ParamRegistry,
    batch_indexes,
    hot_updates: dict[int, list[tuple[str, int]]],
    crash_before_commit: set[int] | None = None,
) -> list[BatchResult]:
    scheduler = Scheduler(
        GLOBAL_LIMIT, PER_BACKEND_LIMIT, list(spec.backends), max_workers=8
    )
    results: list[BatchResult] = []
    try:
        for index in batch_indexes:
            batch = spec.batches[index]
            _apply_hot_updates(index, hot_updates, hosts, registry)

            # 分流 + 派发时冻结参数版本绑定
            hits: list[HitRecord] = []
            bindings = {}
            for request in batch.requests:
                rule = spec.rules[request.rule_id]
                hit = route_request(
                    rule, request.request_id, request.batch_id, request.client_key
                )
                if hit.backend_id not in bindings:
                    bindings[hit.backend_id] = registry.bind(hit.backend_id)
                hits.append(
                    replace(hit, param_version=bindings[hit.backend_id].param_version)
                )

            # 按 (后端, 去向) 分组构造执行负载并投射到工作线程池
            groups: dict[tuple[str, str], list[HitRecord]] = defaultdict(list)
            for hit in hits:
                groups[(hit.backend_id, hit.route)].append(hit)
            futures = []
            for backend_id, route in sorted(groups):
                group = groups[(backend_id, route)]
                binding = bindings[backend_id]
                payload = {
                    "cmd": "exec",
                    "batch_id": batch.batch_id,
                    "route": route,
                    "sample_ids": sorted(h.request_id for h in group),
                    "param_version": binding.param_version,
                }
                job = BatchJob(
                    batch_id=batch.batch_id,
                    backend_id=backend_id,
                    payload=payload,
                    timeout_ms=EXEC_TIMEOUT_MS,
                    max_retries=MAX_RETRIES,
                )
                host = hosts.get(backend_id)
                future = scheduler.submit(
                    job, lambda p, t, h=host: _execute(h, p, t)
                )
                futures.append((backend_id, binding, job, future))

            outcomes = []
            for backend_id, binding, job, future in futures:
                try:
                    outcome = future.result()
                except BackendStaleError as exc:
                    raise e_mixed(
                        "批次在热更新交错中混装两个参数版本",
                        batch_id=batch.batch_id,
                        versions=sorted({exc.bound_version, exc.current_version}),
                    ) from exc
                outcomes.append((backend_id, binding, job, outcome))

            # 回执归一 + 混装检测（同一批次同一后端不得出现两个参数版本）+ 归并
            candidates: list[MergeCandidate] = []
            waits = []
            receipts_for_ledger = []
            for merge_seq, (backend_id, binding, job, outcome) in enumerate(outcomes):
                decl = spec.backends[backend_id]
                ctx = NormContext(
                    sample_ids=tuple(sorted(job.payload["sample_ids"])),
                    param_digest=binding.param_digest,
                    param_version=binding.param_version,
                    quant_profile=decl.quant_profile,
                )
                receipt = decode_and_normalize(outcome.response["frame"], ctx)
                if receipt.param_version != binding.param_version:
                    raise e_mixed(
                        "同一批次同一后端混装两个参数版本",
                        batch_id=batch.batch_id,
                        versions=sorted(
                            {receipt.param_version, binding.param_version}
                        ),
                    )
                candidates.append(MergeCandidate(receipt=receipt, merge_seq=merge_seq))
                receipts_for_ledger.append((backend_id, receipt))
                waits.extend(outcome.waits)
            conclusion = merge_receipts(candidates)

            if crash_before_commit and index in crash_before_commit:
                hosts.kill()
                raise SimulatedCrash(
                    f"在批次 {batch.batch_id} 提交前模拟强制终止"
                )

            version_digests = {
                (bid, b.param_version): b.param_digest for bid, b in bindings.items()
            }
            ledger.commit_batch(
                hits=hits,
                receipts=receipts_for_ledger,
                conclusion=conclusion,
                waits=waits,
                active_versions=registry.active_versions(),
                version_digests=version_digests,
                cursor_value=index + 1,
            )
            results.append(
                BatchResult(batch_id=batch.batch_id, conclusion=conclusion, hits=hits)
            )
    finally:
        scheduler.shutdown()
    return results


def run_session(
    sample_dir,
    db_path,
    hot_updates: dict[int, list[tuple[str, int]]] | None = None,
    crash_before_commit: set[int] | None = None,
) -> SessionReport:
    spec = load_spec(sample_dir)
    ledger = Ledger.open(db_path, input_content_digest(sample_dir))
    report = SessionReport(snapshot_diffs=list(ledger.diffs))
    registry = ParamRegistry(spec.backends)
    hosts = _Hosts(spec)
    hosts.start()
    try:
        report.completed = _run_batches(
            spec,
            ledger,
            hosts,
            registry,
            range(len(spec.batches)),
            hot_updates or {},
            crash_before_commit,
        )
    finally:
        hosts.stop()
        ledger.close()
    return report


def recover_session(sample_dir, db_path) -> SessionReport:
    """强制终止后的恢复：只依据已完整提交的批次；未提交批次标记为未知。

    存在未知批次时不声称能生成退出报告。
    """
    spec = load_spec(sample_dir)
    ledger = Ledger.open(db_path, input_content_digest(sample_dir))
    report = SessionReport(snapshot_diffs=list(ledger.diffs))
    try:
        report.replay_verified = verify_replay(ledger, spec)
        committed = set(ledger.committed_batches())
        report.unknown_batches = [
            batch.batch_id for batch in spec.batches if batch.batch_id not in committed
        ]
        report.exit_report_available = not report.unknown_batches
    finally:
        ledger.close()
    return report


def resume_session(
    sample_dir,
    db_path,
    hot_updates: dict[int, list[tuple[str, int]]] | None = None,
) -> SessionReport:
    """重跑未知批次：跳过已提交批次，确定性重放历史热更新后仅执行缺失部分。"""
    hot_updates = hot_updates or {}
    spec = load_spec(sample_dir)
    ledger = Ledger.open(db_path, input_content_digest(sample_dir))
    report = SessionReport(snapshot_diffs=list(ledger.diffs))
    committed = set(ledger.committed_batches())
    pending = [
        i for i, batch in enumerate(spec.batches) if batch.batch_id not in committed
    ]
    if not pending:
        report.replay_verified = verify_replay(ledger, spec)
        ledger.close()
        return report
    registry = ParamRegistry(spec.backends)
    hosts = _Hosts(spec)
    hosts.start()
    try:
        # 对已提交区间按相同种子确定性重放热更新，恢复参数版本序号与摘要
        first_pending = min(pending)
        for index in range(first_pending):
            _apply_hot_updates(index, hot_updates, hosts, registry)
        report.completed = _run_batches(
            spec, ledger, hosts, registry, pending, hot_updates, None
        )
        report.replay_verified = verify_replay(ledger, spec)
    finally:
        hosts.stop()
        ledger.close()
    return report
