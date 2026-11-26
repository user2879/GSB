"""主编排：构建计划 -> 失效传播 -> 仅失效子集重算 -> 原子提交。

用法：
  python3 -m src.pipeline run --config config/pipeline.json --ports config/ports.json \
      --ledger state/ledger.db [--receipts samples/receipts_r1.txt ...] \
      [--compute-delay-ms N] [--no-workers] [--snapshot]
"""

import argparse
import json
import os

from . import dag, normalize as normalize_mod, receipts as receipts_mod
from .core import (build_plan, compute_receipt_lines, load_config, load_ports,
                   param_spec_dict, spec_digest)
from .ledger import Ledger
from .scheduler import BatchTask, Scheduler


def _config_snapshot(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("__") and k != "workers_by_id"}


def run_pipeline(config_path, ports_path, ledger_path, receipt_files=(),
                 compute_delay_ms=0, use_workers=True, snapshot=False,
                 snapshot_dir=None, scheduler=None):
    cfg = load_config(config_path)
    cfg["__config_path__"] = os.path.abspath(config_path)
    ports = load_ports(ports_path)
    ledger = Ledger(ledger_path, config_snapshot=_config_snapshot(cfg))

    # 1) 注册参数版本，捕获变更前的工作进程规格摘要
    old_worker_digests = ledger.current_worker_digests()
    param_versions = {}
    new_worker_digests = {}
    for wid in sorted(cfg["workers_by_id"]):
        spec = param_spec_dict(wid, cfg["workers_by_id"][wid]["seed"], cfg)
        digest = spec_digest(spec)
        version, _, _ = ledger.register_param_version(wid, digest)
        param_versions[wid] = version
        new_worker_digests[wid] = digest

    # 2) 构建计划并做失效传播
    plan = build_plan(cfg, param_versions)
    old_batches = {bid: b for bid, b in ledger.load_batches().items() if b["status"] == "done"}
    report = dag.compute_invalidation(old_batches, old_worker_digests, plan, new_worker_digests)

    # 派工变更记录为可重放推导的变更行
    for pb in plan:
        cause = report.batch_cause.get(pb.batch_id)
        if not cause:
            continue
        for part in cause.split(";"):
            if part.startswith("worker-reassigned:"):
                ledger.record_reassignment(
                    pb.worker_id, old_batches[pb.batch_id]["param_version"],
                    pb.param_version, new_worker_digests[pb.worker_id], part)

    # 3) 目标子集 = 失效批 + 未完成（未知/新）批；未失效批不触碰
    ledger.mark_unknown(plan)
    if report.retired_batches:
        ledger.mark_retired({
            bid: report.rows_for_batch(bid, old_batches[bid]["request_ids"])
            for bid in report.retired_batches
        })
    done = {bid for bid, b in ledger.load_batches().items() if b["status"] == "done"}
    targets = [pb for pb in plan
               if pb.batch_id in report.batch_cause or pb.batch_id not in done]

    # 4) 历史回执（磁盘三格式并存）
    historical = []
    for path in receipt_files:
        historical.extend(receipts_mod.parse_file(path))
    hist_by_batch = {}
    for rec in historical:
        hist_by_batch.setdefault(rec.batch_id, []).append(rec)

    # 5) 派发目标子集，批完成一个就在同一事务内提交一个（流式）
    committed = []
    wait_stats = {}
    plan_by_id = {pb.batch_id: pb for pb in targets}

    def commit_one(pb, fresh_receipts):
        batch_receipts = list(fresh_receipts) + hist_by_batch.get(pb.batch_id, [])
        views = normalize_mod.normalize(batch_receipts)
        adopted = normalize_mod.adopt(views)
        receipt_rows = []
        adopted_digest = None
        adopted_format = None
        for (bid, rid) in sorted(adopted):
            winner, rejected = adopted[(bid, rid)]
            adopted_digest = winner.output_digest
            adopted_format = winner.src_format
            receipt_rows.append(_receipt_row(bid, rid, winner, 1, None))
            for loser, reason in rejected:
                receipt_rows.append(_receipt_row(bid, rid, loser, 0, reason))
        inv_rows = []
        membership_change = None
        if pb.batch_id in report.batch_cause:
            old = old_batches[pb.batch_id]
            inv_rows = report.rows_for_batch(
                pb.batch_id, set(old["request_ids"]) | set(pb.request_ids))
            if sorted(old["request_ids"]) != sorted(pb.request_ids):
                membership_change = (old["request_ids"], list(pb.request_ids))
        ledger.commit_batch(
            {"batch_id": pb.batch_id, "worker_id": pb.worker_id,
             "param_version": pb.param_version, "adopted_digest": adopted_digest,
             "adopted_format": adopted_format, "request_ids": list(pb.request_ids)},
            receipt_rows, inv_rows, membership_change=membership_change)
        committed.append(pb.batch_id)

    if targets:
        if use_workers:
            own_scheduler = scheduler is None
            sched = scheduler or Scheduler(cfg, ports, compute_delay_ms=compute_delay_ms)
            try:
                tasks = [BatchTask(batch_id=pb.batch_id, worker_id=pb.worker_id,
                                   param_version=pb.param_version,
                                   request_ids=pb.request_ids) for pb in targets]
                for res in sched.run_iter(tasks):
                    if res.error is not None:
                        raise res.error
                    wait_stats[res.batch_id] = (res.wait_start_ms, res.wait_end_ms)
                    commit_one(plan_by_id[res.batch_id], res.receipts)
            finally:
                if own_scheduler:
                    sched.shutdown()
        else:
            for pb in targets:
                lines = compute_receipt_lines(cfg, pb.worker_id, pb.param_version,
                                              pb.batch_id, pb.request_ids,
                                              delay_ms=compute_delay_ms)
                fresh = [receipts_mod.parse_line(line, line_no=i + 1,
                                                 src_file="local://compute")
                         for i, line in enumerate(lines)]
                commit_one(pb, fresh)
    committed.sort()

    snap_path = None
    if snapshot:
        snap_path = ledger.create_snapshot(snapshot_dir or os.path.join(
            os.path.dirname(os.path.abspath(ledger_path)), "snapshots"))

    summary = {
        "param_versions": param_versions,
        "invalidated_batches": list(report.invalidated_batches),
        "retired_batches": list(report.retired_batches),
        "invalidated_requests": list(report.invalidated_requests),
        "unaffected_batches": list(report.unaffected_batches),
        "steps": [list(s) for s in report.steps],
        "committed": committed,
        "wait_stats": wait_stats,
        "snapshot": snap_path,
        "replaced_on_open": ledger.replaced_on_open,
        "progress_cursor": ledger.progress_cursor(),
    }
    ledger.close()
    return summary


def _receipt_row(batch_id, request_id, view, adopted, reject_reason):
    rec = view.receipt
    return (batch_id, request_id, view.src_format, rec.src_file, rec.src_line,
            rec.raw, view.output_digest, adopted, reject_reason,
            normalize_mod.filled_json(view), normalize_mod.basis_json(view))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--ports", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--receipts", nargs="*", default=[])
    run.add_argument("--compute-delay-ms", type=int, default=0)
    run.add_argument("--no-workers", action="store_true")
    run.add_argument("--snapshot", action="store_true")
    args = parser.parse_args(argv)
    summary = run_pipeline(args.config, args.ports, args.ledger,
                           receipt_files=args.receipts,
                           compute_delay_ms=args.compute_delay_ms,
                           use_workers=not args.no_workers,
                           snapshot=args.snapshot)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
