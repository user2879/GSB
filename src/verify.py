"""校验命令：逐项比对增量重算账本与从零全量重建的结论。

用法：
  python3 -m src.verify --config config/pipeline.json --ports config/ports.json \
      --ledger state/ledger.db [--receipts ...]

全量重建在临时账本中从头运行完整流水线（不读取被校验账本）。参数版本序号
是账本按变更历史维护的整数簿记，两次运行的绝对编号可能不同，因此结论按
版本无关的规范形式比对：批次集合、派工归属、请求成员、参数规格摘要与
规范内容摘要（参数版本归一为 0 重算的输出摘要）。全部一致输出 IDENTICAL，
否则定位首个差异的批标识与字段名。
"""

import argparse
import json
import os
import tempfile

from .core import compute_batch
from .ledger import Ledger
from .pipeline import run_pipeline
from .replay import load_config_from_snapshot


def full_rebuild(config_path, ports_path, receipt_files=()):
    """从零全量重建：在临时账本中重跑完整流水线。"""
    tmpdir = tempfile.mkdtemp(prefix="verify-rebuild-")
    rebuild_ledger = os.path.join(tmpdir, "rebuild.db")
    run_pipeline(config_path, ports_path, rebuild_ledger,
                 receipt_files=receipt_files, use_workers=False)
    return rebuild_ledger


def conclusions(ledger_path):
    """提取版本无关的规范结论。"""
    ledger = Ledger(ledger_path)
    try:
        versions = {(r["worker_id"], r["version"]): r["spec_digest"]
                    for r in ledger.query("SELECT worker_id, spec_digest, version"
                                          " FROM param_versions")}
        cfg = load_config_from_snapshot(json.loads(ledger.get_meta("config_snapshot")))
        result = {}
        for b in ledger.query("SELECT batch_id, worker_id, param_version, request_ids"
                              " FROM batches WHERE status='done' ORDER BY batch_id"):
            request_ids = sorted(json.loads(b["request_ids"]))
            content_digest, _ = compute_batch(cfg, b["worker_id"], 0,
                                              b["batch_id"], request_ids)
            result[b["batch_id"]] = {
                "worker_id": b["worker_id"],
                "spec_digest": versions.get((b["worker_id"], b["param_version"])),
                "request_ids": request_ids,
                "content_digest": content_digest,
            }
        return result
    finally:
        ledger.close()


def compare(ledger_path, config_path, ports_path, receipt_files=()):
    incremental = conclusions(ledger_path)
    rebuilt = conclusions(full_rebuild(config_path, ports_path, receipt_files))
    diffs = []
    for bid in sorted(set(incremental) | set(rebuilt)):
        if bid not in incremental:
            diffs.append({"batch_id": bid, "field": "batch", "detail": "增量账本缺失"})
            continue
        if bid not in rebuilt:
            diffs.append({"batch_id": bid, "field": "batch", "detail": "全量重建缺失"})
            continue
        for field in ("worker_id", "spec_digest", "request_ids", "content_digest"):
            if incremental[bid][field] != rebuilt[bid][field]:
                diffs.append({"batch_id": bid, "field": field,
                              "detail": f"增量={incremental[bid][field]}"
                                        f" 全量={rebuilt[bid][field]}"})
                break
    return {"result": "IDENTICAL" if not diffs else "DIFFERENT",
            "compared_batches": sorted(set(incremental) | set(rebuilt)),
            "first_diff": diffs[0] if diffs else None,
            "diff_count": len(diffs)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="verify")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--receipts", nargs="*", default=[])
    args = parser.parse_args(argv)
    result = compare(args.ledger, args.config, args.ports, args.receipts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
