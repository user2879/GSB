"""恢复后一致性重放校验（E_REPLAY）。

仅依据账本内容（配置快照、参数版本表、规格变更表、成员变更表、回执表、
失效轨迹表）重新生成并重算，不修改源码、不依赖账本以外的推断信息。
逐批比较采纳回执摘要、否决理由、失效集合与传播依据；差异定位到首个不一致的
批标识、请求标识与字段名。
"""

import argparse
import json

from . import normalize as normalize_mod, receipts as receipts_mod
from .core import compute_batch, load_config
from .errors import ReplayError
from .ledger import Ledger


def _fail(batch_id, request_id, field, message):
    raise ReplayError(message, batch=batch_id, request=request_id, field=field)


def replay(ledger_path):
    ledger = Ledger(ledger_path)  # 打开即校验格式版本与内容摘要
    try:
        snapshot = ledger.get_meta("config_snapshot")
        if snapshot is None:
            raise ReplayError("账本缺少配置快照，无法重放")
        cfg = load_config_from_snapshot(json.loads(snapshot))
        batches = ledger.query("SELECT * FROM batches ORDER BY batch_id")
        receipts = ledger.query("SELECT * FROM receipts ORDER BY batch_id, request_id,"
                                " src_file, src_line")
        invalidations = ledger.query("SELECT * FROM invalidations"
                                     " ORDER BY batch_id, request_id, step")
        spec_changes = ledger.query("SELECT * FROM spec_changes ORDER BY seq")
        membership_changes = ledger.query("SELECT * FROM membership_changes ORDER BY seq")

        receipts_by_batch = {}
        for row in receipts:
            receipts_by_batch.setdefault(row["batch_id"], []).append(row)
        inv_by_batch = {}
        for row in invalidations:
            inv_by_batch.setdefault(row["batch_id"], []).append(row)

        for batch in batches:
            if batch["status"] != "done":
                continue  # 未知批次要求重跑，不参与重放结论
            bid = batch["batch_id"]
            _replay_batch(cfg, batch, receipts_by_batch.get(bid, []))
            _replay_invalidation(batch, inv_by_batch.get(bid, []),
                                 spec_changes, membership_changes)
        return {"checked_batches": [b["batch_id"] for b in batches if b["status"] == "done"]}
    finally:
        ledger.close()


def load_config_from_snapshot(snapshot):
    cfg = dict(snapshot)
    cfg["workers_by_id"] = {w["worker_id"]: w for w in cfg["workers"]}
    return cfg


def _replay_batch(cfg, batch, receipt_rows):
    bid = batch["batch_id"]
    request_ids = sorted(json.loads(batch["request_ids"]))
    expected_digest, _ = compute_batch(cfg, batch["worker_id"], batch["param_version"],
                                       bid, request_ids)
    if expected_digest != batch["adopted_digest"]:
        _fail(bid, request_ids[0] if request_ids else "-", "adopted_digest",
              f"采纳摘要不一致: 账本={batch['adopted_digest']} 重算={expected_digest}")

    # 由账本回执行重新归一与采纳，逐行比对采纳标记与否决理由
    parsed = [receipts_mod.parse_line(r["raw"], line_no=r["src_line"],
                                      src_file=r["src_file"]) for r in receipt_rows]
    views = normalize_mod.normalize(parsed)
    adopted = normalize_mod.adopt(views)
    expected_rows = {}
    for key in sorted(adopted):
        winner, rejected = adopted[key]
        expected_rows[_row_key(winner.receipt)] = (1, None)
        for loser, reason in rejected:
            expected_rows[_row_key(loser.receipt)] = (0, reason)
    for row in receipt_rows:
        key = (row["src_file"], row["src_line"])
        want = expected_rows.get(key)
        if want is None:
            _fail(bid, row["request_id"], "receipts", "账本回执行无法由重放复现")
        if int(row["adopted"]) != want[0]:
            _fail(bid, row["request_id"], "adopted",
                  f"采纳标记不一致: 账本={row['adopted']} 重放={want[0]}")
        if (row["reject_reason"] or None) != want[1]:
            _fail(bid, row["request_id"], "reject_reason",
                  f"否决理由不一致: 账本={row['reject_reason']} 重放={want[1]}")


def _row_key(rec):
    return (rec.src_file, rec.src_line)


def _replay_invalidation(batch, inv_rows, spec_changes, membership_changes):
    """由规格变更与成员变更表推导失效闭包，与失效轨迹表逐行比对。"""
    bid = batch["batch_id"]
    if not batch["invalidated"]:
        if inv_rows:
            _fail(bid, inv_rows[0]["request_id"], "invalidations",
                  "批次未经历失效却存在失效轨迹行")
        return
    causes = []
    for sc in spec_changes:
        if sc["worker_id"] == batch["worker_id"] and sc["new_version"] == batch["param_version"]:
            causes.append(sc["reason"])
    old_ids = None
    for mc in membership_changes:
        if mc["batch_id"] == bid:
            old_ids = json.loads(mc["old_ids"])
            causes.append("membership-changed")
    expected = []
    if causes:
        cause = ";".join(causes)
        current_ids = json.loads(batch["request_ids"])
        all_ids = sorted(set(old_ids or current_ids) | set(current_ids))
        for rid in all_ids:
            expected.append((bid, rid, 1, cause))
            expected.append((bid, rid, 2, f"batch-invalidated:{bid}"))
            expected.append((bid, rid, 3, f"request-invalidated:{rid}<-{bid}<-{cause}"))
    actual = [(r["batch_id"], r["request_id"], r["step"], r["reason"]) for r in inv_rows]
    if len(actual) != len(expected):
        first = actual[0] if actual else ("-", "-")
        _fail(bid, first[1], "invalidations",
              f"失效轨迹行数不一致: 账本={len(actual)} 重放={len(expected)}")
    for got, want in zip(actual, expected):
        if got != want:
            _fail(got[0], got[1], "invalidations.reason",
                  f"失效传播依据不一致: 账本={got} 重放={want}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="replay")
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args(argv)
    result = replay(args.ledger)
    print(json.dumps({"replay": "OK", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
