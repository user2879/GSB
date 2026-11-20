"""归一: 将 r1/r2/r3 记录整体替换为内部 r3 视图.

归一视图整体重建(非增量合并): 以确定性管线重导出的规范视图为基底,
对记录中已携带的字段逐一校验, 缺失字段按显式声明的默认规则补齐,
补齐字段集合与规则在 meta["filled"] 中可回溯.
"""

from .canon import digest_of
from .errors import E_CHAIN, E_INTEGRITY, LedgerError
from .records import FORMAT_FIELDS

# 显式声明的补齐默认规则.
FILL_RULES = {
    "arrival_seq": "rederive:request-table",
    "backend_fingerprint": "rederive:backend-config",
    "latency_ms": "rederive:latency-model",
    "group_rank": "rederive:group-sort",
    "superseded_refs": "rederive:dedup-trace",
    "fingerprint_snapshot": "rederive:batch-snapshot",
}


def normalize_record(record, pipeline):
    """返回 (view, meta). view 为整体替换后的 r3 视图."""
    fmt = record["fmt"]
    rid = record["request_id"]
    try:
        derived = pipeline.view_for(rid)
    except KeyError:
        raise LedgerError(E_INTEGRITY, f"unknown request_id {rid!r} for current config",
                          record_id=rid, field="request_id", stage="normalize")
    for field in FORMAT_FIELDS[fmt]:
        stored = record[field]
        want = derived[field]
        if stored != want:
            code = E_CHAIN if field == "superseded_refs" else E_INTEGRITY
            raise LedgerError(
                code,
                f"stored {field} does not match re-derived value",
                record_id=rid, field=field, stage="normalize",
                extra={"stored": stored, "derived": want},
            )
    view = dict(derived)
    filled = {f: FILL_RULES[f] for f in FORMAT_FIELDS["r3"] if f not in FORMAT_FIELDS[fmt]}
    meta = {"fmt": fmt, "filled": filled}
    return view, meta


def order_views(views):
    return sorted(views, key=lambda v: (v["arrival_seq"], v["request_id"]))


def view_digest(views):
    return digest_of(order_views([dict(v) for v in views]))
