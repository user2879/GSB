"""本地 JSON 声明加载与强校验（业务要求 1）。

校验按 JSON 声明顺序进行，任一非法项以 ``E_SPEC`` 拒绝，并指明首个非法
对象标识（``object_id``）与字段名（``field``）。
"""

import json
from pathlib import Path

from .digest import canonical, digest
from .errors import SpecError
from .model import (
    COMMITTED,
    OUTCOMES,
    PENDING,
    ROLLED_BACK,
    ReadPoint,
    RecordVersion,
    Spec,
    Transaction,
    VisibilityQuery,
)


def _is_nonneg_int(value) -> bool:
    # 拒绝 bool（bool 是 int 子类）与浮点。
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require(obj, field_name, predicate, object_id):
    if not predicate(obj.get(field_name)):
        raise SpecError(
            f"非法字段 {field_name!r}", object_id=object_id, field=field_name
        )


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"声明文件不存在: {path}", object_id=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"声明文件不是合法 JSON: {exc.msg}",
            object_id=str(path),
            field=f"line {exc.lineno}",
        ) from exc
    if not isinstance(raw, dict):
        raise SpecError("声明根节点必须是对象", object_id=str(path), field="root")
    return build_spec(raw, source=str(path))


def build_spec(raw: dict, source: str = "<dict>") -> Spec:
    for top in ("transactions", "records", "read_points", "queries"):
        if not isinstance(raw.get(top), list):
            raise SpecError(
                f"顶层字段 {top!r} 必须是数组", object_id=source, field=top
            )

    epsilon = raw.get("epsilon", 0)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon < 0:
        raise SpecError("epsilon 必须是非负数值", object_id=source, field="epsilon")
    global_concurrency = raw.get("global_concurrency", 2)
    per_readpoint_quota = raw.get("per_readpoint_quota", 1)
    for name, value in (
        ("global_concurrency", global_concurrency),
        ("per_readpoint_quota", per_readpoint_quota),
    ):
        if not _is_nonneg_int(value) or value == 0:
            raise SpecError(
                f"{name} 必须是正整数", object_id=source, field=name
            )
    spec_version = raw.get("spec_version", 1)
    if not _is_nonneg_int(spec_version):
        raise SpecError("spec_version 必须是非负整数", object_id=source,
                        field="spec_version")

    transactions = _build_transactions(raw["transactions"])
    records, row_chains, max_record_seq_hint = _build_records(
        raw["records"], {t.txn_id: t for t in transactions}
    )
    max_seq = max(
        [t.begin_seq for t in transactions]
        + [t.commit_seq for t in transactions if t.commit_seq is not None]
        + [0]
    )
    read_points = _build_read_points(raw["read_points"], transactions, max_seq)
    queries = _build_queries(raw["queries"], read_points, row_chains)

    return Spec(
        spec_version=int(spec_version),
        epsilon=float(epsilon),
        global_concurrency=int(global_concurrency),
        per_readpoint_quota=int(per_readpoint_quota),
        transactions=tuple(transactions),
        records=tuple(records),
        read_points=tuple(read_points),
        queries=tuple(queries),
        row_chains=row_chains,
        max_seq=max_seq,
    )


def _build_transactions(raw_txns) -> list[Transaction]:
    seen: set[str] = set()
    built: list[Transaction] = []
    for item in raw_txns:
        txn_id = item.get("txn_id") if isinstance(item, dict) else None
        if not isinstance(txn_id, str) or not txn_id:
            raise SpecError("事务缺少 txn_id", object_id=str(txn_id),
                            field="txn_id")
        if txn_id in seen:
            raise SpecError("事务标识重复", object_id=txn_id, field="txn_id")
        seen.add(txn_id)
        _require(item, "begin_seq", _is_nonneg_int, txn_id)
        outcome = item.get("outcome")
        if outcome not in OUTCOMES:
            raise SpecError(
                f"结局必须属于 {OUTCOMES}", object_id=txn_id, field="outcome"
            )
        commit_seq = item.get("commit_seq")
        if outcome == COMMITTED:
            if not _is_nonneg_int(commit_seq):
                raise SpecError(
                    "已提交事务必须带非负整数 commit_seq",
                    object_id=txn_id,
                    field="commit_seq",
                )
            if commit_seq < item["begin_seq"]:
                raise SpecError(
                    "commit_seq 不得早于 begin_seq",
                    object_id=txn_id,
                    field="commit_seq",
                )
        elif commit_seq is not None:
            raise SpecError(
                "仅 committed 事务可带 commit_seq",
                object_id=txn_id,
                field="commit_seq",
            )
        built.append(
            Transaction(
                txn_id=txn_id,
                begin_seq=int(item["begin_seq"]),
                outcome=outcome,
                commit_seq=None if commit_seq is None else int(commit_seq),
            )
        )
    if not built:
        raise SpecError("事务表不能为空", object_id="<transactions>",
                        field="transactions")
    return built


def _build_records(raw_records, txn_index):
    seen_rows: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    built: list[RecordVersion] = []
    for item in raw_records:
        row_key = item.get("row_key") if isinstance(item, dict) else None
        if not isinstance(row_key, str) or not row_key:
            raise SpecError("记录缺少 row_key", object_id=str(row_key),
                            field="row_key")
        if not _is_nonneg_int(item.get("version")):
            raise SpecError("version 必须是非负整数", object_id=row_key,
                            field="version")
        version = int(item["version"])
        pair = (row_key, version)
        if pair in seen_pairs:
            raise SpecError(
                "同一行键的版本号重复", object_id=row_key, field="version"
            )
        seen_pairs.add(pair)
        seen_rows.add(row_key)
        writer = item.get("writer_txn_id")
        if writer not in txn_index:
            raise SpecError(
                "写入事务标识不存在", object_id=row_key, field="writer_txn_id"
            )
        value_digest = item.get("value_digest")
        if not isinstance(value_digest, str) or not value_digest:
            raise SpecError(
                "value_digest 必须是非空字符串",
                object_id=row_key,
                field="value_digest",
            )
        prev = item.get("prev_version")
        if prev is not None and not _is_nonneg_int(prev):
            raise SpecError(
                "prev_version 必须为 null 或非负整数",
                object_id=row_key,
                field="prev_version",
            )
        built.append(
            RecordVersion(
                row_key=row_key,
                writer_txn_id=writer,
                version=version,
                value_digest=value_digest,
                prev_version=None if prev is None else int(prev),
            )
        )

    # 按行键组装版本链并校验引用存在、单头与无环。
    row_chains: dict[str, list[RecordVersion]] = {}
    for rec in built:
        row_chains.setdefault(rec.row_key, []).append(rec)
    for row_key in sorted(row_chains):
        chain = row_chains[row_key]
        by_ver = {r.version: r for r in chain}
        heads = [r for r in chain if r.prev_version is None]
        if len(heads) != 1:
            raise SpecError(
                f"版本链必须恰有一个 null 头版本，当前 {len(heads)} 个",
                object_id=row_key,
                field="prev_version",
            )
        for rec in chain:
            if rec.prev_version is not None and rec.prev_version not in by_ver:
                raise SpecError(
                    f"prev_version={rec.prev_version} 引用的版本不存在",
                    object_id=row_key,
                    field="prev_version",
                )
        # 自每个版本沿引用回溯，遇重复节点即有环。
        for rec in chain:
            visited: set[int] = set()
            cursor = rec.version
            while cursor is not None:
                if cursor in visited:
                    raise SpecError(
                        "版本链沿引用回溯存在环",
                        object_id=row_key,
                        field="prev_version",
                    )
                visited.add(cursor)
                node = by_ver[cursor]
                cursor = node.prev_version
    stable_chains = {
        row_key: tuple(sorted(chain, key=lambda r: r.version))
        for row_key, chain in row_chains.items()
    }
    return built, stable_chains, 0


def _build_read_points(raw_rps, transactions, max_seq):
    txn_index = {t.txn_id: t for t in transactions}
    seen: set[str] = set()
    built: list[ReadPoint] = []
    for item in raw_rps:
        rp_id = item.get("read_point_id") if isinstance(item, dict) else None
        if not isinstance(rp_id, str) or not rp_id:
            raise SpecError("读点缺少 read_point_id", object_id=str(rp_id),
                            field="read_point_id")
        if rp_id in seen:
            raise SpecError("读点标识重复", object_id=rp_id,
                            field="read_point_id")
        seen.add(rp_id)
        if not _is_nonneg_int(item.get("snapshot_seq")):
            raise SpecError(
                "snapshot_seq 必须是非负整数",
                object_id=rp_id,
                field="snapshot_seq",
            )
        snapshot_seq = int(item["snapshot_seq"])
        if snapshot_seq > max_seq:
            raise SpecError(
                f"snapshot_seq={snapshot_seq} 越出事务序号空间(上界 {max_seq})",
                object_id=rp_id,
                field="snapshot_seq",
            )
        reader = item.get("reader_txn_id")
        if reader not in txn_index:
            raise SpecError(
                "读事务标识不存在", object_id=rp_id, field="reader_txn_id"
            )
        built.append(
            ReadPoint(
                read_point_id=rp_id,
                snapshot_seq=snapshot_seq,
                reader_txn_id=reader,
            )
        )
    if not built:
        raise SpecError("读点不能为空", object_id="<read_points>",
                        field="read_points")
    return built


def _build_queries(raw_queries, read_points, row_chains):
    rp_index = {r.read_point_id: r for r in read_points}
    seen: set[str] = set()
    built: list[VisibilityQuery] = []
    for item in raw_queries:
        qid = item.get("query_id") if isinstance(item, dict) else None
        if not isinstance(qid, str) or not qid:
            raise SpecError("查询缺少 query_id", object_id=str(qid),
                            field="query_id")
        if qid in seen:
            raise SpecError("查询标识重复", object_id=qid, field="query_id")
        seen.add(qid)
        rp_id = item.get("read_point_id")
        if rp_id not in rp_index:
            raise SpecError(
                "查询引用了不存在的读点", object_id=qid, field="read_point_id"
            )
        row_key = item.get("row_key")
        if row_key not in row_chains:
            raise SpecError(
                "查询引用了不存在的行键", object_id=qid, field="row_key"
            )
        built.append(
            VisibilityQuery(query_id=qid, read_point_id=rp_id, row_key=row_key)
        )
    if not built:
        raise SpecError("可见性查询不能为空", object_id="<queries>",
                        field="queries")
    return built


# ---- 声明级摘要（账本整体替换识别用，业务要求 7） ----

def spec_digests(spec: Spec) -> dict[str, str]:
    txn_payload = [
        {
            "txn_id": t.txn_id,
            "begin_seq": t.begin_seq,
            "outcome": t.outcome,
            "commit_seq": t.commit_seq,
        }
        for t in sorted(spec.transactions, key=lambda t: t.txn_id)
    ]
    chain_payload = {
        row_key: [
            {
                "version": r.version,
                "writer_txn_id": r.writer_txn_id,
                "value_digest": r.value_digest,
                "prev_version": r.prev_version,
            }
            for r in chain
        ]
        for row_key, chain in sorted(spec.row_chains.items())
    }
    rp_payload = [
        {
            "read_point_id": r.read_point_id,
            "snapshot_seq": r.snapshot_seq,
            "reader_txn_id": r.reader_txn_id,
        }
        for r in sorted(spec.read_points, key=lambda r: r.read_point_id)
    ]
    query_payload = [
        {
            "query_id": q.query_id,
            "read_point_id": q.read_point_id,
            "row_key": q.row_key,
        }
        for q in sorted(spec.queries, key=lambda q: q.query_id)
    ]
    parts = {
        "transactions_digest": digest(txn_payload),
        "chain_digest": digest(chain_payload),
        "readpoints_digest": digest(rp_payload),
        "queries_digest": digest(query_payload),
        "epsilon": spec.epsilon,
        "spec_version": spec.spec_version,
    }
    parts["content_digest"] = digest(canonical(parts))
    return parts
