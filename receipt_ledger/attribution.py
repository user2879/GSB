"""请求批归属归一与重复投递消解。

裁决顺序固定，保证并发调度不会改变结果：

1. 按 (batch_id, worker_id) 聚合回执；
2. 应用 r3 ``supersedes`` 覆盖引用，被覆盖回执退出候选；
3. 同请求候选按 参数版本降序 → 整数耗时升序 → 批次字典序升序 →
   稳定名次升序 → 回执引用升序 决断；
4. 全部候选不可用时输出 E_ATTRIBUTION_UNRESOLVED，其余请求照常归属。
"""

from __future__ import annotations

from collections import defaultdict

from .convert import receipt_ref
from .errors import E_ATTRIBUTION_UNRESOLVED, Failure
from .models import LedgerEntry, Receipt

NEG_INF = -1


def group_receipts(receipts: list[Receipt]) -> dict[tuple[str, str | None], list[Receipt]]:
    groups: dict[tuple[str, str | None], list[Receipt]] = defaultdict(list)
    for receipt in receipts:
        groups[(receipt.batch_id, receipt.worker_id)].append(receipt)
    for group in groups.values():
        group.sort(key=lambda item: (item.line, item.source_file, item.source_version))
    return dict(
        sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    )


def attribute(
    receipts: list[Receipt],
    unknown_versions: set[tuple[str, int]],
    request_seq: dict[str, int],
) -> tuple[list[LedgerEntry], list[Failure], dict]:
    failures: list[Failure] = []
    refs = {receipt_ref(receipt): receipt for receipt in receipts}

    superseded_refs: set[str] = set()
    override_basis: dict[str, list[str]] = defaultdict(list)
    for receipt in receipts:
        for ref in receipt.supersedes:
            target = refs.get(ref)
            if target is None:
                failures.append(
                    Failure(
                        E_ATTRIBUTION_UNRESOLVED,
                        receipt.source_file,
                        receipt.line,
                        f"覆盖引用悬空: {ref}",
                        receipt.injection_id,
                    )
                )
                continue
            if target.request_id != receipt.request_id:
                failures.append(
                    Failure(
                        E_ATTRIBUTION_UNRESOLVED,
                        receipt.source_file,
                        receipt.line,
                        f"覆盖引用 {ref} 不属于请求 {receipt.request_id}",
                        receipt.injection_id,
                    )
                )
                continue
            superseded_refs.add(ref)
            override_basis[receipt.request_id].append(
                f"{receipt_ref(receipt)}>{ref}@{receipt.batch_id}"
            )

    candidates_by_request: dict[str, list[Receipt]] = defaultdict(list)
    for receipt in receipts:
        ref = receipt_ref(receipt)
        if ref in superseded_refs:
            continue
        if _is_unavailable(receipt, unknown_versions):
            continue
        candidates_by_request[receipt.request_id].append(receipt)

    seen_requests = sorted(
        {receipt.request_id for receipt in receipts},
        key=lambda request_id: (request_seq.get(request_id, 1 << 30), request_id),
    )
    entries: list[LedgerEntry] = []
    for request_id in seen_requests:
        candidates = candidates_by_request.get(request_id, [])
        request_receipts = [item for item in receipts if item.request_id == request_id]
        if not candidates:
            failures.append(
                Failure(
                    E_ATTRIBUTION_UNRESOLVED,
                    "attribution",
                    0,
                    f"请求 {request_id} 的全部 {len(request_receipts)} 条回执均不可用",
                )
            )
            continue
        ordered = sorted(candidates, key=_ranking_key)
        winner = ordered[0]
        basis = _build_basis(winner, ordered, override_basis.get(request_id, []))
        request_superseded = tuple(
            sorted(
                ref
                for receipt in request_receipts
                for ref in receipt.supersedes
                if ref in superseded_refs
            )
        )
        entries.append(
            LedgerEntry(
                request_id=request_id,
                seq=request_seq.get(request_id, 1 << 30),
                batch_id=winner.batch_id,
                worker_id=winner.worker_id,
                param_version=winner.param_version,
                source_version=winner.source_version,
                duration_ms=winner.duration_ms,
                output_digest=winner.output_digest,
                rank_in_batch=winner.rank_in_batch,
                integrity_tag=winner.integrity_tag,
                basis=basis,
                candidate_count=len(candidates),
                superseded_refs=request_superseded,
                receipt_ref=receipt_ref(winner),
                invalid=False,
                invalid_reason=None,
                propagation_path=(),
            )
        )

    debug = {
        "batch_groups": [
            {
                "batch_id": batch_id,
                "worker_id": worker_id,
                "receipt_count": len(group),
            }
            for (batch_id, worker_id), group in group_receipts(receipts).items()
        ],
        "superseded_refs": sorted(superseded_refs),
    }
    return entries, failures, debug


def _is_unavailable(receipt: Receipt, unknown_versions: set[tuple[str, int]]) -> bool:
    if receipt.duration_ms is None:
        return False
    if receipt.worker_id is not None and receipt.param_version is not None:
        return (receipt.worker_id, receipt.param_version) in unknown_versions
    return False


def _ranking_key(receipt: Receipt) -> tuple:
    return (
        NEG_INF if receipt.param_version is None else -receipt.param_version,
        1 << 30 if receipt.duration_ms is None else receipt.duration_ms,
        receipt.batch_id,
        1 << 30 if receipt.rank_in_batch is None else receipt.rank_in_batch,
        receipt_ref(receipt),
    )


def _build_basis(winner: Receipt, ordered: list[Receipt], overrides: list[str]) -> str:
    parts = []
    if overrides:
        parts.append("override:" + ",".join(sorted(overrides)))
    parts.append(
        "rank:param_version_desc,duration_ms_asc,batch_id_asc,rank_asc,ref_asc"
    )
    parts.append(f"winner_ref={receipt_ref(winner)}")
    parts.append(f"candidates={len(ordered)}")
    return "|".join(parts)
