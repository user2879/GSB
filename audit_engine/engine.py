"""核心审计引擎：单文件多策略裁决与策略冲突识别。"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Callable, Optional

from .acl import adjudicate
from .budget import BudgetConfig, BudgetGuard
from .errors import (
    E_DENY_SHADOW,
    E_INHERIT,
    E_LINK_ESCAPE,
    E_POLICY_CONFLICT,
    E_TIMEOUT,
    E_TRAVERSAL,
)
from .models import (
    FileResult,
    NormalizationPolicy,
    OP_REQUIRED_BIT,
    Record,
    RecordVerdict,
    RejectedRecord,
)
from .pathnorm import normalize_path
from .truth import summarize as summarize_truth
from .truth import verify_record


def _record_epsilon(record: Record) -> Fraction:
    parsed = record.extra["parsed_permission"]
    total = Fraction(str(parsed["epsilon"]))
    for entry in record.acl:
        total += Fraction(str(entry.epsilon))
    return total


def audit_record(record: Record, root: str, policies: list[NormalizationPolicy]) -> list[RecordVerdict]:
    required_bit = OP_REQUIRED_BIT[record.operation]
    epsilon = _record_epsilon(record)
    verdicts: list[RecordVerdict] = []

    for policy in policies:
        path_result = normalize_path(
            record.target_path, root, policy, record.links
        )
        effective = adjudicate(record, required_bit, base_epsilon=epsilon)
        codes: list[str] = []
        if path_result.blocked_code == E_TRAVERSAL:
            codes.append(E_TRAVERSAL)
        if path_result.blocked_code == E_LINK_ESCAPE or path_result.link_escape:
            codes.append(E_LINK_ESCAPE)
        if effective.shadowed_denies:
            codes.append(E_DENY_SHADOW)
        if effective.inherit_violations:
            codes.append(E_INHERIT)

        verdict = RecordVerdict(
            index=record.index,
            record_id=record.record_id,
            policy_name=policy.name,
            canonical_key=path_result.canonical_key,
            reachable_path=path_result.reachable_path,
            path_error={
                "encoding": path_result.encoding_error,
                "casefold": path_result.casefold_error,
                "link_steps": path_result.link_steps,
                "total": path_result.total_error(),
            },
            effective_permission=effective,
            perm_epsilon=float(epsilon),
            critical=effective.critical,
            codes=codes,
            traversal_depth=path_result.traversal_depth,
        )
        verdicts.append(verdict)
    return verdicts


def detect_policy_conflict(verdicts: list[RecordVerdict]) -> Optional[dict[str, Any]]:
    """同一记录在不同规范化策略下结论不一致。"""
    signatures = {
        v.policy_name: (
            tuple(sorted(v.codes)),
            v.canonical_key,
            v.reachable_path,
            v.critical,
        )
        for v in verdicts
    }
    distinct = {sig[0] for sig in signatures.values()}
    if len(distinct) <= 1:
        return None
    involved = sorted(signatures)
    return {
        "record_id": verdicts[0].record_id,
        "index": verdicts[0].index,
        "policies": involved,
        "signatures": {name: list(sig[0]) for name, sig in signatures.items()},
    }


def audit_file(
    file_id: str,
    path: str,
    root: str,
    records: list[Record],
    rejections: list[RejectedRecord],
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    work_probe: Optional[Callable[[int], None]] = None,
) -> FileResult:
    started = time.monotonic()
    guard = BudgetGuard(budget, work_probe)
    all_verdicts: list[RecordVerdict] = []
    conflicts: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    status = "ok"
    stop_code: Optional[str] = None
    processed = 0

    for record in records:
        stop_code = guard.check(processed)
        if stop_code is not None:
            status = "timeout" if stop_code == E_TIMEOUT else "budget"
            break
        record_verdicts = audit_record(record, root, policies)
        conflict = detect_policy_conflict(record_verdicts)
        if conflict is not None:
            conflicts.append(conflict)
            for verdict in record_verdicts:
                verdict.codes.append(E_POLICY_CONFLICT)
        all_verdicts.extend(record_verdicts)
        verifications.append(verify_record(record, record_verdicts[0]))
        processed += 1

    elapsed = time.monotonic() - started
    file_codes = {code for verdict in all_verdicts for code in verdict.codes}
    file_codes |= {rejection.code for rejection in rejections}
    if stop_code:
        file_codes.add(stop_code)
    file_codes = sorted(file_codes)

    truth_summary = summarize_truth(verifications)
    return FileResult(
        file_id=file_id,
        path=path,
        status=status,
        codes=file_codes,
        verdicts=all_verdicts,
        rejections=rejections,
        elapsed_seconds=elapsed,
        processed_records=processed,
        dropped_records=len(records) - processed,
        truth=truth_summary,
        policy_conflicts=conflicts,
    )
