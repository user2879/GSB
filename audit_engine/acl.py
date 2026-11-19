"""ACL 有效权限裁决。

裁决结果只依赖条目集合而与输入排列顺序无关：
内部按 (source_depth, access, mask, 原始 order) 做全序排序后再处理，
所有输出仅包含语义键（掩码/深度）并做确定性排序。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .models import AclEntry, EffectivePermission, Record
from .permission import accumulate_chain, is_critical


def _canonical_order(entries: Sequence[AclEntry]) -> list[AclEntry]:
    return sorted(
        entries,
        key=lambda e: (e.source_depth, 0 if e.access == "deny" else 1, e.mask, e.order),
    )


def adjudicate(
    record: Record,
    required_bit: int,
    base_epsilon: Fraction = Fraction(0),
) -> EffectivePermission:
    """计算记录主体的有效权限。"""
    subject_entries = [e for e in record.acl if e.subject == record.subject]
    ordered = _canonical_order(subject_entries)

    deny_sources: list[tuple[int, Fraction]] = []
    allow_sources: list[tuple[int, Fraction]] = []
    own_allow_mask = 0
    shadowed: list[dict] = []

    for entry in ordered:
        bound = Fraction(str(entry.epsilon)) if entry.epsilon else Fraction(0)
        if entry.access == "deny":
            deny_sources.append((entry.mask, bound))
        else:
            allow_sources.append((entry.mask, bound))
            if not entry.inherited:
                own_allow_mask |= entry.mask

    denied, deny_epsilon = accumulate_chain(deny_sources)
    allowed, allow_epsilon = accumulate_chain(allow_sources)
    effective = allowed & ~denied
    total_epsilon = deny_epsilon + allow_epsilon + base_epsilon
    critical = is_critical(effective, total_epsilon, required_bit)

    # 拒绝项被同级或更近的允许项覆盖：拒绝位被同深度/更浅深度的允许位抵消
    for entry in ordered:
        if entry.access != "deny":
            continue
        covering_masks = sorted(
            {
                allow.mask
                for allow in ordered
                if allow.access == "allow"
                and allow.source_depth <= entry.source_depth
                and (allow.mask & entry.mask)
            }
        )
        live_bits = entry.mask & effective
        if covering_masks and live_bits == 0:
            shadowed.append(
                {
                    "mask": entry.mask,
                    "source_depth": entry.source_depth,
                    "covered_by_allow_masks": covering_masks,
                }
            )

    inherit_violations = _inherit_violations(ordered, own_allow_mask)
    return EffectivePermission(
        subject=record.subject,
        allowed=allowed,
        denied=denied,
        effective=effective,
        epsilon=float(total_epsilon),
        critical=critical,
        shadowed_denies=sorted(
            shadowed, key=lambda item: (item["source_depth"], item["mask"])
        ),
        inherit_violations=inherit_violations,
    )


def _inherit_violations(
    ordered: Sequence[AclEntry], own_allow_mask: int
) -> list[dict]:
    """继承标记使子路径获得父路径未授权权限。

    规则：存在 inherited allow 授予的位，在所有非继承 allow 中都没有授予，
    且没有对应 deny 约束时，视为继承传播越权。
    """
    own_deny = 0
    for entry in ordered:
        if entry.access == "deny" and not entry.inherited:
            own_deny |= entry.mask
    violations: list[dict] = []
    for entry in ordered:
        if entry.access != "allow" or not entry.inherited:
            continue
        gained = entry.mask & ~own_allow_mask & ~own_deny
        if gained:
            violations.append(
                {"gained_mask": gained, "source_depth": entry.source_depth}
            )
    return sorted(
        violations, key=lambda item: (item["source_depth"], item["gained_mask"])
    )
