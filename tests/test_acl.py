"""ACL 裁决：拒绝覆盖、继承越权与排列顺序无关性。"""

from __future__ import annotations

import random

from audit_engine.acl import adjudicate
from audit_engine.models import (
    AclEntry,
    PERM_DELETE,
    PERM_READ,
    PERM_WRITE,
    Record,
)


def _record(entries, subject="alice", operation="write", rid="R0") -> Record:
    return Record(
        index=0,
        record_id=rid,
        subject=subject,
        operation=operation,
        target_path="/srv/audit-root/x",
        representation="int_mask",
        perm_value=0,
        rounding="truncate",
        links=(),
        acl=tuple(entries),
        truth=None,
    )


def test_deny_shadowed_by_same_depth_allow():
    entries = [
        AclEntry(subject="alice", access="deny", mask=PERM_WRITE, source_depth=1, order=0),
        AclEntry(subject="alice", access="allow", mask=PERM_READ | PERM_WRITE, source_depth=1, order=1),
    ]
    result = adjudicate(_record(entries), PERM_WRITE)
    assert result.shadowed_denies
    assert result.shadowed_denies[0]["covered_by_allow_masks"] == [PERM_READ | PERM_WRITE]


def test_live_deny_is_not_shadowed():
    entries = [
        AclEntry(subject="alice", access="allow", mask=PERM_READ, source_depth=1, order=0),
        AclEntry(subject="alice", access="deny", mask=PERM_DELETE, source_depth=1, order=1),
    ]
    result = adjudicate(_record(entries, operation="read"), PERM_READ)
    assert not result.shadowed_denies
    assert result.effective == PERM_READ


def test_inherit_grant_detected():
    entries = [
        AclEntry(subject="alice", access="allow", mask=PERM_READ | PERM_WRITE,
                 inherited=True, source_depth=0, order=0),
        AclEntry(subject="alice", access="allow", mask=PERM_READ, source_depth=2, order=1),
    ]
    result = adjudicate(_record(entries), PERM_WRITE)
    assert result.inherit_violations
    assert result.inherit_violations[0]["gained_mask"] == PERM_WRITE


def test_inherit_with_local_grant_is_clean():
    entries = [
        AclEntry(subject="alice", access="allow", mask=PERM_READ, inherited=True,
                 source_depth=0, order=0),
        AclEntry(subject="alice", access="allow", mask=PERM_READ, source_depth=2, order=1),
    ]
    result = adjudicate(_record(entries, operation="read"), PERM_READ)
    assert not result.inherit_violations


def test_order_independence():
    entries = [
        AclEntry(subject="alice", access="allow", mask=PERM_READ | PERM_WRITE,
                 source_depth=1, order=0),
        AclEntry(subject="alice", access="deny", mask=PERM_WRITE, source_depth=1, order=1),
        AclEntry(subject="bob", access="allow", mask=PERM_DELETE, source_depth=2, order=2),
        AclEntry(subject="alice", access="allow", mask=PERM_DELETE, inherited=True,
                 source_depth=0, order=3),
    ]
    baseline = adjudicate(_record(entries), PERM_WRITE)
    rng = random.Random(20261119)
    for trial in range(8):
        shuffled = list(entries)
        rng.shuffle(shuffled)
        reordered = [AclEntry(**{**e.__dict__, "order": i}) for i, e in enumerate(shuffled)]
        other = adjudicate(_record(reordered), PERM_WRITE)
        assert other.effective == baseline.effective
        assert other.shadowed_denies == baseline.shadowed_denies
        assert other.inherit_violations == baseline.inherit_violations
