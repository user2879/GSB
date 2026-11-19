"""本地确定性样本生成器。

不使用系统时钟或随机熵；所有数据为固定构造。每条记录显式携带
发送端同步标记 truth（原始路径字节、原始权限数值、预期误差界区间），
验证模块只依据该字段与引擎规范化结果对照。
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from typing import Any

from .models import PERM_DELETE, PERM_EXECUTE, PERM_READ, PERM_WRITE
from .permission import parse_permission

ROOT = "/srv/audit-root"
SYNC_EPOCH = "SYNC-2026-11-19T00:00:00Z"  # 固定同步标记，不取自系统时钟


def _truth(index: int, target_path: str, representation: str, perm_value: Any,
           rounding: str, acl_raw: list[dict[str, Any]]) -> dict[str, Any]:
    """生成发送端真值：误差界区间按与接收端相同的公开解析规则独立计算。"""
    parsed = parse_permission(representation, perm_value, rounding)
    epsilon = Fraction(str(parsed["epsilon"]))
    for entry in acl_raw:
        sub = parse_permission(entry["representation"], entry["mask"], rounding)
        epsilon += Fraction(str(sub["epsilon"]))
    return {
        "sync_marker": f"{SYNC_EPOCH}#REC{index:04d}",
        "raw_path_bytes": target_path.encode("utf-8").hex(),
        "raw_perm_value": perm_value,
        "expected_epsilon": [float(epsilon), float(epsilon)],
    }


def _acl_allow(subject: str, mask: Any, representation: str = "int_mask",
               inherited: bool = False, depth: int = 0) -> dict[str, Any]:
    return {"subject": subject, "access": "allow", "mask": mask,
            "representation": representation, "inherited": inherited, "source_depth": depth}


def _acl_deny(subject: str, mask: Any, representation: str = "int_mask",
              inherited: bool = False, depth: int = 0) -> dict[str, Any]:
    return {"subject": subject, "access": "deny", "mask": mask,
            "representation": representation, "inherited": inherited, "source_depth": depth}


def _record(index: int, rid: str, subject: str, operation: str, target: str,
            representation: str, perm_value: Any, acl: list[dict[str, Any]],
            rounding: str = "truncate", links: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": rid,
        "subject": subject,
        "operation": operation,
        "target_path": target,
        "representation": representation,
        "perm_value": perm_value,
        "rounding": rounding,
        "links": links or [],
        "acl": acl,
        "truth": _truth(index, target, representation, perm_value, rounding, acl),
    }


def _manifest(file_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"file_id": file_id, "root": ROOT, "records": records}


def manifest_traversal() -> dict[str, Any]:
    records = [
        _record(0, "TRV-000", "alice", "read", f"{ROOT}/docs/../secret/key.txt",
                "int_mask", 1, [_acl_allow("alice", PERM_READ, depth=1)]),
        _record(1, "TRV-001", "alice", "write", f"{ROOT}/a/b/../../../../etc/passwd",
                "int_mask", 3, [_acl_allow("alice", PERM_READ | PERM_WRITE, depth=1)]),
        _record(2, "TRV-002", "bob", "read", f"{ROOT}/docs/./report.txt",
                "int_mask", 1, [_acl_allow("bob", PERM_READ, depth=1)]),
    ]
    return _manifest("manifest_traversal", records)


def manifest_link_escape() -> dict[str, Any]:
    links = [
        {"at": f"{ROOT}/docs/outside", "target": "/etc/outside",
         "link_type": "symlink", "steps": 1},
        {"at": f"{ROOT}/docs/hard", "target": f"{ROOT}/var/data.bin",
         "link_type": "hardlink", "steps": 1},
    ]
    records = [
        _record(0, "LNK-000", "carol", "read", f"{ROOT}/docs/outside/file.txt",
                "int_mask", 1, [_acl_allow("carol", PERM_READ, depth=2)], links=links),
        _record(1, "LNK-001", "carol", "execute", f"{ROOT}/docs/hard",
                "int_mask", 4, [_acl_allow("carol", PERM_EXECUTE, depth=2)], links=links),
    ]
    return _manifest("manifest_link_escape", records)


def manifest_representation_mix() -> dict[str, Any]:
    # 八进制 "12.4" => 整数 0o12=10，截断误差 4/8 = 0.5
    records = [
        _record(0, "MIX-000", "dave", "write", f"{ROOT}/app/run.sh",
                "octal_string", "12.4",
                [_acl_allow("dave", "3.2", representation="octal_string", depth=1)],
                rounding="truncate"),
        _record(1, "MIX-001", "dave", "read", f"{ROOT}/app/conf.cfg",
                "octal_string", "7",
                [_acl_deny("dave", 1, depth=1),
                 _acl_allow("dave", 1, depth=2)],
                rounding="truncate"),
        _record(2, "MIX-002", "erin", "delete", f"{ROOT}/app/x.tmp",
                "octal_string", "10.5",
                [_acl_allow("erin", "11", representation="octal_string", depth=1)],
                rounding="nearest"),
        # 临界不确定：允许点估计只含读位(1)；三条八进制来源（记录自身
        # 与两条 ACL）在 truncate 口径下各携带 63/64 截断误差，累计
        # 误差界 189/64 >= 2，恰可翻转写位（位值 2），允许/拒绝在误差
        # 邻域内翻转，禁止给出确定越权结论。
        _record(3, "MIX-003", "peggy", "write", f"{ROOT}/app/border.bin",
                "octal_string", "1.77",
                [
                    _acl_allow("peggy", "1.77", representation="octal_string", depth=1),
                    _acl_allow("peggy", "1.77", representation="octal_string", depth=2),
                ],
                rounding="truncate"),
    ]
    return _manifest("manifest_representation_mix", records)


def manifest_deny_shadow() -> dict[str, Any]:
    records = [
        _record(0, "DSH-000", "frank", "write", f"{ROOT}/shared/plan.doc",
                "int_mask", 3,
                [_acl_deny("frank", PERM_WRITE, depth=1),
                 _acl_allow("frank", PERM_READ | PERM_WRITE, depth=1)],
                ),
        _record(1, "DSH-001", "grace", "read", f"{ROOT}/shared/note.doc",
                "int_mask", 1,
                [_acl_allow("grace", PERM_READ, depth=2),
                 _acl_deny("grace", PERM_DELETE, depth=2)],
                ),
    ]
    return _manifest("manifest_deny_shadow", records)


def manifest_inherit() -> dict[str, Any]:
    records = [
        _record(0, "INH-000", "henry", "write", f"{ROOT}/parent/child/leaf.txt",
                "int_mask", 2,
                [_acl_allow("henry", PERM_READ | PERM_WRITE, inherited=True, depth=0),
                 _acl_allow("henry", PERM_READ, depth=2)],
                ),
        _record(1, "INH-001", "iris", "read", f"{ROOT}/parent/child/ok.txt",
                "int_mask", 1,
                [_acl_allow("iris", PERM_READ, inherited=True, depth=0),
                 _acl_allow("iris", PERM_READ, depth=2)],
                ),
    ]
    return _manifest("manifest_inherit", records)


def manifest_policy_conflict() -> dict[str, Any]:
    # U+00FF（ñ 单码点）与 U+006E U+0303（n + 组合符）NFC 后等价；
    # 大小写变体在 fold_case 策略下与受控对象同键，配合链接产生不同结论。
    links = [
        {"at": f"{ROOT}/data/Café", "target": "/etc/escape",
         "link_type": "symlink", "steps": 1},
    ]
    records = [
        _record(0, "PCF-000", "judy", "read", f"{ROOT}/data/café/secret.txt",
                "int_mask", 1,
                [_acl_allow("judy", PERM_READ, depth=2)], links=links),
        _record(1, "PCF-001", "judy", "read", f"{ROOT}/data/Cafe\u0301/secret.txt",
                "int_mask", 1,
                [_acl_allow("judy", PERM_READ, depth=2)], links=links),
    ]
    return _manifest("manifest_policy_conflict", records)


def manifest_budget() -> dict[str, Any]:
    records = [
        _record(i, f"BUD-{i:03d}", "kevin", "read", f"{ROOT}/bulk/f{i}.bin",
                "int_mask", 1, [_acl_allow("kevin", PERM_READ, depth=1)])
        for i in range(12)
    ]
    return _manifest("manifest_budget", records)


def manifest_timeout() -> dict[str, Any]:
    records = [
        _record(i, f"TO-{i:03d}", "laura", "read", f"{ROOT}/slow/f{i}.bin",
                "int_mask", 1, [_acl_allow("laura", PERM_READ, depth=1)])
        for i in range(6)
    ]
    return _manifest("manifest_timeout", records)


def manifest_parse_errors() -> dict[str, Any]:
    good = _record(0, "PRS-000", "mike", "read", f"{ROOT}/ok.txt",
                   "int_mask", 1, [_acl_allow("mike", PERM_READ, depth=1)])
    bad_operation = {
        "id": "PRS-001", "subject": "mike", "operation": "pwn",
        "target_path": f"{ROOT}/x.txt", "representation": "int_mask",
        "perm_value": 1, "acl": [],
    }
    bad_representation = {
        "id": "PRS-002", "subject": "nina", "operation": "read",
        "target_path": f"{ROOT}/y.txt", "representation": "decimal_string",
        "perm_value": "7", "acl": [],
    }
    bad_path = {
        "id": "PRS-003", "subject": "nina", "operation": "read",
        "target_path": f"{ROOT}/bad\x00name", "representation": "int_mask",
        "perm_value": 1, "acl": [],
    }
    good2 = _record(4, "PRS-004", "oscar", "execute", f"{ROOT}/run.bin",
                    "int_mask", 4, [_acl_allow("oscar", PERM_EXECUTE, depth=1)])
    return _manifest("manifest_parse_errors", [good, bad_operation, bad_representation, bad_path, good2])


GENERATORS = [
    manifest_traversal,
    manifest_link_escape,
    manifest_representation_mix,
    manifest_deny_shadow,
    manifest_inherit,
    manifest_policy_conflict,
    manifest_budget,
    manifest_timeout,
    manifest_parse_errors,
]


def generate_all(output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for factory in GENERATORS:
        manifest = factory()
        target = os.path.join(output_dir, f"{manifest['file_id']}.json")
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        written.append(target)
    return written
