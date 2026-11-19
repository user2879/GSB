"""解析层：非法记录拒绝且不中断整批。"""

from __future__ import annotations

from audit_engine.errors import E_PARSE
from audit_engine.generator import manifest_parse_errors
from audit_engine.parser import parse_manifest


def test_bad_records_rejected_but_batch_continues():
    records, rejections, metadata = parse_manifest(manifest_parse_errors())
    assert len(records) == 2
    assert len(rejections) == 3
    assert all(rejection.code == E_PARSE for rejection in rejections)
    assert [rejection.index for rejection in rejections] == [1, 2, 3]
    assert metadata["file_id"] == "manifest_parse_errors"


def test_duplicate_identifier_rejected():
    payload = {
        "file_id": "dup",
        "root": "/r",
        "records": [
            {"id": "X", "subject": "s", "operation": "read", "target_path": "/r/a",
             "representation": "int_mask", "perm_value": 1, "acl": []},
            {"id": "X", "subject": "s", "operation": "read", "target_path": "/r/b",
             "representation": "int_mask", "perm_value": 1, "acl": []},
        ],
    }
    records, rejections, _ = parse_manifest(payload)
    assert len(records) == 1
    assert rejections[0].code == E_PARSE and "唯一" in rejections[0].reason


def test_operation_whitelist_enforced():
    payload = {
        "file_id": "op",
        "root": "/r",
        "records": [
            {"id": "1", "subject": "s", "operation": "pwn", "target_path": "/r/a",
             "representation": "int_mask", "perm_value": 1, "acl": []},
        ],
    }
    _, rejections, _ = parse_manifest(payload)
    assert rejections[0].index == 0 and rejections[0].code == E_PARSE


def test_representation_whitelist_enforced():
    payload = {
        "file_id": "rep",
        "root": "/r",
        "records": [
            {"id": "1", "subject": "s", "operation": "read", "target_path": "/r/a",
             "representation": "hex", "perm_value": "0x1", "acl": []},
        ],
    }
    _, rejections, _ = parse_manifest(payload)
    assert rejections[0].code == E_PARSE
