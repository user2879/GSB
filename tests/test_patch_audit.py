import pytest

from fwaudit.errors import (
    AuditError, E_PATCH_HEADER, E_PATCH_BOUNDS, E_SLOT_ESCALATION,
    E_WRITE_CHAIN,
)
from fwaudit.generator import build_image, build_patch
from fwaudit.image import parse_image
from fwaudit.patch import parse_patch, audit_patch


def _audit(operations, image_blob, budget, base_image_id=100, corrupt=None):
    patch_blob = build_patch(
        operations, base_image_id=base_image_id, corrupt=corrupt
    )
    patch = parse_patch(patch_blob)
    image = parse_image(image_blob)
    return audit_patch(patch, image, budget)


def test_patch_bad_magic():
    blob = build_patch([], corrupt={"magic": True})
    with pytest.raises(AuditError) as exc:
        parse_patch(blob)
    assert exc.value.code == E_PATCH_HEADER
    assert exc.value.offset == 0


def test_patch_op_count_inconsistent():
    blob = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 4}],
        corrupt={"op_count_delta": 2},
    )
    with pytest.raises(AuditError) as exc:
        parse_patch(blob)
    assert exc.value.code == E_PATCH_HEADER


def test_patch_truncated():
    blob = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 4}],
        corrupt={"truncate": -10},
    )
    with pytest.raises(AuditError):
        parse_patch(blob)


def test_in_bounds_passes(image_blob, budget):
    result = _audit(
        [{"name": "kernel", "write_offset": 0, "write_length": 64},
         {"name": "appfs", "write_offset": 100, "write_length": 200}],
        image_blob, budget,
    )
    assert result["status"] == "pass"
    assert result["codes"] == []
    assert result["coverage_ratio"] == 1.0
    assert all(op["status"] == "pass" for op in result["operations"])


def test_unknown_partition_is_bounds(image_blob, budget):
    result = _audit(
        [{"name": "ghost", "write_offset": 0, "write_length": 8}],
        image_blob, budget,
    )
    assert E_PATCH_BOUNDS in result["codes"]
    assert result["operations"][0]["name"] == "ghost"
    assert result["operations"][0]["findings"][0]["offset"] is not None


def test_write_out_of_partition_bounds(image_blob, budget):
    result = _audit(
        [{"name": "kernel", "write_offset": 500, "write_length": 64}],
        image_blob, budget,
    )
    assert E_PATCH_BOUNDS in result["codes"]
    seq = result["operations"][0]["seq"]
    assert seq == 0


def test_slot_escalation(partitions, budget):
    # kernel 被声明到槽位 1，而基镜像槽位为 0
    partitions[1]["slot"] = 1
    blob = build_image(partitions, base_image_id=100, base_slot=0)
    result = _audit(
        [{"name": "kernel", "write_offset": 0, "write_length": 16}],
        blob, budget,
    )
    assert E_SLOT_ESCALATION in result["codes"]
    finding = next(
        f for f in result["findings"]
        if f["code"] == E_SLOT_ESCALATION
    )
    assert finding["name"] == "kernel"
    assert finding["seq"] == 0


def test_write_chain_merges_and_crosses_boundary(image_blob, budget):
    # 单条都在界内，但合并后触及/跨越边界
    result = _audit(
        [
            {"name": "kernel", "write_offset": 400, "write_length": 60},
            {"name": "kernel", "write_offset": 460, "write_length": 80},
        ],
        image_blob, budget,
    )
    chain_codes = [c["code"] for c in result["write_chains"]]
    assert E_WRITE_CHAIN in chain_codes
    chain = result["write_chains"][0]
    assert chain["operation_seqs"] == [0, 1]
    assert chain["merged_relative_region"][1] > 512


def test_write_chain_order_normalized(image_blob, budget):
    # 逆序输入，合并结论必须与顺序输入一致
    forward = _audit(
        [
            {"name": "kernel", "write_offset": 420, "write_length": 60},
            {"name": "kernel", "write_offset": 480, "write_length": 60},
        ],
        image_blob, budget,
    )
    reverse = _audit(
        [
            {"name": "kernel", "write_offset": 480, "write_length": 60},
            {"name": "kernel", "write_offset": 420, "write_length": 60},
        ],
        image_blob, budget,
    )
    assert forward["write_chains"] == reverse["write_chains"]


def test_non_chain_adjacent_writes_pass(image_blob, budget):
    result = _audit(
        [
            {"name": "kernel", "write_offset": 0, "write_length": 100},
            {"name": "kernel", "write_offset": 100, "write_length": 100},
        ],
        image_blob, budget,
    )
    assert result["write_chains"] == []
    assert result["status"] == "pass"
