"""基于镜像内真值同步标记的独立验证端。

验证端只信任镜像载荷内的 TRUTH_MARKER 与重算摘要，
不信任分区表中的重复声明；两者逐项比对。
"""

from .generator import (
    read_truth_marker, payload_digest, _filler, MARKER_SIZE,
)
from .image import parse_image


def verify_image_truth(blob):
    """返回 (ok, mismatches)。mismatch 列出每个分区的真值偏差。"""
    info = parse_image(blob, path="<verify>")
    table_by_name = {part.name: part for part in info.partitions}
    mismatches = []
    checked = []

    for part in info.partitions:
        marker = read_truth_marker(blob, part.payload_offset)
        if marker is None:
            mismatches.append({
                "name": part.name,
                "reason": "missing_truth_marker",
            })
            continue
        problems = []
        if marker["name"] != part.name:
            problems.append("name: marker=%r table=%r"
                            % (marker["name"], part.name))
        if marker["slot"] != part.slot:
            problems.append("slot: marker=%d table=%d"
                            % (marker["slot"], part.slot))
        if marker["payload_offset"] != part.payload_offset:
            problems.append("offset: marker=%d table=%d"
                            % (marker["payload_offset"],
                               part.payload_offset))
        if marker["payload_length"] != part.payload_length:
            problems.append("length: marker=%d table=%d"
                            % (marker["payload_length"],
                               part.payload_length))
        recomputed = payload_digest(
            part.name, part.payload_offset, part.payload_length
        ).hex()
        if recomputed != marker["declared_digest"]:
            problems.append("digest mismatch")
        expected_body = _filler(
            part.name, part.payload_offset,
            part.payload_length - MARKER_SIZE,
        )
        actual_body = blob[
            part.payload_offset + MARKER_SIZE:part.payload_end
        ]
        if actual_body != expected_body:
            problems.append("payload body deviates from deterministic truth")
        if problems:
            mismatches.append({
                "name": part.name, "reason": "; ".join(problems),
            })
        checked.append({
            "name": part.name,
            "slot": part.slot,
            "payload_region": [part.payload_offset, part.payload_end],
            "declared_digest": marker["declared_digest"],
            "recomputed_digest": recomputed,
            "marker_size": MARKER_SIZE,
        })
    return not mismatches, {"checked": checked, "mismatches": mismatches}
