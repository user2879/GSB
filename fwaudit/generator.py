"""确定性固件镜像与差分补丁样本生成器。

生成器在每个分区载荷首部写入 64 字节真值同步标记 (TRUTH_MARKER)，
记录分区名、槽位、载荷区间与载荷声明摘要，是验证端唯一真值来源。
所有字节序列均由固定种子/索引确定性推导，不依赖时间、随机源或网络。
"""

import hashlib
import struct

from .image import (
    HEADER_SIZE, PART_FIXED_SIZE, RESERVED_SIZE,
)
from .errors import AuditError, E_HEADER, E_PATCH_HEADER

IMAGE_MAGIC = b"FWIM"
PATCH_MAGIC = b"FWDF"
MARKER_MAGIC = b"FWT1"
MARKER_SIZE = 128


def _filler(name, offset, length):
    """由分区名与位置确定性推导的载荷填充字节。"""
    seed = hashlib.sha256(name.encode("utf-8")).digest()
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(seed + struct.pack("<Q", counter)
                                  + struct.pack("<Q", offset)).digest())
        counter += 1
    return bytes(out[:length])


def truth_marker(name, slot, payload_offset, payload_length, digest):
    """构造 128 字节真值同步标记。"""
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > MARKER_SIZE - 20:
        raise ValueError("分区名过长，无法写入真值标记")
    if len(digest) != 32:
        raise ValueError("摘要必须为 32 字节")
    header = (
        MARKER_MAGIC
        + struct.pack("<H", len(name_bytes))
        + name_bytes
    )
    body = struct.pack(
        "<BQQ", slot, payload_offset, payload_length
    ) + digest
    marker = header + body
    if len(marker) > MARKER_SIZE:
        raise ValueError("真值标记超过 128 字节")
    return marker + b"\x00" * (MARKER_SIZE - len(marker))


def read_truth_marker(blob, payload_offset):
    """从载荷首部解析真值标记，非法时返回 None。"""
    if payload_offset + MARKER_SIZE > len(blob):
        return None
    marker = blob[payload_offset:payload_offset + MARKER_SIZE]
    if marker[0:4] != MARKER_MAGIC:
        return None
    name_len = struct.unpack("<H", marker[4:6])[0]
    name = marker[6:6 + name_len].decode("utf-8", errors="replace")
    cursor = 6 + name_len
    slot, poff, plen = struct.unpack("<BQQ", marker[cursor:cursor + 17])
    digest = marker[cursor + 17:cursor + 49]
    return {
        "name": name,
        "slot": slot,
        "payload_offset": poff,
        "payload_length": plen,
        "declared_digest": digest.hex(),
        "raw": marker,
    }


def payload_digest(name, payload_offset, payload_length):
    """对不含真值标记的载荷内容重算摘要（与生成端声明摘要一致）。"""
    body = _filler(
        name, payload_offset, max(0, payload_length - MARKER_SIZE)
    )
    return hashlib.sha256(body).digest()


def build_image(partitions, version=1, base_image_id=1, base_slot=0,
                table_offset=None, corrupt=None):
    """构造镜像字节串。

    partitions: [{"name", "part_type", "slot", "payload_length"}]
    corrupt: None 或 dict，支持键:
      magic, version, declared_delta, table_offset_delta, truncate,
      reserved_tail, bad_name(index), bad_type(index), bad_slot(index),
      force_overflow(index)
    """
    corrupt = corrupt or {}
    if table_offset is None:
        table_size = sum(
            PART_FIXED_SIZE + len(p["name"].encode("utf-8"))
            for p in partitions
        )
        table_offset = HEADER_SIZE

    cursor = table_offset + sum(
        PART_FIXED_SIZE + len(p["name"].encode("utf-8"))
        for p in partitions
    )
    records = bytearray()
    payload_blobs = []
    for index, spec in enumerate(partitions):
        name_bytes = spec["name"].encode("utf-8")
        payload_length = spec["payload_length"]
        if payload_length < MARKER_SIZE:
            raise ValueError("分区 %r 载荷长度至少为 %d"
                             % (spec["name"], MARKER_SIZE))
        payload_offset = cursor
        digest = payload_digest(spec["name"], payload_offset, payload_length)
        body = _filler(
            spec["name"], payload_offset,
            payload_length - MARKER_SIZE
        )
        payload = truth_marker(
            spec["name"], spec["slot"], payload_offset,
            payload_length, digest,
        ) + body

        part_type = spec["part_type"]
        slot = spec["slot"]
        if corrupt.get("bad_type") == index:
            part_type = 250
        if corrupt.get("bad_slot") == index:
            slot = 250
        record = (
            struct.pack("<H", len(name_bytes))
            + struct.pack("<BB", part_type, slot)
            + struct.pack("<QQ", payload_offset, payload_length)
            + name_bytes
        )
        if corrupt.get("bad_name") == index:
            if len(name_bytes) < 2:
                raise ValueError("等长坏 UTF-8 注入要求分区名至少 2 字节")
            broken_name = name_bytes[:-2] + b"\xff\xfe"
            record = (
                struct.pack("<H", len(name_bytes))
                + struct.pack("<BB", part_type, slot)
                + struct.pack("<QQ", payload_offset, payload_length)
                + broken_name
            )
        if corrupt.get("force_overflow") == index:
            record = (
                record[:4]
                + struct.pack("<QQ", (1 << 64) - 4, 8)
                + name_bytes
            )
        records.extend(record)
        payload_blobs.append(payload)
        cursor += payload_length

    declared_length = cursor
    reserved = (
        struct.pack("<I", base_image_id)
        + struct.pack("<B", base_slot)
        + b"\x00"
    )
    reserved = reserved + b"\x00" * (RESERVED_SIZE - len(reserved))
    if corrupt.get("reserved_tail"):
        reserved = reserved[:-1] + b"\x01"

    header = (
        IMAGE_MAGIC
        + struct.pack("<H", corrupt.get("version", version))
        + struct.pack("<I", len(partitions))
        + struct.pack(
            "<Q", table_offset + corrupt.get("table_offset_delta", 0)
        )
        + struct.pack(
            "<Q", declared_length + corrupt.get("declared_delta", 0)
        )
        + reserved
    )
    if len(header) != HEADER_SIZE:  # 防御性自检
        raise AssertionError("header size %d != 32" % len(header))
    if corrupt.get("magic"):
        header = b"XXXX" + header[4:]

    blob = bytearray(table_offset)
    blob[0:HEADER_SIZE] = header
    blob.extend(records)
    for payload in payload_blobs:
        blob.extend(payload)
    if "truncate" in corrupt:
        blob = blob[: max(0, len(blob) + corrupt["truncate"])]
    return bytes(blob)


def build_patch(operations, version=1, base_image_id=1, corrupt=None):
    """构造补丁字节串。

    operations: [{"name", "write_offset", "write_length", "digest"(可选)}]
    corrupt: None 或 dict，支持键: magic, version, op_count_delta, truncate
    """
    corrupt = corrupt or {}
    body = bytearray()
    for index, op in enumerate(operations):
        name_bytes = op["name"].encode("utf-8")
        digest = op.get("digest")
        if digest is None:
            digest = hashlib.sha256(
                b"op-%d-%s" % (index, name_bytes)
            ).digest()
        if len(digest) != 32:
            raise ValueError("操作摘要必须为 32 字节")
        body.extend(
            struct.pack("<H", len(name_bytes))
            + struct.pack("<QI", op["write_offset"], op["write_length"])
            + name_bytes
            + digest
        )
    header = (
        PATCH_MAGIC
        + struct.pack("<H", corrupt.get("version", version))
        + struct.pack(
            "<I", len(operations) + corrupt.get("op_count_delta", 0)
        )
        + struct.pack("<I", base_image_id)
        + b"\x00\x00"
    )
    if len(header) != 16:
        raise AssertionError("patch header size %d != 16" % len(header))
    if corrupt.get("magic"):
        header = b"XXXX" + header[4:]
    blob = header + bytes(body)
    if "truncate" in corrupt:
        blob = blob[: max(0, len(blob) + corrupt["truncate"])]
    return bytes(blob)


def write_sample(path, blob):
    with open(path, "wb") as handle:
        handle.write(blob)
    return path
