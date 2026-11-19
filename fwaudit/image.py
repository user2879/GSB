"""固件镜像头与分区表的严格解析、分区映射重建与结构冲突检测。"""

from dataclasses import dataclass, field

from .binaryio import BinaryView, checked_add_u64
from .errors import (
    AuditError, E_HEADER, E_PART, E_OVERLAP, E_SELFREF,
)

IMAGE_MAGIC = b"FWIM"
HEADER_SIZE = 32
SUPPORTED_VERSIONS = (1, 2)
ALLOWED_PART_TYPES = frozenset([0, 1, 2, 3])
ALLOWED_SLOTS = frozenset([0, 1])
ZERO8 = b"\x00" * 8

# 头部实际为 32 字节：4+2+4+8+8 = 26，保留区实际占 6 字节。
RESERVED_SIZE = HEADER_SIZE - 26
PART_FIXED_SIZE = 20  # u16 名长 + u8 类型 + u8 槽位 + u64 偏移 + u64 长度


@dataclass
class Partition:
    index: int
    name: str
    part_type: int
    slot: int
    payload_offset: int
    payload_length: int
    name_offset: int
    record_offset: int

    @property
    def payload_end(self):
        return self.payload_offset + self.payload_length

    def as_dict(self):
        return {
            "index": self.index,
            "name": self.name,
            "part_type": self.part_type,
            "slot": self.slot,
            "payload_offset": self.payload_offset,
            "payload_length": self.payload_length,
            "payload_end": self.payload_end,
            "name_offset": self.name_offset,
            "record_offset": self.record_offset,
        }


@dataclass
class ImageInfo:
    path: str
    version: int
    part_count: int
    table_offset: int
    declared_length: int
    base_image_id: int
    base_slot: int
    actual_length: int
    table_end: int
    partitions: list
    findings: list = field(default_factory=list)

    def partition_names(self):
        return [p.name for p in self.partitions]

    def get(self, name):
        for part in self.partitions:
            if part.name == name:
                return part
        return None

    @property
    def ok(self):
        return not self.findings

    def as_dict(self):
        return {
            "path": self.path,
            "version": self.version,
            "part_count": self.part_count,
            "table_offset": self.table_offset,
            "declared_length": self.declared_length,
            "actual_length": self.actual_length,
            "base_image_id": self.base_image_id,
            "base_slot": self.base_slot,
            "table_region": [self.table_offset, self.table_end],
            "ok": self.ok,
            "partitions": [p.as_dict() for p in self.partitions],
            "findings": list(self.findings),
        }


def _finding(code, message, offset=None, **extra):
    item = {"code": code, "message": message, "offset": offset}
    item.update(extra)
    return item


def parse_image(data, path="<memory>"):
    """严格解析镜像；任何非法取值抛 AuditError(E_HEADER/E_PART)。"""
    view = BinaryView(data)
    total = len(view)
    if total < HEADER_SIZE:
        raise AuditError(
            E_HEADER,
            "镜像长度 %d 小于 32 字节头部" % total,
            offset=total,
        )

    magic = view.take(4, E_HEADER, "魔数")
    if magic != IMAGE_MAGIC:
        raise AuditError(
            E_HEADER, "魔数应为 b'FWIM'，实际为 %r" % magic, offset=0
        )
    version = view.u16(E_HEADER, "格式版本")
    if version not in SUPPORTED_VERSIONS:
        raise AuditError(
            E_HEADER,
            "格式版本 %d 不在白名单 %s 内" % (version, SUPPORTED_VERSIONS),
            offset=4,
        )
    part_count = view.u32(E_HEADER, "分区数量")
    table_offset = view.u64(E_HEADER, "分区表偏移")
    declared_length = view.u64(E_HEADER, "镜像声明总长")
    reserved = view.take(RESERVED_SIZE, E_HEADER, "保留区")

    if table_offset < HEADER_SIZE:
        raise AuditError(
            E_HEADER,
            "分区表偏移 %d 位于 32 字节头部之内" % table_offset,
            offset=18,
        )
    if table_offset > total:
        raise AuditError(
            E_HEADER,
            "分区表偏移 %d 越过实际文件长度 %d" % (table_offset, total),
            offset=18,
        )
    if declared_length != total:
        raise AuditError(
            E_HEADER,
            "镜像声明总长 %d 与实际文件长度 %d 不一致"
            % (declared_length, total),
            offset=26 - 8,
        )

    # 保留区末 6 字节：前 4 字节承载基镜像标识，其后 1 字节承载基镜像槽位，
    # 末尾 1 字节必须为 0（保留）。
    base_image_id = int.from_bytes(reserved[0:4], "little")
    base_slot = reserved[4]
    if reserved[5] != 0:
        raise AuditError(
            E_HEADER, "保留区尾部保留字节非 0", offset=26 + 5
        )
    if base_slot not in ALLOWED_SLOTS:
        raise AuditError(
            E_HEADER,
            "基镜像槽位 %d 不在白名单 %s 内" % (base_slot, sorted(ALLOWED_SLOTS)),
            offset=26 + 4,
        )

    view.seek(table_offset)
    partitions = []
    seen_names = set()
    for index in range(part_count):
        record_offset = view.pos
        name_len = view.u16(E_PART, "分区名长度")
        if name_len == 0:
            raise AuditError(
                E_PART, "分区名长度为 0", offset=record_offset
            )
        part_type = view.u8(E_PART, "分区类型")
        slot = view.u8(E_PART, "目标槽位")
        payload_offset = view.u64(E_PART, "载荷偏移")
        payload_length = view.u64(E_PART, "载荷长度")
        name_offset = view.pos
        name_bytes = view.take(name_len, E_PART, "分区名")
        try:
            name = name_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise AuditError(
                E_PART,
                "分区 #%d 名称不是合法 UTF-8: %s" % (index, exc.reason),
                offset=name_offset + (exc.start or 0),
            )
        if name in seen_names:
            raise AuditError(
                E_PART, "分区名 %r 重复" % name, offset=name_offset
            )
        seen_names.add(name)

        if part_type not in ALLOWED_PART_TYPES:
            raise AuditError(
                E_PART,
                "分区 %r 类型 %d 不在白名单 %s 内"
                % (name, part_type, sorted(ALLOWED_PART_TYPES)),
                offset=record_offset + 2,
            )
        if slot not in ALLOWED_SLOTS:
            raise AuditError(
                E_PART,
                "分区 %r 槽位 %d 不在白名单 %s 内"
                % (name, slot, sorted(ALLOWED_SLOTS)),
                offset=record_offset + 3,
            )

        payload_end = checked_add_u64(
            payload_offset, payload_length, E_PART,
            record_offset + 4, "分区 %r 载荷偏移加长度" % name,
        )
        if payload_offset < HEADER_SIZE:
            raise AuditError(
                E_PART,
                "分区 %r 载荷偏移 %d 侵入镜像头部" % (name, payload_offset),
                offset=record_offset + 4,
            )
        if payload_end > declared_length:
            raise AuditError(
                E_PART,
                "分区 %r 载荷区间 [%d, %d) 越过镜像声明总长 %d"
                % (name, payload_offset, payload_end, declared_length),
                offset=record_offset + 4,
            )

        partitions.append(
            Partition(
                index=index, name=name, part_type=part_type, slot=slot,
                payload_offset=payload_offset, payload_length=payload_length,
                name_offset=name_offset, record_offset=record_offset,
            )
        )

    table_end = view.pos
    info = ImageInfo(
        path=path, version=version, part_count=part_count,
        table_offset=table_offset, declared_length=declared_length,
        base_image_id=base_image_id, base_slot=base_slot,
        actual_length=total, table_end=table_end, partitions=partitions,
    )
    info.findings = detect_structural_conflicts(info)
    return info


def _intersect(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def detect_structural_conflicts(info):
    """重叠/包含与自引用检测，结论与分区表物理排列顺序无关。"""
    findings = []
    parts = info.partitions
    table_region = (info.table_offset, info.table_end)

    # 自引用：载荷区间覆盖分区表描述区或镜像头部。
    for part in parts:
        if _intersect(part.payload_offset, part.payload_end,
                      info.table_offset, info.table_end):
            findings.append(_finding(
                E_SELFREF,
                "分区 %r 载荷区间 [%d, %d) 覆盖分区表描述区 [%d, %d)"
                % (part.name, part.payload_offset, part.payload_end,
                   info.table_offset, info.table_end),
                offset=part.record_offset + 4,
                partition=part.name,
                payload_region=[part.payload_offset, part.payload_end],
                referenced_region=list(table_region),
            ))
        if _intersect(part.payload_offset, part.payload_end, 0, HEADER_SIZE):
            findings.append(_finding(
                E_SELFREF,
                "分区 %r 载荷区间覆盖镜像头部 [0, %d)" % (part.name, HEADER_SIZE),
                offset=part.record_offset + 4,
                partition=part.name,
                payload_region=[part.payload_offset, part.payload_end],
                referenced_region=[0, HEADER_SIZE],
            ))

    # 两两比较，按名称排序保证输出确定性，与物理顺序无关。
    ordered = sorted(parts, key=lambda p: p.name)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if not _intersect(a.payload_offset, a.payload_end,
                              b.payload_offset, b.payload_end):
                continue
            a_contains_b = (a.payload_offset <= b.payload_offset
                            and b.payload_end <= a.payload_end)
            b_contains_a = (b.payload_offset <= a.payload_offset
                            and a.payload_end <= b.payload_end)
            if a_contains_b and b_contains_a and (
                a.payload_offset == b.payload_offset
                and a.payload_end == b.payload_end
            ):
                relation = "identical"
            elif a_contains_b:
                relation = "contains"
            elif b_contains_a:
                relation = "contained_by"
            else:
                relation = "partial"
            findings.append(_finding(
                E_OVERLAP,
                "分区 %r 与 %r 载荷区间冲突(%s)"
                % (a.name, b.name, relation),
                offset=min(a.record_offset, b.record_offset),
                partitions=[a.name, b.name],
                relation=relation,
                regions={
                    a.name: [a.payload_offset, a.payload_end],
                    b.name: [b.payload_offset, b.payload_end],
                },
            ))
    return findings


def load_image_file(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_image(data, path=path)
