"""差分补丁解析、写入边界校验、跨槽位越权与累积写链检测。"""

from dataclasses import dataclass, field
import time

from .binaryio import BinaryView, checked_add_u64
from .errors import (
    AuditError, E_PATCH_HEADER, E_PATCH_BOUNDS, E_SLOT_ESCALATION,
    E_WRITE_CHAIN, E_TIMEOUT, E_OPS, E_STALE_BASE,
)
PATCH_MAGIC = b"FWDF"
PATCH_HEADER_SIZE = 16
OP_FIXED_SIZE = 14  # u16 名长 + u64 写入偏移 + u32 写入长度
DIGEST_SIZE = 32
SUPPORTED_PATCH_VERSIONS = (1,)


@dataclass
class PatchOp:
    seq: int
    name: str
    write_offset: int
    write_length: int
    digest_hex: str
    record_offset: int
    status: str = "pending"
    findings: list = field(default_factory=list)

    @property
    def write_end(self):
        return self.write_offset + self.write_length

    def as_dict(self):
        return {
            "seq": self.seq,
            "name": self.name,
            "write_offset": self.write_offset,
            "write_length": self.write_length,
            "write_end": self.write_end,
            "digest": self.digest_hex,
            "record_offset": self.record_offset,
            "status": self.status,
            "findings": list(self.findings),
        }


@dataclass
class BudgetConfig:
    image_byte_threshold: int = 4096
    op_count_threshold: int = 16
    chunk_size: int = 8
    max_ops_per_patch: int = 4096
    timeout_seconds: float = 5.0
    concurrency: int = 2
    policy_version: int = 1

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class PatchFile:
    path: str
    version: int
    op_count_declared: int
    base_image_id: int
    actual_length: int
    operations: list

    def as_dict(self):
        return {
            "path": self.path,
            "version": self.version,
            "op_count_declared": self.op_count_declared,
            "base_image_id": self.base_image_id,
            "actual_length": self.actual_length,
        }


def parse_patch(data, path="<memory>"):
    """严格解析补丁；非法时抛 AuditError(E_PATCH_HEADER)。"""
    view = BinaryView(data)
    total = len(view)
    if total < PATCH_HEADER_SIZE:
        raise AuditError(
            E_PATCH_HEADER,
            "补丁长度 %d 小于 16 字节补丁头" % total,
            offset=total,
        )
    magic = view.take(4, E_PATCH_HEADER, "补丁魔数")
    if magic != PATCH_MAGIC:
        raise AuditError(
            E_PATCH_HEADER, "补丁魔数应为 b'FWDF'，实际为 %r" % magic,
            offset=0,
        )
    version = view.u16(E_PATCH_HEADER, "补丁版本")
    if version not in SUPPORTED_PATCH_VERSIONS:
        raise AuditError(
            E_PATCH_HEADER,
            "补丁版本 %d 不在白名单 %s 内"
            % (version, SUPPORTED_PATCH_VERSIONS),
            offset=4,
        )
    op_count = view.u32(E_PATCH_HEADER, "操作数量")
    base_image_id = view.u32(E_PATCH_HEADER, "基镜像标识")
    reserved = view.take(2, E_PATCH_HEADER, "补丁保留区")
    if reserved != b"\x00\x00":
        raise AuditError(
            E_PATCH_HEADER, "补丁头保留区非 0", offset=14
        )

    operations = []
    for seq in range(op_count):
        record_offset = view.pos
        try:
            name_len = view.u16(E_PATCH_HEADER, "目标分区名长度")
            if name_len == 0:
                raise AuditError(
                    E_PATCH_HEADER, "操作 #%d 分区名长度为 0" % seq,
                    offset=record_offset,
                )
            write_offset = view.u64(E_PATCH_HEADER, "写入偏移")
            write_length = view.u32(E_PATCH_HEADER, "写入长度")
            name_bytes = view.take(name_len, E_PATCH_HEADER, "分区名字节")
            name = name_bytes.decode("utf-8", errors="strict")
            digest = view.take(DIGEST_SIZE, E_PATCH_HEADER, "新载荷摘要")
        except UnicodeDecodeError as exc:
            raise AuditError(
                E_PATCH_HEADER,
                "操作 #%d 分区名不是合法 UTF-8" % seq,
                offset=record_offset,
            )
        operations.append(
            PatchOp(
                seq=seq, name=name, write_offset=write_offset,
                write_length=write_length, digest_hex=digest.hex(),
                record_offset=record_offset,
            )
        )

    if view.pos != total:
        raise AuditError(
            E_PATCH_HEADER,
            "操作数量 %d 与文件长度不自洽：解析至偏移 %d，文件总长 %d"
            % (op_count, view.pos, total),
            offset=view.pos,
        )
    return PatchFile(
        path=path, version=version, op_count_declared=op_count,
        base_image_id=base_image_id, actual_length=total,
        operations=operations,
    )


def load_patch_file(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_patch(data, path=path)


def _op_finding(code, message, op, **extra):
    item = {
        "code": code, "message": message, "offset": op.record_offset,
        "seq": op.seq, "name": op.name,
    }
    item.update(extra)
    return item


def _merge_intervals(intervals):
    """区间 (start, end, seqs) 归一化排序后合并。"""
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged = []
    for start, end, seq in ordered:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, seqs = merged[-1]
            merged[-1] = (
                prev_start, max(prev_end, end),
                sorted(seqs + [seq]),
            )
        else:
            merged.append((start, end, [seq]))
    return merged


def audit_patch(patch, image, budget, now=time.monotonic):
    """对单个补丁执行完整审计，返回可 JSON 化的结论字典。

    单补丁内任何异常都被收敛为结论，绝不抛给调用方，保证同批隔离。
    """
    started_at = now()
    result = {
        "path": patch.path,
        "base_image_id": patch.base_image_id,
        "status": "pass",
        "codes": [],
        "op_count": len(patch.operations),
        "operations": [],
        "degraded": False,
        "coverage_ratio": 1.0,
        "uncovered_ops": [],
        "chunk_size": budget.chunk_size,
        "elapsed_seconds": 0.0,
        "findings": [],
        "write_chains": [],
        "stale": False,
    }

    def add_code(code):
        if code not in result["codes"]:
            result["codes"].append(code)

    # 超操作上限：立即判 E_OPS，未校验部分不得报告为通过。
    if len(patch.operations) > budget.max_ops_per_patch:
        add_code(E_OPS)
        result["status"] = "fail"
        result["findings"].append({
            "code": E_OPS,
            "message": "操作数 %d 超过单补丁上限 %d"
                       % (len(patch.operations), budget.max_ops_per_patch),
            "offset": PATCH_HEADER_SIZE + 4,
        })
        for op in patch.operations:
            op.status = "skipped_ops_cap"
            result["operations"].append(op.as_dict())
        result["coverage_ratio"] = 0.0
        result["uncovered_ops"] = [op.seq for op in patch.operations]
        result["elapsed_seconds"] = now() - started_at
        return result

    # 换代失效：基镜像标识或格式版本不匹配当前镜像代次。
    if (patch.base_image_id != image.base_image_id
            or image.version not in (1, 2)):
        add_code(E_STALE_BASE)
        result["stale"] = True
        result["status"] = "fail"
        result["findings"].append({
            "code": E_STALE_BASE,
            "message": "补丁基镜像标识 %d 与当前镜像标识 %d 不匹配"
                       % (patch.base_image_id, image.base_image_id),
            "offset": 12,
            "expected_base_image_id": image.base_image_id,
        })
        for op in patch.operations:
            op.status = "skipped_stale"
            result["operations"].append(op.as_dict())
        result["elapsed_seconds"] = now() - started_at
        return result

    # 降级判定：镜像字节数或操作数超过阈值即分块校验。
    degraded = (
        image.actual_length > budget.image_byte_threshold
        or len(patch.operations) > budget.op_count_threshold
    )
    result["degraded"] = degraded
    chunk_size = max(1, budget.chunk_size)

    by_name = {}
    timed_out = False
    covered = 0
    for op in patch.operations:
        if now() - started_at > budget.timeout_seconds:
            timed_out = True
            op.status = "skipped_timeout"
            result["uncovered_ops"].append(op.seq)
            result["operations"].append(op.as_dict())
            continue

        # 分块模式下仅处理落在完整分块内的操作。
        if degraded and (op.seq // chunk_size) * chunk_size + chunk_size \
                > len(patch.operations):
            op.status = "skipped_degraded"
            result["uncovered_ops"].append(op.seq)
            result["operations"].append(op.as_dict())
            continue

        covered += 1
        target = image.get(op.name)
        if target is None:
            op.status = "fail"
            finding = _op_finding(
                E_PATCH_BOUNDS,
                "操作 #%d 目标分区 %r 不在基镜像分区集中"
                % (op.seq, op.name),
                op,
            )
            op.findings.append(finding)
            result["findings"].append(finding)
            add_code(E_PATCH_BOUNDS)
            result["operations"].append(op.as_dict())
            continue

        op_failed = False

        # 跨槽位越权：分区槽位与补丁基镜像槽位不一致。
        if target.slot != image.base_slot:
            op_failed = True
            finding = _op_finding(
                E_SLOT_ESCALATION,
                "操作 #%d 目标分区 %r 槽位 %d 与基镜像槽位 %d 不一致"
                % (op.seq, op.name, target.slot, image.base_slot),
                op,
                partition_slot=target.slot,
                base_slot=image.base_slot,
            )
            op.findings.append(finding)
            result["findings"].append(finding)
            add_code(E_SLOT_ESCALATION)

        # 相对分区边界：write_offset/write_length 以分区载荷起点为原点。
        rel_end = checked_add_u64(
            op.write_offset, op.write_length, E_PATCH_BOUNDS,
            op.record_offset + 2, "操作 #%d 写入偏移加长度" % op.seq,
        )
        if rel_end > target.payload_length:
            op_failed = True
            finding = _op_finding(
                E_PATCH_BOUNDS,
                "操作 #%d 写入区间 [%d, %d) 超出分区 %r 载荷长度 %d"
                % (op.seq, op.write_offset, rel_end,
                   op.name, target.payload_length),
                op,
                partition_region=[
                    target.payload_offset, target.payload_end
                ],
                absolute_write_region=[
                    target.payload_offset + op.write_offset,
                    target.payload_offset + rel_end,
                ],
            )
            op.findings.append(finding)
            result["findings"].append(finding)
            add_code(E_PATCH_BOUNDS)

        op.status = "fail" if op_failed else "pass"
        if op_failed:
            result["status"] = "fail"
        result["operations"].append(op.as_dict())
        by_name.setdefault(op.name, []).append(
            (op.write_offset, rel_end, op.seq)
        )

    total_ops = len(patch.operations)
    result["coverage_ratio"] = (
        round(covered / total_ops, 6) if total_ops else 1.0
    )

    # 累积写链：同一分区多操作归一化合并后跨界或覆盖描述区。
    for name in sorted(by_name):
        target = image.get(name)
        if target is None:
            continue
        for start, end, seqs in _merge_intervals(by_name[name]):
            chain_violation = False
            reasons = []
            if end > target.payload_length:
                chain_violation = True
                reasons.append("合并区间端点 %d 跨越分区边界 %d"
                               % (end, target.payload_length))
            abs_start = target.payload_offset + start
            abs_end = target.payload_offset + end
            if target.payload_offset < image.table_end and abs_start < image.table_end:
                chain_violation = True
                reasons.append("合并绝对区间 [%d, %d) 覆盖分区表描述区"
                               % (abs_start, abs_end))
            if chain_violation and len(seqs) >= 1:
                chain = {
                    "code": E_WRITE_CHAIN,
                    "name": name,
                    "operation_seqs": seqs,
                    "merged_relative_region": [start, end],
                    "merged_absolute_region": [abs_start, abs_end],
                    "partition_region": [
                        target.payload_offset, target.payload_end
                    ],
                    "reasons": reasons,
                }
                result["write_chains"].append(chain)
                result["findings"].append({
                    "code": E_WRITE_CHAIN,
                    "message": "分区 %r 越权写入链: %s"
                               % (name, "; ".join(reasons)),
                    "offset": target.record_offset,
                    **chain,
                })
                add_code(E_WRITE_CHAIN)
                result["status"] = "fail"
                for seq in seqs:
                    for op_view in result["operations"]:
                        if op_view["seq"] == seq and op_view["status"] == "pass":
                            op_view["status"] = "fail_chain"

    if timed_out:
        add_code(E_TIMEOUT)
        result["status"] = "fail"
        result["findings"].append({
            "code": E_TIMEOUT,
            "message": "补丁审计实测耗时超过 %.3f 秒" % budget.timeout_seconds,
            "offset": None,
            "timeout_seconds": budget.timeout_seconds,
        })

    if degraded:
        result["coverage_note"] = (
            "分块降级：仅校验完整分块内操作，未覆盖面见 uncovered_ops"
        )
    result["elapsed_seconds"] = round(now() - started_at, 6)
    return result
