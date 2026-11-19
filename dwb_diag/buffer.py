"""双写缓冲文件(DWB)的镜像编码、追加与扫描校验。

每条镜像 = 56 字节定长头 + 页帧内容:
    magic 4s(DWBM) | page_no I | page_size I | seq Q | length I | digest 32s | payload(length)

解码校验(要求 1):
  - 魔数/截断等结构性错误
  - 页号在容量范围内
  - 页大小属于受支持集合
  - 摘要与内容匹配
  - 写入序号在整个缓冲内严格递增
任一不合法以 E_PAGE 整体拒绝, 并稳定输出首个出错镜像的缓冲偏移与页号。
恢复扫描使用容错模式: 结构错误仍抛 E_PAGE, 摘要不匹配的镜像仅计数丢弃。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DWB_DIGEST_LEN,
    DWB_HEADER_SIZE,
    DWB_MAGIC,
    SUPPORTED_PAGE_SIZES,
)
from .errors import PageFormatError
from .paging import Page
from .util import fsync_file, sha256_digest

_HEADER = struct.Struct("<4sIIQ I 32s")


@dataclass(frozen=True)
class Mirror:
    offset: int
    page_no: int
    page_size: int
    seq: int
    frame: bytes
    digest: bytes

    def to_row(self) -> dict[str, object]:
        return {
            "offset": self.offset,
            "page_no": self.page_no,
            "page_size": self.page_size,
            "seq": self.seq,
            "digest": self.digest.hex(),
        }


@dataclass
class ScanResult:
    mirrors: list[Mirror]
    usable: list[Mirror]          # 摘要匹配的镜像(按文件顺序)
    discarded: list[Mirror]       # 结构合法但摘要不匹配而丢弃的镜像
    discarded_count: int
    max_seq: int
    latest_per_page: dict[int, Mirror]


class DoubleWriteBuffer:
    def __init__(self, path: Path, capacity: int) -> None:
        self.path = Path(path)
        self.capacity = capacity

    # ---------- 写入 ----------
    def append_mirror(self, page_no: int, page_size: int, seq: int, frame: bytes) -> Mirror:
        if not isinstance(page_no, int) or not (0 <= page_no < self.capacity):
            raise PageFormatError(
                f"page_no {page_no} out of capacity {self.capacity}",
                offset=self.path.stat().st_size if self.path.exists() else 0,
                page_no=page_no,
            )
        if page_size not in SUPPORTED_PAGE_SIZES:
            raise PageFormatError(
                f"unsupported page_size {page_size}",
                offset=self.path.stat().st_size if self.path.exists() else 0,
                page_no=page_no,
            )
        if len(frame) != page_size:
            raise PageFormatError(
                "frame length differs from page_size",
                offset=self.path.stat().st_size if self.path.exists() else 0,
                page_no=page_no,
            )
        digest = sha256_digest(frame)
        header = _HEADER.pack(
            DWB_MAGIC, page_no, page_size, seq, len(frame), digest
        )
        assert len(header) == DWB_HEADER_SIZE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        offset = self.path.stat().st_size if self.path.exists() else 0
        with open(self.path, "ab") as fh:
            fh.write(header)
            fh.write(frame)
            fsync_file(fh)
        return Mirror(offset, page_no, page_size, seq, frame, digest)

    def file_digest_hex(self) -> str:
        from .util import sha256_hex

        if not self.path.exists():
            return sha256_hex(b"")
        with open(self.path, "rb") as fh:
            return sha256_hex(fh.read())

    # ---------- 读取 ----------
    def _read_raw(self) -> bytes:
        if not self.path.exists():
            return b""
        with open(self.path, "rb") as fh:
            return fh.read()

    def scan(self, *, strict_digest: bool) -> ScanResult:
        """扫描缓冲。

        strict_digest=True : 摘要不匹配同样抛 E_PAGE(校验命令/截断注入)。
        strict_digest=False: 结构错误抛 E_PAGE; 摘要不匹配镜像计入丢弃(恢复)。
        """
        blob = self._read_raw()
        pos = 0
        prev_seq: int | None = None
        mirrors: list[Mirror] = []
        usable: list[Mirror] = []
        discarded: list[Mirror] = []

        while pos < len(blob):
            record_offset = pos
            if len(blob) - pos < DWB_HEADER_SIZE:
                raise PageFormatError(
                    "truncated mirror header",
                    offset=record_offset,
                    page_no=None,
                    reason="truncated_header",
                )
            magic, page_no, page_size, seq, length, digest = _HEADER.unpack(
                blob[pos : pos + DWB_HEADER_SIZE]
            )
            if magic != DWB_MAGIC:
                raise PageFormatError(
                    "bad mirror magic",
                    offset=record_offset,
                    page_no=page_no,
                    reason="bad_magic",
                )
            if not (0 <= page_no < self.capacity):
                raise PageFormatError(
                    f"page_no {page_no} out of capacity {self.capacity}",
                    offset=record_offset,
                    page_no=page_no,
                    reason="page_no_range",
                )
            if page_size not in SUPPORTED_PAGE_SIZES:
                raise PageFormatError(
                    f"unsupported page_size {page_size}",
                    offset=record_offset,
                    page_no=page_no,
                    reason="page_size",
                )
            if length != page_size:
                raise PageFormatError(
                    f"length {length} != page_size {page_size}",
                    offset=record_offset,
                    page_no=page_no,
                    reason="length_mismatch",
                )
            frame_start = pos + DWB_HEADER_SIZE
            frame_end = frame_start + length
            if frame_end > len(blob):
                raise PageFormatError(
                    "truncated mirror payload",
                    offset=record_offset,
                    page_no=page_no,
                    reason="truncated_payload",
                )
            if prev_seq is not None and seq <= prev_seq:
                raise PageFormatError(
                    f"seq {seq} not strictly greater than previous {prev_seq}",
                    offset=record_offset,
                    page_no=page_no,
                    reason="seq_not_increasing",
                )
            frame = blob[frame_start:frame_end]
            mirror = Mirror(record_offset, page_no, page_size, seq, frame, digest)
            actual = sha256_digest(frame)
            if actual != digest:
                if strict_digest:
                    raise PageFormatError(
                        "mirror digest mismatch",
                        offset=record_offset,
                        page_no=page_no,
                        reason="digest_mismatch",
                    )
                discarded.append(mirror)
            else:
                if not Page.structural_valid(frame, page_size):
                    # 摘要通过则结构必然合法; 保留防御性分支
                    if strict_digest:
                        raise PageFormatError(
                            "mirror frame structurally invalid",
                            offset=record_offset,
                            page_no=page_no,
                            reason="frame_invalid",
                        )
                    discarded.append(mirror)
                else:
                    usable.append(mirror)
            mirrors.append(mirror)
            prev_seq = seq
            pos = frame_end

        latest: dict[int, Mirror] = {}
        for mirror in usable:
            # 严格递增 => 后出现者序号更大
            latest[mirror.page_no] = mirror
        max_seq = max((m.seq for m in mirrors), default=0)
        return ScanResult(
            mirrors=mirrors,
            usable=usable,
            discarded=discarded,
            discarded_count=len(discarded),
            max_seq=max_seq,
            latest_per_page=dict(sorted(latest.items())),
        )

    def validate_strict(self) -> ScanResult:
        """要求 1 的严格校验: 任一非法即 E_PAGE 整体拒绝。"""
        return self.scan(strict_digest=True)
