"""页帧编解码与主数据文件的定长页寻址。

页帧(固定 page_size 字节)布局:
    magic(4s) = PG01 | payload_len(I) | payload(...) | 0x00 填充
双写缓冲镜像与主文件槽位存放的都是同一份页帧字节, 因此
"页内容摘要"统一为 sha256(整帧), 缓冲镜像摘要与主文件槽位摘要可逐页直接比对。
"""

from __future__ import annotations

import struct
from pathlib import Path

from .config import PAGE_HEADER_SIZE, PAGE_MAGIC, SUPPORTED_PAGE_SIZES
from .errors import PageFormatError
from .util import fsync_file, sha256_digest, sha256_hex

_LEN = struct.Struct("<I")


class Page:
    """固定大小页的编解码助手。"""

    def __init__(self, page_size: int) -> None:
        if page_size not in SUPPORTED_PAGE_SIZES:
            raise ValueError(f"unsupported page_size={page_size}")
        self.page_size = page_size
        self.capacity_payload = page_size - PAGE_HEADER_SIZE

    def encode(self, payload: bytes) -> bytes:
        if len(payload) > self.capacity_payload:
            raise ValueError(
                f"payload {len(payload)} exceeds page payload capacity {self.capacity_payload}"
            )
        frame = PAGE_MAGIC + _LEN.pack(len(payload)) + payload
        if len(frame) < self.page_size:
            frame += b"\x00" * (self.page_size - len(frame))
        return frame

    def decode_payload(self, frame: bytes) -> bytes:
        if len(frame) != self.page_size:
            raise ValueError("frame length does not match page_size")
        if frame[:4] != PAGE_MAGIC:
            raise ValueError("bad page magic")
        (payload_len,) = _LEN.unpack(frame[4:8])
        if payload_len > self.capacity_payload:
            raise ValueError("payload length out of page bounds")
        return frame[8 : 8 + payload_len]

    @staticmethod
    def structural_valid(frame: bytes, page_size: int) -> bool:
        """不依赖外部摘要的结构判定: 魔数与长度边界。"""
        if len(frame) != page_size:
            return False
        if frame[:4] != PAGE_MAGIC:
            return False
        try:
            (payload_len,) = _LEN.unpack(frame[4:8])
        except struct.error:
            return False
        if payload_len > page_size - PAGE_HEADER_SIZE:
            return False
        # 填充区必须为零(半写混入脏字节可被识别)
        if frame[8 + payload_len :].count(0) != (page_size - 8 - payload_len):
            return False
        return True

    @staticmethod
    def digest(frame: bytes) -> bytes:
        return sha256_digest(frame)

    @staticmethod
    def digest_hex(frame: bytes) -> str:
        return sha256_hex(frame)


class MainDataFile:
    """主数据文件: 按页号定长寻址, 槽位 = page_no * page_size。"""

    def __init__(self, path: Path, page_size: int, capacity: int) -> None:
        if page_size not in SUPPORTED_PAGE_SIZES:
            raise PageFormatError(
                f"unsupported page_size {page_size}", offset=-1, page_no=None
            )
        self.path = Path(path)
        self.page_size = page_size
        self.capacity = capacity
        self.page = Page(page_size)

    def _check_page_no(self, page_no: int) -> None:
        if not isinstance(page_no, int) or not (0 <= page_no < self.capacity):
            raise PageFormatError(
                f"page_no {page_no} out of capacity {self.capacity}",
                offset=-1,
                page_no=page_no,
            )

    def exists(self) -> bool:
        return self.path.exists()

    def expected_length(self) -> int:
        return self.capacity * self.page_size

    def ensure_file(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as fh:
                fh.truncate(self.expected_length())
                fsync_file(fh)

    def read_slot(self, page_no: int) -> bytes | None:
        """返回整帧; 从未写入的全零槽位返回 None。"""
        self._check_page_no(page_no)
        if not self.path.exists():
            return None
        with open(self.path, "rb") as fh:
            fh.seek(page_no * self.page_size)
            raw = fh.read(self.page_size)
        if len(raw) < self.page_size:
            # 文件被截断, 视为半写槽位
            return raw + b"\x00" * (self.page_size - len(raw))
        if raw == b"\x00" * self.page_size:
            return None
        return raw

    def write_slot(self, page_no: int, frame: bytes) -> None:
        self._check_page_no(page_no)
        if len(frame) != self.page_size:
            raise PageFormatError(
                "frame length mismatch", offset=-1, page_no=page_no
            )
        self.ensure_file()
        with open(self.path, "r+b") as fh:
            fh.seek(page_no * self.page_size)
            fh.write(frame)
            fsync_file(fh)

    def slot_intact(self, page_no: int, expected_digest: bytes | None) -> bool:
        """主文件槽位是否完好: 结构合法且与期望摘要一致。"""
        frame = self.read_slot(page_no)
        if frame is None:
            return False
        if not Page.structural_valid(frame, self.page_size):
            return False
        if expected_digest is not None and Page.digest(frame) != expected_digest:
            return False
        return True

    def iter_digests(self) -> dict[int, str]:
        """当前主文件全部已写页的摘要集合(页号升序)。"""
        out: dict[int, str] = {}
        if not self.path.exists():
            return out
        for page_no in range(self.capacity):
            frame = self.read_slot(page_no)
            if frame is not None:
                out[page_no] = Page.digest_hex(frame)
        return dict(sorted(out.items()))
