"""确定性块内容生成与摘要登记（需求 2）。

内容生成规则（显式、可复现）：
  以 SHA-256(seed || "|" || block_id) 为初始状态，按计数器模式扩展：
  chunk_i = SHA-256(state || u64le(i))，顺序拼接并截断到块字节数。
  该规则仅依赖 (seed, block_id, bytes)，相同声明多次生成逐位一致。

块摘要规则（显式、可复现）：
  digest = SHA256( "RB1|" || utf8(block_id) || "|" || u64le(bytes) || "|" || content )
  摘要须在修复前逐块登记（BlockRegistry）。
"""

from __future__ import annotations

import hashlib
import os
import struct

from .errors import E_INTEGRITY, WorkbenchError
from .spec import BlockSpec

_CONTENT_CHUNK = 64  # 每轮 SHA-256 扩展 32 字节，两次拼接成 64


def generate_content(seed: str, block_id: str, byte_count: int) -> bytes:
    """按固定整数规则生成确定性块内容。"""
    state = hashlib.sha256(f"{seed}|{block_id}".encode("utf-8")).digest()
    out = bytearray()
    counter = 0
    while len(out) < byte_count:
        for _ in range(2):
            out.extend(hashlib.sha256(state + struct.pack("<Q", counter)).digest())
            counter += 1
    return bytes(out[:byte_count])


def block_digest(block_id: str, byte_count: int, content: bytes) -> str:
    """显式可复现的块摘要规则。"""
    h = hashlib.sha256()
    h.update(b"RB1|")
    h.update(block_id.encode("utf-8"))
    h.update(b"|")
    h.update(struct.pack("<Q", byte_count))
    h.update(b"|")
    h.update(content)
    return h.hexdigest()


def segment_digest(block_id: str, start: int, end: int, segment: bytes) -> str:
    """批次字节段摘要：SHA256("RB1S|" || block_id || u64le(start) || u64le(end) || segment)。"""
    h = hashlib.sha256()
    h.update(b"RB1S|")
    h.update(block_id.encode("utf-8"))
    h.update(struct.pack("<Q", start))
    h.update(struct.pack("<Q", end))
    h.update(segment)
    return h.hexdigest()


class BlockRegistry:
    """块内容文件与修复前摘要登记。"""

    def __init__(self, blocks_dir: str):
        self._dir = blocks_dir
        os.makedirs(self._dir, exist_ok=True)
        self._digests: dict[str, str] = {}
        self._sizes: dict[str, int] = {}

    def path_for(self, block_id: str) -> str:
        return os.path.join(self._dir, f"{block_id}.bin")

    @property
    def digests(self) -> dict:
        return dict(self._digests)

    def materialize(self, block: BlockSpec) -> str:
        """生成块内容写入本地文件并在修复前登记摘要；幂等且逐位一致。"""
        content = generate_content(block.seed, block.block_id, block.byte_count)
        path = self.path_for(block.block_id)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(content)
        os.replace(tmp, path)
        digest = block_digest(block.block_id, block.byte_count, content)
        self._digests[block.block_id] = digest
        self._sizes[block.block_id] = block.byte_count
        return digest

    def register_existing(self, block: BlockSpec) -> str:
        """不重写文件，仅从磁盘读取并登记摘要（用于恢复路径）。"""
        path = self.path_for(block.block_id)
        with open(path, "rb") as fh:
            content = fh.read()
        if len(content) != block.byte_count:
            raise WorkbenchError(
                E_INTEGRITY,
                f"块文件长度不符: {block.block_id}",
                block_id=block.block_id,
                field="bytes",
                expected=block.byte_count,
                actual=len(content),
            )
        digest = block_digest(block.block_id, block.byte_count, content)
        self._digests[block.block_id] = digest
        self._sizes[block.block_id] = block.byte_count
        return digest

    def verify(self, block_id: str) -> None:
        """完整性核对：磁盘当前摘要须等于修复前登记摘要，否则 E_INTEGRITY。"""
        path = self.path_for(block_id)
        with open(path, "rb") as fh:
            content = fh.read()
        actual = block_digest(block_id, len(content), content)
        expected = self._digests.get(block_id)
        if expected is None or actual != expected:
            raise WorkbenchError(
                E_INTEGRITY,
                f"块内容摘要不符: {block_id}",
                block_id=block.block_id if hasattr(block_id, "block_id") else block_id,
                field="digest",
                expected=expected,
                actual=actual,
            )

    def read_segment(self, block_id: str, start: int, end: int) -> bytes:
        path = self.path_for(block_id)
        with open(path, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start)
