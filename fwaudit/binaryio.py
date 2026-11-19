"""带边界校验的小端二进制只读视图。"""

import struct

from .errors import AuditError

UINT64_MAX = (1 << 64) - 1


class BinaryView:
    def __init__(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        self._data = bytes(data)
        self._pos = 0

    @property
    def data(self):
        return self._data

    @property
    def pos(self):
        return self._pos

    def __len__(self):
        return len(self._data)

    def seek(self, pos):
        if not 0 <= pos <= len(self._data):
            raise AuditError(
                "E_HEADER", "偏移 0x%x 越过文件边界" % pos, offset=pos
            )
        self._pos = pos

    def take(self, size, code, what):
        end = self._pos + size
        if end > len(self._data):
            raise AuditError(
                code,
                "读取 %s 需要 %d 字节，但文件在偏移 0x%x 处结束"
                % (what, size, self._pos),
                offset=self._pos,
            )
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def u8(self, code, what):
        return self.take(1, code, what)[0]

    def u16(self, code, what):
        return struct.unpack("<H", self.take(2, code, what))[0]

    def u32(self, code, what):
        return struct.unpack("<I", self.take(4, code, what))[0]

    def u64(self, code, what):
        return struct.unpack("<Q", self.take(8, code, what))[0]

    def slice(self, start, end, code, what):
        if start < 0 or end < start or end > len(self._data):
            raise AuditError(
                code,
                "切片 %s [%d, %d) 越过文件边界(总长 %d)"
                % (what, start, end, len(self._data)),
                offset=max(0, start),
            )
        return self._data[start:end]


def checked_add_u64(a, b, code, offset, what):
    """无符号 64 位加法，溢出抛 AuditError。"""
    if a < 0 or b < 0 or a > UINT64_MAX or b > UINT64_MAX:
        raise AuditError(code, "%s 取值非法" % what, offset=offset)
    if a > UINT64_MAX - b:
        raise AuditError(
            code,
            "%s: 0x%x + 0x%x 溢出 64 位无符号整型" % (what, a, b),
            offset=offset,
        )
    return a + b
