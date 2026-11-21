"""显式声明的稳定哈希组合规则与分区映射。

规则名固定为 ``FNV1A32_V1``：
- FNV-1a 32 位，offset basis ``0x811c9dc5``，prime ``0x01000193``。
- 整数分组键按 8 字节小端有符号编码（int64 域）。
- 浮点分组键先按会话缩放因子向零截断量化为 int64，再走整数编码。
- 复合分组键按声明列顺序逐列写入“类型标签字节 + 8 字节载荷”，
  全程维护同一个哈希状态，实现显式组合。
- 分区数仅支持 2 的幂集合 [1,2,4,...,1024]，映射 ``h mod n``。
- 子分区再分发在同一规则上叠加固定标签 ``0xA5`` 与子分区序号种子，
  保证同键恒收敛到唯一子分区。

整个模块不使用 Python 内建 ``hash()``，也不依赖字典/集合遍历顺序。
"""

from .fixedpoint import quantize

FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF
TAG_INT = 0x11
TAG_FLOAT = 0x22
TAG_NULL = 0x00
SUBPART_TAG = 0xA5
RULE_NAME = "FNV1A32_V1"
SUPPORTED_PARTITION_COUNTS = tuple(2**k for k in range(0, 11))


class Fnv1a32:
    def __init__(self, seed: int = FNV_OFFSET):
        self._state = int(seed) & MASK32

    def update_bytes(self, data: bytes) -> None:
        h = self._state
        for byte in data:
            h = ((h ^ byte) * FNV_PRIME) & MASK32
        self._state = h

    def digest(self) -> int:
        return self._state


def _encode_int64(value: int) -> bytes:
    v = int(value)
    return v.to_bytes(8, byteorder="little", signed=True)


def hash_group_key(encoded_key, seed: int = FNV_OFFSET) -> int:
    """对已编码分组键（列标签+载荷元组序列）计算 FNV-1a 32 位。"""
    hasher = Fnv1a32(seed)
    for tag, payload in encoded_key:
        hasher.update_bytes(bytes((tag,)))
        hasher.update_bytes(payload)
    return hasher.digest()


def encode_group_columns(columns, row_index, key_columns, scale: int):
    """按 key_columns 声明顺序把一行的分组键编码为标签载荷序列。

    columns: {列名: Column}；空值输出 TAG_NULL + 8 个 0 字节。
    """
    encoded = []
    for col_name in key_columns:
        col = columns[col_name]
        if col.bitmap[row_index] == 0:
            encoded.append((TAG_NULL, b"\x00" * 8))
            continue
        raw = col.values[row_index]
        if col.dtype == "int64":
            encoded.append((TAG_INT, _encode_int64(int(raw))))
        elif col.dtype == "float64":
            encoded.append((TAG_FLOAT, _encode_int64(quantize(float(raw), scale))))
        else:  # pragma: no cover - 规格校验已拦截
            raise ValueError(f"不支持的分组键类型 {col.dtype}")
    return tuple(encoded)


def validate_partition_count(n) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n not in SUPPORTED_PARTITION_COUNTS:
        raise ValueError(
            f"分区数必须属于受支持整数集合 {list(SUPPORTED_PARTITION_COUNTS)}"
        )
    return n


def map_partition(key_hash: int, partition_count: int) -> int:
    return key_hash % partition_count


def subpartition_hash(key_hash: int, subpartition_index: int) -> int:
    """热点再分发使用的二级哈希：叠加固定标签与子分区序号。"""
    hasher = Fnv1a32((key_hash ^ SUBPART_TAG) & MASK32)
    hasher.update_bytes(subpartition_index.to_bytes(4, "little", signed=False))
    return hasher.digest()


def map_subpartition(key_hash: int, subpartition_count: int) -> int:
    return subpartition_hash(key_hash, 0) % subpartition_count
