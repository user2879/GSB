"""确定性分片内容生成与 NumPy 定点校验。

规则全部显式、可复现，不依赖 dict 顺序、集合顺序或批次切分：

1. 分片字节流由 (内容键, 分片内容种子, 分片序号, 分片字节数) 决定，
   与节点、加载顺序、批次切分无关——同一内容分片在任何节点上字节逐位相同。
2. 字节流用 numpy.random.default_rng(派生整数种子) 的 PCG64 生成，
   种子由字段经 64 位 FNV-1a 显式混合得到。
3. 校验摘要绑定节点标识、内容键、分片序号与内容字节：
   - 内容滚动哈希（FNV-1a over bytes）
   - 以节点随机种子派生样本，经会话缩放因子量化为整数，NumPy 完成
     定点加总和平方和，模拟"节点声明的整数定点校验口径"
   - 汇总为十六进制摘要字符串
"""

import hashlib

import numpy as np

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a_64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def mix_seed(parts) -> int:
    """把任意整数/字符串序列混合为 64 位整数种子（显式、顺序敏感）。"""
    h = FNV_OFFSET
    for part in parts:
        if isinstance(part, str):
            data = part.encode("utf-8")
        elif isinstance(part, int):
            data = int(part).to_bytes(8, "little", signed=True)
        else:
            raise TypeError("unsupported seed part: %r" % type(part))
        h = fnv1a_64(data + h.to_bytes(8, "little"))
    return h


def shard_id(content_key: str, shard_index: int) -> str:
    return f"{content_key}#s{shard_index:04d}"


def shard_byte_size(total_bytes: int, shard_count: int) -> int:
    return total_bytes // shard_count


def generate_shard_bytes(content_key: str, content_seed: int, shard_index: int,
                         byte_size: int) -> bytes:
    """生成确定性分片字节，仅取决于内容侧字段。"""
    seed = mix_seed([content_key, int(content_seed), int(shard_index),
                     int(byte_size)])
    rng = np.random.default_rng(seed)
    # 分块生成，避免一次性大分配，结果仍逐位确定。
    chunks = []
    remaining = byte_size
    while remaining > 0:
        n = min(remaining, 1 << 16)
        chunks.append(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
        remaining -= n
    return b"".join(chunks)


def fixed_point_check(node_seed: int, node_id: str, content_key: str,
                      shard_index: int, payload: bytes,
                      session_scale: int) -> dict:
    """节点子进程内执行的 NumPy 整数定点校验计算。

    样本由 (节点种子, 内容键, 分片序号) 确定性派生，经会话缩放因子量化为
    非负整数后做定点加总/平方和；再与内容字节的滚动哈希绑定。
    """
    sample_seed = mix_seed([node_id, int(node_seed), content_key,
                            int(shard_index)])
    rng = np.random.default_rng(sample_seed)
    n = max(1, len(payload))
    # 均匀样本 [0,1)，乘会话缩放因子后截断为 int64 整数（量化口径）。
    raw = rng.random(n, dtype=np.float64)
    quantized = np.floor(raw * int(session_scale)).astype(np.int64)
    fp_sum = int(np.sum(quantized, dtype=np.int64))
    fp_sqsum = int(np.sum(quantized * quantized, dtype=np.int64))
    byte_hash = fnv1a_64(payload)
    return {
        "sample_count": int(n),
        "session_scale": int(session_scale),
        "fp_sum": fp_sum,
        "fp_sqsum": fp_sqsum,
        "byte_hash": f"0x{byte_hash:016x}",
    }


def compute_digest(node_id: str, node_seed: int, content_key: str,
                   shard_index: int, payload: bytes,
                   session_scale: int) -> dict:
    """完整摘要：绑定节点标识、内容键、分片序号与内容字节。"""
    fp = fixed_point_check(node_seed, node_id, content_key, shard_index,
                           payload, session_scale)
    binding = "|".join([
        node_id,
        content_key,
        str(shard_index),
        fp["byte_hash"],
        str(fp["fp_sum"]),
        str(fp["fp_sqsum"]),
        str(fp["sample_count"]),
        str(fp["session_scale"]),
    ])
    digest = hashlib.sha256(binding.encode("utf-8")).hexdigest()
    return {
        "node_id": node_id,
        "shard_id": shard_id(content_key, shard_index),
        "content_key": content_key,
        "shard_index": shard_index,
        "digest": digest,
        **fp,
    }
