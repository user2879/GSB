"""确定性对象内容生成与摘要校验。

默认后端使用标准库 random.Random（字符串种子经 UTF-8 + SHA-512 派生，
跨平台、跨进程稳定），不依赖 numpy；安装 numpy 后可选择 numpy 后端。
所有内容按对象键与固定整数种子独立派生，逐字节可复现。
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterator

from .errors import IntegrityError

DIGEST_ALGO = "sha256"
DEFAULT_CHUNK = 64 * 1024


def object_seed(seed: int, key: str) -> bytes:
    """对象级确定性种子：固定整数种子 + 对象键。"""
    return f"{int(seed)}|{key}".encode("utf-8")


def iter_content(seed: int, key: str, size: int,
                 chunk_size: int = DEFAULT_CHUNK,
                 backend: str = "python") -> Iterator[bytes]:
    """按确定性序列产出 size 字节内容，分块生成避免大对象驻留内存。"""
    if size < 0:
        raise ValueError("size 必须为非负整数")
    remaining = size
    if backend == "python":
        rng = random.Random(object_seed(seed, key))
        while remaining > 0:
            n = min(chunk_size, remaining)
            yield rng.randbytes(n)
            remaining -= n
    elif backend == "numpy":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise IntegrityError("numpy 后端不可用，请改用 python 后端或安装 numpy",
                                 field="backend") from exc
        # 用摘要把字符串对象键折叠为确定性整数序列，避免 Python 字符串哈希随机化
        mix = hashlib.sha256(object_seed(seed, key)).digest()
        root_seed = int.from_bytes(mix[:16], "big")
        rng = np.random.Generator(np.random.PCG64(root_seed))
        while remaining > 0:
            n = min(chunk_size, remaining)
            yield rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            remaining -= n
    else:
        raise ValueError(f"未知内容后端: {backend}")


def expected_digest(seed: int, key: str, size: int,
                    backend: str = "python") -> str:
    """流式计算按种子应生成内容的 SHA-256 摘要。"""
    hasher = hashlib.new(DIGEST_ALGO)
    for chunk in iter_content(seed, key, size, backend=backend):
        hasher.update(chunk)
    return hasher.hexdigest()


def digest_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK) -> str:
    hasher = hashlib.new(DIGEST_ALGO)
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_object_file(path: str | Path, seed: int, key: str, size: int,
                      backend: str = "python") -> str:
    """确定性生成对象内容并落盘，返回内容摘要。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.new(DIGEST_ALGO)
    with path.open("wb") as fh:
        for chunk in iter_content(seed, key, size, backend=backend):
            hasher.update(chunk)
            fh.write(chunk)
        fh.flush()
    return hasher.hexdigest()


def verify_file(path: str | Path, expected: str, *, field: str = "content_digest"
                ) -> None:
    """校验磁盘文件摘要与期望摘要一致，失败抛 E_INTEGRITY。"""
    p = Path(path)
    if not p.is_file():
        raise IntegrityError(f"对象文件缺失: {p}", field=field,
                             details={"path": str(p), "expected_digest": expected})
    actual = digest_file(p)
    if actual != expected:
        raise IntegrityError(
            f"内容摘要不一致: {p.name}",
            field=field,
            details={"path": str(p), "expected_digest": expected,
                     "actual_digest": actual},
        )
