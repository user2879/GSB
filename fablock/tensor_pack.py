"""小端二进制张量包读写与加载校验。

二进制布局（全部小端）：
    magic      4B   b"FATP"
    version    u32  (=1)
    meta_len   u32
    meta       meta_len 字节 JSON（pad_rule、各张量 orig_shape、seed 等）
    n_tensors  u32
    重复 n_tensors 次：
        name_len   u16 + name 字节（utf-8）
        ndim       u32 + ndim 个 u64 形状维度
        dtype      u32（0=float32, 1=float64）
        nbytes     u64
        data       nbytes 字节小端数据
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .config import Issue, OperatorConfig
from .errors import Code, OperatorError

MAGIC = b"FATP"
PACK_VERSION = 1
DTYPE_TO_CODE = {np.dtype("<f4"): 0, np.dtype("<f8"): 1}
CODE_TO_DTYPE = {0: np.dtype("<f4"), 1: np.dtype("<f8")}

REQUIRED_TENSORS = ("Q", "K", "V")


def write_pack(path, tensors: dict, meta: dict) -> None:
    meta_blob = json.dumps(meta, sort_keys=True).encode("utf-8")
    parts = [MAGIC, struct.pack("<II", PACK_VERSION, len(meta_blob)), meta_blob,
             struct.pack("<I", len(tensors))]
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32:
            arr = arr.astype("<f4")
        elif arr.dtype == np.float64:
            arr = arr.astype("<f8")
        if arr.dtype not in DTYPE_TO_CODE:
            raise ValueError(f"不支持的 dtype: {arr.dtype}")
        name_b = name.encode("utf-8")
        payload = arr.tobytes(order="C")
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<I", arr.ndim))
        parts.append(struct.pack(f"<{arr.ndim}Q", *arr.shape))
        parts.append(struct.pack("<IQ", DTYPE_TO_CODE[arr.dtype], len(payload)))
        parts.append(payload)
    with open(path, "wb") as fh:
        fh.write(b"".join(parts))


def read_pack(path):
    """返回 (tensors: dict[str, ndarray], meta: dict)。

    形状描述与数据长度不一致时抛 E_SHAPE_DATA_MISMATCH。
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != MAGIC:
        raise OperatorError(Code.E_SHAPE_DATA_MISMATCH, "张量包 magic 不匹配")
    version, meta_len = struct.unpack_from("<II", blob, 4)
    if version != PACK_VERSION:
        raise OperatorError(Code.E_SHAPE_DATA_MISMATCH,
                            f"不支持的张量包版本 {version}")
    off = 12
    meta = json.loads(blob[off:off + meta_len].decode("utf-8"))
    off += meta_len
    (n_tensors,) = struct.unpack_from("<I", blob, off)
    off += 4
    tensors = {}
    for _ in range(n_tensors):
        (name_len,) = struct.unpack_from("<H", blob, off)
        off += 2
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        (ndim,) = struct.unpack_from("<I", blob, off)
        off += 4
        shape = struct.unpack_from(f"<{ndim}Q", blob, off)
        off += 8 * ndim
        code, nbytes = struct.unpack_from("<IQ", blob, off)
        off += 12
        dtype = CODE_TO_DTYPE[code]
        expected = int(np.prod(shape)) * dtype.itemsize
        if expected != nbytes or off + nbytes > len(blob):
            raise OperatorError(
                Code.E_SHAPE_DATA_MISMATCH,
                f"张量 {name} 形状描述 {shape} 需要 {expected} 字节，"
                f"实际数据块 {nbytes} 字节")
        arr = np.frombuffer(blob[off:off + nbytes], dtype=dtype).reshape(shape)
        tensors[name] = np.array(arr, copy=True)
        off += nbytes
    return tensors, meta


def _block_index(position0: int, block_len: int) -> int:
    return position0 // block_len


def validate_pack(tensors: dict, meta: dict, cfg: OperatorConfig) -> list:
    """加载校验层：返回 Issue 列表（不抛异常，便于故障隔离）。

    Issue.context 中携带受影响的 query_blocks / key_blocks，供续算隔离。
    """
    issues = []
    for name in REQUIRED_TENSORS:
        if name not in tensors:
            issues.append(Issue(Code.E_TENSOR_MISSING,
                                f"缺失必需张量 {name}", {"tensor": name}))
    if any(i.code == Code.E_TENSOR_MISSING for i in issues):
        return issues

    q, k, v = tensors["Q"], tensors["K"], tensors["V"]

    # 非有限元素 → 定位到块
    for name, arr, kind in (("Q", q, "query"), ("K", k, "key"), ("V", v, "key")):
        bad = np.argwhere(~np.isfinite(arr))
        if bad.size:
            pos = tuple(int(x) for x in bad[0])
            blk = _block_index(pos[0], cfg.query_block_len if kind == "query"
                               else cfg.key_block_len)
            ctx = {"tensor": name, "position": pos}
            if kind == "query":
                ctx["query_blocks"] = sorted({
                    _block_index(int(r), cfg.query_block_len) for r in bad[:, 0]})
            else:
                ctx["key_blocks"] = sorted({
                    _block_index(int(r), cfg.key_block_len) for r in bad[:, 0]})
            issues.append(Issue(Code.E_NONFINITE,
                                f"张量 {name} 含非有限元素（首处 {pos}）", ctx))

    # 头维度：正偶数且与配置一致
    for name, arr in (("Q", q), ("K", k), ("V", v)):
        if arr.ndim != 2 or arr.shape[1] != cfg.head_dim:
            issues.append(Issue(
                Code.E_HEAD_DIM_MISMATCH,
                f"张量 {name} 头维度与配置不符",
                {"tensor": name,
                 "actual": int(arr.shape[1]) if arr.ndim == 2 else None,
                 "expected": cfg.head_dim}))

    # 序列长度按块长补齐规则标注
    pad_rule = meta.get("pad_rule")
    orig = meta.get("orig_shape", {})
    for name, arr, blk in (("Q", q, cfg.query_block_len),
                           ("K", k, cfg.key_block_len),
                           ("V", v, cfg.key_block_len)):
        if pad_rule != "zero":
            issues.append(Issue(Code.E_PAD_RULE_VIOLATION,
                                "张量包缺少 zero 补齐规则标注",
                                {"pad_rule": pad_rule}))
            break
        if arr.shape[0] % blk != 0:
            issues.append(Issue(
                Code.E_PAD_RULE_VIOLATION,
                f"张量 {name} 序列长度 {arr.shape[0]} 未按块长 {blk} 补齐",
                {"tensor": name}))
        o = orig.get(name)
        if o is None or int(o[0]) > arr.shape[0]:
            issues.append(Issue(
                Code.E_PAD_RULE_VIOLATION,
                f"张量 {name} 缺少有效原始序列长度标注",
                {"tensor": name, "orig_shape": o}))
    return issues
