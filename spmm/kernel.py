"""SpMM 参照内核与稀疏行分块内核（含累加顺序变体）。"""

from __future__ import annotations

import hashlib
import struct
import time

import numpy as np

ORDER_ASC = "asc"
ORDER_DESC = "desc"
ORDER_ABS_DESC = "abs_desc"
ORDERS = (ORDER_ASC, ORDER_DESC, ORDER_ABS_DESC)

PRECISIONS = ("fp32", "fp64")

EMPTY_ROW_MARKER = "EMPTY_ROW"


def _accum_dtype(precision):
    return np.float32 if precision == "fp32" else np.float64


def b_row_abs_keys(b):
    """每个 B 行的 |元素| 之和，作为 abs_desc 累加顺序的排序键。"""
    return np.abs(np.asarray(b, dtype=np.float64)).sum(axis=1)


def _chunk_permutation(cols, abs_keys, order):
    n = int(cols.shape[0])
    if order == ORDER_ASC:
        return np.arange(n)
    if order == ORDER_DESC:
        return np.arange(n - 1, -1, -1)
    # abs_desc：按对应 B 行元素绝对值之和降序，平局按列索引升序，保证确定性。
    keys = abs_keys[cols]
    return np.lexsort((cols, -keys))


def row_fingerprint(row_idx, nnz, order, out_row):
    """行结束指纹：绑定行号、非零元个数、累加顺序标识与输出字节。"""
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<qq", int(row_idx), int(nnz)))
    h.update(order.encode("utf-8"))
    h.update(np.ascontiguousarray(out_row, dtype=np.float32).tobytes())
    return h.hexdigest()


def block_fingerprint(block_id, row_fps):
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<q", int(block_id)))
    for fp in row_fps:
        h.update(fp.encode("utf-8"))
    return h.hexdigest()


def compute_row(a, b, c_row, i, alpha, beta, precision, nnz_chunk, order, abs_keys):
    """按块内累加顺序计算 C[i] = alpha * A[i,:] @ B + beta * C[i]。"""
    dt = _accum_dtype(precision)
    s, e = int(a.indptr[i]), int(a.indptr[i + 1])
    cols = a.indices[s:e]
    vals = a.values[s:e]
    nnz = int(e - s)
    n = int(b.shape[1])
    acc = np.zeros(n, dtype=dt)
    n_chunks = 0
    for start in range(0, nnz, nnz_chunk):
        ccols = cols[start : start + nnz_chunk]
        cvals = vals[start : start + nnz_chunk]
        perm = _chunk_permutation(ccols, abs_keys, order)
        partial = np.zeros(n, dtype=dt)
        for j in perm:
            partial += dt(cvals[j]) * b[int(ccols[j])]
        acc += partial
        n_chunks += 1
    out = (dt(alpha) * acc + dt(beta) * c_row.astype(dt, copy=False)).astype(np.float32)
    meta = {
        "row": int(i),
        "nnz": nnz,
        "order": order,
        "precision": precision,
        "chunks": n_chunks,
        "empty": nnz == 0,
        "marker": EMPTY_ROW_MARKER if nnz == 0 else "",
        "fingerprint": row_fingerprint(i, nnz, order, out),
    }
    return out, meta


def compute_block(block_id, row0, row1, a, b, c, config, abs_keys, sleep_ms=0):
    """计算一个输出行块，返回块结果、行元数据、依赖集合与产出指纹。"""
    t0 = time.perf_counter()
    rows = []
    metas = []
    b_rows = set()
    for i in range(row0, row1):
        out, meta = compute_row(
            a, b, c[i], i,
            config.alpha, config.beta, config.precision,
            config.nnz_chunk_size, config.accumulation_order, abs_keys,
        )
        rows.append(out)
        metas.append(meta)
        b_rows.update(int(x) for x in a.row_cols(i))
    if rows:
        c_block = np.stack(rows).astype(np.float32, copy=False)
    else:
        c_block = np.zeros((0, b.shape[1]), dtype=np.float32)
    fp = block_fingerprint(block_id, [m["fingerprint"] for m in metas])
    if sleep_ms:
        time.sleep(sleep_ms / 1000.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "block_id": int(block_id),
        "row0": int(row0),
        "row1": int(row1),
        "C": c_block,
        "row_metas": metas,
        "fingerprint": fp,
        "deps": {
            "a_rows": list(range(int(row0), int(row1))),
            "b_rows": sorted(b_rows),
            "param_fp": config.fingerprint(),
        },
        "elapsed_ms": elapsed_ms,
    }


def reference_spmm(a, b, c, alpha, beta):
    """单精度参照内核：升序、单一块、fp32 累加，结果不做重排。"""
    abs_keys = b_row_abs_keys(b)
    out = np.empty((a.n_rows, b.shape[1]), dtype=np.float32)
    metas = []
    for i in range(a.n_rows):
        row, meta = compute_row(
            a, b, c[i], i, alpha, beta, "fp32",
            max(1, a.n_cols), ORDER_ASC, abs_keys,
        )
        out[i] = row
        metas.append(meta)
    return out, metas
