"""CSR 稀疏结构构造、校验与行级编辑。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .errors import E_CSR_INVALID, E_NONFINITE, E_SHAPE_MISMATCH, SpmmError

DISTRIBUTIONS = ("uniform", "powerlaw", "extreme")


@dataclass
class CSRMatrix:
    """M x K 的 CSR 稀疏矩阵，列索引在每行内严格递增。"""

    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    shape: tuple

    @property
    def n_rows(self):
        return int(self.shape[0])

    @property
    def n_cols(self):
        return int(self.shape[1])

    @property
    def nnz(self):
        return int(self.indices.shape[0])

    def row_nnz(self, i):
        return int(self.indptr[i + 1] - self.indptr[i])

    def row_cols(self, i):
        return self.indices[self.indptr[i] : self.indptr[i + 1]]

    def row_vals(self, i):
        return self.values[self.indptr[i] : self.indptr[i + 1]]


def validate_csr(a: CSRMatrix):
    """校验 CSR 结构规则，非法时抛出 E_CSR_INVALID 并记录行号与违规字段。"""
    violations = []
    m, k = int(a.shape[0]), int(a.shape[1])
    indptr = np.asarray(a.indptr)
    indices = np.asarray(a.indices)
    values = np.asarray(a.values)
    if indptr.shape[0] != m + 1:
        violations.append({
            "row": -1,
            "field": "indptr",
            "detail": f"indptr 长度 {indptr.shape[0]} 与行数 {m} 不一致",
        })
        raise SpmmError(E_CSR_INVALID, "CSR 行偏移数组长度非法", violations)
    if int(indptr[0]) != 0:
        violations.append({"row": 0, "field": "indptr", "detail": "indptr[0] 必须为 0"})
    for i in range(m):
        if int(indptr[i + 1]) < int(indptr[i]):
            violations.append({
                "row": i,
                "field": "indptr",
                "detail": f"行偏移非单调: indptr[{i + 1}]={int(indptr[i + 1])} < indptr[{i}]={int(indptr[i])}",
            })
    if int(indptr[-1]) != indices.shape[0] or indices.shape[0] != values.shape[0]:
        violations.append({
            "row": -1,
            "field": "nnz",
            "detail": (
                f"非零元总数不一致: indptr[-1]={int(indptr[-1])}, "
                f"indices={indices.shape[0]}, values={values.shape[0]}"
            ),
        })
    if violations:
        raise SpmmError(E_CSR_INVALID, "CSR 行偏移/非零元总数校验失败", violations)
    for i in range(m):
        s, e = int(indptr[i]), int(indptr[i + 1])
        cols = indices[s:e]
        if cols.size == 0:
            continue
        if int(cols.min()) < 0 or int(cols.max()) >= k:
            violations.append({
                "row": i,
                "field": "indices",
                "detail": f"列索引越界 [0,{k}): min={int(cols.min())}, max={int(cols.max())}",
            })
        if cols.size > 1 and bool(np.any(cols[1:] <= cols[:-1])):
            violations.append({
                "row": i,
                "field": "indices_order",
                "detail": "列索引未严格递增",
            })
    if violations:
        raise SpmmError(E_CSR_INVALID, "CSR 列索引校验失败", violations)


def check_finite(name, arr):
    """校验数组元素均为有限值，非法时抛出 E_NONFINITE。"""
    arr = np.asarray(arr)
    bad = np.argwhere(~np.isfinite(arr))
    if bad.shape[0] > 0:
        loc = [int(x) for x in bad[0]]
        raise SpmmError(
            E_NONFINITE,
            f"{name} 含非有限元素",
            [{"field": name, "index": loc, "detail": "NaN 或 Inf"}],
        )


def validate_shapes(a: CSRMatrix, b: np.ndarray, c: np.ndarray):
    """校验 A/B/C 形状一致性，非法时抛出 E_SHAPE_MISMATCH。"""
    if b.ndim != 2 or b.shape[0] != a.n_cols:
        raise SpmmError(
            E_SHAPE_MISMATCH,
            "B 的行数必须等于 A 的列数",
            [{"field": "B", "detail": f"A.cols={a.n_cols}, B.shape={tuple(b.shape)}"}],
        )
    if c.ndim != 2 or c.shape != (a.n_rows, b.shape[1]):
        raise SpmmError(
            E_SHAPE_MISMATCH,
            "C 的形状必须为 (A.rows, B.cols)",
            [{"field": "C", "detail": f"期望 {(a.n_rows, b.shape[1])}, 实际 {tuple(c.shape)}"}],
        )


def _row_nnz_counts(distribution, m, k, rng):
    if distribution == "uniform":
        hi = max(1, min(k, int(np.ceil(k * 0.2))))
        return rng.integers(0, hi + 1, size=m)
    if distribution == "powerlaw":
        raw = rng.pareto(1.5, size=m) * max(1.0, k * 0.05)
        return np.clip(raw.astype(np.int64), 0, k)
    if distribution == "extreme":
        counts = np.zeros(m, dtype=np.int64)
        heavy = rng.random(m) < 0.06
        counts[heavy] = rng.integers(max(1, k // 2), k + 1, size=int(heavy.sum()))
        light = ~heavy
        counts[light] = rng.integers(0, 2, size=int(light.sum()))
        return counts
    raise SpmmError(E_CSR_INVALID, f"未知非零元分布: {distribution}",
                    [{"field": "distribution", "detail": distribution}])


def generate_csr(m, k, distribution, seed):
    """按固定种子与分布确定性生成 CSR 矩阵，列索引严格递增。"""
    rng = np.random.default_rng(seed)
    counts = _row_nnz_counts(distribution, m, k, rng)
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    values = np.empty(total, dtype=np.float32)
    for i in range(m):
        s, e = int(indptr[i]), int(indptr[i + 1])
        cnt = e - s
        if cnt == 0:
            continue
        cols = np.sort(rng.choice(k, size=cnt, replace=False))
        indices[s:e] = cols
        values[s:e] = rng.standard_normal(cnt).astype(np.float32)
    a = CSRMatrix(indptr=indptr, indices=indices, values=values, shape=(m, k))
    validate_csr(a)
    return a


def _rows_of(a: CSRMatrix):
    return [(a.row_cols(i).copy(), a.row_vals(i).copy()) for i in range(a.n_rows)]


def _build_from_rows(rows, k):
    m = len(rows)
    indptr = np.zeros(m + 1, dtype=np.int64)
    for i, (cols, _) in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(cols)
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    values = np.empty(total, dtype=np.float32)
    for i, (cols, vals) in enumerate(rows):
        s, e = int(indptr[i]), int(indptr[i + 1])
        indices[s:e] = cols
        values[s:e] = vals
    a = CSRMatrix(indptr=indptr, indices=indices, values=values, shape=(m, k))
    validate_csr(a)
    return a


def replace_rows(a: CSRMatrix, replacements):
    """替换指定行的非零元集合，replacements: {row: (cols, vals)}。"""
    rows = _rows_of(a)
    for r, (cols, vals) in replacements.items():
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float32)
        order = np.argsort(cols, kind="stable")
        rows[int(r)] = (cols[order], vals[order])
    return _build_from_rows(rows, a.n_cols)


def delete_rows(a: CSRMatrix, rows_to_delete):
    """删除指定行（按旧行号），后续行重新编号。"""
    drop = set(int(r) for r in rows_to_delete)
    rows = [row for i, row in enumerate(_rows_of(a)) if i not in drop]
    return _build_from_rows(rows, a.n_cols)


def append_rows(a: CSRMatrix, new_rows):
    """在末尾追加新行，new_rows: [(cols, vals), ...]。"""
    rows = _rows_of(a)
    for cols, vals in new_rows:
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float32)
        order = np.argsort(cols, kind="stable")
        rows.append((cols[order], vals[order]))
    return _build_from_rows(rows, a.n_cols)


def fingerprint_inputs(a: CSRMatrix, b: np.ndarray, c: np.ndarray):
    """输入指纹：覆盖 A 结构/取值、B、C 与形状。"""
    h = hashlib.blake2b(digest_size=16)
    for arr in (a.indptr, a.indices, a.values, np.asarray(b), np.asarray(c)):
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype).encode())
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    return h.hexdigest()
