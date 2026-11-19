"""增量结果与全量重建的逐元素/逐字段一致性比较。"""

from __future__ import annotations

import numpy as np


def compare_runs(res_a, res_b):
    """比较两次运行的 C 与行结束指纹，返回差异 dict（空 dict 表示完全一致）。"""
    diffs = {}
    if res_a.C.shape != res_b.C.shape:
        diffs["shape"] = {"a": list(res_a.C.shape), "b": list(res_b.C.shape)}
        return diffs
    a_nan = np.isnan(res_a.C)
    b_nan = np.isnan(res_b.C)
    same = (res_a.C == res_b.C) | (a_nan & b_nan)
    if not bool(same.all()):
        bad = np.argwhere(~same)
        diffs["C_mismatch"] = {
            "count": int(bad.shape[0]),
            "first": [int(x) for x in bad[0]],
        }
    fps_a = [m.get("fingerprint") for m in res_a.row_metas]
    fps_b = [m.get("fingerprint") for m in res_b.row_metas]
    if fps_a != fps_b:
        bad_rows = [i for i, (x, y) in enumerate(zip(fps_a, fps_b)) if x != y]
        diffs["row_fingerprints"] = {"mismatch_rows": bad_rows[:8], "count": len(bad_rows)}
    return diffs
