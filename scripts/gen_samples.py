"""生成确定性样本：三种非零元分布的 CSR 稀疏矩阵与稠密矩阵。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spmm.datasets import generate_all


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    metas = generate_all(data_dir)
    for dist, meta in metas.items():
        print(f"[{dist}] seed={meta['seed']} M={meta['M']} K={meta['K']} "
              f"N={meta['N']} nnz={meta['nnz']} "
              f"row_nnz=[{meta['row_nnz_min']},{meta['row_nnz_max']}]")
    print(f"样本已写入 {data_dir}")


if __name__ == "__main__":
    main()
