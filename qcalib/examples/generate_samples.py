"""生成本地确定性样本（固定种子，规模由 sample_config.json 控制）。

产物：
  samples/feature_table.jsonl     特征表（含缺失标记）
  samples/groups_config.json      逐组标定配置
  samples/calib_v1.txt            v1 定宽标定结果
  samples/calib_v2.json           v2 JSON 标定结果
  samples/sample_config.json      本次生成所用规模配置（回写）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from qcalib.enums import V1_LINE_WIDTH
from qcalib.exchange import (
    CalibEntry,
    encode_v1_line,
    write_v2,
)

SEED = 20261124


def _make_config(n_rows: int, n_groups: int, cols_per_group: int) -> Dict[str, Any]:
    columns: List[str] = []
    groups: List[Dict[str, Any]] = []
    partition = [f"g{idx}" for idx in range(n_groups)]
    for gi in range(n_groups):
        cols = [f"f{gi}_{ci}" for ci in range(cols_per_group)]
        columns.extend(cols)
        groups.append(
            {
                "group_id": f"g{gi}",
                "columns": cols,
                "scheme": "symmetric" if gi % 2 == 0 else "asymmetric",
                "q_lo_permille": 25,
                "q_hi_permille": 975,
                "qbits": 8,
                "fbits": 2 + (gi % 3),
                "rounding": "half_up",
                "saturation": "clip",
                "abs_tol": 0.15,
                "rel_tol": 0.10,
            }
        )
    return {
        "seed": SEED,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "cols_per_group": cols_per_group,
        "columns": columns,
        "partition": partition,
        "groups": groups,
    }


def generate(out_dir: Path, n_rows: int, n_groups: int, cols_per_group: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    spec = _make_config(n_rows, n_groups, cols_per_group)

    # 1) 特征表 JSONL（不同分布 + 确定性缺失 + 极端值触发饱和）
    table_path = out_dir / "feature_table.jsonl"
    with table_path.open("w", encoding="utf-8") as handle:
        for ri in range(n_rows):
            row: Dict[str, Any] = {}
            for ci, col in enumerate(spec["columns"]):
                base = rng.normal(loc=ci * 0.15 - 0.4, scale=1.0 + 0.2 * ci)
                if ri == n_rows - 1:  # 末行注入极端值，保证出现饱和截断
                    base = base + np.sign(base or 1.0) * (12.0 + ci)
                value = None if (ri + ci) % 11 == 0 else float(base)
                row[col] = value
            handle.write(json.dumps({"row_id": f"r{ri:04d}", "values": row}, sort_keys=True) + "\n")

    # 2) 逐组标定配置
    config_payload = {
        "partition": spec["partition"],
        "groups": [{k: v for k, v in g.items()} for g in spec["groups"]],
    }
    (out_dir / "groups_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 3) v1 定宽 + v2 JSON 标定结果（确定性伪参数）
    entries: List[CalibEntry] = []
    for gi, group in enumerate(spec["groups"]):
        for ci, col in enumerate(group["columns"]):
            permille = int(250 + ((gi * 7 + ci * 13) % 400))  # 0.250 .. 0.649
            scheme = group["scheme"]
            zero_point = 0 if scheme == "symmetric" else int(8 + (gi * 3 + ci) % 40)
            entry = CalibEntry(
                group_id=group["group_id"],
                column=col,
                scale=permille / 1000.0,
                zero_point=zero_point,
                qbits=8,
                scheme=scheme,
                rounding="half_up",
                precision_label="permille-v1",
                per_channel=True,
                source_version=1,
                scale_permille=permille,
            )
            from qcalib.fingerprint import version_id
            from qcalib.exchange import _entry_fp_payload

            entry.version = version_id(1, _entry_fp_payload(entry))
            entries.append(entry)

    v1_lines = "\n".join(encode_v1_line(e) for e in entries)
    (out_dir / "calib_v1.txt").write_text(v1_lines + "\n", encoding="utf-8")

    # v2 用浮点（含非千分整除的精细尺度，制造降级残差）
    v2_entries = []
    for entry in entries:
        v2 = CalibEntry(
            group_id=entry.group_id,
            column=entry.column,
            scale=round(entry.scale + 0.00037, 6),  # 非整数 permille
            zero_point=entry.zero_point,
            qbits=entry.qbits,
            scheme=entry.scheme,
            rounding="half_even",
            precision_label="float-native-v2",
            per_channel=True,
            source_version=2,
        )
        from qcalib.fingerprint import version_id
        from qcalib.exchange import _entry_fp_payload

        v2.version = version_id(2, _entry_fp_payload(v2))
        v2_entries.append(v2)
    write_v2(out_dir / "calib_v2.json", v2_entries)

    (out_dir / "sample_config.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "n_rows": n_rows,
        "n_groups": n_groups,
        "cols_per_group": cols_per_group,
        "v1_line_width": V1_LINE_WIDTH,
        "entries": len(entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成确定性量化标定样本")
    parser.add_argument("--out", default="samples", help="样本输出目录")
    parser.add_argument("--rows", type=int, default=40, help="特征表行数")
    parser.add_argument("--groups", type=int, default=3, help="列组数量")
    parser.add_argument("--cols", type=int, default=3, help="每组列数")
    args = parser.parse_args()
    summary = generate(Path(args.out), args.rows, args.groups, args.cols)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
