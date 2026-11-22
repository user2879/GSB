"""固定种子本地样本生成：行记录、Gather 请求、配置、编辑序列与异常注入。"""
import json
import os

import numpy as np

SEED = 20261122
BLOCKS = 4
ROWS_PER_BLOCK = 8
DIM = 6
VALID_ROW_IDS = list(range(100, 100 + BLOCKS * ROWS_PER_BLOCK))


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_samples(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    row_records = []
    table_snapshot = {}
    idx = 0
    for block_id in range(BLOCKS):
        for offset in range(ROWS_PER_BLOCK):
            row_id = VALID_ROW_IDS[idx]
            idx += 1
            base = rng.normal(0.0, 2.0, size=DIM)
            if row_id == VALID_ROW_IDS[3]:
                base = base + rng.choice([-1, 1]) * 40.0
            vector = [float(x) for x in base]
            row_records.append(
                {
                    "row_id": row_id,
                    "block_id": block_id,
                    "offset": offset,
                    "vector": vector,
                }
            )
            table_snapshot[row_id] = {
                "block_id": block_id,
                "offset": offset,
                "vector": vector,
            }

    # ---- 异常注入行
    bad_dim_row = dict(row_records[0])
    bad_dim_row = {
        "row_id": 90001,
        "block_id": 0,
        "offset": 90,
        "vector": [1.0] * (DIM - 1),
    }
    nonfinite_row = {
        "row_id": 90002,
        "block_id": 1,
        "offset": 91,
        "vector": [1.0] * (DIM - 1) + [float("inf")],
    }
    malformed_line = '{"row_id": 90003, "block_id": 2, "offset": 92, "vector": '
    dup_row = {
        "row_id": VALID_ROW_IDS[0],
        "block_id": 3,
        "offset": 93,
        "vector": [0.5] * DIM,
    }
    conflict_row = {
        "row_id": 90004,
        "block_id": 0,
        "offset": 0,
        "vector": [0.25] * DIM,
    }
    negative_row = {
        "row_id": 90005,
        "block_id": -1,
        "offset": 0,
        "vector": [0.25] * DIM,
    }

    rows_path = os.path.join(out_dir, "rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as handle:
        for record in row_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bad_dim_row, ensure_ascii=False) + "\n")
        handle.write(json.dumps(nonfinite_row, ensure_ascii=False) + "\n")
        handle.write(malformed_line + "\n")
        handle.write(json.dumps(dup_row, ensure_ascii=False) + "\n")
        handle.write(json.dumps(conflict_row, ensure_ascii=False) + "\n")
        handle.write(json.dumps(negative_row, ensure_ascii=False) + "\n")

    # ---- Gather 请求（固定确定性序列）
    requests = []
    request_specs = [
        ("rq_sum_01", [VALID_ROW_IDS[0], VALID_ROW_IDS[5], VALID_ROW_IDS[12]], "SUM"),
        ("rq_mean_01", [VALID_ROW_IDS[1], VALID_ROW_IDS[6], VALID_ROW_IDS[13], VALID_ROW_IDS[20]], "MEAN"),
        ("rq_sum_02", [VALID_ROW_IDS[31], VALID_ROW_IDS[0], VALID_ROW_IDS[3]], "SUM"),
        ("rq_mean_02", [VALID_ROW_IDS[7], VALID_ROW_IDS[15], VALID_ROW_IDS[23]], "MEAN"),
        ("rq_cross_01", [VALID_ROW_IDS[2], VALID_ROW_IDS[10], VALID_ROW_IDS[18], VALID_ROW_IDS[26]], "SUM"),
        ("rq_missing_01", [VALID_ROW_IDS[0], 777777, VALID_ROW_IDS[1]], "MEAN"),
        ("rq_big_01", [VALID_ROW_IDS[3], VALID_ROW_IDS[3], VALID_ROW_IDS[3]], "SUM"),
        ("rq_mean_03", [VALID_ROW_IDS[4], VALID_ROW_IDS[14]], "MEAN"),
    ]
    for request_id, seq, mode in request_specs:
        requests.append({"request_id": request_id, "id_sequence": seq, "agg_mode": mode})

    bad_requests = [
        {"request_id": "rq_bad_mode", "id_sequence": [VALID_ROW_IDS[0]], "agg_mode": "PRODUCT"},
        {"request_id": "rq_empty", "id_sequence": [], "agg_mode": "SUM"},
        {"request_id": "rq_sum_01", "id_sequence": [VALID_ROW_IDS[1]], "agg_mode": "SUM"},
    ]
    requests_path = os.path.join(out_dir, "requests.jsonl")
    with open(requests_path, "w", encoding="utf-8") as handle:
        for record in requests:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in bad_requests:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    config = {
        "name": "deterministic-workbench",
        "dim": DIM,
        "quant_level": 16,
        "fractional_bits": 8,
        "accumulator_headroom_bits": 4,
        "concurrency": 3,
        "point_budget": 12,
        "rounding": "HALF_UP",
        "agg_modes": ["SUM", "MEAN"],
        "tolerance": 0.02,
        "top_k": 3,
        "alternate_gear": {"quant_level": 12, "fractional_bits": 4},
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        import yaml

        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    zero_budget = dict(config)
    zero_budget["point_budget"] = 0
    zero_budget["name"] = "zero-budget-invalid"
    with open(os.path.join(out_dir, "config_zero_budget.yaml"), "w", encoding="utf-8") as handle:
        import yaml

        yaml.safe_dump(zero_budget, handle, sort_keys=False, allow_unicode=True)

    # ---- 编辑命令序列：行替换 / 块替换 / 追加块 / 档位切换 / 位宽调整 / 聚合切换
    new_vector_a = [float(x) for x in rng.normal(5.0, 0.5, size=DIM)]
    new_vector_b = [float(x) for x in rng.normal(-3.0, 0.7, size=DIM)]
    edits = [
        {
            "op": "replace_row",
            "row_id": VALID_ROW_IDS[0],
            "block_id": 0,
            "offset": 0,
            "vector": new_vector_a,
        },
        {
            "op": "replace_block",
            "block_id": 2,
            "rows": [
                {
                    "row_id": rid,
                    "offset": table_snapshot[rid]["offset"],
                    "vector": [float(x) for x in rng.normal(1.0, 0.4, size=DIM)],
                }
                for rid in VALID_ROW_IDS[16:24]
            ],
        },
        {
            "op": "append_block",
            "block_id": BLOCKS,
            "rows": [
                {
                    "row_id": 500 + offset,
                    "offset": offset,
                    "vector": [float(x) for x in rng.normal(0.0, 1.0, size=DIM)],
                }
                for offset in range(4)
            ],
        },
        {"op": "switch_gear", "quant_level": 12, "fractional_bits": 4},
        {"op": "switch_gear", "quant_level": 16, "fractional_bits": 8},
        {"op": "switch_agg", "request_id": "rq_sum_02", "agg_mode": "MEAN"},
        {
            "op": "replace_row",
            "row_id": VALID_ROW_IDS[5],
            "block_id": 0,
            "offset": 5,
            "vector": new_vector_b,
        },
        {"op": "switch_bits", "fractional_bits": 6},
        {"op": "switch_bits", "fractional_bits": 8},
    ]
    _write_jsonl(os.path.join(out_dir, "edits.jsonl"), edits)

    # 最终表（供全量重建对照）：应用结构类编辑（不含 gear/agg 切换）
    final_state_rows = [dict(record) for record in row_records]
    return {
        "rows_path": rows_path,
        "requests_path": requests_path,
        "config_path": os.path.join(out_dir, "config.yaml"),
        "zero_budget_config_path": os.path.join(out_dir, "config_zero_budget.yaml"),
        "edits_path": os.path.join(out_dir, "edits.jsonl"),
        "valid_row_ids": VALID_ROW_IDS,
        "dim": DIM,
        "seed": SEED,
    }
