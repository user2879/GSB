"""生成示例算子配置、异常注入清单与固定种子确定性张量包。

用法：python scripts/generate_samples.py
所有样本均为本地程序生成，不依赖外部下载或真实模型权重。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fablock.config import OperatorConfig
from fablock.dataset import SampleDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OPERATOR_CONFIG = {
    "query_block_len": 8,
    "key_block_len": 8,
    "head_dim": 8,
    "burst_query_blocks": [1],
    "fixed_point_frac_bits": 12,
    "quantization_level_bits": 16,
    "accumulator_int_bits": 64,
    "stability_offset": 0.0,
    "tolerances": {"score": 0.05, "softmax_state": 0.05, "output": 0.05},
    "thread_pool_workers": 4,
    "point_budget_per_batch": 100000,
    "snapshot_commit_interval_batches": 1,
}

INJECTION_MANIFEST = {
    "injections": [
        {"id": "INJ-001", "type": "missing_tensor", "target": {"tensor": "V"}},
        {"id": "INJ-002", "type": "head_dim_mismatch",
         "target": {"tensor": "Q", "head_dim": 6}},
        {"id": "INJ-003", "type": "nonfinite",
         "target": {"tensor": "K", "row": 10, "col": 3}},
        {"id": "INJ-004", "type": "budget_below_min_block", "target": {}},
        {"id": "INJ-005", "type": "frac_bits_out_of_range",
         "target": {"value": 40}},
        {"id": "INJ-006", "type": "block_fault",
         "target": {"stage": "matmul", "query_block": 2, "key_block": 1}},
        {"id": "INJ-007", "type": "block_fault",
         "target": {"stage": "softmax", "query_block": 0, "key_block": 2}},
    ]
}


def main():
    cfg = OperatorConfig.from_dict(OPERATOR_CONFIG)
    issues = cfg.validate()
    assert not issues, [i.to_dict() for i in issues]

    configs_dir = os.path.join(ROOT, "configs")
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    cfg_path = os.path.join(configs_dir, "operator_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(OPERATOR_CONFIG, fh, ensure_ascii=False, indent=2)

    inj_path = os.path.join(configs_dir, "injection_manifest.json")
    with open(inj_path, "w", encoding="utf-8") as fh:
        json.dump(INJECTION_MANIFEST, fh, ensure_ascii=False, indent=2)

    ds = SampleDataset(seed=20240501, s_q=20, s_k=24, head_dim=cfg.head_dim,
                       query_block_len=cfg.query_block_len,
                       key_block_len=cfg.key_block_len,
                       burst_query_blocks=cfg.burst_query_blocks)
    pack_path = os.path.join(data_dir, "sample_pack.fatp")
    pack_fp = ds.write_pack(pack_path)

    print(f"配置: {cfg_path} (hash={cfg.config_hash()})")
    print(f"注入清单: {inj_path} ({len(INJECTION_MANIFEST['injections'])} 项)")
    print(f"张量包: {pack_path} (fp={pack_fp}, "
          f"Q{ds.Q.shape} K{ds.K.shape} V{ds.V.shape}, seed={ds.seed})")


if __name__ == "__main__":
    main()
