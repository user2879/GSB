"""确定性样本生成：所有数值由固定规则产生（不使用随机数）。"""

import json
from pathlib import Path


def call_sequence(function_id, function_version, minutes=12):
    seq = []
    for minute in range(minutes):
        # 调用数：基线 + 日内正弦状确定性波动（整数规则）
        calls = 160 + 140 * ((minute * 37) % 11) // 10 + (minute * 7) % 5
        # 突发万分比：在 5%~35% 之间按确定性规则摆动
        burst_pm = 500 + ((minute * 1729) % 3001)
        seq.append(
            {
                "function_id": function_id,
                "function_version": function_version,
                "minute": minute,
                "calls": calls,
                "burst_permyriad": burst_pm,
            }
        )
    return seq


def base_manifest(declaration_version, function_version, scale, warmup_pm):
    fid = "fn-order-processor"
    return {
        "manifest_id": "order-processor-manifest",
        "declaration_version": declaration_version,
        "scale": scale,
        "functions": [
            {
                "function_id": fid,
                "function_version": function_version,
                "memory_mib": 512,
                "init_ms": 420,
                "concurrency": 10,
                "min_instances": 2,
                "max_instances": 40,
            }
        ],
        "call_sequence": call_sequence(fid, function_version),
        "policy": {
            "instance_concurrency_cap": 10,
            "global_concurrency_cap": 180,
            "reserved_instances": 4,
            "warmup_permyriad": warmup_pm,
            "global_memory_mib": 16384,
        },
    }


def write_samples(sample_dir):
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    # 场景 1：首版声明（预热 25%）
    v1 = base_manifest(declaration_version=1, function_version=1, scale=10000, warmup_pm=2500)
    # 场景 2：修改预热比例（->40%），函数版本与声明版本同时单调递增
    v2 = base_manifest(declaration_version=2, function_version=2, scale=10000, warmup_pm=4000)
    # 场景 3：缩放因子扰动（10000 -> 100000），声明版本继续递增
    v3 = base_manifest(declaration_version=3, function_version=3, scale=100000, warmup_pm=4000)
    paths = []
    for name, doc in (("manifest_v1.json", v1), ("manifest_v2.json", v2), ("manifest_v3.json", v3)):
        p = sample_dir / name
        p.write_text(json.dumps(doc, sort_keys=True, indent=2), encoding="utf-8")
        paths.append(p)
    return paths
