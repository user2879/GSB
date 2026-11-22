"""生成两版确定性样本清单与节点注册表（全部落在 data/ 内）。

清单刻意覆盖全部差异类型：
* ``base_rate`` 取值变更，并沿派生链影响 ``throughput``、``score``；
* ``noise_floor`` 派生系数变更（系数放大场景）；
* ``weighted_mix`` 目标节点变更；
* ``spare_knob`` 在候选版删除，``canary_boost`` 在候选版新增；
* ``node-c`` 注册 50% 丢包率，用于验证真实超时与重试隔离。
"""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(data_dir: str = "data") -> None:
    root = Path(data_dir)
    registry = {
        "nodes": [
            {"node_id": "node_a", "host": "127.0.0.1", "port": 19101, "fail_rate": 0.0},
            {"node_id": "node_b", "host": "127.0.0.1", "port": 19102, "fail_rate": 0.0},
            {"node_id": "node_c", "host": "127.0.0.1", "port": 19103, "fail_rate": 0.5},
        ]
    }
    baseline = {
        "version": 1,
        "strategy": "rolling",
        "items": {
            "base_rate": {
                "derived": False,
                "value": 1.25,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 2,
            },
            "noise_floor": {
                "derived": False,
                "value": 0.4,
                "targets": ["node_a", "node_b"],
                "work_repeat": 1,
            },
            "spare_knob": {
                "derived": False,
                "value": 7.0,
                "targets": ["node_a"],
                "work_repeat": 1,
            },
            "throughput": {
                "derived": True,
                "coeffs": {"base_rate": 2.0},
                "const": 0.5,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 2,
            },
            "weighted_mix": {
                "derived": True,
                "coeffs": {"base_rate": 1.0, "noise_floor": -1.0},
                "const": 0.0,
                "targets": ["node_a", "node_b"],
                "work_repeat": 1,
            },
            "score": {
                "derived": True,
                "coeffs": {"throughput": 1.5, "weighted_mix": 0.5},
                "const": 0.1,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 3,
            },
        },
    }
    candidate = {
        "version": 2,
        "strategy": "canary",
        "items": {
            "base_rate": {
                "derived": False,
                "value": 3.0,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 2,
            },
            "noise_floor": {
                "derived": False,
                "value": 0.4,
                "targets": ["node_a", "node_b"],
                "work_repeat": 1,
            },
            "canary_boost": {
                "derived": False,
                "value": 2.25,
                "targets": ["node_c"],
                "work_repeat": 1,
            },
            "throughput": {
                "derived": True,
                "coeffs": {"base_rate": 2.0},
                "const": 0.5,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 2,
            },
            "weighted_mix": {
                "derived": True,
                "coeffs": {"base_rate": 1.0, "noise_floor": -2.0},
                "const": 0.0,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 1,
            },
            "score": {
                "derived": True,
                "coeffs": {"throughput": 3.0, "weighted_mix": 0.5},
                "const": 0.1,
                "targets": ["node_a", "node_b", "node_c"],
                "work_repeat": 3,
            },
        },
    }
    write_json(root / "registry.json", registry)
    write_json(root / "baseline.json", baseline)
    write_json(root / "candidate.json", candidate)
    print(f"已生成: {root / 'registry.json'}")
    print(f"已生成: {root / 'baseline.json'}")
    print(f"已生成: {root / 'candidate.json'}")


if __name__ == "__main__":
    main()
