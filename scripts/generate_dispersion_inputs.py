"""生成“节点执行离散”专用清单：两版取值完全相同，仅候选版增加确定性休眠。

节点收到的样本与配置值逐元素相同，因此聚合摘要必然一致；只有整数毫秒
耗时在候选版增加，用于触发仅耗时类 E_REGRESSION 并验证离散归因。
"""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(data_dir: str = "data/dispersion") -> None:
    root = Path(data_dir)
    registry = {
        "nodes": [
            {"node_id": "node_a", "host": "127.0.0.1", "port": 19201, "fail_rate": 0.0},
            {"node_id": "node_b", "host": "127.0.0.1", "port": 19202, "fail_rate": 0.0},
        ]
    }
    common_item = {
        "derived": False,
        "value": 1.5,
        "targets": ["node_a", "node_b"],
        "work_repeat": 1,
    }
    baseline = {"version": 1, "strategy": "broadcast", "items": {"latency_sensitive": dict(common_item)}}
    candidate_item = dict(common_item)
    candidate_item["extra_sleep_ms"] = 80
    candidate = {"version": 2, "strategy": "broadcast", "items": {"latency_sensitive": candidate_item}}
    write_json(root / "registry.json", registry)
    write_json(root / "baseline.json", baseline)
    write_json(root / "candidate.json", candidate)
    print(f"已生成离散场景输入于 {root}")


if __name__ == "__main__":
    main()
