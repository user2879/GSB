"""按固定种子在本地生成算子链、输入张量描述、执行配置与异常注入配置。

不依赖任何外部数据或网络服务。运行：
    python -m opdiag.samplegen --out demo/sample
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

GLOBAL_SEED = 20261125


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_chain() -> dict:
    # 设计要点：
    # - 多个激活直连转置 -> 融合候选
    # - 同层并行节点（A1/A2 都消费 in0）验证线程池并发
    # - fan-out 后再原地写回 -> E_INPLACE_CONFLICT
    # - 独立分支承载注入，互不污染主路径
    nodes = [
        {"id": "A1", "op": "relu", "input_ref": "in0",
         "params": {"slope": 0.1}},
        {"id": "A2", "op": "gelu", "input_ref": "in0", "params": {}},
        {"id": "A3", "op": "sigmoid", "input_ref": "in1",
         "params": {"scale": 1.5}},
        {"id": "T1", "op": "transpose", "perm": [1, 0]},
        {"id": "T2", "op": "transpose", "perm": [1, 0]},
        {"id": "A4", "op": "tanh", "params": {"gain": 1.2}},
        {"id": "A5", "op": "relu", "inplace": True,
         "params": {"slope": 0.0}},
        {"id": "T3", "op": "transpose", "perm": [0, 1]},
        {"id": "A6", "op": "relu", "input_ref": "in2",
         "params": {"slope": 0.2}},
        {"id": "A7", "op": "gelu", "input_ref": "in2", "params": {}},
        {"id": "A8", "op": "sigmoid", "params": {"scale": 0.8}},
        {"id": "A9", "op": "tanh", "input_ref": "in1",
         "params": {"gain": 0.9}},
        {"id": "A10", "op": "relu", "input_ref": "in2",
         "params": {"slope": 0.05}},
    ]
    edges = [
        ["A1", "T1"], ["A2", "T2"], ["A3", "A4"], ["T2", "A5"],
        ["A5", "T3"], ["A6", "A7" if False else "A8"],
        ["A8", "A9" if False else "A10"],
    ]
    # 上面的三元表达式仅为可读性占位，这里显式给出确定边：
    edges = [
        ["A1", "T1"], ["A2", "T2"], ["A3", "A4"], ["T2", "A5"],
        ["A5", "T3"], ["A4", "A8"], ["A8", "A10"],
    ]
    return {"chain_id": "demo-chain-001", "seed": GLOBAL_SEED,
            "nodes": nodes, "edges": edges}


def build_inputs() -> dict:
    return {
        "in0": {"shape": [4, 6], "distribution": "uniform", "low": -2.0,
                "high": 2.0, "seed": GLOBAL_SEED + 1},
        "in1": {"shape": [4, 6], "distribution": "normal",
                "seed": GLOBAL_SEED + 2},
        "in2": {"shape": [3, 5], "distribution": "uniform", "low": -1.5,
                "high": 1.5, "seed": GLOBAL_SEED + 3},
    }


def build_execution() -> dict:
    return {
        "precision": "fp32",
        "bit_width": 16,
        "quant_mode": "uniform",
        "tolerance": 1e-5,
        "max_workers": 3,
        "element_budget": 120,
        "snapshot_interval": 2,
        "enable_fusion": True,
        "output_dir": "output",
    }


def build_injections() -> dict:
    return {"injections": [
        {"id": "INJ-01", "node_id": "A6", "stage": "compute",
         "fault": "nonfinite"},
        {"id": "INJ-02", "node_id": "T3", "stage": "validate",
         "fault": "duplicate_perm"},
        {"id": "INJ-03", "node_id": "A9", "stage": "validate",
         "fault": "activation_param"},
        {"id": "INJ-04", "node_id": "A7", "stage": "compute",
         "fault": "impl_version_missing"},
        {"id": "INJ-05", "node_id": "A2", "stage": "quantize",
         "fault": "nonfinite"},
    ]}


def build_run_config() -> dict:
    return {
        "chain_file": "chain.json",
        "inputs_file": "inputs.json",
        "injection_file": "injections.json",
        "execution": build_execution(),
    }


def generate(out_dir: str):
    _write(os.path.join(out_dir, "chain.json"), build_chain())
    _write(os.path.join(out_dir, "inputs.json"), build_inputs())
    _write(os.path.join(out_dir, "execution.json"), build_execution())
    _write(os.path.join(out_dir, "injections.json"), build_injections())
    _write(os.path.join(out_dir, "run_config.json"), build_run_config())
    # 健康基线（无注入），供全量重建与增量一致性对照
    healthy = build_run_config()
    healthy.pop("injection_file")
    healthy["execution"]["output_dir"] = "output_healthy"
    _write(os.path.join(out_dir, "run_config_healthy.json"), healthy)
    rng = np.random.default_rng(GLOBAL_SEED)
    return rng


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m opdiag.samplegen")
    ap.add_argument("--out", default="demo/sample")
    args = ap.parse_args(argv)
    generate(args.out)
    print(f"[samplegen] wrote deterministic sample to {args.out}")


if __name__ == "__main__":
    main()
