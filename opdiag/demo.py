"""端到端可重复演示：样本生成 -> 多批执行 -> 注入隔离 -> 编辑序列一致性。

运行：python -m opdiag.demo
"""

from __future__ import annotations

import copy
import json
import os
import shutil

from . import samplegen
from .engine import Engine
from .verify import run_consistency_probe


def _node(chain, nid):
    return next(n for n in chain["nodes"] if n["id"] == nid)


def build_edits():
    """覆盖需求 7 的全部编辑类型。每个 edit 返回 (chain, cfg)。"""

    def edit_param(chain, cfg):
        # 1) 节点参数修改：A1 relu slope 0.1 -> 0.25
        _node(chain, "A1")["params"]["slope"] = 0.25
        return chain, cfg

    def edit_order(chain, cfg):
        # 2) 节点顺序调整：仅改变 nodes 数组声明顺序（拓扑由边决定，
        #    位次稳定，但参数指纹上下文变化由内容指纹捕捉）
        chain = copy.deepcopy(chain)
        chain["nodes"].sort(key=lambda n: n["id"], reverse=True)
        return chain, cfg

    def edit_subgraph(chain, cfg):
        # 3) 子图替换：把 A4 tanh 的 gain 与 A8 sigmoid scale 一起替换
        chain = copy.deepcopy(chain)
        _node(chain, "A4")["params"]["gain"] = 2.0
        _node(chain, "A8")["params"]["scale"] = 1.1
        return chain, cfg

    def edit_precision(chain, cfg):
        # 4) 精度档位切换 fp32 -> fp64（真实改变中间值与定点整数）
        cfg = copy.deepcopy(cfg)
        cfg["precision"] = "fp64"
        return chain, cfg

    def edit_inplace(chain, cfg):
        # 5) 原地写回标记变更：给独立分支 A6 加 inplace（A6 无消费者，安全）
        chain = copy.deepcopy(chain)
        _node(chain, "A6")["inplace"] = True
        return chain, cfg

    def edit_bitwidth(chain, cfg):
        # 附加：定点位宽切换 16 -> 8，验证整数表示真实改变
        cfg = copy.deepcopy(cfg)
        cfg["bit_width"] = 8
        cfg["quant_mode"] = "pow2"
        return chain, cfg

    return [edit_param, edit_order, edit_subgraph, edit_precision,
            edit_inplace, edit_bitwidth]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "demo"))
    sample = os.path.join(root, "sample")
    samplegen.generate(sample)
    chain = json.load(open(os.path.join(sample, "chain.json"),
                           encoding="utf-8"))
    inputs_desc = json.load(open(os.path.join(sample, "inputs.json"),
                                 encoding="utf-8"))
    cfg = samplegen.build_execution()
    injections = samplegen.build_injections()["injections"]

    workroot = os.path.join(root, "probe")
    shutil.rmtree(workroot, ignore_errors=True)
    result = run_consistency_probe(
        chain, inputs_desc, cfg, None, workroot, build_edits())

    # 注入隔离单独验证（带注入的健康链，逐步多批直到全部命中）
    inj_dir = os.path.join(root, "injection_run")
    shutil.rmtree(inj_dir, ignore_errors=True)
    hit_codes = set()
    for _ in range(6):
        eng = Engine(chain, inputs_desc, cfg, injections, base_dir=inj_dir)
        art = eng.run()
        hit_codes = {h["code"] for h in art.report["injection_hits"]}
        if len(hit_codes) >= 4:
            break
    injection_report = art.report

    summary = {
        "consistency": result,
        "injection_hit_codes": sorted(hit_codes),
        "injection_hits": injection_report["injection_hits"],
    }
    out = os.path.join(root, "demo_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("[demo] consistency =", result["consistent"])
    if result["diffs"]:
        for d in result["diffs"]:
            print("  DIFF:", d)
    print("[demo] injection codes =", sorted(hit_codes))
    print("[demo] summary ->", out)
    return 0 if result["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
