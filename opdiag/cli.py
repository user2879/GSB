"""命令行入口：python -m opdiag --config <run_config.json>"""

from __future__ import annotations

import argparse
import json
import os

from .engine import Engine


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve(base, p):
    return p if os.path.isabs(p) else os.path.join(base, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opdiag",
                                 description="激活/转置算子链诊断组件")
    ap.add_argument("--config", required=True, help="执行配置 JSON 路径")
    ap.add_argument("--generate-sample", action="store_true",
                    help="按固定种子生成示例输入（由 samplegen 提供）")
    args = ap.parse_args(argv)

    cfg_path = os.path.abspath(args.config)
    base = os.path.dirname(cfg_path)
    run_cfg = _load(cfg_path)

    chain = _load(_resolve(base, run_cfg["chain_file"]))
    inputs_desc = _load(_resolve(base, run_cfg["inputs_file"]))
    injections = []
    if run_cfg.get("injection_file"):
        inj_path = _resolve(base, run_cfg["injection_file"])
        if os.path.exists(inj_path):
            injections = _load(inj_path).get("injections", [])

    engine = Engine(chain, inputs_desc,
                    run_cfg.get("execution", {}), injections,
                    base_dir=base)
    artifacts = engine.run()
    out_dir = os.path.abspath(os.path.join(
        base, run_cfg.get("output_dir",
                          run_cfg.get("execution", {}).get(
                              "output_dir", engine.cfg["output_dir"]))))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(artifacts.report, f, ensure_ascii=False, indent=2,
                  sort_keys=True)
    print(f"[opdiag] report -> {report_path}")
    print(f"[opdiag] versions -> {artifacts.report['global_version']} "
          f"reused={len(artifacts.report['reuse_list'])} "
          f"recomputed={len(artifacts.report['recompute_list'])} "
          f"pending={len(artifacts.report['pending_list'])} "
          f"failures={len(artifacts.report['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
