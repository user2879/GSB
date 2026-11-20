"""实验台运行入口。

用法：
  python main.py generate                      # 生成确定性样本
  python main.py run [--workers N] [--budget B] [--output DIR]
  python main.py verify [--output DIR]         # 串行 vs 并行逐字段比对
"""

import argparse
import json
import os
import subprocess
import sys

from fusion_bench.config import load_config
from fusion_bench.errors import E_CONFIG_INVALID, PipelineError
from fusion_bench.injection import InjectionPlan
from fusion_bench.pipeline import (EXIT_CONFIG_INVALID, EXIT_OK, Pipeline)

COMPARE_ARTIFACTS = (
    "results.jsonl", "failures.jsonl", "budget_ledger.jsonl",
    "cache_evictions.jsonl", "cache_recomputes.jsonl",
    "pending_queue.json", "summary.json",
)


def _run_once(args, output_dir, workers=None, budget=None):
    try:
        cfg = load_config(args.config)
    except PipelineError as exc:
        os.makedirs(output_dir, exist_ok=True)
        record = {"sample_id": "<run>", "stage": exc.stage,
                  "code": E_CONFIG_INVALID, "message": exc.message,
                  "injection_id": None}
        with open(os.path.join(output_dir, "failures.jsonl"), "w") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"E_CONFIG_INVALID: {exc.message}", file=sys.stderr)
        return EXIT_CONFIG_INVALID
    plan = InjectionPlan.load(args.inject)
    pipe = Pipeline(cfg, plan, args.samples, output_dir,
                    budget_override=budget,
                    workers_override=workers,
                    stage_delay_ms=args.stage_delay_ms)
    pipe.install_signal_handlers()
    return pipe.run()


def cmd_generate(args):
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "scripts", "generate_samples.py")
    cmd = [sys.executable, script, "--output", args.samples,
           "--count", str(args.count)]
    return subprocess.call(cmd)


def cmd_run(args):
    return _run_once(args, args.output,
                     workers=args.workers, budget=args.budget)


def _read_artifact(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def cmd_verify(args):
    """串行（workers=1）与并行（workers=配置值）全量执行并逐字段比对。"""
    serial_dir = os.path.join(args.output, "serial")
    parallel_dir = os.path.join(args.output, "parallel")
    code_s = _run_once(args, serial_dir, workers=1)
    code_p = _run_once(args, parallel_dir, workers=None)
    report = {"serial_exit": code_s, "parallel_exit": code_p,
              "artifacts": {}, "identical": True}
    for name in COMPARE_ARTIFACTS:
        a = _read_artifact(os.path.join(serial_dir, name))
        b = _read_artifact(os.path.join(parallel_dir, name))
        same = a is not None and a == b
        report["artifacts"][name] = "identical" if same else "DIFF"
        if not same:
            report["identical"] = False
    if code_s != code_p:
        report["identical"] = False
    out_path = os.path.join(args.output, "verify_report.json")
    os.makedirs(args.output, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return EXIT_OK if report["identical"] else 1


def main():
    parser = argparse.ArgumentParser(description="多模态特征融合实验台")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    parser.add_argument("--inject", default="configs/injection.yaml")
    parser.add_argument("--samples", default="data/samples")
    parser.add_argument("--stage-delay-ms", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--count", type=int, default=32)

    p_run = sub.add_parser("run")
    p_run.add_argument("--output", default="output/run")
    p_run.add_argument("--workers", type=int, default=None)
    p_run.add_argument("--budget", type=int, default=None)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--output", default="output/verify")

    args = parser.parse_args()
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
