"""工作台命令行入口。

子命令：
  generate  按固定种子生成样本 JSONL 与默认配置 YAML、注入配置；
  run       运行回测（--fresh 全量，否则增量复用缓存），写出交换文件与报告；
  convert   在 v1/v2 交换文件间转换并生成/消费旁路；
  report    仅根据最近运行重算诊断报告。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .exchange.build import build_ir
from .exchange.converter import fingerprint_v2
from .exchange.v1_codec import write_v1, fingerprint_v1
from .exchange.v2_codec import write_v2
from .inject import load_injections
from .runner import run_pipeline
from .sample import default_specs, write_jsonl
from .report import build_report, write_report

DEFAULT_CONFIG_YAML = """\
backtest:
  window_length: 24
  window_step: 12
  horizon: 6
models:
  - name: naive
    weight: 0.2
    lag_order: 0
  - name: moving_average
    weight: 0.3
    lag_order: 5
  - name: autoregressive
    weight: 0.5
    lag_order: 3
features:
  lag_orders: [1, 2, 3]
missing:
  strategy: forward_fill
quantization:
  frac_bits: 4
  gear: 1
exchange:
  version: v1
execution:
  max_workers: 3
  observation_budget: 100000
  series_quota: 100000
  snapshot_every_windows: 4
  temp_dir: .tmp
"""

DEFAULT_INJECTION_JSON = {
    "injections": [
        {"id": "INJ-001", "kind": "irregular_timestamp", "series_id": "S001", "window_index": 1},
        {"id": "INJ-002", "kind": "non_finite_value", "series_id": "S002", "window_index": 2},
        {"id": "INJ-003", "kind": "window_too_long", "series_id": "S003", "window_index": 0},
        {"id": "INJ-004", "kind": "quant_gear_out_of_range", "series_id": "S000", "window_index": 0},
    ]
}


def _write_default_assets(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(sample_dir / "samples.jsonl", default_specs())
    (sample_dir / "config.yaml").write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    (sample_dir / "injections.json").write_text(
        json.dumps(DEFAULT_INJECTION_JSON, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit_exchange(cfg, run, out_dir: Path) -> dict[str, str]:
    ir = build_ir(cfg, run.results)
    paths: dict[str, str] = {}
    if cfg.exchange_version == "v1":
        ir.version_fingerprint = fingerprint_v1(ir)
        v1_path = out_dir / "exchange.v1.txt"
        write_v1(ir, v1_path)
        paths["v1"] = str(v1_path)
    else:
        ir.version_fingerprint = fingerprint_v2(ir)
        v2_path = out_dir / "exchange.v2.json"
        write_v2(ir, v2_path)
        paths["v2"] = str(v2_path)
    return paths


def cmd_generate(args) -> None:
    sample_dir = Path(args.sample_dir)
    _write_default_assets(sample_dir)
    print(f"已生成样本与配置到 {sample_dir}")


def cmd_run(args) -> None:
    cfg = load_config(Path(args.sample_dir) / "config.yaml")
    injections = load_injections(Path(args.sample_dir) / "injections.json") \
        if args.inject else []
    work_dir = Path(args.work_dir)
    run = run_pipeline(cfg, Path(args.sample_dir) / "samples.jsonl",
                       work_dir, injections=injections, fresh=args.fresh)
    out_dir = work_dir / "exchange"
    paths = _emit_exchange(cfg, run, out_dir)
    report = build_report(cfg, run, paths)
    report_path = write_report(work_dir / "diagnostic_report.json", report)
    print(json.dumps({
        "mode": "fresh" if args.fresh else "incremental",
        "windows": len(run.results),
        "failures": len(run.failures),
        "exchange": paths,
        "report": str(report_path),
        "cursors": run.cursors,
    }, ensure_ascii=False, indent=2))


def cmd_convert(args) -> None:
    from .exchange.converter import convert_file
    summary = convert_file(args.src, args.dst)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="多传感器时序预测可重复演示工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="生成确定性样本与默认配置")
    g.add_argument("--sample-dir", default="data")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="运行回测并产出交换文件与诊断报告")
    r.add_argument("--sample-dir", default="data")
    r.add_argument("--work-dir", default="work")
    r.add_argument("--fresh", action="store_true", help="忽略缓存从空白全量重建")
    r.add_argument("--inject", action="store_true", help="启用注入配置")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("convert", help="v1/v2 交换文件双向转换")
    c.add_argument("src")
    c.add_argument("dst")
    c.set_defaults(func=cmd_convert)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
