"""命令行入口：python -m relx [data_dir] [output_dir]。"""

import argparse
import os

from .pipeline import run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地确定性中文关系抽取演示")
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("data_dir", nargs="?",
                        default=os.path.join(default_root, "data"))
    parser.add_argument("output_dir", nargs="?",
                        default=os.path.join(default_root, "output"))
    args = parser.parse_args(argv)
    summary = run_demo(args.data_dir, args.output_dir)
    print("=" * 68)
    print("RELX 本地确定性关系抽取演示完成")
    print("=" * 68)
    print(f"样本句子数: {summary['sample_info']['sentence_count']}")
    print(f"异常样本行: {summary['sample_info']['invalid_line_count']}")
    print(f"装载失败清单: {len(summary['load_failures'])} 条")
    print(f"全量结果句数: {summary['full_rebuild']['sentence_results']}")
    print(f"预算批次数: {summary['full_rebuild']['budget_batches']}")
    print(f"账本核对一致: {summary['full_rebuild']['ledger_consistent']}")
    print(f"增量与全量逐字段一致: {summary['incremental_vs_full_identical']}")
    print(f"串行与并发逐字段一致: {summary['concurrency_identical']}")
    print(f"结果目录: {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
