"""命令行入口：样本生成、审计与报告输出，退出码区分结论。"""

import argparse
import json
import os
import sys

from .errors import AuditError, E_ARGS
from .engine import AuditEngine
from .generator import build_image, build_patch, write_sample
from .image import load_image_file
from .patch import BudgetConfig
from .report import build_report, write_outputs, render_graph_text
from .verify import verify_image_truth

EXIT_ALL_PASS = 0
EXIT_PARTIAL_FAIL = 1
EXIT_ARG_ERROR = 2


def _read_inputs(paths):
    inputs = []
    for path in paths:
        with open(path, "rb") as handle:
            inputs.append((path, handle.read()))
    return inputs


def cmd_generate(args):
    os.makedirs(args.samples_dir, exist_ok=True)
    partitions = [
        {"name": "bootloader", "part_type": 0, "slot": 0,
         "payload_length": 256},
        {"name": "kernel", "part_type": 1, "slot": 0,
         "payload_length": 512},
        {"name": "appfs", "part_type": 2, "slot": 0,
         "payload_length": 768},
    ]
    image_v1 = build_image(partitions, version=1, base_image_id=100)
    write_sample(os.path.join(args.samples_dir, "base_v1.fwim"), image_v1)

    image_v2 = build_image(partitions, version=2, base_image_id=200)
    write_sample(os.path.join(args.samples_dir, "base_v2.fwim"), image_v2)

    good_patch = build_patch(
        [
            {"name": "kernel", "write_offset": 64, "write_length": 32},
            {"name": "appfs", "write_offset": 128, "write_length": 64},
        ],
        base_image_id=100,
    )
    write_sample(os.path.join(args.samples_dir, "good.fwdf"), good_patch)

    bounds_patch = build_patch(
        [{"name": "kernel", "write_offset": 500, "write_length": 64}],
        base_image_id=100,
    )
    write_sample(
        os.path.join(args.samples_dir, "bounds.fwdf"), bounds_patch
    )

    stale_patch = build_patch(
        [{"name": "kernel", "write_offset": 0, "write_length": 16}],
        base_image_id=999,
    )
    write_sample(
        os.path.join(args.samples_dir, "stale.fwdf"), stale_patch
    )

    malformed_image = build_image(
        partitions, base_image_id=100, corrupt={"magic": True}
    )
    write_sample(
        os.path.join(args.samples_dir, "malformed.fwim"), malformed_image
    )
    print(json.dumps({
        "samples_dir": args.samples_dir,
        "files": sorted(os.listdir(args.samples_dir)),
    }, ensure_ascii=False, indent=2))
    return EXIT_ALL_PASS


def _build_budget(args):
    return BudgetConfig(
        image_byte_threshold=args.image_byte_threshold,
        op_count_threshold=args.op_count_threshold,
        chunk_size=args.chunk_size,
        max_ops_per_patch=args.max_ops,
        timeout_seconds=args.timeout,
        concurrency=args.concurrency,
    )


def cmd_audit(args):
    budget = _build_budget(args)
    os.makedirs(args.reports_dir, exist_ok=True)
    engine = AuditEngine(args.workdir, budget=budget)

    try:
        image_inputs = _read_inputs(args.images)
        patch_inputs = _read_inputs(args.patches)
    except OSError as exc:
        print(json.dumps({"code": E_ARGS, "message": str(exc)}),
              file=sys.stderr)
        return EXIT_ARG_ERROR

    engine_output = engine.run(image_inputs, patch_inputs)

    # 报告以本批最新可解析镜像为准；全部解析失败时单独输出错误报告。
    latest_path, latest_blob = image_inputs[-1]
    try:
        image_info = load_image_file(latest_path)
    except AuditError as exc:
        payload = {
            "report_version": 1,
            "fatal_parse_error": exc.as_dict(),
            "parse_errors": engine_output["parse_errors"],
            "patches": engine_output["results"],
        }
        report_path = os.path.join(args.reports_dir, "audit_report.json")
        from .storage import atomic_write_json
        atomic_write_json(report_path, payload)
        print(json.dumps(payload["fatal_parse_error"], ensure_ascii=False))
        return EXIT_PARTIAL_FAIL

    report = build_report(image_info, engine_output, budget)
    report_path = os.path.join(args.reports_dir, "audit_report.json")
    graph_path = os.path.join(args.reports_dir, "partition_map.json")
    text_graph_path = os.path.join(args.reports_dir, "partition_map.txt")
    write_outputs(report, report_path, graph_path)
    from .storage import atomic_write
    atomic_write(text_graph_path, render_graph_text(report["graph"]))

    try:
        ok, truth = verify_image_truth(latest_blob)
    except AuditError as exc:
        ok, truth = False, {"error": exc.as_dict(), "checked": [],
                            "mismatches": []}
    report["truth_verification"] = truth
    from .storage import atomic_write_json
    atomic_write_json(report_path, report)

    print(json.dumps({
        "report": report_path,
        "graph": graph_path,
        "text_graph": text_graph_path,
        "summary": report["summary"],
        "truth_ok": ok,
    }, ensure_ascii=False, indent=2))
    if report["summary"]["overall_status"] == "pass":
        return EXIT_ALL_PASS
    return EXIT_PARTIAL_FAIL


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fwaudit", description="固件镜像分区映射审计引擎"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本")
    gen.add_argument("--samples-dir", default="samples")
    gen.set_defaults(func=cmd_generate)

    audit = sub.add_parser("audit", help="审计镜像与补丁")
    audit.add_argument("--images", nargs="+", required=True)
    audit.add_argument("--patches", nargs="*", default=[])
    audit.add_argument("--reports-dir", default="reports")
    audit.add_argument("--workdir", default=".fwaudit_state")
    audit.add_argument("--image-byte-threshold", type=int, default=4096)
    audit.add_argument("--op-count-threshold", type=int, default=16)
    audit.add_argument("--chunk-size", type=int, default=8)
    audit.add_argument("--max-ops", type=int, default=4096)
    audit.add_argument("--timeout", type=float, default=5.0)
    audit.add_argument("--concurrency", type=int, default=2)
    audit.set_defaults(func=cmd_audit)
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except AuditError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return EXIT_PARTIAL_FAIL
    except SystemExit:
        raise
    except (OSError, ValueError) as exc:
        print(json.dumps({"code": E_ARGS, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return EXIT_ARG_ERROR


if __name__ == "__main__":
    sys.exit(main())
