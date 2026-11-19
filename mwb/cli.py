"""命令行入口：退出码 0 正常 / 2 可处理终止 / 1 业务错误。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .atomicio import read_json
from .errors import WorkbenchError
from .parse import parse_score
from .parse_edit import parse_edit_script
from .parse_params import parse_expression
from .report import (
    REPORT_NAME,
    TIMELINE_NAME,
    compare_timelines,
    export_results,
    full_recompute_timeline,
    verify_persisted_chunks,
)
from .runner import EXIT_GRACEFUL_TERMINATION, EXIT_OK, WorkbenchRunner
from .samples import CHUNK_SIZE, generate_workspace


def _emit_error(err: WorkbenchError) -> int:
    # 业务错误：输出带 E_xxx 业务码的 JSON，到 stderr，不产生持久化输出
    print(json.dumps(err.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
    return 1


def _cmd_generate(args) -> int:
    paths = generate_workspace(args.samples_dir)
    print(json.dumps({"ok": True, "samples": paths}, ensure_ascii=False, indent=2))
    return EXIT_OK


def _load_inputs(args):
    score = parse_score(args.score)
    params = parse_expression(args.expression)
    edits, concurrency, memory_budget = parse_edit_script(args.edits)
    score_dict = read_json(args.score)
    params_dict = read_json(args.expression)
    edit_script_dict = read_json(args.edits)
    return score, params, edits, concurrency, memory_budget, score_dict, params_dict, edit_script_dict


def _cmd_run(args) -> int:
    try:
        score, params, edits, concurrency, memory_budget, score_dict, params_dict, edit_script_dict = _load_inputs(args)
        runner = WorkbenchRunner(
            score=score,
            base_params=params,
            edits=edits,
            state_root=args.state_dir,
            chunk_size=args.chunk_size,
            concurrency=concurrency,
            memory_budget_chunks=memory_budget,
            score_dict=score_dict,
            params_dict=params_dict,
            edit_script_dict=edit_script_dict,
            tolerances={"duration": args.duration_tol, "velocity": args.velocity_tol},
        )
        runner.stop.install_signals()
        result = runner.run()
        if result["status"] == "interrupted":
            print(
                json.dumps(
                    {"ok": True, "status": "interrupted", "step_cursor": result["step_cursor"],
                     "version_id": runner.version},
                    ensure_ascii=False, indent=2,
                )
            )
            return EXIT_GRACEFUL_TERMINATION

        manifest = result["manifest"]
        # 最终逐块摘要校验（E_DIGEST 时由调用方回退补算）
        try:
            chunks = verify_persisted_chunks(runner.store, manifest)
        except WorkbenchError as digest_error:
            # 回退补算：剔除坏块后重算缺失块，再校验一次
            if digest_error.code == "E_DIGEST":
                good = []
                for entry in list(manifest["chunks"]):
                    try:
                        runner.store.verify_chunk(entry["chunk_index"])
                        good.append(entry)
                    except WorkbenchError:
                        bad_path = runner.store.chunk_path(entry["chunk_index"])
                        if os.path.exists(bad_path):
                            os.unlink(bad_path)
                manifest["chunks"] = good
                params2, placed2 = runner._params_at(manifest["step_cursor"])
                have = {c["chunk_index"] for c in good}
                runner._run_chunk_pool([i for i in range(manifest["chunk_count"]) if i not in have],
                                       params2, placed2, manifest)
                runner.store.write_manifest(manifest)
                chunks = verify_persisted_chunks(runner.store, manifest)
            else:
                raise

        # 与从头全量重演比对
        full_chunks = full_recompute_timeline(score, result["params_final"], args.chunk_size)
        consistency = compare_timelines(chunks, full_chunks, args.duration_tol, args.velocity_tol)
        report = export_results(
            args.output_dir, score, runner, manifest, chunks, consistency, runner.impact_log
        )
        print(json.dumps({"ok": True, "report": os.path.join(args.output_dir, REPORT_NAME),
                          "timeline": os.path.join(args.output_dir, TIMELINE_NAME),
                          "consistency": consistency, "version_id": runner.version},
                         ensure_ascii=False, indent=2))
        return EXIT_OK
    except WorkbenchError as err:
        return _emit_error(err)


def _cmd_gui(args) -> int:
    try:
        from .gui import launch_gui
        launch_gui(args.output_dir)
    except Exception as exc:  # 环境缺少 tkinter 或输出目录缺失
        print(json.dumps({"ok": False, "code": "E_IO", "message": f"无法启动桌面视图: {exc}"},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mwb", description="演奏表达时间轴诊断工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性本地样本")
    gen.add_argument("--samples-dir", default="samples")
    gen.set_defaults(func=_cmd_generate)

    run = sub.add_parser("run", help="展开、增量重演、中断恢复并输出诊断报告")
    run.add_argument("--score", default="samples/score.json")
    run.add_argument("--expression", default="samples/expression.json")
    run.add_argument("--edits", default="samples/edits.json")
    run.add_argument("--state-dir", default="state")
    run.add_argument("--output-dir", default="output")
    run.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    run.add_argument("--duration-tol", type=int, default=0)
    run.add_argument("--velocity-tol", type=int, default=0)
    run.set_defaults(func=_cmd_run)

    gui = sub.add_parser("gui", help="启动本地桌面诊断视图（只读已持久化结果）")
    gui.add_argument("--output-dir", default="output")
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
