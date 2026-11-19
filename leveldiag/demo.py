"""可重复演示入口与命令行。

子命令：
  demo        一条命令生成四类样本，输出哈希与账本，校验两次运行逐字节一致，
              并演示全部六个业务错误码与增量重生成闭包。
  generate    从 YAML 规格生成工件
  recompute   对工件目录运行独立复算
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

import yaml

from .errors import ErrorCode, LevelDiagError
from .generator import generate_from_dict, run_pipeline, write_artifacts
from .incremental import incremental_regenerate
from .recompute import recompute
from .samples import all_samples, boundary_spec, mutate, normal_spec
from .spec import parse_spec, spec_from_dict
from .util import canonical_dumps, file_hash


def _bytewise_equal(d1: Path, d2: Path) -> tuple[bool, list[str]]:
    files1 = {str(p.relative_to(d1)) for p in d1.rglob("*") if p.is_file()}
    files2 = {str(p.relative_to(d2)) for p in d2.rglob("*") if p.is_file()}
    if files1 != files2:
        return False, sorted(files1 ^ files2)
    for rel in sorted(files1):
        if not filecmp.cmp(d1 / rel, d2 / rel, shallow=False):
            return False, [rel]
    return True, []


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_demo(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"samples": {}, "byte_identical": {}, "error_codes": {},
                    "incremental": {}}
    samples = all_samples()

    # 1) 四类样本，每份生成两次，逐字节比对
    for name, spec_dict in samples.items():
        run_a = root / name / "run_a"
        run_b = root / name / "run_b"
        _remove_tree(run_a)
        _remove_tree(run_b)
        manifest_a = generate_from_dict(spec_dict, run_a)
        generate_from_dict(spec_dict, run_b)
        same, diff = _bytewise_equal(run_a, run_b)
        ledger = json.loads((run_a / "ledger.json").read_text(encoding="utf-8"))
        boundary = json.loads((run_a / "boundary_report.json").read_text(
            encoding="utf-8"))
        report["samples"][name] = {
            "spec_hash": manifest_a["spec_hash"],
            "grid_hash": manifest_a["grid_hash"],
            "ledger_hash": manifest_a["ledger_hash"],
            "ledger_summary": ledger["summary"],
            "boundary_items": len(boundary["boundary_items"]),
            "out_dir": str(run_a.relative_to(root)),
        }
        report["byte_identical"][name] = {"identical": same, "diff_files": diff}

        # 独立复算
        recompute(run_a)

    # boundary 样本在严格模式下必须抛 E_BOUNDARY_AMBIGUOUS（工件已写出）
    strict_dir = root / "boundary" / "strict"
    _remove_tree(strict_dir)
    try:
        generate_from_dict(boundary_spec(), strict_dir, strict_boundary=True)
        raise AssertionError("boundary 样本应触发 E_BOUNDARY_AMBIGUOUS")
    except LevelDiagError as exc:
        report["error_codes"][exc.code] = exc.message

    # 2) 业务错误码演示
    for code, msg in _demo_error_codes(root).items():
        report["error_codes"].setdefault(code, msg)

    # 3) 增量重生成：只改密度上限阈值，闭包应为 ledger 三件套
    inc_dir = root / "incremental" / "work"
    _remove_tree(inc_dir)
    base = normal_spec()
    generate_from_dict(base, inc_dir)
    changed = mutate(base, "resources.density_cap", 0.12)
    inc_report = incremental_regenerate(base, changed, inc_dir)
    # 增量后复算仍须通过
    recompute(inc_dir)
    report["incremental"]["threshold_change"] = inc_report

    # 量化步长变更：闭包应包含 mapping/grid/ledger
    inc_dir2 = root / "incremental" / "step_work"
    _remove_tree(inc_dir2)
    generate_from_dict(base, inc_dir2)
    changed_step = mutate(base, "quantization.density_step", 0.25)
    step_report = incremental_regenerate(base, changed_step, inc_dir2)
    recompute(inc_dir2)
    report["incremental"]["step_change"] = step_report

    (root / "demo_report.json").write_bytes(
        canonical_dumps(report) + b"\n")
    return report


def _demo_error_codes(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}

    # E_SPEC_INVALID：缺失字段 + 权重超容忍 + 点重合 + 区域越界
    bad = normal_spec()
    del bad["resources"]["density_cap"]
    try:
        spec_from_dict(bad)
    except LevelDiagError as exc:
        found[exc.code] = f"{exc.message} | 字段路径={exc.details.get('field_path')}"

    bad = mutate(normal_spec(), "terrain.weights.grass", 0.6 + 5e-9)
    try:
        spec_from_dict(bad)
    except LevelDiagError as exc:
        found.setdefault(exc.code, exc.message)

    bad = mutate(normal_spec(), "points.target", [1, 1])
    try:
        spec_from_dict(bad)
    except LevelDiagError as exc:
        found.setdefault(exc.code, exc.message)

    bad = mutate(normal_spec(), "regions[1].cells",
                 [[x, y] for x in range(5, 11) for y in range(0, 8)])
    try:
        spec_from_dict(bad)
    except LevelDiagError as exc:
        found.setdefault(exc.code, exc.message)

    # E_QUANT_BOUNDS：非法步长（构造 >1 的量化结果）
    bad = mutate(normal_spec(), "quantization.elevation_step", 0.6)
    # 0.6 仍在域内，改用直接触发越界的规格：步长 1.5
    bad = mutate(normal_spec(), "quantization.elevation_step", 1.5)
    try:
        spec = spec_from_dict(bad)
        run_pipeline(spec)
    except LevelDiagError as exc:
        found[exc.code] = exc.message

    # E_LEDGER_STALE：篡改 ledger 后复算
    stale_dir = root / "error_demo" / "stale"
    _remove_tree(stale_dir)
    generate_from_dict(normal_spec(), stale_dir)
    ledger_path = stale_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["entries"][0]["status"] = "FAIL"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    try:
        recompute(stale_dir)
    except LevelDiagError as exc:
        found[exc.code] = exc.message

    # E_LEDGER_MISMATCH：保持文件哈希自洽，但写入错误结论
    mismatch_dir = root / "error_demo" / "mismatch"
    _remove_tree(mismatch_dir)
    manifest = generate_from_dict(normal_spec(), mismatch_dir)
    ledger_path = mismatch_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    target = next(e for e in ledger["entries"]
                  if e["kind"] == "resource_density_cap")
    target["status"] = "FAIL"  # 与复算口径冲突
    from .util import content_hash
    ledger["ledger_hash"] = content_hash({
        "spec_hash": ledger["spec_hash"],
        "continuous_hash": ledger["continuous_hash"],
        "mapping_hash": ledger["mapping_hash"],
        "grid_hash": ledger["grid_hash"],
        "entries": ledger["entries"]})
    ledger_path.write_bytes(canonical_dumps(ledger) + b"\n")
    manifest_path = mismatch_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ledger_hash"] = ledger["ledger_hash"]
    manifest["files"]["ledger.json"] = file_hash(ledger_path)
    manifest_path.write_bytes(canonical_dumps(manifest) + b"\n")
    try:
        recompute(mismatch_dir)
    except LevelDiagError as exc:
        found[exc.code] = exc.message

    # E_CLOSURE_MISMATCH：注入一次声明/实际不一致的调用
    from . import incremental as inc_mod
    original = inc_mod.declared_closure
    inc_mod.declared_closure = lambda paths: {
        "scope": "partial", "files": ["ledger.json"],
        "ledger_entries": ["*"]}
    closure_dir = root / "error_demo" / "closure"
    _remove_tree(closure_dir)
    generate_from_dict(normal_spec(), closure_dir)
    try:
        incremental_regenerate(normal_spec(),
                               mutate(normal_spec(),
                                      "resources.density_cap", 0.12),
                               closure_dir)
    except LevelDiagError as exc:
        found[exc.code] = exc.message
    finally:
        inc_mod.declared_closure = original

    # E_BOUNDARY_AMBIGUOUS 由 run_demo 主流程的 strict 模式演示
    expected = {ErrorCode.E_SPEC_INVALID, ErrorCode.E_QUANT_BOUNDS,
                ErrorCode.E_LEDGER_STALE, ErrorCode.E_LEDGER_MISMATCH,
                ErrorCode.E_CLOSURE_MISMATCH}
    missing = expected - set(found)
    if missing:
        raise AssertionError(f"演示未覆盖错误码: {missing}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leveldiag",
        description="连续域约束求解与数值误差传播诊断实验环境")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo", help="生成四类样本并执行全部验证")
    p_demo.add_argument("--out", default="demo_out", help="演示输出目录")
    p_gen = sub.add_parser("generate", help="从 YAML 规格生成工件")
    p_gen.add_argument("spec")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--strict-boundary", action="store_true")
    p_rec = sub.add_parser("recompute", help="独立复算工件目录")
    p_rec.add_argument("out_dir")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "demo":
            report = run_demo(Path(args.out))
            for name, info in report["samples"].items():
                print(f"[sample] {name}: grid={info['grid_hash'][:12]} "
                      f"ledger={info['ledger_hash'][:12]} "
                      f"summary={info['ledger_summary']}")
            for name, info in report["byte_identical"].items():
                print(f"[determinism] {name}: identical={info['identical']}")
            for code, msg in sorted(report["error_codes"].items()):
                print(f"[error-code] {code}: {msg}")
            print("[incremental] threshold_change:",
                  report["incremental"]["threshold_change"]["rewritten_files"])
            print("[incremental] step_change:",
                  report["incremental"]["step_change"]["rewritten_files"])
            print(f"[demo] 完整报告: {Path(args.out) / 'demo_report.json'}")
            return 0
        if args.cmd == "generate":
            text = Path(args.spec).read_text(encoding="utf-8")
            spec = parse_spec(text, args.spec)
            result = run_pipeline(spec)
            write_artifacts(spec, result, args.out, source_yaml=text)
            if args.strict_boundary and result["ledger"]["summary"]["uncertain"]:
                raise LevelDiagError(
                    ErrorCode.E_BOUNDARY_AMBIGUOUS,
                    "存在不确定判定",
                    {"ids": result["ledger"]["summary"]["uncertain_ids"]})
            print(f"[generate] 工件已写入 {args.out}")
            return 0
        if args.cmd == "recompute":
            result = recompute(args.out_dir)
            print(f"[recompute] 复算 {result['checked']} 条判定，"
                  f"不一致 {result['mismatch_count']} 条，"
                  f"汇总 {result['recomputed_summary']}")
            return 0
    except LevelDiagError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2,
                         sort_keys=True), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
