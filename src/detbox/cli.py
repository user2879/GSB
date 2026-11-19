"""Command line entry points for the candidate exchange verifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DetectionParams, MatchConfig, load_injections
from .errors import DetboxError
from .fingerprint import global_config_hash
from .models import ClassTable, FORMAT_V1, FORMAT_V2
from .pipeline import Pipeline
from .report import build_report, write_report
from .samplegen import generate_dataset
from .store import SNAPSHOT_VERSION
from .util import atomic_write_json, read_json


def _load_class_table(dataset_root: Path) -> ClassTable:
    gt_path = dataset_root / "ground_truth.json"
    if gt_path.exists():
        gt = read_json(gt_path)
        return ClassTable.from_dict(gt["class_table"])
    from .config import DEFAULT_CLASS_TABLE

    return DEFAULT_CLASS_TABLE


def _dimensions(dataset_root: Path) -> tuple[int, int]:
    gt = read_json(dataset_root / "ground_truth.json")
    return int(gt["width"]), int(gt["height"])


def cmd_generate(args) -> int:
    root = generate_dataset(args.dataset)
    print(f"generated dataset at {root}")
    return 0


def _build_pipeline(args, fmt: str | None = None, strict: bool = False) -> Pipeline:
    dataset_root = Path(args.dataset)
    workdir = Path(args.workdir)
    params = DetectionParams.load(args.detection_params)
    match_config = MatchConfig.load(args.match_config)
    classes = _load_class_table(dataset_root)
    width, height = _dimensions(dataset_root)
    injections = load_injections(args.injections)
    overrides = read_json(args.overrides) if args.overrides else {}
    return Pipeline(
        dataset_root=dataset_root,
        workdir=workdir,
        fmt=fmt or args.format,
        detection_params=params,
        match_config=match_config,
        classes=classes,
        width=width,
        height=height,
        injections=injections,
        overrides=overrides,
        strict_resume=strict,
    )


def cmd_run(args) -> int:
    pipeline = _build_pipeline(args, strict=args.strict_resume)
    result = pipeline.run()
    report = build_report(result, pipeline.classes, pipeline.width, pipeline.height)
    report_path = Path(args.report)
    write_report(report_path, report)
    print(json.dumps({
        "exit_code": result["exit_code"],
        "report": str(report_path),
        "failures": len(result["failures"]),
        "hits": len(result["injection_hits"]),
        "reused": len(result["reused"]),
        "recomputed": len(result["recomputed"]),
    }, indent=2))
    return result["exit_code"]


def cmd_rebuild(args) -> int:
    workdir = Path(args.workdir)
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir)
    args.strict_resume = False
    return cmd_run(args)


def cmd_check(args) -> int:
    dataset_root = Path(args.dataset)
    width, height = _dimensions(dataset_root)
    params = DetectionParams.load(args.detection_params)
    match_config = MatchConfig.load(args.match_config)
    classes = _load_class_table(dataset_root)
    global_hash = global_config_hash(
        str(dataset_root.resolve()),
        params.to_dict(),
        match_config.to_dict(),
        classes.to_dict(),
        width,
        height,
    )
    from .store import SnapshotStore

    store = SnapshotStore(args.workdir)
    try:
        store.verify_resume(SNAPSHOT_VERSION, global_hash, strict=True)
    except DetboxError as exc:
        print(json.dumps({"resume": False, "code": exc.code, "message": str(exc)}, indent=2))
        return 2
    print(json.dumps({"resume": True, "config_hash": global_hash}, indent=2))
    return 0


def cmd_edit(args) -> int:
    edit = read_json(args.edit_file)
    journal_path = Path(args.workdir) / "edit_journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(journal_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(edit, sort_keys=True) + "\n")
    apply_edit(Path(args.dataset), edit)
    print(json.dumps({"applied": edit.get("kind"), "journal": str(journal_path)}, indent=2))
    return 0


def apply_edit(dataset_root: Path, edit: dict) -> None:
    kind = edit["kind"]
    if kind == "switch_format":
        return
    if kind == "replace_frame_candidates":
        overrides_path = dataset_root / "frame_overrides.json"
        data = read_json(overrides_path) if overrides_path.exists() else {"frames": {}}
        data["frames"][f"{edit['sequence']}:{edit['frame']}"] = edit["payload"]
        atomic_write_json(overrides_path, data)
    elif kind == "replace_image":
        target = dataset_root / "images" / edit["sequence"] / f"frame_{edit['frame']:04d}.png"
        Path(target).write_bytes(Path(edit["source_png"]).read_bytes())
    elif kind == "update_iou":
        cfg_path = dataset_root / "match_config.json"
        cfg = read_json(cfg_path)
        cfg["iou_threshold"] = float(edit["iou_threshold"])
        atomic_write_json(cfg_path, cfg)
    elif kind == "update_pool_capacity":
        cfg_path = dataset_root / "detection_params.json"
        cfg = read_json(cfg_path)
        cfg["pool_capacity"] = int(edit["pool_capacity"])
        atomic_write_json(cfg_path, cfg)
    elif kind == "update_weights":
        cfg_path = dataset_root / "detection_params.json"
        cfg = read_json(cfg_path)
        cfg["score_weights"] = dict(edit["score_weights"])
        atomic_write_json(cfg_path, cfg)
    else:
        raise DetboxError(f"unknown edit kind: {kind}")


def cmd_convert(args) -> int:
    from .exchange import load_exchange_file, write_exchange_file

    classes = _load_class_table(Path(args.dataset))
    width, height = _dimensions(Path(args.dataset))
    frames, failures = load_exchange_file(
        args.input, args.input_format, classes, width, height
    )
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        return 1
    write_exchange_file(args.output, args.output_format, frames, classes)
    print(json.dumps({"converted": args.output, "frames": len(frames)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detbox")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_generate)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", required=True)
        sp.add_argument("--workdir", required=True)
        sp.add_argument("--detection-params", required=True)
        sp.add_argument("--match-config", required=True)
        sp.add_argument("--format", choices=[FORMAT_V1, FORMAT_V2], default=FORMAT_V1)
        sp.add_argument("--injections")
        sp.add_argument("--overrides")
        sp.add_argument("--report", default="report.json")
        sp.add_argument("--strict-resume", action="store_true")

    p = sub.add_parser("run")
    add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("rebuild")
    add_common(p)
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("check")
    p.add_argument("--dataset", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--detection-params", required=True)
    p.add_argument("--match-config", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("edit")
    p.add_argument("--dataset", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--edit-file", required=True)
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("convert")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--input-format", choices=[FORMAT_V1, FORMAT_V2], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-format", choices=[FORMAT_V1, FORMAT_V2], required=True)
    p.set_defaults(func=cmd_convert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DetboxError as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}, indent=2), file=sys.stderr)
        return 2
