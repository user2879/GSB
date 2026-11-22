from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .codec_v1 import parse_v1
from .codec_v2 import encode_v2, parse_v2
from .config import load_config
from .convert import downgrade_v2, restore_v1_with_sidecar, v1_to_v2_regions
from .diagnostics import cross_version_diagnostics, validate_injection_cases
from .errors import WorkbenchError
from .image_registry import load_images
from .manifest import load_edit_manifest
from .reporting import build_diagnostics, write_report
from .runner import ScenarioRunner, load_commands, load_resume_state
from .samples import generate_sample_workspace
from .util import atomic_write_bytes, read_json


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _safe_clear(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise WorkbenchError("E_CONFIG_INVALID", f"refusing to clear path outside workspace: {path}", stage="cli")
    shutil.rmtree(resolved_path, ignore_errors=True)


def _root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_images(sample_dir: Path, config: Any):
    image_dir = sample_dir / "images" if (sample_dir / "images").is_dir() else sample_dir
    return load_images(image_dir, expected_size=config.image_size)


def _manifest_keys(sample_dir: Path, images: Any) -> tuple[list[Any], set[str]]:
    manifest = load_edit_manifest(sample_dir / "edits.json", images=images)
    if manifest.failures:
        raise WorkbenchError(
            "E_STRUCTURE_INVALID",
            f"edit manifest has {len(manifest.failures)} failures",
            stage="manifest",
        )
    return manifest.regions, {region.region_key for region in manifest.regions}


def command_generate_samples(args: argparse.Namespace) -> int:
    result = generate_sample_workspace(Path(args.root))
    _json_print(result)
    return 0


def command_validate_injections(args: argparse.Namespace) -> int:
    root = Path(args.root)
    sample_dir = _root_path(root, args.sample_dir) if args.sample_dir else root / "samples"
    config = load_config(_root_path(root, args.config))
    images = _resolve_images(sample_dir, config)
    _, keys = _manifest_keys(sample_dir, images)
    results = validate_injection_cases(root, images=images, manifest_keys=keys)
    _json_print({"cases": results, "all_hit": all(item["hit"] for item in results)})
    return 0 if all(item["hit"] for item in results) else 1


def command_convert(args: argparse.Namespace) -> int:
    root = Path(args.root)
    sample_dir = _root_path(root, args.sample_dir) if args.sample_dir else root / "samples"
    config = (
        load_config(_root_path(root, args.config)) if args.config else load_config(sample_dir / "config.json")
    )
    images = _resolve_images(sample_dir, config)
    _, keys = _manifest_keys(sample_dir, images)
    source = _root_path(root, args.input)
    source_text = source.read_text(encoding="utf-8")
    output = _root_path(root, args.output)
    sidecar_out = (
        _root_path(root, args.sidecar_out)
        if args.sidecar_out
        else output.with_suffix(output.suffix + ".sidecar.json")
    )

    if args.from_version == 1 and args.to_version == 2:
        parsed = parse_v1(source_text, images=images, manifest_keys=keys, source=str(source))
        if parsed.failures:
            raise WorkbenchError("E_STRUCTURE_INVALID", f"v1 parse produced {len(parsed.failures)} failures", stage="convert")
        if args.sidecar_in:
            restored = restore_v1_with_sidecar(parsed.regions, read_json(_root_path(root, args.sidecar_in)))
            upgraded = restored.regions
            summary = {"sidecar_hits": restored.sidecar_hits, "sidecar_misses": restored.sidecar_misses}
        else:
            upgraded = v1_to_v2_regions(parsed.regions)
            summary = {"sidecar_hits": 0, "sidecar_misses": 0}
        text = encode_v2(upgraded)
        atomic_write_bytes(output, text.encode("utf-8"))
    elif args.from_version == 2 and args.to_version == 1:
        parsed = parse_v2(source_text, images=images, manifest_keys=keys, source=str(source))
        if parsed.failures:
            raise WorkbenchError("E_STRUCTURE_INVALID", f"v2 parse produced {len(parsed.failures)} failures", stage="convert")
        converted = downgrade_v2(parsed.regions)
        atomic_write_bytes(output, converted.text.encode("utf-8"))
        atomic_write_bytes(sidecar_out, (converted.sidecar_json or "").encode("utf-8"))
        summary = {"sidecar_hash": converted.sidecar_hash, "sidecar_path": str(sidecar_out)}
    else:
        raise WorkbenchError("E_STRUCTURE_INVALID", "conversion must be 1->2 or 2->1", stage="convert")
    _json_print({"input": str(source), "output": str(output), **summary})
    return 0


def command_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    sample_dir = _root_path(root, args.sample_dir) if args.sample_dir else root / "samples"
    state_dir = _root_path(root, args.state_dir) if args.state_dir else root / "state" / "incremental"
    output_dir = _root_path(root, args.output_dir) if args.output_dir else root / "outputs" / "incremental"
    report_dir = _root_path(root, args.report_dir) if args.report_dir else root / "report"
    config = load_config(_root_path(root, args.config))
    if args.reset_state and args.resume:
        raise WorkbenchError("E_CONFIG_INVALID", "--reset-state and --resume are mutually exclusive", stage="cli")
    if args.reset_state:
        _safe_clear(root, state_dir)
        _safe_clear(root, output_dir)
    initial_state = load_resume_state(state_dir) if args.resume else None
    if initial_state is not None:
        config = initial_state.config
    commands_path = _root_path(root, args.commands) if args.commands else None
    commands = load_commands(commands_path) if commands_path else [{"id": "CMD-RUN", "op": "noop"}]
    images = _resolve_images(sample_dir, config)
    scenario = ScenarioRunner(
        sample_dir=sample_dir,
        state_dir=state_dir,
        output_dir=output_dir,
        reference_root=state_dir / "reference",
    ).run(
        commands=commands,
        config=config,
        exchange_version=args.exchange_version,
        drain_after=not args.batch,
        verify_each_command=not args.no_verify_each_command,
        keep_reference=args.keep_reference,
        initial_state=initial_state,
    )
    cross_work = state_dir / "cross_version_diagnostics"
    cross = cross_version_diagnostics(
        regions=list(scenario.final_state.regions),
        images=images,
        config=scenario.final_state.config,
        work_root=cross_work,
    )
    injections = validate_injection_cases(
        root,
        images=images,
        manifest_keys={region.region_key for region in scenario.final_state.regions},
    )
    diagnostics = build_diagnostics(
        scenario=scenario,
        cross_version=cross,
        injections=injections,
        config=scenario.final_state.config,
        images=images,
        sample_dir=sample_dir,
        state_dir=state_dir,
        output_dir=output_dir,
        report_dir=report_dir,
    )
    report_paths = write_report(
        report_dir=report_dir,
        diagnostics=diagnostics,
        scenario=scenario,
        cross_version=cross,
        images=images,
        sample_dir=sample_dir,
        output_dir=output_dir,
    )
    summary = {
        "ok": diagnostics["equivalence"]["all_commands_semantic_equal"],
        "commands": len(scenario.commands),
        "final_content_fingerprint": scenario.final_result.content_fingerprint,
        "final_pending_queue": scenario.final_result.pending_queue,
        "reused_regions": scenario.final_result.reused_regions,
        "recomputed_regions": scenario.final_result.recomputed_regions,
        "max_active_tasks": scenario.final_result.max_active_tasks,
        "overall_max_active_tasks": diagnostics["run_summary"]["max_active_tasks"],
        "worker_limit": scenario.final_state.config.thread_pool_workers,
        "equivalence": diagnostics["equivalence"]["all_commands_semantic_equal"],
        "injection_hits": diagnostics["injection_hits"],
        "boundary_samples": len(diagnostics["boundary_samples"]),
        "business_codes": diagnostics["business_codes"],
        "report": report_paths,
        "environment": diagnostics["environment"],
    }
    _json_print(summary)
    all_injections_hit = all(item["hit"] for item in injections)
    return 0 if summary["equivalence"] and all_injections_hit else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-edit-workbench",
        description="Deterministic local image-edit exchange and incremental diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="generate deterministic PNG and exchange samples")
    generate.add_argument("--root", default=".")
    generate.set_defaults(func=command_generate_samples)

    validate = subparsers.add_parser("validate-injections", help="run independent malformed exchange injection cases")
    validate.add_argument("--root", default=".")
    validate.add_argument("--sample-dir")
    validate.add_argument("--config", default="config/default.json")
    validate.set_defaults(func=command_validate_injections)

    convert = subparsers.add_parser("convert", help="convert v1<->v2 with sidecar degradation metadata")
    convert.add_argument("--root", default=".")
    convert.add_argument("--sample-dir")
    convert.add_argument("--config")
    convert.add_argument("--from-version", type=int, choices=(1, 2), required=True)
    convert.add_argument("--to-version", type=int, choices=(1, 2), required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--sidecar-in")
    convert.add_argument("--sidecar-out")
    convert.set_defaults(func=command_convert)

    run = subparsers.add_parser("run", help="execute a command sequence and write diagnostics/report")
    run.add_argument("--root", default=".")
    run.add_argument("--sample-dir")
    run.add_argument("--config", default="config/default.json")
    run.add_argument("--commands")
    run.add_argument("--exchange-version", type=int, choices=(1, 2), default=2)
    run.add_argument("--state-dir")
    run.add_argument("--output-dir")
    run.add_argument("--report-dir")
    run.add_argument("--reset-state", action="store_true")
    run.add_argument("--no-verify-each-command", action="store_true")
    run.add_argument("--batch", action="store_true", help="process only one budget batch per command")
    run.add_argument("--keep-reference", action="store_true")
    run.add_argument("--resume", action="store_true", help="resume saved scenario state and persisted pending queue")
    run.set_defaults(func=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "stage": exc.stage, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "code": "E_CLI", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
