"""Command-line interface for validation, rendering, conversion and demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .demo import lease_recovery_demo, run_demo
from .errors import BusinessError
from .formats import load_sidecar, parse_v1, parse_v2, v1_to_v2, v2_to_v1
from .renderer import Renderer
from .samples import create_sample_workspace
from .util import atomic_write_text, canonical_json
from .workspace import Workspace, load_commands


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _default_paths(config_path: str | Path) -> tuple[Path, Path, Path]:
    base = Path(config_path).resolve().parent
    return base / ".mdet_cache", base / "ledger" / "render.jsonl", base / "out"


def _validate(args: argparse.Namespace) -> int:
    workspace = Workspace.load(args.config)
    command_count = 0
    if args.commands:
        commands = load_commands(args.commands)
        for command in commands:
            workspace = workspace.apply_command(command)
            command_count += 1
    _print({
        "ok": True,
        "config": str(Path(args.config).resolve()),
        "resolution": [workspace.config.width, workspace.config.height],
        "pixel_mode": workspace.config.pixel_mode,
        "condition_dim": workspace.config.condition_dim,
        "vocabulary_dim": workspace.vocabulary.dimension,
        "vocabulary_sha256": workspace.vocabulary.digest,
        "motion_table_sha256": workspace.motion.digest,
        "motion_blocks": [workspace.motion.block_rows, workspace.motion.block_cols],
        "frames": {
            str(index): {
                "prompt": item.prompt.as_dict(),
                "image": None if item.image_path is None else str(item.image_path),
            }
            for index, item in sorted(workspace.frames.items())
        },
        "fixed_point": workspace.config.fixed.as_dict(),
        "quantization": workspace.config.fixed.as_dict(),
        "commands_validated": command_count,
    })
    return 0


def _render(args: argparse.Namespace) -> int:
    default_cache, default_ledger, default_out = _default_paths(args.config)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache
    ledger_path = Path(args.ledger) if args.ledger else default_ledger
    output_dir = Path(args.out) if args.out else default_out
    workspace = Workspace.load(args.config)
    renderer = Renderer(workspace, cache_dir=cache_dir, ledger_path=ledger_path)
    commands = load_commands(args.commands) if args.commands else []
    edits: list[dict[str, Any]] = []
    try:
        if args.replay_edits and commands:
            result = renderer.run(output_dir=output_dir, max_batches=args.max_batches)
            edits.append({"edit": "initial", "result": result.as_dict()})
            for index, command in enumerate(commands, start=1):
                workspace = workspace.apply_command(command)
                renderer.workspace = workspace
                result = renderer.run(output_dir=output_dir, max_batches=args.max_batches)
                edits.append({"edit": index, "command": command, "result": result.as_dict()})
                if not result.complete:
                    break
        else:
            for command in commands:
                workspace = workspace.apply_command(command)
            renderer.workspace = workspace
            result = renderer.run(output_dir=output_dir, max_batches=args.max_batches)
    finally:
        renderer.close()
    payload = result.as_dict()
    payload["command_count"] = len(commands)
    if edits:
        payload["edits"] = edits
    _print(payload)
    if not result.complete:
        return 3
    return 0


def _init_samples(args: argparse.Namespace) -> int:
    manifest = create_sample_workspace(args.root)
    _print({"ok": True, **manifest})
    return 0


def _demo(args: argparse.Namespace) -> int:
    report = run_demo(args.root, regenerate=not args.no_regenerate)
    _print({
        "ok": report["ok"],
        "demo_report": str(Path(report["demo_root"]) / "demo_report.json"),
        "comparison": report["comparison"],
        "lease_ok": report["lease_recovery"]["ok"],
        "final_incremental_ledger": report["final_incremental_ledger"],
        "cache_keys_match": report["incremental_cache_keys_sha256"] == report["full_cache_keys_sha256"],
        "scheduler_batches": len(report["final_incremental"]["scheduler_batches"]),
        "max_observed_workers": report["max_observed_workers_overall"],
        "frames": {
            frame: {
                "pixel_digest": value["pixel_digest"],
                "png_digest": value["png_digest"],
                "blocks": len(value["blocks"]),
            }
            for frame, value in report["final_incremental"]["frames"].items()
        },
    })
    return 0 if report["ok"] else 4


def _lease_demo(args: argparse.Namespace) -> int:
    params = None
    if args.params:
        raw = json.loads(Path(args.params).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise BusinessError("E_LEASE_CONFIG", "lease params must be an object")
        params = raw
    report = lease_recovery_demo(params)
    _print(report)
    return 0 if report["ok"] else 4


def _v2_to_v1(args: argparse.Namespace) -> int:
    source = Path(args.input)
    raw = json.loads(source.read_text(encoding="utf-8"))
    v1_text, sidecar = v2_to_v1(raw)
    output = Path(args.output)
    sidecar_path = Path(args.sidecar)
    atomic_write_text(output, v1_text)
    atomic_write_text(sidecar_path, json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _print({"ok": True, "v1": str(output), "sidecar": str(sidecar_path),
            "records": len(parse_v1(v1_text)), "v1_sha256": sidecar["v1_document_sha256"]})
    return 0


def _v1_to_v2(args: argparse.Namespace) -> int:
    source = Path(args.input)
    v1_text = source.read_text(encoding="utf-8")
    sidecar = load_sidecar(args.sidecar) if args.sidecar else None
    document = v1_to_v2(v1_text, sidecar)
    parse_v2(document)
    output = Path(args.output)
    atomic_write_text(output, json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _print({"ok": True, "v2": str(output), "records": len(document["records"]),
            "restored_from_sidecar": bool(sidecar)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdet-render",
                                     description="Deterministic fixed-point multimodal renderer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-samples", help="generate a deterministic sample workspace")
    init.add_argument("--root", default="samples")
    init.set_defaults(handler=_init_samples)

    validate = subparsers.add_parser("validate", help="validate config, assets and optional commands")
    validate.add_argument("--config", required=True)
    validate.add_argument("--commands")
    validate.set_defaults(handler=_validate)

    render = subparsers.add_parser("render", help="render or incrementally replay edits")
    render.add_argument("--config", required=True)
    render.add_argument("--commands")
    render.add_argument("--cache-dir")
    render.add_argument("--ledger")
    render.add_argument("--out")
    render.add_argument("--replay-edits", action="store_true")
    render.add_argument("--max-batches", type=int)
    render.set_defaults(handler=_render)

    demo = subparsers.add_parser("demo", help="run full incremental/full-equivalence/lease demo")
    demo.add_argument("--root", default="samples")
    demo.add_argument("--no-regenerate", action="store_true")
    demo.set_defaults(handler=_demo)

    lease = subparsers.add_parser("lease-demo", help="exercise lease generation/expiry/reclamation")
    lease.add_argument("--params")
    lease.set_defaults(handler=_lease_demo)

    to_v1 = subparsers.add_parser("v2-to-v1", help="convert v2 JSON to v1 fixed-width lines + sidecar")
    to_v1.add_argument("--input", required=True)
    to_v1.add_argument("--output", required=True)
    to_v1.add_argument("--sidecar", required=True)
    to_v1.set_defaults(handler=_v2_to_v1)

    to_v2 = subparsers.add_parser("v1-to-v2", help="convert v1 fixed-width lines to v2 JSON")
    to_v2.add_argument("--input", required=True)
    to_v2.add_argument("--output", required=True)
    to_v2.add_argument("--sidecar")
    to_v2.set_defaults(handler=_v1_to_v2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BusinessError as exc:
        _print(exc.as_dict())
        return exc.exit_code
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _print({"ok": False, "code": "E_INTERNAL", "message": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
