"""Command-line interface for validation, rendering, edits, and recovery."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, b64_decode, load_json
from .config import load_render_config
from .errors import (
    E_ENUM_MISSING,
    E_SNAPSHOT_INCOMPLETE,
    BusinessError,
    fail,
)
from .exchange import (
    ExchangeDocument,
    parse_v1,
    validate_v2_document,
    v1_to_v2,
    v2_to_v1,
)
from .renderer import Renderer
from .samplegen import generate_samples
from .snapshot import SnapshotStore
from .workspace import Workspace, load_workspace


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _profile_bits(config: Any) -> dict[str, int]:
    return {name: profile.fraction_bits for name, profile in config.quant_profiles.items()}


def _write_png_outputs(renderer: Renderer, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for frame_key, output in sorted(renderer.outputs.items(), key=lambda item: int(item[0])):
        frame_number = int(frame_key)
        frame_dir = root / f"frame_{frame_number:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "frame.png"
        atomic_write_bytes(frame_path, b64_decode(output["png_b64"]))
        written.append(str(frame_path))
        for chunk in output.get("chunks", []):
            chunk_path = frame_dir / f"chunk_{int(chunk['index']):02d}.png"
            atomic_write_bytes(chunk_path, b64_decode(chunk["png_b64"]))
            written.append(str(chunk_path))
    atomic_write_json(
        root / "output_manifest.json",
        {
            "frames": [
                {
                    "frame": int(frame_key),
                    "png_sha256": output["png_sha256"],
                    "pixel_digest": output["pixel_digest"],
                    "chunks": [
                        {
                            "index": chunk["index"],
                            "row_range": chunk["row_range"],
                            "png_sha256": chunk["png_sha256"],
                        }
                        for chunk in output.get("chunks", [])
                    ],
                }
                for frame_key, output in sorted(renderer.outputs.items(), key=lambda item: int(item[0]))
            ],
            "files": written,
        },
    )


def _write_exchange_bundle(renderer: Renderer, directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records = renderer.export_stage_records("denoise")
    if not records:
        return {"skipped": "no completed denoise records"}
    document = ExchangeDocument(stage="denoise", records=records, provenance={"source": "renderer"}).v2_document()
    v2_path = root / "denoise_v2.json"
    atomic_write_json(v2_path, document)
    profiles = _profile_bits(renderer.config)
    text, sidecar = v2_to_v1(
        document,
        noise_schemes=renderer.config.noise_schemes,
        profiles=profiles,
        allow_missing=False,
    )
    v1_path = root / "denoise_v1.txt"
    sidecar_path = root / "denoise_v1.sidecar.json"
    from .canonical import atomic_write_text

    atomic_write_text(v1_path, text)
    atomic_write_json(sidecar_path, sidecar)
    restored = v1_to_v2(
        text,
        sidecar=sidecar,
        noise_code_map=renderer.config.noise_schemes,
        profile_bits=profiles,
    )
    original_records = document["records"]
    restored_records = restored["records"]
    exact = len(original_records) == len(restored_records) and all(
        left["guidance"] == right["guidance"]
        and left["refs"] == right["refs"]
        and left["noise_scheme"] == right["noise_scheme"]
        and left["quant_profile"] == right["quant_profile"]
        for left, right in zip(original_records, restored_records)
    )
    return {
        "v2": str(v2_path),
        "v1": str(v1_path),
        "sidecar": str(sidecar_path),
        "records": len(records),
        "round_trip_exact": exact,
        "overridden_reference_count": sidecar.get("overridden_reference_count"),
    }


def _render_report(renderer: Renderer, result: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    report = {
        **extra,
        "render": result,
        "output_manifest": {
            frame_key: {
                "png_sha256": output["png_sha256"],
                "pixel_digest": output["pixel_digest"],
                "chunks": [
                    {"index": chunk["index"], "row_range": chunk["row_range"], "png_sha256": chunk["png_sha256"]}
                    for chunk in output.get("chunks", [])
                ],
            }
            for frame_key, output in sorted(renderer.outputs.items(), key=lambda item: int(item[0]))
        },
    }
    return report


def cmd_generate_samples(args: argparse.Namespace) -> int:
    result = generate_samples(args.root)
    _print({"ok": True, "generated": result})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.config, args.conditions)
    applied = []
    if args.edits:
        workspace, applied = workspace.apply_commands(args.edits)
    result = {
        "ok": True,
        "config_hash": workspace.config_hash,
        "input_hash": workspace.input_hash,
        "config_version": workspace.config.exchange_version,
        "quant_profile": workspace.config.quant_profile.name,
        "frames": len(workspace.conditions.frames),
        "condition_dim": workspace.config.condition_dim,
        "latent_dim": workspace.config.latent_dim,
        "denoise_steps": workspace.config.denoise_steps,
        "chunk_rows": workspace.config.chunk_rows,
        "vocabulary_terms": len(workspace.conditions.vocab.terms),
        "applied_commands": applied,
    }
    _print(result)
    return 0


def cmd_validate_exchange(args: argparse.Namespace) -> int:
    config = load_render_config(args.config)
    if args.version == "v2":
        document = load_json(args.input)
        parsed = validate_v2_document(
            document,
            allow_missing=args.allow_missing,
            noise_schemes=config.noise_schemes,
            quant_profiles=config.quant_profiles,
        )
        records = parsed.records
        if not args.allow_missing:
            for index, record in enumerate(records):
                if record.missing_fields:
                    fail(E_ENUM_MISSING, "v2 record has missing fields", index=index, fields=record.missing_fields)
        result = {
            "ok": True,
            "version": "v2",
            "stage": parsed.stage,
            "records": len(records),
            "missing_fields": [record.missing_fields for record in records if record.missing_fields],
        }
    else:
        text = Path(args.input).read_text(encoding="utf-8")
        sidecar = load_json(args.sidecar) if args.sidecar else None
        parsed = parse_v1(
            text,
            sidecar=sidecar,
            noise_code_map=config.noise_schemes,
            profile_bits=_profile_bits(config),
        )
        if not args.allow_missing:
            for index, record in enumerate(parsed.records):
                if record.missing_fields:
                    fail(E_ENUM_MISSING, "v1 record has missing fields", index=index, fields=record.missing_fields)
        result = {
            "ok": True,
            "version": "v1",
            "stage": parsed.stage,
            "records": len(parsed.records),
            "missing_fields": [record.missing_fields for record in parsed.records if record.missing_fields],
            "sidecar_restored": parsed.provenance.get("sidecar_restored", False),
        }
    _print(result)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    config = load_render_config(args.config)
    profiles = _profile_bits(config)
    output_path = Path(args.output)
    if args.source_version == "v2":
        document = load_json(args.input)
        text, sidecar = v2_to_v1(
            document,
            noise_schemes=config.noise_schemes,
            profiles=profiles,
            allow_missing=args.allow_missing,
        )
        sidecar_path = Path(args.sidecar_output or (str(output_path) + ".sidecar.json"))
        from .canonical import atomic_write_text

        atomic_write_text(output_path, text)
        atomic_write_json(sidecar_path, sidecar)
        _print(
            {
                "ok": True,
                "direction": "v2->v1",
                "output": str(output_path),
                "sidecar": str(sidecar_path),
                "residual_count": len(sidecar.get("downgrade_residuals", [])),
                "overridden_reference_count": sidecar.get("overridden_reference_count", 0),
            }
        )
    else:
        text = Path(args.input).read_text(encoding="utf-8")
        sidecar = load_json(args.sidecar) if args.sidecar else None
        document = v1_to_v2(
            text,
            sidecar=sidecar,
            noise_code_map=config.noise_schemes,
            profile_bits=profiles,
        )
        atomic_write_json(output_path, document)
        _print(
            {
                "ok": True,
                "direction": "v1->v2",
                "output": str(output_path),
                "sidecar_restored": document.get("provenance", {}).get("sidecar_restored", False),
                "records": len(document.get("records", [])),
            }
        )
    return 0


def _write_structured(path: Path, value: Any) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            fail("E_CONFIG_INVALID", "writing YAML requires PyYAML", path=str(path))
        from .canonical import atomic_write_text

        atomic_write_text(path, yaml.safe_dump(value, sort_keys=True, allow_unicode=True))
    else:
        atomic_write_json(path, value)


def cmd_edit(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.config, args.conditions)
    edited, applied = workspace.apply_commands(args.commands)
    config_out = Path(args.out_config or args.config)
    conditions_out = Path(args.out_conditions or args.conditions)
    _write_structured(config_out, edited.config_raw)
    _write_structured(conditions_out, edited.manifest_raw)
    _print(
        {
            "ok": True,
            "applied_commands": applied,
            "config": str(config_out),
            "conditions": str(conditions_out),
            "config_hash": edited.config_hash,
            "input_hash": edited.input_hash,
        }
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.config, args.conditions)
    applied: list[dict[str, Any]] = []
    if args.edits:
        workspace, applied = workspace.apply_commands(args.edits)
    renderer = Renderer(workspace, args.snapshot_dir, budget_override=args.budget)
    result = renderer.render()
    report = _render_report(
        renderer,
        result,
        {
            "command": "render",
            "config": str(args.config),
            "conditions": str(args.conditions),
            "applied_commands": applied,
        },
    )
    if args.output_dir:
        _write_png_outputs(renderer, args.output_dir)
    if args.exchange_dir:
        report["exchange"] = _write_exchange_bundle(renderer, args.exchange_dir)
    if args.report:
        atomic_write_json(args.report, report)
    _print({"ok": result["ok"], "code": result["code"], **{key: result[key] for key in (
        "generation", "global_version", "config_hash", "input_hash", "max_workers",
        "max_observed_concurrency", "executed_task_count", "ledger_summary", "budget"
    )}, "pending_count": len(result["pending"]), "frames": result["frames"], "applied_commands": applied})
    return 0 if result["ok"] else 3


def cmd_resume(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshot_dir)
    committed = store.load_committed()
    if committed is None:
        fail(E_SNAPSHOT_INCOMPLETE, "no committed snapshot exists to resume", snapshot_dir=args.snapshot_dir)
    workspace_raw = committed.state.get("workspace")
    if not isinstance(workspace_raw, dict):
        fail(E_SNAPSHOT_INCOMPLETE, "committed snapshot has no workspace input", generation=committed.generation)
    workspace = Workspace.from_serialised(workspace_raw)
    renderer = Renderer(workspace, args.snapshot_dir, budget_override=args.budget)
    result = renderer.render()
    report = _render_report(
        renderer,
        result,
        {"command": "resume", "resumed_from": committed.generation},
    )
    if args.output_dir:
        _write_png_outputs(renderer, args.output_dir)
    if args.exchange_dir:
        report["exchange"] = _write_exchange_bundle(renderer, args.exchange_dir)
    if args.report:
        atomic_write_json(args.report, report)
    _print(
        {
            "ok": result["ok"],
            "code": result["code"],
            "resumed_from": committed.generation,
            "new_generation": result["generation"],
            "budget": result["budget"],
            "pending": result["pending"],
            "frames": result["frames"],
            "ledger_summary": result["ledger_summary"],
            "uncommitted": result["uncommitted"],
        }
    )
    return 0 if result["ok"] else 3


def cmd_status(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshot_dir)
    committed = store.load_committed()
    uncommitted = store.uncommitted_report()
    if committed is None:
        _print({"ok": False, "code": E_SNAPSHOT_INCOMPLETE, "uncommitted": uncommitted})
        return 4
    pending = committed.state.get("pending", [])
    pending_codes = {str(item.get("code")) for item in pending if isinstance(item, dict)}
    if pending and "E_BUDGET_PENDING" in pending_codes:
        status_code = "E_BUDGET_PENDING"
        exit_code = 3
    elif pending:
        status_code = E_SNAPSHOT_INCOMPLETE
        exit_code = 4
    else:
        status_code = "OK"
        exit_code = 0
    result = {
        "ok": not pending,
        "code": status_code,
        "generation": committed.generation,
        "global_version": committed.manifest.get("global_version"),
        "config_hash": committed.manifest.get("config_hash"),
        "input_hash": committed.manifest.get("input_hash"),
        "cursor": committed.state.get("cursor", {}),
        "pending": pending,
        "budget": committed.state.get("budget", {}),
        "max_observed_concurrency": committed.state.get("max_observed_concurrency"),
        "uncommitted": uncommitted,
    }
    _print(result)
    return exit_code


def _find_frame_artifact(state: dict[str, Any], frame: int) -> dict[str, Any] | None:
    logical_id = f"frame_decode:{frame}"
    descriptor = state.get("descriptors", {}).get(logical_id, {})
    cache_key = descriptor.get("cache_key") if isinstance(descriptor, dict) else None
    if cache_key and cache_key in state.get("artifacts", {}):
        artifact = state["artifacts"][cache_key]
        if isinstance(artifact, dict) and "step_refs" in artifact:
            return artifact
    found: list[dict[str, Any]] = []
    for artifact in state.get("artifacts", {}).values():
        if (
            isinstance(artifact, dict)
            and artifact.get("logical_id") == logical_id
            and "step_refs" in artifact
        ):
            found.append(artifact)
    if not found:
        return None
    return sorted(found, key=lambda item: str(item.get("cache_key")))[-1]


def cmd_trace(args: argparse.Namespace) -> int:
    left = SnapshotStore(args.snapshot_a).load_committed()
    right = SnapshotStore(args.snapshot_b).load_committed()
    if left is None or right is None:
        fail(E_SNAPSHOT_INCOMPLETE, "both snapshots must have a committed generation")
    frame_a = _find_frame_artifact(left.state, args.frame)
    frame_b = _find_frame_artifact(right.state, args.frame)
    if frame_a is None or frame_b is None:
        fail(E_SNAPSHOT_INCOMPLETE, "frame artifact is missing from one snapshot", frame=args.frame)
    steps_a = {item["step"]: item for item in frame_a.get("step_refs", [])}
    steps_b = {item["step"]: item for item in frame_b.get("step_refs", [])}
    first_step = None
    for step in sorted(set(steps_a) | set(steps_b)):
        if steps_a.get(step, {}).get("artifact_digest") != steps_b.get(step, {}).get("artifact_digest"):
            first_step = step
            break
    result = {
        "ok": True,
        "frame": args.frame,
        "pixels_equal": frame_a.get("pixel_digest") == frame_b.get("pixel_digest"),
        "left_pixel_digest": frame_a.get("pixel_digest"),
        "right_pixel_digest": frame_b.get("pixel_digest"),
        "first_diverging_step": first_step,
        "left_steps": steps_a,
        "right_steps": steps_b,
    }
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmfm", description="Deterministic fixed-point multimodal renderer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="write deterministic local samples")
    generate.add_argument("--root", default=".")
    generate.set_defaults(func=cmd_generate_samples)

    validate = subparsers.add_parser("validate", help="validate config, vocabulary, conditions, and edits")
    validate.add_argument("--config", required=True)
    validate.add_argument("--conditions", required=True)
    validate.add_argument("--edits")
    validate.set_defaults(func=cmd_validate)

    validate_exchange = subparsers.add_parser("validate-exchange", help="validate a v1 or v2 intermediate document")
    validate_exchange.add_argument("--config", required=True)
    validate_exchange.add_argument("--version", choices=["v1", "v2"], required=True)
    validate_exchange.add_argument("--input", required=True)
    validate_exchange.add_argument("--sidecar")
    validate_exchange.add_argument("--allow-missing", action="store_true", help="allow records with missing enum fields")
    validate_exchange.set_defaults(func=cmd_validate_exchange)

    convert = subparsers.add_parser("convert", help="convert between v1 fixed lines and v2 JSON")
    convert.add_argument("--config", required=True)
    convert.add_argument("--source-version", choices=["v1", "v2"], required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--sidecar")
    convert.add_argument("--sidecar-output")
    convert.add_argument("--allow-missing", action="store_true")
    convert.set_defaults(func=cmd_convert)

    edit = subparsers.add_parser("edit", help="apply a validated edit command stream")
    edit.add_argument("--config", required=True)
    edit.add_argument("--conditions", required=True)
    edit.add_argument("--commands", required=True)
    edit.add_argument("--out-config")
    edit.add_argument("--out-conditions")
    edit.set_defaults(func=cmd_edit)

    render = subparsers.add_parser("render", help="incrementally render into a snapshot store")
    render.add_argument("--config", required=True)
    render.add_argument("--conditions", required=True)
    render.add_argument("--snapshot-dir", default="snapshots")
    render.add_argument("--edits")
    render.add_argument("--budget", type=int)
    render.add_argument("--output-dir")
    render.add_argument("--exchange-dir")
    render.add_argument("--report")
    render.set_defaults(func=cmd_render)

    resume = subparsers.add_parser("resume", help="resume from the last atomically committed generation")
    resume.add_argument("--snapshot-dir", default="snapshots")
    resume.add_argument("--budget", type=int)
    resume.add_argument("--output-dir")
    resume.add_argument("--exchange-dir")
    resume.add_argument("--report")
    resume.set_defaults(func=cmd_resume)

    status = subparsers.add_parser("status", help="inspect the committed snapshot and pending queue")
    status.add_argument("--snapshot-dir", default="snapshots")
    status.set_defaults(func=cmd_status)

    trace = subparsers.add_parser("trace-diff", help="trace a frame pixel difference to the first differing denoise step")
    trace.add_argument("--snapshot-a", required=True)
    trace.add_argument("--snapshot-b", required=True)
    trace.add_argument("--frame", type=int, required=True)
    trace.set_defaults(func=cmd_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return int(exc.exit_code or 2)
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"}, ensure_ascii=False))
        return 130
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "E_INTERNAL",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
