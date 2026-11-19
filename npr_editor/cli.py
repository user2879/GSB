"""Command-line application entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .acceptance import run_acceptance
from .binary_format import decode_binary, encode_binary
from .compare import ComparisonStore, VersionTrace
from .errors import NprError
from .geometry import GeometryEngine
from .migration import migrate_document
from .model import Project, dump_project
from .png import encode_png
from .render import RENDER_MODES, RenderConfig, RenderEngine
from .samples import write_samples
from .snapshots import SnapshotStore, atomic_write_bytes


def _load(path: str | Path) -> tuple[dict[str, Any], Any, Project]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise NprError("E_SCHEMA_VERSION", "project JSON cannot be read", path=str(path)) from exc
    migration = migrate_document(raw)
    return raw, migration, migration.project


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _max_vertex_error(project_a: Project, project_b: Project) -> float:
    engine_a = GeometryEngine(project_a)
    engine_b = GeometryEngine(project_b)
    maximum = 0.0
    for shape in project_a.ordered_shapes():
        a = engine_a.ensure(shape.id).vertices
        b = engine_b.ensure(shape.id).vertices
        if a.shape != b.shape:
            return float("inf")
        maximum = max(maximum, float(np.abs(a - b).max()))
    return maximum


def cmd_samples(args: argparse.Namespace) -> int:
    manifest = write_samples(args.out)
    _print(
        {
            "out": str(Path(args.out).resolve()),
            "project_hash": manifest["project_hash"],
            "files": len(manifest["files"]),
            "binary_variants": manifest["binary_variants"],
        }
    )
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    source_raw, _, project = _load(args.project)
    encoded = encode_binary(
        project,
        byte_order=args.endian,
        index_width=args.index_width,
        format_version=args.format_version,
        include_project_metadata=not args.no_metadata,
    )
    decoded = decode_binary(encoded.data)
    error = _max_vertex_error(project, decoded.project)
    tolerance = float(args.tolerance)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out / "skeleton.npbs", encoded.data)
    _write_json(out / "roundtrip_project.json", decoded.project.to_dict())
    report = {
        "source_schema_version": source_raw.get("schema_version"),
        "json_hash": f"json://sha256:{project.content_hash()}",
        "binary_hash": f"bin://sha256:{encoded.content_hash}",
        "roundtrip_json_hash": f"json://sha256:{decoded.project_hash}",
        "format_version": encoded.format_version,
        "byte_order": encoded.byte_order,
        "index_width": encoded.index_width,
        "width_encoding": encoded.width_encoding,
        "vertex_count": encoded.vertex_count,
        "index_count": encoded.index_count,
        "shape_count": encoded.shape_count,
        "max_vertex_error": error,
        "tolerance": tolerance,
        "project_metadata_preserved": decoded.parsed.metadata is not None,
        "passed": error <= tolerance
        and (
            decoded.parsed.metadata is None
            or decoded.project_hash == project.content_hash()
        ),
    }
    _write_json(out / "roundtrip_report.json", report)
    _print(report)
    return 0 if report["passed"] else 1


def cmd_migrate(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = migrate_document(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = dump_project(result.project, indent=2)
    atomic_write_bytes(out, text.encode("utf-8"))
    report = result.to_dict()
    report["output"] = str(out.resolve())
    report["project_id"] = result.project.project_id
    if args.report:
        _write_json(Path(args.report), report)
    _print(report)
    return 0


def _render_config(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(width=args.width, height=args.height, supersample=args.supersample)


def cmd_render(args: argparse.Namespace) -> int:
    _, _, project = _load(args.project)
    result = RenderEngine(project, _render_config(args)).render(args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = encode_png(result.image)
    atomic_write_bytes(out, data)
    _print({"out": str(out.resolve()), **result.report(), "mode": args.mode})
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    _, _, project = _load(args.project)
    config = _render_config(args)
    engine = RenderEngine(project, config)
    result = engine.render(args.mode)
    store = SnapshotStore(args.workspace)
    snapshot = store.commit(project, mode=args.mode, result=result, engine=engine)
    _print(
        {
            "version_stamp": snapshot.version_stamp,
            "snapshot_hash": snapshot.snapshot_hash,
            "project_hash": snapshot.project_hash,
            "mode": snapshot.mode,
            "path": str(snapshot.path),
            "render_pixel_sha256": result.pixel_sha256,
        }
    )
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.workspace)
    recovery = store.recover()
    if recovery.latest is None:
        raise NprError("E_SNAPSHOT_STALE", "no committed snapshot is available")
    project = store.load_project(recovery.latest)
    engine = RenderEngine(project, _render_config(args))
    restored = (
        store.restore_render_cache(recovery.latest, engine, args.mode)
        if args.mode == recovery.latest.mode
        else []
    )
    result = engine.render(args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out, encode_png(result.image))
    _print(
        {
            "recovery": recovery.to_dict(),
            "restored_cache_shapes": sorted(restored),
            "render": result.report(),
            "out": str(out.resolve()),
        }
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left_raw, left_migration, left_project = _load(args.left)
    right_raw, right_migration, right_project = _load(args.right)
    left_trace = VersionTrace(
        version_stamp=left_project.version_stamp,
        source_schema_version=left_raw.get("schema_version"),
        migration_actions=[action.to_dict() for action in left_migration.actions],
    )
    right_trace = VersionTrace(
        version_stamp=right_project.version_stamp,
        source_schema_version=right_raw.get("schema_version"),
        migration_actions=[action.to_dict() for action in right_migration.actions],
    )
    store = ComparisonStore(args.workspace)
    result, path = store.save(
        left_project,
        right_project,
        left_trace,
        right_trace,
        config=_render_config(args),
    )
    _print({"path": str(path), "result": result})
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    report = run_acceptance(args.workspace, out_dir=args.out)
    _print(
        {
            "passed": report["summary"]["passed"],
            "check_count": report["summary"]["check_count"],
            "report_hash": report["report_hash"],
            "out": str(Path(args.out or Path(args.workspace) / "artifacts" / "acceptance").resolve()),
        }
    )
    return 0 if report["summary"]["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npr-editor",
        description="Deterministic NPR project interchange, rendering and recovery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    samples = sub.add_parser("samples", help="generate deterministic local samples")
    samples.add_argument("--out", default="samples")
    samples.set_defaults(func=cmd_samples)

    roundtrip = sub.add_parser("roundtrip", help="convert project JSON to skeleton and back")
    roundtrip.add_argument("--project", required=True)
    roundtrip.add_argument("--out", required=True)
    roundtrip.add_argument("--endian", choices=("little", "big"), default="little")
    roundtrip.add_argument("--index-width", type=int, choices=(2, 4), default=4)
    roundtrip.add_argument("--format-version", type=int, choices=(1, 2), default=2)
    roundtrip.add_argument("--no-metadata", action="store_true")
    roundtrip.add_argument("--tolerance", type=float, default=1e-6)
    roundtrip.set_defaults(func=cmd_roundtrip)

    migrate = sub.add_parser("migrate", help="migrate a project JSON document to schema v3")
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--out", required=True)
    migrate.add_argument("--report")
    migrate.set_defaults(func=cmd_migrate)

    render = sub.add_parser("render", help="render a project to deterministic PNG")
    render.add_argument("--project", required=True)
    render.add_argument("--mode", choices=RENDER_MODES, default="lineart")
    render.add_argument("--out", required=True)
    render.add_argument("--width", type=int, default=240)
    render.add_argument("--height", type=int, default=160)
    render.add_argument("--supersample", type=int, choices=(1, 2, 4), default=1)
    render.set_defaults(func=cmd_render)

    snapshot = sub.add_parser("snapshot", help="atomically commit a rendered snapshot")
    snapshot.add_argument("--project", required=True)
    snapshot.add_argument("--workspace", required=True)
    snapshot.add_argument("--mode", choices=RENDER_MODES, default="lineart")
    snapshot.add_argument("--width", type=int, default=240)
    snapshot.add_argument("--height", type=int, default=160)
    snapshot.add_argument("--supersample", type=int, choices=(1, 2, 4), default=1)
    snapshot.set_defaults(func=cmd_snapshot)

    restore = sub.add_parser("restore", help="recover the latest committed snapshot")
    restore.add_argument("--workspace", required=True)
    restore.add_argument("--out", required=True)
    restore.add_argument("--mode", choices=RENDER_MODES, default="lineart")
    restore.add_argument("--width", type=int, default=240)
    restore.add_argument("--height", type=int, default=160)
    restore.add_argument("--supersample", type=int, choices=(1, 2, 4), default=1)
    restore.set_defaults(func=cmd_restore)

    compare = sub.add_parser("compare", help="compare two project versions and persist evidence")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--workspace", required=True)
    compare.add_argument("--width", type=int, default=240)
    compare.add_argument("--height", type=int, default=160)
    compare.add_argument("--supersample", type=int, choices=(1, 2, 4), default=1)
    compare.set_defaults(func=cmd_compare)

    acceptance = sub.add_parser("acceptance", help="run the full local acceptance suite")
    acceptance.add_argument("--workspace", default=".")
    acceptance.add_argument("--out")
    acceptance.set_defaults(func=cmd_acceptance)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NprError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {"code": "E_FORMAT_DATA", "message": str(exc), "details": {}},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
