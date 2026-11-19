"""Command-line interface for the offline audio fingerprint workbench."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from .audio import find_ffmpeg
from .config import DEFAULT_CONFIG, LEGACY_CONFIG, FingerprintConfig
from .demo import run_demo
from .errors import AudioWorkbenchError
from .fixtures import generate_fixtures
from .library import Library


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _config_for_version(version: str) -> FingerprintConfig:
    if version == "fp-v1":
        return LEGACY_CONFIG
    if version == "fp-v2":
        return DEFAULT_CONFIG
    raise ValueError(f"unknown fingerprint version: {version}")


def _discover_audio(paths: list[str]) -> list[Path]:
    extensions = {".wav", ".ogg", ".flac", ".mp3", ".opus", ".oga"}
    discovered: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            discovered.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in extensions
            )
        elif path.is_file() and path.suffix.lower() in extensions:
            discovered.append(path)
    return sorted({item.resolve() for item in discovered}, key=lambda item: item.as_posix())


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fingerprint-version",
        choices=("fp-v1", "fp-v2"),
        default="fp-v2",
        help="fingerprint encoding version (fp-v1 is the migratable legacy format)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiofp",
        description="Deterministic offline audio fingerprint and similar-segment retrieval workbench",
    )
    parser.add_argument("--version", action="version", version="audiofp 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect local dependencies and GUI readiness")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    generate = subparsers.add_parser("generate", help="generate the deterministic synthetic fixture set")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261119)

    build = subparsers.add_parser("build", help="incrementally import local audio files into an index")
    build.add_argument("--index", required=True)
    build.add_argument("--audio", nargs="+", required=True, help="files or directories")
    build.add_argument("--root", default=None, help="root used for relative source paths")
    _add_config_option(build)

    query = subparsers.add_parser("query", help="query an index with a local audio fragment")
    query.add_argument("--index", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--no-variants", action="store_true")
    query.add_argument("--no-auto-migrate", action="store_true")
    query.add_argument("--json", action="store_true")
    _add_config_option(query)

    inventory = subparsers.add_parser("list", help="list logical tracks and format sources")
    inventory.add_argument("--index", required=True)
    inventory.add_argument("--json", action="store_true")
    _add_config_option(inventory)

    delete = subparsers.add_parser("delete", help="delete one logical group or one format source")
    delete.add_argument("--index", required=True)
    selection = delete.add_mutually_exclusive_group(required=True)
    selection.add_argument("--group")
    selection.add_argument("--source")
    _add_config_option(delete)

    migrate = subparsers.add_parser("migrate", help="migrate a compatible index encoding version")
    migrate.add_argument("--index", required=True)
    migrate.add_argument("--to", choices=("fp-v2",), default="fp-v2")

    demo = subparsers.add_parser("demo", help="run the complete deterministic acceptance demo")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument(
        "--no-repeat",
        action="store_true",
        help="skip the second byte-for-byte reproducibility run (faster)",
    )

    ui = subparsers.add_parser("ui", help="start the PySide6 workbench or the portable terminal UI")
    ui.add_argument("--index", default="artifacts/demo/index/current.db")
    ui.add_argument("--portable", action="store_true", help="force the terminal portable UI")
    ui.add_argument("--smoke", action="store_true", help="construct the GUI offscreen and exit")
    _add_config_option(ui)
    return parser


def _doctor_payload() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for module in ("numpy", "scipy", "soundfile", "PySide6"):
        try:
            imported = __import__(module)
            modules[module] = {
                "available": True,
                "version": str(getattr(imported, "__version__", "unknown")),
            }
        except Exception as exc:
            modules[module] = {"available": False, "error": str(exc)}
    ffmpeg = find_ffmpeg()
    ffmpeg_version = None
    if ffmpeg:
        process = subprocess.run(
            [ffmpeg, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        ffmpeg_version = (process.stdout or process.stderr).splitlines()[0]
    has_display = bool(os.environ.get("DISPLAY")) or sys.platform.startswith("win") or sys.platform == "darwin"
    gui_status = {
        "pyside6_available": modules["PySide6"]["available"],
        "display_environment": os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"),
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM"),
        "display_available": has_display,
        "portable_fallback_required": not (modules["PySide6"]["available"] and has_display),
    }
    return {
        "ok": all(item["available"] for item in modules.values()) and bool(ffmpeg),
        "offline_only": True,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "modules": modules,
        "ffmpeg": {"available": bool(ffmpeg), "path": ffmpeg, "version": ffmpeg_version},
        "gui": gui_status,
        "cache_used_for_demo": False,
    }


def _cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_for_version(args.fingerprint_version)
    files = _discover_audio(args.audio)
    if not files:
        raise AudioWorkbenchError("No supported local audio files were found", "E_NO_INPUT", {}, 2)
    imported = []
    with Library(args.index, config) as library:
        for path in files:
            item = library.import_file(path, root=args.root)
            imported.append(item.to_dict())
        stats = library.store.stats()
    return {"ok": True, "imported_count": len(imported), "imported": imported, "stats": stats}


def _cmd_query(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_for_version(args.fingerprint_version)
    with Library(args.index, config, auto_migrate=not args.no_auto_migrate) as library:
        result = library.query_file(
            args.query,
            top_k=args.top_k,
            search_variants=not args.no_variants,
        )
    return result


def _cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_for_version(args.fingerprint_version)
    with Library(args.index, config) as library:
        return {
            "ok": True,
            "stats": library.store.stats(),
            "groups": library.store.list_groups(),
        }


def _cmd_delete(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_for_version(args.fingerprint_version)
    with Library(args.index, config) as library:
        if args.group:
            result = library.store.delete_group(args.group)
        else:
            result = library.store.delete_source(args.source)
        result["stats"] = library.store.stats()
        return result


def _cmd_migrate(args: argparse.Namespace) -> dict[str, Any]:
    target = _config_for_version(args.to)
    before: dict[str, Any] = {}
    if Path(args.index).is_file():
        connection = sqlite3.connect(args.index)
        try:
            before = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT key, value FROM meta WHERE key IN "
                    "('fingerprint_version','config_digest','encoding_space','revision','content_hash')"
                )
            }
        finally:
            connection.close()
    with Library(args.index, target, auto_migrate=True) as library:
        after = library.store.stats()
    return {
        "ok": True,
        "before": before,
        "after": after,
        "target_version": target.version,
        "idempotent_rule": "reopening with the same configuration does not change revision or content hash",
    }


def _cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = generate_fixtures(args.out, seed=args.seed, clean=True)
    return {
        "ok": True,
        "out": args.out,
        "seed": args.seed,
        "tracks": len(manifest["tracks"]),
        "formats": sum(len(track["formats"]) for track in manifest["tracks"]),
        "broken_samples": len(manifest["broken_samples"]),
        "variants": len(manifest["variants"]),
        "manifest_sha256": __import__("hashlib").sha256(
            (Path(args.out) / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def _cmd_demo(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = run_demo(Path(args.out), seed=args.seed, verify_repeat=not args.no_repeat)
    summary = {
        "ok": bool(report["ok"])
        and (report.get("repeatability", {}).get("ok") is not False),
        "out": args.out,
        "seed": args.seed,
        "steps": {name: bool(step.get("ok")) for name, step in report["steps"].items()},
        "step_hashes": {
            name: step.get("content_hash") for name, step in report["steps"].items()
        },
        "report": str(Path(args.out) / "reports" / "demo_report.json"),
        "markdown_report": str(Path(args.out) / "reports" / "demo_report.md"),
        "repeatability": report.get("repeatability"),
    }
    return summary, 0 if summary["ok"] else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    from .ui import main as ui_main

    return ui_main(
        index_path=args.index,
        config=_config_for_version(args.fingerprint_version),
        portable=args.portable,
        smoke=args.smoke,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            payload = _doctor_payload()
            print(_json(payload) if args.json else _doctor_text(payload))
            return 0 if payload["ok"] else 1
        if args.command == "generate":
            print(_json(_cmd_generate(args)))
            return 0
        if args.command == "build":
            print(_json(_cmd_build(args)))
            return 0
        if args.command == "query":
            print(_json(_cmd_query(args)))
            return 0
        if args.command == "list":
            payload = _cmd_list(args)
            print(_json(payload) if args.json else _list_text(payload))
            return 0
        if args.command == "delete":
            print(_json(_cmd_delete(args)))
            return 0
        if args.command == "migrate":
            print(_json(_cmd_migrate(args)))
            return 0
        if args.command == "demo":
            payload, code = _cmd_demo(args)
            print(_json(payload))
            return code
        if args.command == "ui":
            return _cmd_ui(args)
        parser.error(f"unsupported command: {args.command}")
        return 2
    except AudioWorkbenchError as exc:
        print(_json(exc.to_dict()), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print(_json({"ok": False, "error": {"code": "E_INTERRUPTED", "message": "Interrupted"}}))
        return 130


def _doctor_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Python: {payload['python']['version']} ({payload['python']['implementation']})",
        "Offline-only demo: yes",
    ]
    for name, value in payload["modules"].items():
        lines.append(f"{name}: {'ok ' + value.get('version', '') if value['available'] else 'MISSING'}")
    lines.append(
        f"ffmpeg: {'ok ' + str(payload['ffmpeg']['path']) if payload['ffmpeg']['available'] else 'MISSING'}"
    )
    gui = payload["gui"]
    lines.append(
        "GUI: "
        + (
            "ready"
            if not gui["portable_fallback_required"]
            else "portable fallback required (no usable display/PySide6)"
        )
    )
    return "\n".join(lines)


def _list_text(payload: dict[str, Any]) -> str:
    lines = [
        f"groups={payload['stats']['groups']} sources={payload['stats']['sources']} "
        f"fingerprints={payload['stats']['fingerprints']} revision={payload['stats']['revision']}"
    ]
    for group in payload["groups"]:
        formats = ", ".join(source["format_signature"] for source in group["sources"])
        lines.append(f"{group['group_id']}  {group['display_label']}  [{formats}]")
    return "\n".join(lines)
