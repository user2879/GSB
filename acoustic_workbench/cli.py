"""Command-line entry points for the local acoustic diagnostics workbench."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .demo import run_demo
from .diagnostics import manifest_from_state
from .engine import IncrementalEngine
from .errors import (
    ERROR_CODES,
    E_DEPENDENCY_STALE,
    E_GRID_BOUNDS,
    E_PARAM_RANGE,
    E_SNAPSHOT_CONFLICT,
    E_SPH_ORDER,
    E_TILE_HASH,
    WorkbenchError,
)
from .gl_preview import check_offscreen_fbo
from .hashing import atomic_write_bytes, sha256_bytes
from .rir import check_wav_roundtrip
from .scene import parse_patch_value, scene_hash
from .store import WorkspaceStore
from .webapp import run_dashboard


def _workspace(value: str) -> Path:
    return Path(value).resolve()


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _open_engine(path: Path) -> IncrementalEngine:
    return IncrementalEngine.open(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustic-workbench", description="Incremental room-acoustic diagnostics workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create and build a workspace")
    init.add_argument("--workspace", default="artifacts/workbench")

    probe = sub.add_parser("probe-wav", help="write and read back a 48 kHz / 16-bit / 4-channel WAV probe")
    probe.add_argument("--workspace", default="artifacts/workbench")

    submit = sub.add_parser("submit", help="submit one parameter patch")
    submit.add_argument("assignments", nargs="+", metavar="KEY=VALUE")
    submit.add_argument("--workspace", default="artifacts/workbench")
    submit.add_argument("--expected-revision", type=int)

    show = sub.add_parser("show", help="show the latest graph state summary")
    show.add_argument("--workspace", default="artifacts/workbench")

    manifest = sub.add_parser("manifest", help="write the current per-node hash manifest")
    manifest.add_argument("--workspace", default="artifacts/workbench")
    manifest.add_argument("--output")

    rebuild = sub.add_parser("rebuild-full", help="force a full graph rebuild in place")
    rebuild.add_argument("--workspace", default="artifacts/workbench")

    verify = sub.add_parser("verify", help="reopen and verify workspace hashes/readiness")
    verify.add_argument("--workspace", default="artifacts/workbench")
    verify.add_argument("--full-compare", action="store_true")
    verify.add_argument("--gl-probe", action="store_true")

    demo = sub.add_parser("demo", help="run the fixed deterministic change sequence twice and compare with a full rebuild")
    demo.add_argument("--output-dir", default="artifacts/demo")

    diagnose = sub.add_parser("diagnose", help="serve the read-only diagnostic dashboard")
    diagnose.add_argument("--workspace", default="artifacts/demo/run_a")
    diagnose.add_argument("--host", default="127.0.0.1")
    diagnose.add_argument("--port", type=int, default=8765)
    diagnose.add_argument("--open-browser", action="store_true")

    gl_probe = sub.add_parser("gl-probe", help="test local PyOpenGL/GLFW offscreen FBO readback")
    gl_probe.add_argument("--workspace", default="artifacts/workbench")

    inject = sub.add_parser("inject-error", help="reproduce a business error code")
    inject.add_argument("code", choices=ERROR_CODES)
    inject.add_argument("--workspace", default="artifacts/workbench")
    return parser


def _parse_assignments(assignments: Sequence[str]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise WorkbenchError(E_PARAM_RANGE, "assignment must use KEY=VALUE", value=assignment)
        key, raw = assignment.split("=", 1)
        patch[key.strip()] = parse_patch_value(key.strip(), raw.strip())
    return patch


def _cmd_init(path: Path) -> None:
    engine = _open_engine(path)
    _print_json(
        {
            "workspace": str(path),
            "revision": engine.scene["revision"],
            "scene_hash": scene_hash(engine.scene),
            "nodes": len(engine.state["nodes"]),
            "edges": len(engine.state["edges"]),
            "snapshot": engine.last_snapshot_name,
            "snapshot_hash": engine.last_snapshot_hash,
            "build": engine.state["build"],
        }
    )


def _cmd_probe_wav(path: Path) -> None:
    store = WorkspaceStore(path)
    store.ensure()
    result = check_wav_roundtrip(store.output_dir / "wav_roundtrip_probe.wav")
    _print_json(result)


def _cmd_submit(path: Path, assignments: Sequence[str], expected_revision: int | None) -> None:
    patch = _parse_assignments(assignments)
    engine = _open_engine(path)
    report = engine.submit(patch, expected_revision=expected_revision)
    _print_json({"workspace": str(path), "patch": patch, "build": report, "scene": engine.scene})


def _cmd_show(path: Path) -> None:
    engine = _open_engine(path)
    _print_json(
        {
            "revision": engine.scene["revision"],
            "scene_hash": scene_hash(engine.scene),
            "versions": engine.versions,
            "build": engine.state["build"],
            "rir_bundle": (engine.store.output_dir / "rir_metadata.json").exists(),
            "snapshot": engine.last_snapshot_name,
        }
    )


def _cmd_manifest(path: Path, output: str | None) -> None:
    engine = _open_engine(path)
    data = engine.manifest_bytes()
    if output:
        target = Path(output).resolve()
        atomic_write_bytes(target, data)
        print(str(target))
    else:
        sys.stdout.buffer.write(data)


def _cmd_verify(path: Path, full_compare: bool, gl_probe: bool) -> None:
    engine = _open_engine(path)
    reopened = _open_engine(path)
    manifest_equal = engine.manifest_bytes() == reopened.manifest_bytes()
    wav_path = path / "outputs" / "room_rir_48k_16bit_4ch.wav"
    from .rir import read_pcm16_wav_bytes

    samples, sample_rate, channels = read_pcm16_wav_bytes(wav_path.read_bytes())
    result: dict[str, Any] = {
        "workspace": str(path),
        "revision": engine.scene["revision"],
        "scene_hash": scene_hash(engine.scene),
        "nodes": len(engine.state["nodes"]),
        "restart_manifest_identical": manifest_equal,
        "wav": {
            "sample_rate": sample_rate,
            "bit_depth": 16,
            "channels": channels,
            "frames": int(samples.shape[0]),
            "sha256": sha256_bytes(wav_path.read_bytes()),
        },
    }
    if full_compare:
        with tempfile.TemporaryDirectory(prefix="acoustic-full-compare-") as temp_dir:
            temp = Path(temp_dir)
            store = WorkspaceStore(temp)
            store.ensure()
            store.save_scene_document(engine.scene, engine.versions)
            full = _open_engine(temp)
            result["full_compare"] = {
                "manifest_equal": engine.manifest_bytes() == full.manifest_bytes(),
                "incremental_manifest_sha256": sha256_bytes(engine.manifest_bytes()),
                "full_manifest_sha256": sha256_bytes(full.manifest_bytes()),
                "full_build_closure_size": full.state["build"]["closure_size"],
            }
    if gl_probe:
        result["offscreen_gl"] = check_offscreen_fbo()
    engine.store.export_output("verification.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    _print_json(result)


def _cmd_demo(output_dir: str) -> None:
    run_demo(output_dir)


def _cmd_diagnose(path: Path, host: str, port: int, open_browser: bool) -> None:
    run_dashboard(path, host=host, port=port, open_browser=open_browser)


def _cmd_gl_probe(path: Path) -> None:
    result = check_offscreen_fbo()
    store = WorkspaceStore(path)
    store.ensure()
    store.export_output("opengl_probe.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _print_json(result)


def _cmd_inject_error(path: Path, code: str) -> None:
    engine = _open_engine(path)
    result: dict[str, Any] = {"requested_code": code, "workspace": str(path)}
    if code == E_SPH_ORDER:
        _expect_error(result, lambda: engine.submit({"reconstruction.order": 99}, expected_revision=engine.scene["revision"]))
    elif code == E_GRID_BOUNDS:
        _expect_error(result, lambda: engine.submit({"grid.width": 7}, expected_revision=engine.scene["revision"]))
    elif code == E_PARAM_RANGE:
        _expect_error(result, lambda: engine.submit({"wall.xmax.alpha": 1.5}, expected_revision=engine.scene["revision"]))
    elif code == E_SNAPSHOT_CONFLICT:
        stale = engine.scene["revision"] - 1
        _expect_error(result, lambda: engine.submit({"wall.xmax.alpha": 0.30}, expected_revision=stale))
        probe = engine.submit({"wall.xmax.alpha": 0.30}, expected_revision=engine.scene["revision"])
        result["workspace_usable_revision"] = probe["revision"]
    elif code in {E_TILE_HASH, E_DEPENDENCY_STALE}:
        copy_root = path.parent / f"injected_{code.lower()}"
        if copy_root.exists():
            shutil.rmtree(copy_root)
        shutil.copytree(path, copy_root)
        copy_engine = _open_engine(copy_root)
        node_type = "tile" if code == E_TILE_HASH else "sph"
        victim = next(node for node in copy_engine.state["nodes"].values() if node["type"] == node_type)
        blob_path = copy_engine.store.blob_path(victim["artifact_hash"])
        blob_path.write_bytes(b"corrupted-for-error-injection")
        _expect_error(result, lambda: _open_engine(copy_root))
        reopened = _open_engine(path)
        result["original_workspace_revision"] = reopened.scene["revision"]
        result["copy_workspace"] = str(copy_root)
    else:
        raise WorkbenchError(E_PARAM_RANGE, "unsupported error injection", code=code)
    reopened = _open_engine(path)
    result["workspace_usable"] = True
    result["workspace_revision"] = reopened.scene["revision"]
    _print_json(result)


def _expect_error(result: dict[str, Any], action) -> None:
    try:
        action()
    except WorkbenchError as exc:
        result["observed_code"] = exc.code
        result["message"] = exc.message
        result["details"] = exc.details
        if exc.code != result["requested_code"]:
            raise
        return
    raise AssertionError(f"expected {result['requested_code']} but no error was raised")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            _cmd_init(_workspace(args.workspace))
        elif args.command == "probe-wav":
            _cmd_probe_wav(_workspace(args.workspace))
        elif args.command == "submit":
            _cmd_submit(_workspace(args.workspace), args.assignments, args.expected_revision)
        elif args.command == "show":
            _cmd_show(_workspace(args.workspace))
        elif args.command == "manifest":
            _cmd_manifest(_workspace(args.workspace), args.output)
        elif args.command == "rebuild-full":
            engine = _open_engine(_workspace(args.workspace))
            _print_json(engine.rebuild_full())
        elif args.command == "verify":
            _cmd_verify(_workspace(args.workspace), args.full_compare, args.gl_probe)
        elif args.command == "demo":
            _cmd_demo(args.output_dir)
        elif args.command == "diagnose":
            _cmd_diagnose(_workspace(args.workspace), args.host, args.port, args.open_browser)
        elif args.command == "gl-probe":
            _cmd_gl_probe(_workspace(args.workspace))
        elif args.command == "inject-error":
            _cmd_inject_error(_workspace(args.workspace), args.code)
        else:
            parser.error("unknown command")
    except WorkbenchError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
