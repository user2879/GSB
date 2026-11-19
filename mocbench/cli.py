import argparse
import json
import sys

from .canonical import version_digest
from .engine import EXIT_SIGINT, InterruptController, SolveEngine
from .errors import BenchError, exit_code_for
from .parser import load_bundle
from .persistence import verify_state
from .report import build_poses, build_report, export_atomic
from .samplegen import write_sample

try:
    from .app import run_window, STATE_PLACEHOLDER, STATE_SOLVED
except ImportError:
    run_window = None
from .solver import solve_frame


def _write_error(exc):
    sys.stdout.write(json.dumps(exc.to_json(), ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


def _serialize(frames):
    return frames


def cmd_generate(args):
    path = write_sample(args.input, damping=args.damping)
    print(json.dumps({"ok": True, "sample": path}, ensure_ascii=False))
    return 0


def _run_solve(input_path, state_dir, output_dir, attach_signals):
    bundle = load_bundle(input_path)
    controller = InterruptController()
    if attach_signals:
        controller.attach(state_dir)
    engine = SolveEngine(bundle, state_dir, controller=controller, progress=_print_progress)
    try:
        engine.run()
    finally:
        controller.detach()
    frames = engine.load_all_frames()
    report = build_report(frames, engine.fill_result, bundle, engine.version)
    poses = build_poses(frames, bundle, engine.version)
    export_info = export_atomic(output_dir, report, poses)
    print(
        json.dumps(
            {
                "ok": True,
                "version_digest": engine.version,
                "frame_count": engine.frame_count,
                "chunks": len(engine.chunk_plans),
                "passed_frames": report["passed_frame_count"],
                "unconverged_frames": report["unconverged_frame_count"],
                "export": export_info,
                "report": f"{output_dir}/report.json",
                "poses": f"{output_dir}/poses.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _print_progress(event):
    sys.stderr.write(json.dumps({"progress": event}, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def cmd_solve(args):
    return _run_solve(args.input, args.state_dir, args.output, attach_signals=True)


def cmd_resume(args):
    return _run_solve(args.input, args.state_dir, args.output, attach_signals=True)


def cmd_verify(args):
    bundle = load_bundle(args.input)
    digest = version_digest(bundle.raw)
    verify_state(args.state_dir, digest, bundle.curves.frame_count, bundle.params.chunk_size)
    print(
        json.dumps(
            {
                "ok": True,
                "version_digest": digest,
                "state_dir": args.state_dir,
                "frame_count": bundle.curves.frame_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_serial_check(args):
    bundle = load_bundle(args.input)
    engine = SolveEngine(bundle, args.state_dir)
    engine.prepare_state()
    serial_frames = []
    for index, start, ids in engine.chunk_plans:
        chunk = []
        for frame_index in ids:
            frame_local = {
                name: engine.fill_result.rotations[name][frame_index]
                for name in bundle.skeleton.order
            }
            chunk.append(
                solve_frame(
                    frame_index,
                    frame_local,
                    bundle.curves.root_positions[frame_index],
                    bundle,
                )
            )
        serial_frames.extend(chunk)
    engine.run()
    parallel_frames = engine.load_all_frames()
    max_delta = _max_pose_delta(serial_frames, parallel_frames)
    print(
        json.dumps(
            {
                "ok": True,
                "serial_vs_parallel_max_abs_delta": max_delta,
                "tolerance": bundle.params.tolerance,
                "within_tolerance": max_delta <= bundle.params.tolerance,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if max_delta <= bundle.params.tolerance else 1


def _max_pose_delta(serial_frames, stored_frames):
    import numpy as np

    by_index = {frame["frame"]: frame for frame in serial_frames}
    worst = 0.0
    for stored in stored_frames:
        direct = by_index[stored["frame"]]
        for name, rotation in direct["local_rotations"].items():
            worst = max(worst, float(np.max(np.abs(np.asarray(rotation) - np.asarray(stored["local_rotations"][name])))))
        worst = max(
            worst,
            float(np.max(np.abs(direct["root_position"] - np.asarray(stored["root_position"])))),
        )
        for name, position in direct["world_positions"].items():
            worst = max(
                worst,
                float(np.max(np.abs(position - np.asarray(stored["world_positions"][name])))),
            )
    return worst


def cmd_gui(args):
    if run_window is None:
        raise BenchError("E_INPUT", "PySide6 is not available")
    app, window = run_window(args.input, args.state_dir)
    return app.exec()


def cmd_ui_selftest(args):
    if run_window is None:
        raise BenchError("E_INPUT", "PySide6 is not available")
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app, window = run_window(args.input, args.state_dir)
    observed = {}
    placeholder_states = {}
    for probe in (25, 35, 7):
        window.slider.setValue(probe)
        app.processEvents()
        placeholder_states[probe] = window.canvas.state
        observed[probe] = None
    deadline = time.time() + 60
    snapshots = {}
    while time.time() < deadline:
        app.processEvents()
        for probe in observed:
            if observed[probe] is None and probe in window.solved_frames:
                window.slider.setValue(probe)
                app.processEvents()
                observed[probe] = json.dumps(window.solved_frames[probe], sort_keys=True)
                snapshots[probe] = observed[probe]
        if not window.worker.isRunning():
            break
        time.sleep(0.01)
    app.processEvents()
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)
    for probe, snapshot in list(observed.items()):
        window.slider.setValue(probe)
        app.processEvents()
        assert snapshot is not None, f"frame {probe} never became available"
        later = json.dumps(window.solved_frames[probe], sort_keys=True)
        assert later == snapshot, f"frame {probe} changed after background completion"
    window.slider.setValue(25)
    app.processEvents()
    solved_state = window.canvas.state
    immutable = window._immutable_check()
    all_cached = sorted(window.solved_frames) == list(range(window.frame_count))
    placeholder_ok = all(value == STATE_PLACEHOLDER for value in placeholder_states.values())
    payload = window.export_ui_state(args.state_export)
    if args.screenshot:
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(window.size())
        window.render(pixmap)
        pixmap.save(args.screenshot)
    window.close()
    app.processEvents()
    result = {
        "ok": bool(immutable and all_cached and solved_state == STATE_SOLVED and placeholder_ok),
        "placeholder_before_solve": placeholder_states,
        "state_after_solve": solved_state,
        "all_frames_cached": all_cached,
        "all_frames_immutable": immutable,
        "watched_frames_stable": list(observed),
        "cached_frame_count": len(payload["cached_frames"]),
        "state_export": args.state_export,
        "screenshot": args.screenshot,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def build_parser():
    parser = argparse.ArgumentParser(prog="mocbench", description="Motion capture constraint workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic sample")
    gen.add_argument("--input", default="samples/sample.json")
    gen.add_argument("--damping", type=float, default=0.9)
    gen.set_defaults(func=cmd_generate)

    solve = sub.add_parser("solve", help="solve all chunks and export report/poses")
    solve.add_argument("--input", required=True)
    solve.add_argument("--state-dir", required=True)
    solve.add_argument("--output", required=True)
    solve.set_defaults(func=cmd_solve)

    resume = sub.add_parser("resume", help="resume from persisted chunks")
    resume.add_argument("--input", required=True)
    resume.add_argument("--state-dir", required=True)
    resume.add_argument("--output", required=True)
    resume.set_defaults(func=cmd_resume)

    verify = sub.add_parser("verify", help="verify state digest and chunk integrity")
    verify.add_argument("--input", required=True)
    verify.add_argument("--state-dir", required=True)
    verify.set_defaults(func=cmd_verify)

    serial = sub.add_parser("serial-check", help="compare serial vs parallel solve")
    serial.add_argument("--input", required=True)
    serial.add_argument("--state-dir", required=True)
    serial.set_defaults(func=cmd_serial_check)

    gui = sub.add_parser("gui", help="launch interactive workbench window")
    gui.add_argument("--input", required=True)
    gui.add_argument("--state-dir", required=True)
    gui.set_defaults(func=cmd_gui)

    selftest = sub.add_parser("ui-selftest", help="offscreen UI consistency self-test")
    selftest.add_argument("--input", required=True)
    selftest.add_argument("--state-dir", required=True)
    selftest.add_argument("--state-export", required=True)
    selftest.add_argument("--screenshot", default=None)
    selftest.set_defaults(func=cmd_ui_selftest)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchError as exc:
        _write_error(exc)
        if exc.code == "E_INTERRUPTED":
            return exc.details.get("exit_code", EXIT_SIGINT)
        return exit_code_for(exc.code)


if __name__ == "__main__":
    sys.exit(main())
