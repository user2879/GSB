"""Reproduce the local acceptance evidence written to artifacts/verification.json."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from h2burst.engine import Engine
from h2burst.errors import WorkbenchError
from h2burst.frames import decode_frame
from h2burst.model import MAX_WINDOW, SessionSpec, digest_value, load_session, parse_session
from h2burst.replay import replay_from_snapshot, replay_from_state
from h2burst.snapshot import SnapshotStore
from h2burst.viewmodel import WorkbenchViewModel


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ROOT / "samples" / "minimal.json",
    ROOT / "samples" / "burst_demo.json",
    ROOT / "samples" / "scale_1000.json",
]


def run_tests() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    output = process.stdout + process.stderr
    ran_match = re.search(r"Ran (\d+) tests?", output)
    failure_match = re.search(r"failures=(\d+)", output)
    error_match = re.search(r"errors=(\d+)", output)
    skipped_match = re.search(r"skipped=(\d+)", output)
    run = int(ran_match.group(1)) if ran_match else 0
    failures = int(failure_match.group(1)) if failure_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    return {
        "run": run,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "successful": process.returncode == 0,
    }


def tkinter_probe() -> dict[str, Any]:
    try:
        import tkinter

        try:
            root = tkinter.Tk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
            return {"available": True, "detail": f"Tk {tkinter.TkVersion} launched successfully"}
        except Exception as exc:
            return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}


def maybe_error(action) -> dict[str, Any]:
    try:
        action()
    except WorkbenchError as exc:
        return {
            "observed": True,
            "field": exc.field,
            "event_seq": exc.event_seq,
            "frame_seq": exc.frame_seq,
            "message": exc.message,
        }
    return {"observed": False, "field": None, "event_seq": None, "frame_seq": None}


def error_probes(raw_minimal: dict[str, Any], spec_minimal: SessionSpec, full_minimal: Engine) -> dict[str, Any]:
    invalid_spec = copy.deepcopy(raw_minimal)
    invalid_spec["streams"][0]["stream_id"] = 2

    def frame_probe() -> None:
        wire = b"\x00\x00\x0a" + bytes((0, 0)) + b"\x00\x00\x00\x01" + b"abc"
        decode_frame(wire, frame_seq=9)

    flow_raw = copy.deepcopy(raw_minimal)
    flow_raw["connection"]["initial_window"] = MAX_WINDOW
    flow_raw["connection"]["initial_stream_window"] = 65535
    flow_raw["events"] = [
        {"seq": 0, "arrival_ms": 0, "type": "window_update", "scope": "connection", "increment": 1}
    ]
    flow_spec = parse_session(flow_raw)

    state_raw = copy.deepcopy(raw_minimal)
    state_raw["events"] = [
        {"seq": 0, "arrival_ms": 0, "type": "rst_stream", "stream_id": 1, "error_code": 8},
        {"seq": 1, "arrival_ms": 1, "type": "rst_stream", "stream_id": 1, "error_code": 8},
    ]
    state_spec = parse_session(state_raw)

    replay_spec = spec_minimal
    replay_partial = Engine(replay_spec)
    replay_partial.step()
    replay_state = replay_partial.to_state_dict()
    replay_state.pop("state_digest")
    replay_state["connection_window"] += 1
    replay_state["state_digest"] = digest_value(replay_state)

    def snapshot_probe(path: Path) -> None:
        with SnapshotStore(path) as store:
            info = store.save(Engine(spec_minimal), "tamper")
            store.connection.execute(
                "UPDATE snapshots SET content_digest=? WHERE id=?",
                ("0" * 64, info.snapshot_id),
            )
            store.connection.commit()
            store.load(info.snapshot_id)

    def flow_probe() -> None:
        engine = Engine(flow_spec)
        engine.step()

    def state_probe() -> None:
        engine = Engine(state_spec)
        engine.step()
        engine.step()

    with tempfile.TemporaryDirectory() as temp:
        snapshot_path = Path(temp) / "snapshot-probe.db"
        probes = {
            "E_SPEC": maybe_error(lambda: parse_session(invalid_spec)),
            "E_FRAME": maybe_error(frame_probe),
            "E_FLOW": maybe_error(flow_probe),
            "E_STATE": maybe_error(state_probe),
            "E_REPLAY": maybe_error(
                lambda: replay_from_state(replay_spec, replay_state, reference_engine=full_minimal)
            ),
            "E_SNAPSHOT": maybe_error(lambda: snapshot_probe(snapshot_path)),
        }
    return probes


def sample_evidence(path: Path, temp_dir: Path) -> dict[str, Any]:
    spec = load_session(path)
    reference = Engine(spec).run()
    repeat = Engine(spec).run()
    deterministic_repeat = (
        reference.run_digest() == repeat.run_digest()
        and [frame.to_dict() for frame in reference.frames]
        == [frame.to_dict() for frame in repeat.frames]
    )
    target_index = 1 if len(spec.events) > 2 else max(0, len(spec.events) // 2)
    if len(spec.events) > 10:
        target_index = len(spec.events) // 2
    partial = Engine(spec)
    while partial.event_cursor < target_index:
        partial.step()

    db_path = temp_dir / f"{path.stem}.db"
    with SnapshotStore(db_path) as store:
        info = store.save(partial, f"verification-event-{partial.next_event_seq - 1}")
        _loaded_info, loaded_spec, loaded_state = store.load(info.snapshot_id)
        restored_state = Engine.from_state_dict(loaded_spec, loaded_state)
        restore_matches = restored_state.run_digest() == partial.run_digest()
        restore = store.restore(info.snapshot_id, spec, reference)
        replay = replay_from_snapshot(
            store,
            info.snapshot_id,
            reference_spec=spec,
            reference_engine=reference,
        )
        entries = len(store.list())
        snapshot_info = info.to_dict()

    view_model = WorkbenchViewModel(spec, snapshot_db=db_path)
    try:
        digest_before = view_model.engine.run_digest()
        first_render = view_model.render_signature()
        second_render = view_model.render_signature()
        view_model_stable = first_render == second_render and view_model.engine.run_digest() == digest_before
    finally:
        view_model.close()

    summary = reference.summary()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "session_id": spec.session_id,
        "declared_version": spec.declared_version,
        "content_digest": spec.content_digest,
        "streams": len(spec.streams),
        "events": len(spec.events),
        "scheduling_quantum": spec.connection.scheduling_quantum,
        "frames": summary["frames"],
        "transitions": summary["transitions"],
        "warnings": summary["warnings"],
        "preemption_warnings": summary["preemption_warnings"],
        "strategy_switches": summary["strategy_switches"],
        "current_strategy": summary["current_strategy"],
        "strategy_selection": summary["strategy_selection"],
        "run_digest": summary["run_digest"],
        "deterministic_repeat": deterministic_repeat,
        "snapshot": snapshot_info,
        "snapshot_restore_matches_current": restore_matches,
        "snapshot_whole_replacement": restore.whole_replacement,
        "snapshot_entries": entries,
        "replay": replay.to_dict(),
        "view_model_stable_and_read_only": view_model_stable,
    }


def whole_replacement_probe(temp_dir: Path) -> dict[str, Any]:
    base = json.loads((ROOT / "samples" / "minimal.json").read_text(encoding="utf-8"))
    old = copy.deepcopy(base)
    old["session_id"] = "old-session"
    old["declared_version"] = 1
    old["streams"] = [{"stream_id": 1, "priority": 1, "request_body_bytes": 10}]
    old["events"] = []
    old_spec = parse_session(old)
    old_engine = Engine(old_spec).run(drain=False)

    current = copy.deepcopy(base)
    current["session_id"] = "current-session"
    current["declared_version"] = 2
    current["streams"] = [{"stream_id": 9, "priority": 1, "request_body_bytes": 10}]
    current["events"] = []
    current_spec = parse_session(current)
    current_engine = Engine(current_spec).run(drain=False)

    with SnapshotStore(temp_dir / "replacement.db") as store:
        info = store.save(old_engine, "old-checkpoint")
        result = store.restore(info.snapshot_id, current_spec, current_engine)
        rows = store.list()
    return {
        "whole_replacement": result.whole_replacement,
        "version_mismatch": result.version_mismatch,
        "content_digest_mismatch": result.content_digest_mismatch,
        "added": len(result.diff["added"]),
        "removed": len(result.diff["removed"]),
        "changed": len(result.diff["changed"]),
        "backup_snapshot_id": result.backup_snapshot_id,
        "all_snapshots_read_only": all(row.read_only for row in rows),
        "replacement_session": result.spec.session_id,
        "replacement_streams": sorted(result.engine.streams),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "verification.json"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    tests = run_tests() if not args.skip_tests else {"run": 0, "successful": True, "skipped": True}
    raw_minimal = json.loads(SAMPLES[0].read_text(encoding="utf-8"))
    spec_minimal = load_session(SAMPLES[0])
    full_minimal = Engine(spec_minimal).run()
    payload: dict[str, Any] = {
        "ok": bool(tests["successful"]),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "tkinter": tkinter_probe(),
        "tests": tests,
        "error_code_probes": error_probes(raw_minimal, spec_minimal, full_minimal),
        "samples": [],
        "whole_replacement_probe": None,
        "notes": [
            "No network service, container, or external account is used.",
            "Replay compares target stream, data bytes, both window ledgers, strategy, event trigger, business code, and stream transitions.",
            "The configured NumPy seed drives deterministic scale sample generation; runtime scheduling uses integer-only apportionment.",
            "Tk availability is measured on this host. The Windows UI path is also covered through side-effect-free view-model tests.",
        ],
    }
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for sample in SAMPLES:
            payload["samples"].append(sample_evidence(sample, temp_dir))
        payload["whole_replacement_probe"] = whole_replacement_probe(temp_dir)

    payload["ok"] = payload["ok"] and all(item["deterministic_repeat"] for item in payload["samples"])
    payload["ok"] = payload["ok"] and all(item["replay"]["match"] for item in payload["samples"])
    payload["ok"] = payload["ok"] and all(item["view_model_stable_and_read_only"] for item in payload["samples"])
    payload["ok"] = payload["ok"] and all(item["observed"] for item in payload["error_code_probes"].values())

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "output": str(output), "tests": tests["run"]}, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
