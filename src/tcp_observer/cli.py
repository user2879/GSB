from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical import canonical_text, sha256_hex, write_canonical_json
from .config_paths import ROOT, resolve_path
from .engine import ObservationWorkspace, build_observation, diff_script_entries
from .errors import ObserverError
from .ledger import Ledger
from .pipeline import commit_observation
from .receipts import (
    NormalizationDefaults,
    decode_and_normalize_file,
    write_receipt_file,
)
from .recovery import recover_and_replay
from .replay import compare_edit_items, compare_items
from .scheduler import deterministic_queue_demo
from .scripting import dupack_windows, find_first_gap, validate_for_run
from .snapshot import assert_version_monotonic, open_or_replace_snapshot, visible_snapshot_view
from .spec import SessionSpec, load_session
from .transport import SRC_DIR, bind_listener, run_live_exchange


def _print_json(value: Any) -> None:
    print(canonical_text(value))


def _error_exit(exc: ObserverError) -> int:
    _print_json({"ok": False, "error": exc.to_dict()})
    return 2


def cmd_validate_spec(args: argparse.Namespace) -> int:
    try:
        spec = load_session(args.config)
    except ObserverError as exc:
        return _error_exit(exc)
    _print_json({"ok": True, "session_id": spec.session_id, "script_digest": spec.script_digest})
    return 0


def cmd_check_script(args: argparse.Namespace) -> int:
    spec = load_session(args.config)
    gap = find_first_gap(spec)
    windows = dupack_windows(spec)
    try:
        validate_for_run(spec)
    except ObserverError as exc:
        return _error_exit(exc)
    _print_json({"ok": True, "first_gap": None if gap is None else gap.__dict__, "dupack_windows": windows})
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    try:
        normalized = decode_and_normalize_file(args.file, NormalizationDefaults())
    except ObserverError as exc:
        return _error_exit(exc)
    if args.out:
        write_canonical_json(args.out, normalized)
    _print_json({"ok": True, "count": len(normalized), "out": args.out})
    return 0


def generate_receipt_samples(spec: SessionSpec, result: dict[str, Any], samples_dir: Path) -> dict[str, list[str]]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    records = result["batches"][:3]
    paths: dict[str, list[str]] = {}
    for version, filename in [(1, "receipts_o1.txt"), (2, "receipts_o2.jsonl"), (3, "receipts_o3.jsonl")]:
        path = samples_dir / filename
        write_receipt_file(path, records, version)
        paths[f"o{version}"] = [str(path.relative_to(ROOT))]
    return paths


def normalize_generated_samples(samples_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for version, filename in [(1, "receipts_o1.txt"), (2, "receipts_o2.jsonl"), (3, "receipts_o3.jsonl")]:
        normalized = decode_and_normalize_file(samples_dir / filename)
        output[f"o{version}"] = normalized
    return output


def verify_error_fixtures() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    spec_cases = [
        ("E_SPEC delay line", ROOT / "samples/invalid/spec_bad_delay.json"),
        ("E_SPEC dupack compatibility", ROOT / "samples/invalid/spec_bad_dupack.json"),
    ]
    for label, path in spec_cases:
        try:
            load_session(path)
            checks.append({"case": label, "ok": False, "reason": "expected E_SPEC"})
        except ObserverError as exc:
            checks.append({"case": label, "ok": exc.code == "E_SPEC", "error": exc.to_dict()})

    for label, config, expected_code in [
        ("E_SCRIPT gap", ROOT / "configs/session_gap.json", "E_SCRIPT"),
        ("E_DUPACK cluster", ROOT / "configs/session_dupack_cluster.json", "E_DUPACK"),
    ]:
        try:
            validate_for_run(load_session(config))
            checks.append({"case": label, "ok": False, "reason": f"expected {expected_code}"})
        except ObserverError as exc:
            checks.append({"case": label, "ok": exc.code == expected_code, "error": exc.to_dict()})

    format_cases = [
        ("unknown version", ROOT / "samples/invalid/o1_unknown.txt", "format_version"),
        ("o1 width", ROOT / "samples/invalid/o1_width.txt", "fixed_width"),
        ("o2 missing field", ROOT / "samples/invalid/o2_missing.jsonl", "srtt_ms"),
        ("o2 negative duration", ROOT / "samples/invalid/o2_negative_duration.jsonl", "duration_ms"),
    ]
    for label, path, expected_field in format_cases:
        try:
            decode_and_normalize_file(path)
            checks.append({"case": label, "ok": False, "reason": "expected E_FORMAT"})
        except ObserverError as exc:
            checks.append(
                {
                    "case": label,
                    "ok": exc.code == "E_FORMAT" and exc.field_name == expected_field and exc.line is not None,
                    "error": exc.to_dict(),
                }
            )
    return checks


def run_transport_check(spec: SessionSpec) -> dict[str, Any]:
    exchange = run_live_exchange(spec)
    responses = exchange["responses"]
    matched = []
    for index, response in enumerate(responses):
        expected_seq = spec.initial_seq + index
        if response.get("type") == "error":
            raise ObserverError("E_FORMAT", "peer returned protocol error", details=response)
        if response.get("event_seq") != expected_seq:
            raise ObserverError(
                "E_SCRIPT",
                "live peer response event sequence mismatch",
                event_seq=expected_seq,
                details={"actual": response.get("event_seq")},
            )
        matched.append(expected_seq)
    return {
        "ok": True,
        "response_count": len(responses),
        "matched_event_seqs": matched,
        "hello_peer_id": exchange["hello"].get("peer_id"),
        "hello_script_digest_matches": exchange["hello"].get("script_digest") == spec.script_digest,
        "real_socket_timeouts": {"handshake_timeout_ms": spec.handshake_timeout_ms, "read_timeout_ms": spec.read_timeout_ms},
    }


def port_and_timeout_checks(spec: SessionSpec) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    holder = bind_listener("127.0.0.1", 0)
    occupied_port = holder.getsockname()[1]
    try:
        try:
            bind_listener("127.0.0.1", occupied_port)
            results.append({"case": "E_PORT occupied bind", "ok": False})
        except ObserverError as exc:
            results.append({"case": "E_PORT occupied bind", "ok": exc.code == "E_PORT", "error": exc.to_dict()})
    finally:
        holder.close()

    silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    silent_port = silent.getsockname()[1]

    def accept_silent() -> None:
        try:
            conn, _ = silent.accept()
            threading.Event().wait(2.0)
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=accept_silent, daemon=True)
    thread.start()
    short_spec = replace(spec, handshake_timeout_ms=40, peer_port=silent_port)
    try:
        from .transport import connect_handshake

        try:
            connect_handshake(short_spec, silent_port)
            results.append({"case": "E_TIMEOUT real handshake recv timeout", "ok": False})
        except ObserverError as exc:
            results.append({"case": "E_TIMEOUT real handshake recv timeout", "ok": exc.code == "E_TIMEOUT", "error": exc.to_dict()})
    finally:
        silent.close()
    return results


def build_edit_reports(spec: SessionSpec) -> dict[str, Any]:
    changed_seq = 1012
    old_entry = spec.entry_for_seq(changed_seq)
    assert old_entry is not None
    changed_entry = replace(old_entry, delay_ms=4)
    new_entries = tuple(changed_entry if entry.event_seq == changed_seq else entry for entry in spec.entries)
    from .spec import canonical_script_digest

    edited_spec = replace(
        spec,
        entries=new_entries,
        script_digest=sha256_hex(canonical_script_digest(new_entries, spec.script_version)),
    )
    workspace = ObservationWorkspace(spec)
    script_plan = workspace.apply_open_window_script_edit(edited_spec, changed_seq)
    expected_edit_result = build_observation(edited_spec)
    equality = compare_edit_items(expected_edit_result, script_plan["result"], changed_seq)

    seal_spec = replace(spec, seal_length=6)
    seal_workspace = ObservationWorkspace(spec)
    seal_plan = seal_workspace.apply_seal_length_view_edit(seal_spec)

    append_workspace = ObservationWorkspace(spec, initial_events=spec.event_count - 1)
    append_plan = append_workspace.append_next_event_incrementally()
    append_equality = compare_items(build_observation(spec), append_plan["result"])
    return {
        "new_event_arrival": {
            "new_event_seq": append_plan["new_event_seq"],
            "affected_items": append_plan["affected_items"],
            "reused_item_count": append_plan["reused_item_count"],
            "affected_batch_id": append_plan["affected_batch_id"],
            "affected_window_id": append_plan["affected_window_id"],
            "sealed_windows_byte_unchanged": append_plan["sealed_windows_byte_unchanged"],
            "replay_equality": append_equality,
        },
        "script_edit": {
            "changed_event_seq": changed_seq,
            "changed_field": "entries.delay_ms",
            "old_delay_ms": old_entry.delay_ms,
            "new_delay_ms": 4,
            "affected_items": script_plan["affected_items"],
            "recomputed_items": script_plan["recomputed_items"],
            "reused_item_count": script_plan["reused_item_count"],
            "sealed_windows_byte_unchanged": script_plan["sealed_windows_byte_unchanged"],
            "sealed_window_digests_before": {str(k): v for k, v in script_plan["sealed_window_digests_before"].items()},
            "sealed_window_digests_after": {str(k): v for k, v in script_plan["sealed_window_digests_after"].items()},
            "propagation_steps": script_plan["propagation_steps"],
            "replay_equality": equality,
            "edited_summary_digest": script_plan["result"]["summary"]["content_digest"],
        },
        "seal_length_edit": {
            "old_seal_length": seal_plan["old_seal_length"],
            "new_seal_length": seal_plan["new_seal_length"],
            "affected_view_items": seal_plan["affected_view_items"],
            "metric_items_recomputed": seal_plan["metric_items_recomputed"],
            "sealed_windows_byte_unchanged": seal_plan["sealed_window_digests_unchanged"],
            "basis": seal_plan["basis"],
        },
    }


def run_crash_demo(config: str, db: str, crash_batch: int) -> int:
    spec = load_session(config)
    result = build_observation(spec)
    commit_observation(spec, result, db, reset=True, crash_before_batch=crash_batch)
    return 0


def execute_demo(out_dir: Path, config: str | Path | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_session(config or ROOT / "configs/session.json")
    validate_for_run(spec)

    sample_paths = generate_receipt_samples(spec, build_observation(spec), ROOT / "samples")
    normalized = normalize_generated_samples(ROOT / "samples")
    write_canonical_json(out_dir / "normalized_receipts.json", normalized)

    error_fixtures = verify_error_fixtures()
    if not all(item["ok"] for item in error_fixtures):
        raise ObserverError("E_VERIFY", "one or more error-code fixtures failed", details={"checks": error_fixtures})

    transport_report = run_transport_check(spec)
    transport_report["error_code_checks"] = port_and_timeout_checks(spec)
    if not all(item["ok"] for item in transport_report["error_code_checks"]):
        raise ObserverError("E_VERIFY", "port/timeout checks failed", details=transport_report)
    write_canonical_json(out_dir / "transport_report.json", transport_report)

    result = build_observation(spec)
    write_canonical_json(out_dir / "conclusions.json", result)
    write_canonical_json(out_dir / "summary.json", result["summary"])

    scheduler_demo = deterministic_queue_demo()
    write_canonical_json(out_dir / "scheduler_demo.json", scheduler_demo)

    edit_report = build_edit_reports(spec)
    write_canonical_json(out_dir / "edit_recompute.json", edit_report)

    snapshot_path = out_dir / "snapshot-current.json"
    snapshot_v1 = open_or_replace_snapshot(snapshot_path, spec, result)
    spec_v2 = load_session(ROOT / "configs/session_v2.json")
    result_v2 = build_observation(spec_v2)
    assert_version_monotonic(spec.script_version, spec_v2.script_version, spec.scale_version, spec_v2.scale_version)
    snapshot_replacement = open_or_replace_snapshot(snapshot_path, spec_v2, result_v2)
    snapshot_report = {
        "first_action": snapshot_v1["action"],
        "replacement_action": snapshot_replacement["action"],
        "merge_performed": snapshot_replacement["merge_performed"],
        "backup_path": Path(snapshot_replacement["backup_path"]).name if snapshot_replacement.get("backup_path") else None,
        "old_content_digest": snapshot_replacement.get("old_content_digest"),
        "new_content_digest": snapshot_replacement["snapshot"]["content_digest"],
        "reasons": snapshot_replacement.get("reasons"),
    }
    write_canonical_json(out_dir / "snapshot_replacement.json", snapshot_report)

    ledger_path = out_dir / "ledger.sqlite"
    ledger = commit_observation(spec, result, ledger_path, reset=True)
    ledger_export = ledger.export_core_jsonl(out_dir / "ledger_core.jsonl")
    baseline_records = ledger.core_records()
    ledger.close()

    recovery_dir = out_dir / "recovery"
    recovery_dir.mkdir(exist_ok=True)
    crashed_db = recovery_dir / "crashed.sqlite"
    if crashed_db.exists():
        crashed_db.unlink()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env.get("PYTHONPATH", ""))
    crash_cmd = [
        sys.executable,
        "-m",
        "tcp_observer.cli",
        "crash-demo",
        "--config",
        str(Path(config or ROOT / "configs/session.json").resolve()),
        "--db",
        str(crashed_db),
        "--crash-batch",
        "3",
    ]
    crash_proc = subprocess.run(crash_cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    forced_kill_ok = crash_proc.returncode == -9
    if not forced_kill_ok:
        raise ObserverError(
            "crash subprocess was not force-killed with SIGKILL",
            details={"returncode": crash_proc.returncode, "stdout": crash_proc.stdout[-500:], "stderr": crash_proc.stderr[-1000:]},
        )

    recovered_ledger, recovery_report, _replayed = recover_and_replay(spec, crashed_db, result)
    recovered_export = recovered_ledger.export_core_jsonl(recovery_dir / "recovered_ledger_core.jsonl")
    recovered_ledger.close()
    recovery_report["forced_kill_returncode"] = crash_proc.returncode
    recovery_report["core_ledger_byte_match"] = ledger_export.read_bytes() == recovered_export.read_bytes()
    if not recovery_report["core_ledger_byte_match"]:
        raise ObserverError("E_REPLAY", "recovered core ledger does not byte-match uninterrupted ledger")
    write_canonical_json(recovery_dir / "recovery_report.json", recovery_report)

    repeat_db = recovery_dir / "repeat.sqlite"
    repeat_ledger = commit_observation(spec, build_observation(spec), repeat_db, reset=True)
    repeat_export = repeat_ledger.export_core_jsonl(recovery_dir / "repeat_ledger_core.jsonl")
    repeat_ledger.close()
    repeat_match = ledger_export.read_bytes() == repeat_export.read_bytes()
    if not repeat_match:
        raise ObserverError("E_VERIFY", "repeat ledger export is not byte-identical")

    report = {
        "ok": True,
        "sample_paths": sample_paths,
        "receipt_normalization_counts": {key: len(value) for key, value in normalized.items()},
        "error_fixture_count": len(error_fixtures),
        "all_error_fixtures_ok": True,
        "transport": {k: v for k, v in transport_report.items() if k != "error_code_checks"},
        "port_timeout_check_count": len(transport_report["error_code_checks"]),
        "summary_content_digest": result["summary"]["content_digest"],
        "conclusions_sha256": sha256_hex(write_canonical_json(out_dir / "conclusions.json", result).read_bytes()),
        "ledger_core_sha256": sha256_hex(ledger_export.read_bytes()),
        "ledger_record_count": len(baseline_records),
        "scheduler_peak": scheduler_demo["peak_in_flight"],
        "affected_script_items": edit_report["script_edit"]["affected_items"],
        "sealed_windows_byte_unchanged": edit_report["script_edit"]["sealed_windows_byte_unchanged"],
        "snapshot_action": snapshot_report["replacement_action"],
        "version_monotonic_ok": True,
        "recovery": recovery_report,
        "repeat_ledger_byte_match": repeat_match,
    }
    write_canonical_json(out_dir / "validation_report.json", report)
    return report


def cmd_all(args: argparse.Namespace) -> int:
    try:
        report = execute_demo(Path(args.out), args.config)
    except ObserverError as exc:
        return _error_exit(exc)
    _print_json(report)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        spec = load_session(args.config)
        validate_for_run(spec)
        exchange = run_live_exchange(spec) if not args.no_transport else None
        result = build_observation(spec, exchange)
        out = Path(args.out)
        write_canonical_json(out / "summary.json", result["summary"])
        write_canonical_json(out / "conclusions.json", result)
        ledger = commit_observation(spec, result, out / "ledger.sqlite", reset=True)
        ledger.export_core_jsonl(out / "ledger_core.jsonl")
        ledger.close()
    except ObserverError as exc:
        return _error_exit(exc)
    _print_json({"ok": True, "out": args.out, "digest": result["summary"]["content_digest"]})
    return 0


def cmd_crash_demo(args: argparse.Namespace) -> int:
    return run_crash_demo(args.config, args.db, args.crash_batch)


def cmd_recover(args: argparse.Namespace) -> int:
    try:
        spec = load_session(args.config)
        expected = build_observation(spec)
        ledger, report, _ = recover_and_replay(spec, args.db, expected)
        if args.out:
            ledger.export_core_jsonl(Path(args.out) / "recovered_ledger_core.jsonl")
            write_canonical_json(Path(args.out) / "recovery_report.json", report)
        ledger.close()
    except ObserverError as exc:
        return _error_exit(exc)
    _print_json(report)
    return 0


def cmd_snapshot_show(args: argparse.Namespace) -> int:
    try:
        _print_json({"ok": True, "view": visible_snapshot_view(args.path)})
    except ObserverError as exc:
        return _error_exit(exc)
    return 0


def cmd_gui(_args: argparse.Namespace) -> int:
    from .gui import launch

    launch()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcp-observer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-spec")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_validate_spec)

    p = sub.add_parser("check-script")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_check_script)

    p = sub.add_parser("normalize")
    p.add_argument("--file", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("generate-samples")
    p.add_argument("--config", default=str(ROOT / "configs/session.json"))
    p.add_argument("--samples-dir", default=str(ROOT / "samples"))
    p.set_defaults(func=lambda a: (_print_json(generate_receipt_samples(load_session(a.config), build_observation(load_session(a.config)), Path(a.samples_dir))) or 0))

    p = sub.add_parser("run")
    p.add_argument("--config", default=str(ROOT / "configs/session.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--no-transport", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("all")
    p.add_argument("--config", default=str(ROOT / "configs/session.json"))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("crash-demo")
    p.add_argument("--config", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--crash-batch", type=int, default=3)
    p.set_defaults(func=lambda a: run_crash_demo(a.config, a.db, a.crash_batch))

    p = sub.add_parser("recover")
    p.add_argument("--config", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("snapshot-show")
    p.add_argument("--path", required=True)
    p.set_defaults(func=cmd_snapshot_show)

    p = sub.add_parser("gui")
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
