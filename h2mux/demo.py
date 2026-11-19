"""Repeatable, deterministic three-scenario demonstration."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .declaration import load_declaration, load_script
from .replay import replay_from_snapshot
from .session import H2Session
from .snapshot import SnapshotStore, restore_snapshot
from .util import digest_value


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUTPUT = ROOT / "artifacts" / "demo"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_summary(session: H2Session) -> dict[str, Any]:
    summary = session.summary()
    frame_projection = [
        {
            key: record.get(key)
            for key in (
                "direction",
                "frame_type",
                "target_stream_id",
                "payload_bytes",
                "flow_bytes",
                "conn_send_window_after",
                "stream_send_window_after",
                "state_after",
                "code",
            )
        }
        for record in session.ledger.records
    ]
    return {
        "strategy": summary["strategy"],
        "initial_strategy": summary.get("initial_strategy"),
        "frame_count": summary["frame_count"],
        "ledger_records": summary["ledger_records"],
        "frame_sequence_digest": digest_value(frame_projection),
        "completion": summary["completion"],
        "stream_states": [
            {
                "stream_id": stream["stream_id"],
                "state": stream["state"],
                "bytes_sent": stream["bytes_sent"],
                "bytes_received": stream["bytes_received"],
                "send_window": stream["send_window"],
                "recv_window": stream["recv_window"],
                "parent_id": stream["parent_id"],
                "weight": stream["weight"],
            }
            for stream in summary["streams"]
        ],
        "empty_wait_count": len(summary["empty_waits"]),
        "queue_wait_count": len(summary["queue_waits"]),
        "diagnostics": summary["diagnostics"],
    }


def run_demo(output_dir: Path = OUTPUT) -> dict[str, Any]:
    declaration = load_declaration(SAMPLES / "session.json")
    script = load_script(SAMPLES / "frames.json", declaration)
    truncated_script = load_script(SAMPLES / "frames_truncated.json", declaration)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_1 = H2Session(declaration, script, strategy="weighted_round_robin").run()
    scenario_1_summary = _compact_summary(scenario_1)
    _write_json(output_dir / "scenario-1-single-strategy.json", scenario_1_summary)
    _write_json(output_dir / "scenario-1-ledger.json", scenario_1.ledger.to_dict())

    scenario_2 = H2Session(declaration, script, strategy="weighted_round_robin")
    switch_event = 10
    scenario_2.run(stop_after_events=switch_event)
    scenario_2.set_strategy("window_priority", script_index=switch_event)
    scenario_2.run()
    scenario_2_summary = _compact_summary(scenario_2)
    scenario_2_summary["switch"] = {
        "at_script_index": switch_event,
        "from": "weighted_round_robin",
        "to": "window_priority",
        "control": list(scenario_2.ledger.controls),
    }
    _write_json(output_dir / "scenario-2-strategy-switch.json", scenario_2_summary)
    _write_json(output_dir / "scenario-2-ledger.json", scenario_2.ledger.to_dict())

    snapshot_dir = output_dir / "snapshots"
    store = SnapshotStore(snapshot_dir)
    snapshot_name = "demo-session.json"
    interrupted = H2Session(declaration, script, strategy="weighted_round_robin")
    interrupted.run(stop_after_events=9)
    first_path = store.save(interrupted, snapshot_name, label="stage 1 - pre-reprioritization")
    interrupted.run(stop_after_events=15)
    second_path = store.save(interrupted, snapshot_name, label="stage 2 - before final scheduling")
    backup_path = Path(str(second_path) + ".bak")
    backup_mode = stat.S_IMODE(os.stat(backup_path).st_mode) if backup_path.exists() else None
    backup_read_only = bool(backup_mode is not None and not (backup_mode & stat.S_IWRITE))

    restoration = restore_snapshot(
        second_path,
        script,
        current_declaration=declaration,
        current_strategy="weighted_round_robin",
    )
    restored = restoration.session
    restored.run()
    replay = replay_from_snapshot(
        str(second_path),
        script,
        current_declaration=declaration,
        current_strategy="weighted_round_robin",
    )
    restoration_report = {
        "snapshot_path": str(second_path),
        "backup_path": str(backup_path),
        "backup_exists": backup_path.exists(),
        "backup_mode_octal": oct(backup_mode) if backup_mode is not None else None,
        "backup_read_only": backup_read_only,
        "first_snapshot_path": str(first_path),
        "restore": restoration.to_dict(),
        "restored_completion": restored.completion_report(),
        "replay": replay.to_dict(),
        "reproducible": replay.status in {"MATCH", "TRUNCATED"},
    }
    _write_json(output_dir / "scenario-3-snapshot-replay.json", restoration_report)

    truncated = H2Session(declaration, truncated_script, strategy="weighted_round_robin").run()
    truncated_report = {
        "script": "samples/frames_truncated.json",
        "completion": truncated.completion_report(),
        "attribution": "script_tail_truncated" if truncated.completion_report()["script_tail_truncated"] else "complete",
        "frame_count": truncated.frame_seq,
    }
    _write_json(output_dir / "tail-truncation.json", truncated_report)

    summary = {
        "format_version": 1,
        "declaration_version": declaration.declaration_version,
        "declaration_digest": declaration.content_digest,
        "script_digest": script.source_digest,
        "scenario_order": [
            "small_single_strategy_send",
            "strategy_switch_send",
            "snapshot_save_restore_replay",
        ],
        "scenario_1": scenario_1_summary,
        "scenario_2": scenario_2_summary,
        "scenario_3": restoration_report,
        "tail_truncation": truncated_report,
        "all_checks_passed": bool(
            scenario_1.completion_report()["complete"]
            and scenario_2.completion_report()["complete"]
            and restored.completion_report()["complete"]
            and replay.status == "MATCH"
            and backup_read_only
            and truncated.completion_report()["script_tail_truncated"]
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "diagnostics.json", scenario_1.diagnostics())
    return summary


def main() -> int:
    summary = run_demo()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
