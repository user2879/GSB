"""Reproducible non-UI acceptance run using only 127.0.0.1 sockets."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tcp_workbench.config import load_session_config
from tcp_workbench.runner import ingest_events, load_event_script, run_loopback_session
from tcp_workbench.service import SessionService
from tcp_workbench.snapshot import seed_snapshot
from tcp_workbench.structured_log import StructuredLogger

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def run_event_acceptance(db_path: Path, log_path: Path) -> dict:
    config, script = load_session_config(SAMPLES / "session.json")
    service = SessionService.open(
        db_path, config, script, logger=StructuredLogger(log_path)
    )
    try:
        events = load_event_script(SAMPLES / "events.json")
        results = ingest_events(service, events)
        service.verify_incremental_equivalence()
        baseline = service.save_snapshot("acceptance-v1")
        seeded = []
        for path in sorted((SAMPLES / "snapshots").glob("*.json")):
            seed = json.loads(path.read_text(encoding="utf-8"))
            seeded.append(
                seed_snapshot(
                    service.ledger,
                    seed,
                    mono_ms=seed["created_mono_ms"],
                    base_dir=path.parent,
                )
            )
        verified = [
            service.verify_snapshot(snapshot_id) for snapshot_id in [baseline, *seeded]
        ]
        v2_config, v2_script = load_session_config(
            SAMPLES / "versions" / "session_v2.json",
            peer_script_override=SAMPLES / "versions" / "peer_script_v2.json",
        )
        replacement = service.evolve(v2_config, v2_script)
        service.verify_incremental_equivalence()
        return {
            "events_ingested": len(results),
            "incremental_equals_full": True,
            "baseline_snapshot_id": baseline,
            "historical_snapshot_ids": seeded,
            "snapshots_verified": len(verified),
            "evolved_to": f"C{service.config.config_version}/S{service.script.script_version}",
            "replacement_fields": len(replacement.affected_fields),
            "archived_fields": replacement.archived_fields,
        }
    finally:
        service.close()


def run_loopback_acceptance(db_path: Path, log_path: Path) -> dict:
    config, script = load_session_config(SAMPLES / "session.json")
    service = SessionService.open(
        db_path, config, script, logger=StructuredLogger(log_path)
    )
    try:
        transcript, peer = run_loopback_session(service, port=0, send_segments=7)
        service.verify_incremental_equivalence()
        return {
            "peer": f"127.0.0.1:{peer.port}",
            "rounds": transcript.rounds,
            "responses": len(transcript.peer_responses),
            "dropped_seqs": list(transcript.dropped_seqs),
            "reordered_responses": sum(
                1 for item in transcript.peer_responses if item.get("reordered")
            ),
            "event_count": len(service.events),
            "incremental_equals_full": True,
            "final_state": service.items["connection_state"].value,
        }
    finally:
        service.close()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tcpdiag-acceptance-") as directory:
        temp = Path(directory)
        result = {
            "event_script": run_event_acceptance(
                temp / "events.sqlite3", temp / "events.jsonl"
            ),
            "loopback_peer": run_loopback_acceptance(
                temp / "loopback.sqlite3", temp / "loopback.jsonl"
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
