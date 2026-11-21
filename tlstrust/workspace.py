"""Local workspace preparation and validation-run orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .certfamily import Family, generate_family, load_family
from .eventlog import JsonlEventLog
from .handshake import capture_real_handshake, write_input_snapshot
from .ledger import Ledger
from .trust import TrustDocument, demo_revisions, load_trust_document, write_trust_document
from .verify import validate_snapshot


WORKSPACE_MARKER = ".tlstrust-workspace"


@dataclass(frozen=True)
class Workspace:
    root: Path
    family: Family
    trust_path: Path
    ledger_path: Path
    snapshot_dir: Path
    session_dir: Path
    log_path: Path

    def open_ledger(self, *, auto_recover: bool = True) -> Ledger:
        return Ledger(self.ledger_path, auto_recover=auto_recover)

    def event_log(self) -> JsonlEventLog:
        return JsonlEventLog(self.log_path)

    def save_input_snapshot(self, snapshot: dict[str, Any], name: str = "input.json") -> Path:
        path = self.snapshot_dir / name
        write_input_snapshot(path, snapshot)
        return path


def _prepare_directory(root: Path, fresh: bool) -> None:
    if fresh and root.exists() and any(root.iterdir()):
        marker = root / WORKSPACE_MARKER
        if not marker.exists():
            raise ValueError(
                f"refusing to clear non-workbench directory: {root}; choose an empty path or use --reuse"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / WORKSPACE_MARKER
    if not marker.exists():
        marker.write_text("tlstrust workspace v1\n", encoding="ascii")


def prepare_workspace(
    root: str | Path,
    *,
    roots: int = 2,
    intermediate_levels: int = 2,
    leaves: int = 1,
    validity_days: int = 90,
    port: int = 0,
    seed: str = "tlstrust-deterministic-v1",
    key_bits: int = 2048,
    not_before_ms: int | None = None,
    fresh: bool = False,
) -> Workspace:
    root_path = Path(root)
    _prepare_directory(root_path, fresh)
    family_dir = root_path / "family"
    family_json = family_dir / "family.json"
    if family_json.exists():
        family = load_family(family_json)
    else:
        kwargs: dict[str, Any] = {
            "roots": roots,
            "intermediate_levels": intermediate_levels,
            "leaves": leaves,
            "validity_days": validity_days,
            "port": port,
            "seed": seed,
            "key_bits": key_bits,
        }
        if not_before_ms is not None:
            kwargs["not_before_ms"] = not_before_ms
        family = generate_family(family_dir, **kwargs)
    snapshot_dir = root_path / "snapshots"
    session_dir = root_path / "sessions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = root_path / "logs" / "events.jsonl"
    trust_path = root_path / "trust_revisions.json"
    revisions = demo_revisions(family)
    write_trust_document(trust_path, revisions)
    trust_document = load_trust_document(trust_path, family)
    workspace = Workspace(
        root=root_path,
        family=family,
        trust_path=trust_path,
        ledger_path=root_path / "ledger.sqlite3",
        snapshot_dir=snapshot_dir,
        session_dir=session_dir,
        log_path=log_path,
    )
    ledger = workspace.open_ledger()
    try:
        ensure_trust_version(ledger, trust_document, source_path=str(trust_path))
    finally:
        ledger.close()
    return workspace


def ensure_trust_version(
    ledger: Ledger,
    document: TrustDocument,
    *,
    source_path: str | None = None,
) -> tuple[int, bool]:
    current = ledger.current_trust_version()
    if current is not None and current["revision_chain_digest"] == document.chain_digest:
        return int(current["version"]), False
    created = ledger.create_trust_version(
        document,
        operation="INITIAL" if current is None else "REBUILD",
        source_path=source_path,
    )
    return int(created["version"]), True


def run_validation(
    ledger: Ledger,
    document: TrustDocument,
    snapshot: dict[str, Any],
    *,
    source_path: str | None = None,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
    crash_after: int | None = None,
) -> dict[str, Any]:
    version, switched = ensure_trust_version(ledger, document, source_path=source_path)
    run_id = ledger.start_run(version, snapshot, document.revision_numbers)
    results: list[dict[str, Any]] = []
    for revision in document.revisions:
        result = validate_snapshot(snapshot, revision)
        ledger.record_revision_result(run_id, result)
        results.append(result)
        if on_result is not None:
            on_result(revision.revision, result)
        if crash_after is not None and len(results) >= crash_after:
            return {
                "run_id": run_id,
                "trust_version": version,
                "results": results,
                "crashed": True,
                "switched_trust_version": switched,
            }
    session = ledger.finalize_run(run_id)
    results = ledger.committed_results(run_id)
    with ledger._lock:
        flip_rows = ledger.conn.execute(
            "SELECT flip_json FROM flips WHERE run_id = ? AND committed = 1 ORDER BY ordinal",
            (run_id,),
        ).fetchall()
    import json

    return {
        "run_id": run_id,
        "trust_version": version,
        "switched_trust_version": switched,
        "results": results,
        "flips": [json.loads(row["flip_json"]) for row in flip_rows],
        "session": session,
        "crashed": False,
    }


def capture_snapshot(workspace: Workspace, *, leaf_id: str | None = None, port: int = 0) -> dict[str, Any]:
    snapshot = capture_real_handshake(workspace.family, leaf_id=leaf_id, port=port)
    workspace.save_input_snapshot(snapshot, f"input-{snapshot['leaf_id']}.json")
    return snapshot
