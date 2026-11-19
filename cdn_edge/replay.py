from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .content import ContentRepository, content_digest_from_versions
from .errors import PipelineError
from .fixed import FixedGrid
from .ledger import Ledger
from .orchestrator import run_session
from .spec import SourceUpdateSpec, parse_spec
from .verify import first_difference


def _source_updates_from_payload(payload: list[dict[str, Any]]) -> list[SourceUpdateSpec]:
    updates: list[SourceUpdateSpec] = []
    for raw in payload:
        dependencies = None
        if "derivation" in raw:
            from .spec import Dependency

            derivation = raw.get("derivation") or {}
            dependencies = tuple(
                Dependency(str(item["path"]), int(item["coefficient"]))
                for item in derivation.get("inputs", [])
            )
        body = None
        if "body_hex" in raw:
            body = bytes.fromhex(str(raw["body_hex"]))
        updates.append(
            SourceUpdateSpec(
                at_ms=int(raw["at_ms"]),
                path=str(raw["path"]),
                etag=raw.get("etag"),
                body=body,
                ttl_seconds=raw.get("ttl_seconds"),
                dependencies=dependencies,
            )
        )
    return updates


def _node_for_metric_path(path: str) -> str:
    marker = "$.nodes."
    if path.startswith(marker):
        return path[len(marker) :].split(".", 1)[0]
    if path.startswith("$.aggregate"):
        return "__aggregate__"
    if path.startswith("$.derived_weights"):
        return "derived_weights"
    if path.startswith("$.requests["):
        return "requests"
    return "batch"


def _replay_mismatch(
    *,
    session_id: str,
    batch_id: int,
    node_id: str,
    field: str,
    expected: Any,
    actual: Any,
    basis: str,
) -> PipelineError:
    return PipelineError(
        "E_REPLAY",
        "replayed batch differs from the recorded snapshot lineage",
        path=f"session={session_id}/batch={batch_id}/node={node_id}",
        field=field,
        details={
            "first_inconsistency": {
                "batch_id": batch_id,
                "node_id": node_id,
                "field": field,
                "expected": expected,
                "actual": actual,
            },
            "basis": basis,
        },
    )


def replay_snapshot(
    ledger_path: str,
    snapshot_id: str,
    *,
    grid_override: FixedGrid | None = None,
    current_spec_path: str | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    ledger = Ledger(ledger_path, create=False)
    try:
        snapshot = ledger.get_snapshot(snapshot_id)
        ledger.verify_snapshot_integrity(snapshot)
        payload = snapshot["payload"]
        snapshot_grid = FixedGrid(int(snapshot["grid_power"]))
        if grid_override is not None and grid_override != snapshot_grid:
            raise PipelineError(
                "E_REPLAY",
                "precision grid differs from the snapshot; replay is not reproducible",
                path=snapshot_id,
                field="grid",
                details={
                    "reason": "grid_mismatch",
                    "snapshot_grid": snapshot_grid.as_dict(),
                    "requested_grid": grid_override.as_dict(),
                    "not_reproducible_basis": "every fixed-point metric is defined on the snapshot grid",
                },
            )
        session_id = str(snapshot["session_id"])
        session = ledger.get_session(session_id)
        spec = parse_spec(
            payload["spec"],
            grid_power=snapshot_grid.power,
            source_name=f"snapshot:{snapshot_id}",
        )
        updates = _source_updates_from_payload(payload.get("source_updates", []))
        request_index = int(snapshot["cursor"].get("next_request_index", 0))
        if request_index > len(spec.requests):
            raise PipelineError(
                "E_REPLAY",
                "snapshot cursor exceeds the recorded request sequence",
                path=snapshot_id,
                field="cursor.next_request_index",
                details={"cursor": request_index, "request_count": len(spec.requests)},
            )
        if current_spec_path is not None:
            from .spec import load_spec

            current_spec = load_spec(current_spec_path, grid_power=snapshot_grid.power)
            at_ms = (
                spec.requests[request_index].at_ms
                if request_index < len(spec.requests)
                else (spec.requests[-1].at_ms if spec.requests else 0)
            )
            current_repo = ContentRepository(current_spec)
            for update in updates:
                current_repo.apply_update(update)
            current_versions = current_repo.versions_at(at_ms)
            current_digest = content_digest_from_versions(current_versions)
            if current_digest != snapshot["content_digest"]:
                raise PipelineError(
                    "E_REPLAY",
                    "content version set differs from the snapshot; replay is not reproducible",
                    path=snapshot_id,
                    field="content_digest",
                    details={
                        "reason": "content_digest_mismatch",
                        "snapshot_content_digest": snapshot["content_digest"],
                        "current_content_digest": current_digest,
                        "at_ms": at_ms,
                        "not_reproducible_basis": "the request sequence was recorded against a different effective content-version set",
                    },
                )
        node_states = payload.get("node_states", {})
        if work_dir is None:
            manager = tempfile.TemporaryDirectory(prefix="cdn-edge-replay-")
            replay_work = Path(manager.name)
        else:
            manager = None
            replay_work = Path(work_dir)
        try:
            result = run_session(
                spec,
                ledger_path,
                work_dir=replay_work,
                record_ledger=False,
                seed_node_states=node_states,
                start_request_index=request_index,
                source_updates=updates,
            )
        finally:
            if manager is not None:
                manager.cleanup()
        replayed_batches = result["batches"]
        expected_batch_ids = [
            int(row["batch_id"]) for row in ledger.get_batch_rows(session_id) if int(row["batch_id"]) >= int(snapshot["batch_id"]) + (1 if snapshot["batch_id"] else 0)
        ]
        if int(snapshot["batch_id"]) == 0:
            expected_batch_ids = [int(row["batch_id"]) for row in ledger.get_batch_rows(session_id)]
        replayed_ids = [int(batch["batch_id"]) for batch in replayed_batches]
        if replayed_ids != expected_batch_ids:
            first_expected = expected_batch_ids[0] if expected_batch_ids else None
            first_actual = replayed_ids[0] if replayed_ids else None
            raise _replay_mismatch(
                session_id=session_id,
                batch_id=first_expected if first_expected is not None else 0,
                node_id="batch",
                field="batch_ids",
                expected=expected_batch_ids,
                actual=replayed_ids,
                basis="replay must produce every batch after the snapshot cursor in the same order",
            )
        for replay_batch in replayed_batches:
            batch_id = int(replay_batch["batch_id"])
            recorded = ledger.get_batch(session_id, batch_id)
            recorded_requests = ledger.get_batch_requests(session_id, batch_id)
            recorded_request_dicts = [record.as_dict() for record in recorded_requests]
            diff = first_difference(replay_batch["metrics"], recorded["metrics"])
            if diff:
                path = str(diff.get("path", "$"))
                raise _replay_mismatch(
                    session_id=session_id,
                    batch_id=batch_id,
                    node_id=_node_for_metric_path(path),
                    field=path,
                    expected=diff.get("expected", diff.get("expected_length", diff.get("expected_keys"))),
                    actual=diff.get("actual", diff.get("actual_length", diff.get("actual_keys"))),
                    basis="recomputed fixed-point statistics and error bounds differ from the immutable snapshot lineage",
                )
            diff = first_difference(replay_batch["requests"], recorded_request_dicts)
            if diff:
                path = str(diff.get("path", "$"))
                request_index_in_batch = 0
                if path.startswith("$.requests") or path.startswith("$["):
                    try:
                        if path.startswith("$["):
                            request_index_in_batch = int(path[2:].split("]", 1)[0])
                    except ValueError:
                        request_index_in_batch = 0
                node_id = (
                    recorded_request_dicts[request_index_in_batch]["node_id"]
                    if recorded_request_dicts and request_index_in_batch < len(recorded_request_dicts)
                    else "requests"
                )
                raise _replay_mismatch(
                    session_id=session_id,
                    batch_id=batch_id,
                    node_id=node_id,
                    field=path,
                    expected=diff.get("expected"),
                    actual=diff.get("actual"),
                    basis="per-request hit source, version, TTL or cache entry version differs",
                )
            if replay_batch["content_digest"] != recorded["content_digest"]:
                raise _replay_mismatch(
                    session_id=session_id,
                    batch_id=batch_id,
                    node_id="session",
                    field="content_digest",
                    expected=recorded["content_digest"],
                    actual=replay_batch["content_digest"],
                    basis="replayed effective content version set differs from the recorded batch",
                )
            if replay_batch["cursor"] != recorded["cursor"]:
                raise _replay_mismatch(
                    session_id=session_id,
                    batch_id=batch_id,
                    node_id="session",
                    field="cursor",
                    expected=recorded["cursor"],
                    actual=replay_batch["cursor"],
                    basis="progress cursor differs from the recorded batch",
                )
        return {
            "ok": True,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "restored_from_batch": int(snapshot["batch_id"]),
            "grid": snapshot_grid.as_dict(),
            "content_digest": snapshot["content_digest"],
            "checked_batches": [int(batch["batch_id"]) for batch in replayed_batches],
            "checked_requests": sum(len(batch["requests"]) for batch in replayed_batches),
            "first_inconsistency": None,
            "conclusion": "all replayed batches, node metrics, error bounds and request records match",
            "not_reproducible_basis": None,
        }
    finally:
        ledger.close()
