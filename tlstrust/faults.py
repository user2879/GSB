"""Deterministic, explicitly reported fault injection for diagnostics."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .errors import E_CHAIN, E_REVOKE, E_SNAPSHOT, WorkbenchError
from .handshake import capture_real_handshake
from .ledger import Ledger
from .trust import demo_revisions, parse_trust_document
from .workspace import Workspace, capture_snapshot, prepare_workspace, run_validation


INJECTION_KINDS = ("chain", "revoke", "snapshot")


def _record_error(workspace: Workspace, kind: str, error: WorkbenchError) -> dict[str, Any]:
    workspace.event_log().error(f"fault_injection.{kind}", error)
    result = error.to_dict()
    result["injection"] = kind
    result["log_path"] = str(workspace.log_path)
    return result


def inject_chain(workspace: Workspace) -> dict[str, Any]:
    try:
        capture_real_handshake(workspace.family, inject_chain_fingerprint=True)
    except WorkbenchError as exc:
        if exc.code != E_CHAIN:
            raise
        return _record_error(workspace, "chain", exc)
    raise AssertionError("chain injection did not raise E_CHAIN")


def inject_revoke(workspace: Workspace) -> dict[str, Any]:
    revisions = [revision.content() for revision in demo_revisions(workspace.family)]
    # A syntactically valid hexadecimal serial that is deliberately absent from
    # the certificate family.
    revisions[2]["revoked_serial_numbers"] = ["DEADBEEFDEADBEEF"]
    try:
        parse_trust_document({"revisions": revisions}, workspace.family)
    except WorkbenchError as exc:
        if exc.code != E_REVOKE:
            raise
        return _record_error(workspace, "revoke", exc)
    raise AssertionError("revocation injection did not raise E_REVOKE")


def _ensure_session(workspace: Workspace, ledger: Ledger) -> dict[str, Any]:
    snapshots = ledger.list_session_snapshots(current_only=True)
    if snapshots:
        return ledger.load_session_snapshot(snapshots[-1]["content_digest"])
    current = ledger.current_trust_document()
    if current is None:
        raise WorkbenchError(E_SNAPSHOT, "workspace has no current trust revisions", field_path="trust_version")
    snapshot = capture_snapshot(workspace)
    run = run_validation(ledger, current, snapshot, source_path=str(workspace.trust_path))
    return run["session"]["payload"]


def inject_snapshot(workspace: Workspace) -> dict[str, Any]:
    ledger = workspace.open_ledger()
    try:
        payload = _ensure_session(workspace, ledger)
        tampered = copy.deepcopy(payload)
        tampered["trust_version"] = int(tampered["trust_version"]) + 1000
        try:
            ledger.validate_session_payload(tampered)
        except WorkbenchError as exc:
            if exc.code != E_SNAPSHOT:
                raise
            return _record_error(workspace, "snapshot", exc)
    finally:
        ledger.close()
    raise AssertionError("snapshot injection did not raise E_SNAPSHOT")


def run_injection(
    kind: str,
    root: str | Path,
    *,
    fresh: bool = False,
    key_bits: int = 2048,
    port: int = 0,
) -> dict[str, Any]:
    if kind not in INJECTION_KINDS:
        raise ValueError(f"injection kind must be one of {INJECTION_KINDS}")
    workspace = prepare_workspace(
        root,
        fresh=fresh,
        key_bits=key_bits,
        port=port,
    )
    if kind == "chain":
        return inject_chain(workspace)
    if kind == "revoke":
        return inject_revoke(workspace)
    return inject_snapshot(workspace)
