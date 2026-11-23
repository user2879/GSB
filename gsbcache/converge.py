"""Post-restart persisted-replica digest convergence check."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .content import artifact_filename
from .errors import ConvergeError, LedgerError
from .ledger import Ledger
from .spec import load_spec
from .util import sha256_bytes


def _loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value if value is not None else default


def check_convergence(
    spec_path: str | Path,
    ledger_path: str | Path,
    nodes_root: str | Path,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    try:
        ledger = Ledger(ledger_path, spec, read_only=True)
    except LedgerError as exc:
        raise ConvergeError(
            "ledger cannot be read for convergence validation",
            field=exc.details.get("field", "ledger"),
            expected=exc.details.get("expected"),
            actual=exc.details.get("actual"),
            differences=exc.details.get("differences", []),
        ) from exc
    nodes_root = Path(nodes_root)
    try:
        rows = ledger.conn.execute("SELECT * FROM leases ORDER BY node_id,lease_id").fetchall()
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            node_id = item["node_id"]
            keys = _loads(item["artifact_keys"], [])
            digests = _loads(item["digests"], {})
            for key in keys:
                pair = pairs.setdefault(node_id, {}).setdefault(
                    key,
                    {"expected": digests.get(key), "ledger_digests": set(), "lease_ids": []},
                )
                pair["lease_ids"].append(item["lease_id"])
                pair["ledger_digests"].add(digests.get(key))
        deviations: list[dict[str, Any]] = []
        checked = 0
        node_report: dict[str, dict[str, Any]] = {}
        for node_id in sorted(pairs):
            node_dir = nodes_root / node_id
            node_checked = 0
            node_deviations: list[dict[str, Any]] = []
            for key in sorted(pairs[node_id]):
                pair = pairs[node_id][key]
                expected = pair["expected"]
                path = node_dir / "artifacts" / artifact_filename(key)
                checked += 1
                node_checked += 1
                if not path.exists():
                    deviation = {
                        "node_id": node_id,
                        "artifact_key": key,
                        "field": "persisted_content",
                        "expected": expected,
                        "actual": None,
                        "reason": "persisted_artifact_missing",
                        "path": str(path),
                    }
                    deviations.append(deviation)
                    node_deviations.append(deviation)
                    continue
                actual = sha256_bytes(path.read_bytes())
                if len(pair["ledger_digests"]) > 1:
                    deviation = {
                        "node_id": node_id,
                        "artifact_key": key,
                        "field": "ledger_digests",
                        "expected": sorted(value for value in pair["ledger_digests"] if value is not None),
                        "actual": actual,
                        "reason": "ledger_records_disagree_for_node_and_artifact",
                        "path": str(path),
                    }
                    deviations.append(deviation)
                    node_deviations.append(deviation)
                if actual != expected:
                    deviation = {
                        "node_id": node_id,
                        "artifact_key": key,
                        "field": "digest",
                        "expected": expected,
                        "actual": actual,
                        "reason": "persisted_content_digest_mismatch_after_restart",
                        "path": str(path),
                    }
                    deviations.append(deviation)
                    node_deviations.append(deviation)
            node_report[node_id] = {
                "checked_keys": node_checked,
                "ok": not node_deviations,
                "deviating_artifact_keys": sorted({item["artifact_key"] for item in node_deviations}),
                "deviations": node_deviations,
            }
        if deviations:
            deviating_nodes = sorted({item["node_id"] for item in deviations})
            key_sets = {
                node_id: sorted(
                    {item["artifact_key"] for item in deviations if item["node_id"] == node_id}
                )
                for node_id in deviating_nodes
            }
            first = sorted(deviations, key=lambda item: (item["node_id"], item["artifact_key"], item["field"]))[0]
            raise ConvergeError(
                "persisted replica digests do not converge with the arbitration ledger",
                artifact_key=first["artifact_key"],
                node_id=first["node_id"],
                field=first["field"],
                expected=first["expected"],
                actual=first["actual"],
                reason=first["reason"],
                deviating_nodes=deviating_nodes,
                deviating_artifact_keys=key_sets,
                deviations=deviations,
                nodes=node_report,
            )
        return {
            "ok": True,
            "code": "OK",
            "ledger": str(ledger_path),
            "nodes_root": str(nodes_root),
            "checked_artifacts": checked,
            "nodes": node_report,
            "conclusion": "all persisted node replicas converge with ledger digests",
        }
    finally:
        ledger.close()
