"""Independent replay/verification of the latest committed session."""

from __future__ import annotations

from pathlib import Path

from .errors import BizError, ErrorCode
from .manifest import canonical_json, load_manifest
from .planner import build_plan, plan_fingerprint
from .runtime import load_node_declaration
from .session import SessionStore


def _raise_replay(reason: str, **details) -> None:
    payload = {"reason": reason, **details}
    raise BizError(
        ErrorCode.E_REPLAY,
        f"replay not reproducible: {reason}",
        payload,
    )


def replay_latest(
    db_path: str | Path,
    manifest_path: str | Path,
    nodes_path: str | Path,
) -> dict:
    """Recompute the plan and node digests for the latest committed version.

    Reads the SQLite session store independently and compares stored plan and
    receipts against freshly recomputed values. On mismatch the first
    inconsistent node and field are reported deterministically.
    """
    with SessionStore(db_path) as store:
        committed = store.latest_committed_version()
        if committed is None:
            _raise_replay("no committed configuration version found in session database")
        config_version = committed["config_version"]
        stored_plan = committed["plan"]

        manifest = load_manifest(manifest_path)
        if manifest["format_version"] != 3:
            _raise_replay(
                "current manifest is not the required v3 format",
                manifest_format_version=manifest["format_version"],
            )
        manifest_versions = {entry["config_version"] for entry in manifest["entries"]}
        if config_version not in manifest_versions:
            _raise_replay(
                "committed config version is missing from the current manifest",
                missing_config_version=config_version,
                available_config_versions=sorted(manifest_versions),
            )

        node_specs = load_node_declaration(nodes_path)
        node_ids = [spec["id"] for spec in node_specs]
        stored_nodes = {slice_payload["node"] for slice_payload in stored_plan["slices"]}
        declared_nodes = set(node_ids)
        downgraded = sorted(stored_nodes - declared_nodes)
        if downgraded:
            _raise_replay(
                "node downgrade detected: nodes present in the stored plan are absent "
                "from the current declaration",
                downgraded_nodes=downgraded,
            )

        recomputed = build_plan(manifest, node_ids)
        if plan_fingerprint(recomputed) != plan_fingerprint(stored_plan):
            first_diff = _first_plan_diff(stored_plan, recomputed)
            _raise_replay(
                "recomputed distribution plan differs from the stored plan",
                first_diff=first_diff,
            )

        session_id = committed["session_id"]
        latest_receipts = store.latest_successful_receipts(session_id)
        receipt_diffs = _compare_receipts(stored_plan, latest_receipts)
        if receipt_diffs:
            _raise_replay(
                "stored receipts do not match recomputed expected digests",
                first_diff=receipt_diffs[0],
                all_diffs=receipt_diffs,
            )

        return {
            "ok": True,
            "config_version": config_version,
            "session_id": session_id,
            "plan_fingerprint": plan_fingerprint(recomputed),
            "verified_nodes": sorted(stored_nodes),
        }


def _first_plan_diff(stored: dict, recomputed: dict) -> dict:
    if stored.get("config_version") != recomputed.get("config_version"):
        return {
            "node": None,
            "field": "config_version",
            "stored": stored.get("config_version"),
            "recomputed": recomputed.get("config_version"),
        }
    stored_slices = {item["node"]: item for item in stored["slices"]}
    recomputed_slices = {item["node"]: item for item in recomputed["slices"]}
    for node_id in sorted(set(stored_slices) | set(recomputed_slices)):
        if node_id not in stored_slices:
            return {"node": node_id, "field": "slice", "stored": None,
                    "recomputed": "present"}
        if node_id not in recomputed_slices:
            return {"node": node_id, "field": "slice", "stored": "present",
                    "recomputed": None}
        left, right = stored_slices[node_id], recomputed_slices[node_id]
        for field in ("entries", "expected_digest"):
            if canonical_json(left.get(field)) != canonical_json(right.get(field)):
                return {
                    "node": node_id,
                    "field": f"slice.{field}",
                    "stored": left.get(field),
                    "recomputed": right.get(field),
                }
    if canonical_json(stored.get("derivations")) != canonical_json(
        recomputed.get("derivations")
    ):
        return {
            "node": None,
            "field": "derivations",
            "stored": stored.get("derivations"),
            "recomputed": recomputed.get("derivations"),
        }
    return {"node": None, "field": "plan", "stored": "diff", "recomputed": "diff"}


def _compare_receipts(plan: dict, latest_receipts: dict) -> list[dict]:
    diffs = []
    for slice_payload in plan["slices"]:
        node_id = slice_payload["node"]
        expected = slice_payload["expected_digest"]
        receipt = latest_receipts.get(node_id)
        if receipt is None:
            diffs.append(
                {
                    "node": node_id,
                    "field": "receipt",
                    "expected_digest": expected,
                    "actual_digest": None,
                    "reason": "no successful receipt stored",
                }
            )
            continue
        actual = receipt.get("digest")
        if actual != expected:
            diffs.append(
                {
                    "node": node_id,
                    "field": "receipt.digest",
                    "expected_digest": expected,
                    "actual_digest": actual,
                    "reason": "digest mismatch",
                }
            )
    return diffs
