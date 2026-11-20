from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import fail
from .formats import CURRENT_VERSION, Manifest, decode_manifest_data
from .ledger import Ledger, verify_manifest_against_ledger
from .view import canonical_view, first_difference, view_bytes, view_digest


@dataclass(frozen=True)
class NormalizedManifest:
    view: dict[str, Any]
    view_bytes: bytes
    view_digest: str
    report: dict[str, Any]


def _filled_fields(version: int) -> list[str]:
    common = ["manifest_id", "partition_mode", "parent_manifest"]
    if version == 1:
        return sorted(["rng_state", "shards[*].momentum_digest", "shards[*].data_cursor", *common])
    if version == 2:
        return sorted(["shards[*].data_cursor", *common])
    return []


def upgrade_manifest_to_current(
    manifest: Manifest,
    ledger: Ledger,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[Manifest, list[str], int]:
    path_text = str(manifest_path or manifest.source_path) if (manifest_path or manifest.source_path) else None
    verified = verify_manifest_against_ledger(manifest, ledger, manifest_path=path_text)
    truth = ledger.entry(manifest.global_step)
    upgraded = decode_manifest_data(
        {"format_version": CURRENT_VERSION, **truth.view},
        source_path=manifest.source_path,
    )
    return upgraded, _filled_fields(manifest.version), verified


def _restore_failure(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    manifest_path: str | None,
    reason: str,
) -> None:
    mismatch = first_difference(expected, actual) or first_difference(actual, expected) or {}
    fail(
        "E_RESTORE",
        "normalized metadata view is not byte-identical to the truth view",
        path=mismatch.get("field_path"),
        manifest_path=manifest_path,
        shard_index=mismatch.get("shard_index"),
        details={"reason": reason, "first_mismatch": mismatch},
    )


def normalize_manifest(
    manifest: Manifest,
    ledger: Ledger,
    *,
    expected_view: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> NormalizedManifest:
    path_text = str(manifest_path or manifest.source_path) if (manifest_path or manifest.source_path) else None
    verified_digest_count = verify_manifest_against_ledger(
        manifest, ledger, manifest_path=path_text
    )
    truth = ledger.entry(manifest.global_step)

    # Rebuild a fresh complete view from the ledger truth. Historical fields are
    # filled, never merged into a mutable copy of the source record.
    rebuilt = canonical_view(truth.view)
    rebuilt_bytes = view_bytes(rebuilt)
    rebuilt_digest = view_digest(rebuilt)
    truth_bytes = view_bytes(truth.view)
    if rebuilt_bytes != truth_bytes:
        _restore_failure(
            truth.view,
            rebuilt,
            manifest_path=path_text,
            reason="deterministic_rebuild_mismatch",
        )

    if expected_view is not None:
        expected_bytes = view_bytes(expected_view)
        if rebuilt_bytes != expected_bytes:
            _restore_failure(
                expected_view,
                rebuilt,
                manifest_path=path_text,
                reason="expected_current_view_mismatch",
            )

    filled = _filled_fields(manifest.version)
    report = {
        "ok": True,
        "operation": "normalize",
        "manifest_path": path_text,
        "source_format_version": manifest.version,
        "target_format_version": CURRENT_VERSION,
        "upgrade_trace": [
            {
                "manifest_path": path_text,
                "from_version": manifest.version,
                "to_version": CURRENT_VERSION,
                "mode": "in_memory_whole_replace",
            }
        ],
        "filled_fields": filled,
        "verified_digest_count": verified_digest_count,
        "view_digest": rebuilt_digest,
        "view_sha256": hashlib.sha256(rebuilt_bytes).hexdigest(),
        "byte_length": len(rebuilt_bytes),
        "whole_view_replaced": True,
    }
    return NormalizedManifest(
        view=rebuilt,
        view_bytes=rebuilt_bytes,
        view_digest=rebuilt_digest,
        report=report,
    )


def compare_manifests(
    left: Manifest,
    right: Manifest,
    ledger: Ledger,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    left_normalized = normalize_manifest(left, ledger)
    right_normalized = normalize_manifest(right, ledger)

    left_view = left_normalized.view
    right_view = right_normalized.view
    return compare_views(
        left_view,
        right_view,
        left_label=left_label,
        right_label=right_label,
        left_view_digest=left_normalized.view_digest,
        right_view_digest=right_normalized.view_digest,
    )


def compare_views(
    left_view: dict[str, Any],
    right_view: dict[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
    left_view_digest: str | None = None,
    right_view_digest: str | None = None,
) -> dict[str, Any]:
    left_bytes = view_bytes(left_view)
    right_bytes = view_bytes(right_view)
    if left_bytes != right_bytes:
        mismatch = first_difference(left_view, right_view) or first_difference(right_view, left_view) or {}
        fail(
            "E_DIVERGE",
            "normalized metadata views diverge",
            path=mismatch.get("field_path"),
            shard_index=mismatch.get("shard_index"),
            details={
                "first_mismatch": mismatch,
                "left": left_label,
                "right": right_label,
                "left_view_digest": left_view_digest or view_digest(left_view),
                "right_view_digest": right_view_digest or view_digest(right_view),
            },
        )
    return {
        "ok": True,
        "operation": "compare",
        "equal": True,
        "left": left_label,
        "right": right_label,
        "global_step": left_view["global_step"],
        "shard_count": left_view["shard_count"],
        "partition_mode": left_view["partition_mode"],
        "view_digest": left_view_digest or view_digest(left_view),
        "compared_fields": [
            "global_step",
            "rng_state",
            "shards[*].parameter_digest",
            "shards[*].momentum_digest",
            "shards[*].data_cursor",
        ],
    }


def compare_upgrade_paths(
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    direct_manifest: Manifest,
    ledger: Ledger,
) -> dict[str, Any]:
    v1_report = normalize_manifest(v1_manifest, ledger, manifest_path=v1_manifest.source_path)
    v2_report = normalize_manifest(v2_manifest, ledger, manifest_path=v2_manifest.source_path)
    direct_report = normalize_manifest(direct_manifest, ledger, manifest_path=direct_manifest.source_path)

    v1_vs_direct = compare_manifests(
        v1_manifest, direct_manifest, ledger, left_label="v1-upgrade", right_label="v3-direct"
    )
    v2_vs_direct = compare_manifests(
        v2_manifest, direct_manifest, ledger, left_label="v2-upgrade", right_label="v3-direct"
    )
    byte_identical = (
        v1_report.view_bytes == direct_report.view_bytes == v2_report.view_bytes
    )
    if not byte_identical:
        left = v1_report.view if v1_report.view_bytes != direct_report.view_bytes else v2_report.view
        mismatch = first_difference(left, direct_report.view) or {}
        fail(
            "E_DIVERGE",
            "upgrade and direct views are not byte-identical",
            path=mismatch.get("field_path"),
            shard_index=mismatch.get("shard_index"),
            details={"first_mismatch": mismatch},
        )
    return {
        "ok": True,
        "operation": "compare_upgrade_paths",
        "byte_identical": True,
        "v1_vs_direct": v1_vs_direct,
        "v2_vs_direct": v2_vs_direct,
        "view_digest": direct_report.view_digest,
        "filled_fields": {
            "v1": v1_report.report["filled_fields"],
            "v2": v2_report.report["filled_fields"],
            "v3": direct_report.report["filled_fields"],
        },
    }
