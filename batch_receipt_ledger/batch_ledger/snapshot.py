from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .hashing import file_fingerprint, sha256_json
from .engine import config_fingerprint, effective_config

MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "CURRENT"


def atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_json(value)


def fingerprint_receipt_files(input_dir: str | Path | None) -> list[dict[str, Any]]:
    if input_dir is None:
        return []
    root = Path(input_dir)
    fingerprints = []
    receipt_root = root / "receipts"
    if receipt_root.exists():
        for path in sorted(receipt_root.rglob("*.jsonl")):
            item = file_fingerprint(path)
            item["path"] = str(path.relative_to(root))
            fingerprints.append(item)
    return fingerprints


def fingerprint_state_receipts(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("receipts", [])
    return [{"path": "state:receipts", "sha256": sha256_json(raw), "records": len(raw)}]


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / CURRENT_NAME

    def commit(
        self,
        report: dict[str, Any],
        state: dict[str, Any],
        *,
        input_dir: str | Path | None = None,
        command_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        config = effective_config(state)
        config_hash = config_fingerprint(config)
        receipt_fingerprints = (
            fingerprint_receipt_files(input_dir)
            if input_dir is not None
            else fingerprint_state_receipts(state)
        )
        snapshot_id = (
            f"snapshot-v{int(state.get('global_version', 1))}-"
            f"{config_hash.split(':', 1)[1][:10]}-"
            f"{report['report_hash'].split(':', 1)[1][:12]}"
        )
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.snapshots_dir))
        try:
            stages = {
                "parse": {
                    "schema_version": "parse-v1",
                    "workers": report["workers"],
                    "planned_batches": report["planned_batches"],
                    "failures": report["failures"],
                    "receipts": report["receipts"],
                },
                "attribution": {
                    "schema_version": "attribution-v1",
                    "decisions": report["decisions"],
                    "worker_aggregates": report["worker_aggregates"],
                    "batch_stats": report["batch_stats"],
                },
                "resources": {
                    "schema_version": "resources-v1",
                    "resource_entries": report["resource_entries"],
                    "invalidation": report["invalidation"],
                },
                "ledger": report,
            }
            products: dict[str, Any] = {}
            for stage_name, product in stages.items():
                file_name = f"{stage_name}.json"
                digest = atomic_write_json(staging / file_name, product)
                products[stage_name] = {
                    "version": product["schema_version"] if isinstance(product, dict) and "schema_version" in product else f"{stage_name}-v1",
                    "file": file_name,
                    "sha256": digest,
                }

            manifest = {
                "manifest_version": "snapshot-manifest-v1",
                "snapshot_id": snapshot_id,
                "global_version": int(state.get("global_version", 1)),
                "config_hash": config_hash,
                "receipt_fingerprints": receipt_fingerprints,
                "stage_products": products,
                "report_hash": report["report_hash"],
                "command_results": command_results or [],
            }
            atomic_write_json(staging / MANIFEST_NAME, manifest)
            final_dir = self.snapshots_dir / snapshot_id
            if final_dir.exists():
                # Content-addressed identical commit may already be present.
                import shutil

                shutil.rmtree(staging)
            else:
                os.replace(staging, final_dir)

            pointer = {"snapshot_id": snapshot_id, "manifest": f"snapshots/{snapshot_id}/{MANIFEST_NAME}"}
            atomic_write_json(self.current_path, pointer)
            return manifest
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)

    def current_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        return json.loads(self.current_path.read_text(encoding="utf-8"))

    def current_manifest(self) -> dict[str, Any] | None:
        pointer = self.current_pointer()
        if not pointer:
            return None
        manifest_path = self.root / pointer["manifest"]
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_current_report(self) -> dict[str, Any] | None:
        manifest = self.current_manifest()
        if not manifest:
            return None
        ledger_path = self.root / manifest["stage_products"]["ledger"]["file"]
        # Stage paths are relative to the snapshot directory.
        snapshot_dir = self.root / Path(self.current_pointer()["manifest"]).parent
        ledger_path = snapshot_dir / manifest["stage_products"]["ledger"]["file"]
        if not ledger_path.exists():
            return None
        return json.loads(ledger_path.read_text(encoding="utf-8"))

    def recovery_decision(
        self,
        *,
        expected_global_version: int,
        expected_config_hash: str,
        expected_receipt_fingerprints: list[dict[str, Any]],
        verify_hashes: bool = True,
    ) -> dict[str, Any]:
        manifest = self.current_manifest()
        if manifest is None:
            return {"resume": False, "decision": "rebuild", "reason": "no_committed_manifest"}
        if int(manifest.get("global_version", -1)) != int(expected_global_version):
            return {"resume": False, "decision": "rebuild", "reason": "global_version_changed", "manifest": manifest}
        if manifest.get("config_hash") != expected_config_hash:
            return {"resume": False, "decision": "rebuild", "reason": "config_hash_changed", "manifest": manifest}
        if manifest.get("receipt_fingerprints") != expected_receipt_fingerprints:
            return {"resume": False, "decision": "rebuild", "reason": "receipt_fingerprints_changed", "manifest": manifest}

        if verify_hashes:
            snapshot_dir = self.root / Path(self.current_pointer()["manifest"]).parent
            for product in manifest.get("stage_products", {}).values():
                path = snapshot_dir / product["file"]
                if not path.exists():
                    return {"resume": False, "decision": "rebuild", "reason": "stage_file_missing", "manifest": manifest}
                actual = sha256_json(json.loads(path.read_text(encoding="utf-8")))
                if actual != product["sha256"]:
                    return {"resume": False, "decision": "rebuild", "reason": "stage_hash_mismatch", "manifest": manifest}
        return {"resume": True, "decision": "resume", "reason": "manifest_and_fingerprints_match", "manifest": manifest}
